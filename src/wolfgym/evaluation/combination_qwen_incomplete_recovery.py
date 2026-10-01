"""Bounded recovery for an empty, unmetered provider response; frozen v1 unchanged."""
import json
import time
from wolfgym.evaluation.combination_host import read_json, file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import canonical_json, digest
from wolfgym.evaluation.combination_qwen_pilot import MODEL, CANONICAL, RESERVATION
from wolfgym.evaluation.combination_qwen_recovery import RecoveryJournal, retryable_failure, pooled_totals, verify_prefix
from wolfgym.evaluation.combination_requests import RequestLimitError, AmbiguousRequestError, TransportFailure

VERSION = 'combination-qwen-empty-response-recovery-v2'


def incomplete_receipt(path):
    if not path.exists(): return False
    value = read_json(path)
    # Do not retry a valid refusal, a completed answer missing usage, or a model/price violation.
    choices = value.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict): return False
    message = choices[0].get('message')
    return (value.get('model') in (MODEL,CANONICAL) and value.get('provider')=='Alibaba'
            and isinstance(value.get('id'),str) and value['id'].startswith('gen-')
            and value.get('usage') is None and isinstance(message,dict)
            and message.get('content') in (None,'') and message.get('refusal') is None
            and choices[0].get('finish_reason')=='stop')


def eligible_failure(failure_path, payload):
    return (retryable_failure(failure_path,payload) or
            incomplete_receipt(failure_path.with_name(digest(payload)+'.json')))


class IncompleteRecoveryJournal(RecoveryJournal):
    def __init__(self, output, manifest, job, amendment, supplement, **kwargs):
        self.supplement = supplement
        super().__init__(output,manifest,job,amendment,**kwargs)

    def verify_incident(self):
        if file_hash(self.output/'recovery_stop.json') != self.supplement['acknowledged_stop_sha256']:
            raise ValueError('original incident stop marker changed')

    def __enter__(self):
        super().__enter__()
        try:
            verify_prefix(self.output,self.supplement)
            self.verify_incident()
            return self
        except BaseException:
            self.close(); raise

    def stop(self, reason):
        durable_json(self.output/'incomplete_recovery_stop.json',{'job':self.job['id'],'reason':reason})

    def request(self, scope, day, payload, transport):
        if not isinstance(scope, str) or not scope or type(day) is not int or not 0 <= day < 30:
            raise ValueError('invalid scope/day')
        key, encoded = digest([scope,day]), canonical_json(payload)
        while True:
            rows = self.db.execute('SELECT * FROM attempts WHERE request_key=? ORDER BY id',(key,)).fetchall()
            if any(r['payload'] != encoded for r in rows): raise ValueError('request input changed during replay')
            if rows and rows[-1]['state']=='success': return json.loads(rows[-1]['response'])['text']
            canonical = transport.directory/f'{digest(payload)}.json'
            if rows:
                last = rows[-1]
                evidence = self.evidence(last['id'])
                if last['state']=='pending' and (evidence/f'{digest(payload)}.json').exists():
                    if incomplete_receipt(evidence/f'{digest(payload)}.json'):
                        with self.db:
                            self.db.execute('UPDATE attempts SET state=?,error_type=? WHERE id=?',
                                            ('fatal','IncompleteProviderResponse',last['id']))
                        continue
                    try: return self.settle(last,payload,transport)
                    except Exception:
                        self.stop('invalid saved provider receipt'); raise
                if last['state']=='pending':
                    raise AmbiguousRequestError('pending attempt has no saved receipt; diagnose before retry')
                if canonical.exists(): raise AmbiguousRequestError('unexpected canonical receipt; no POST')
                old_count = len(self.amendment['original_ledgers'].get(self.job['id'],{}).get('attempts',[]))
                failure = (transport.directory if last['id'] <= old_count else evidence)/f'{digest(payload)}.failure.json'
                if last['state']!='fatal' or not eligible_failure(failure,payload):
                    self.stop('failure is neither an eligible network error nor an empty unmetered response')
                    raise TransportFailure('failure is not eligible for bounded retry')
                if len(rows)>=3:
                    self.stop('two supplemental attempts exhausted')
                    raise TransportFailure('two supplemental attempts exhausted; reservations retained')
                self.pause((5,15)[len(rows)-1])
            elif canonical.exists():
                raise AmbiguousRequestError('orphan provider receipt; no POST')
            with self.budget_lock():
                self.verify_incident()
                if (self.output/'incomplete_recovery_stop.json').exists(): raise TransportFailure('recovery is stopped')
                totals = pooled_totals(self.output,self.manifest)
                if (totals['overrun'] or totals['attempts']>=self.manifest['max_requests_total'] or
                        totals['charged_or_reserved_microusd']+RESERVATION>self.manifest['max_cost_microusd_total']):
                    raise RequestLimitError('pooled count/cost limit reached; no POST')
                with self.db:
                    cursor = self.db.execute('INSERT INTO attempts(request_key,payload,state,charged,created_unix) VALUES(?,?,?,?,?)',
                                            (key,encoded,'pending',RESERVATION,time.time()))
                attempt_id = cursor.lastrowid
            evidence = self.evidence(attempt_id)
            started = time.monotonic()
            try:
                # The frozen transport still sends exactly one POST, without SDK retries.
                self.transport_factory(transport.key,evidence)(payload)
                row = self.db.execute('SELECT * FROM attempts WHERE id=?',(attempt_id,)).fetchone()
                answer = self.settle(row,payload,transport)
            except Exception as exc:
                with self.db:
                    self.db.execute('UPDATE attempts SET state=?,error_type=?,elapsed_seconds=? WHERE id=?',
                                    ('fatal',type(exc).__name__,time.monotonic()-started,attempt_id))
                if not eligible_failure(evidence/f'{digest(payload)}.failure.json',payload):
                    self.stop('non-network/provider validation failure; reservation retained')
                    raise TransportFailure('non-network/provider validation failure') from None
                continue
            with self.db:
                self.db.execute('UPDATE attempts SET elapsed_seconds=? WHERE id=?',(time.monotonic()-started,attempt_id))
            return answer
