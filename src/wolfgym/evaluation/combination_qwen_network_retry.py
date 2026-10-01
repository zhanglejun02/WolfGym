"""One additional attempt on each of two user-authorized exhausted requests."""
import json
import time
from wolfgym.evaluation.combination_host import read_json,file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest,canonical_json
from wolfgym.evaluation.combination_qwen_pilot import RESERVATION
from wolfgym.evaluation.combination_qwen_dev32 import DevelopmentJournal
from wolfgym.evaluation.combination_qwen_network_diagnostic import DiagnosticTransport
from wolfgym.evaluation.combination_qwen_recovery import pooled_totals,verify_prefix
from wolfgym.evaluation.combination_qwen_incomplete_recovery import incomplete_receipt,eligible_failure
from wolfgym.evaluation.combination_requests import RequestLimitError,TransportFailure,AmbiguousRequestError


class NetworkRetryJournal(DevelopmentJournal):
    def __init__(self,output,manifest,job,supplement,*,pause=time.sleep,transport_factory=DiagnosticTransport):
        self.supplement=supplement
        super().__init__(output,manifest,job,pause=pause,transport_factory=transport_factory)

    def __enter__(self):
        super().__enter__()
        try:
            verify_prefix(self.output,self.supplement);self.verify_incident();return self
        except BaseException:
            self.close();raise

    def verify_incident(self):
        super().verify_incident()
        if file_hash(self.output/'incomplete_recovery_stop.json')!=self.supplement['acknowledged_stop_sha256']:
            raise ValueError('old stop evidence changed')

    def request_cap(self,key):
        return 4 if key in self.supplement['one_extra_attempt_keys'].get(self.job['id'],[]) else 3

    def stop(self,reason):
        durable_json(self.output/'network_retry_stop.json',{'job':self.job['id'],'reason':reason})

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
                if len(rows)>=self.request_cap(key):
                    self.stop('authorized attempt limit exhausted')
                    raise TransportFailure('authorized attempt limit exhausted; reservations retained')
                self.pause((5,15,30)[len(rows)-1])
            elif canonical.exists():
                raise AmbiguousRequestError('orphan provider receipt; no POST')
            with self.budget_lock():
                self.verify_incident()
                if (self.output/'network_retry_stop.json').exists(): raise TransportFailure('recovery is stopped')
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
