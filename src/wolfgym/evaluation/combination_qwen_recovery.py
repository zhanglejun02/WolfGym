"""Explicit recovery amendment for the frozen Qwen calibration, never hidden retries."""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sqlite3
import time

from wolfgym.evaluation.combination_host import read_json
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import canonical_json, digest
from wolfgym.evaluation.combination_qwen_pilot import QwenTransport, decode_reply, RESERVATION
from wolfgym.evaluation.combination_requests import (
    RequestJournal, RequestLimitError, AmbiguousRequestError, TransportFailure)

VERSION = 'combination-qwen-explicit-recovery-v1'


def ledger_snapshot(path):
    if not Path(path).exists(): return None
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        return {'metadata': [dict(x) for x in db.execute('SELECT * FROM metadata ORDER BY id')],
                'attempts': [dict(x) for x in db.execute('SELECT * FROM attempts ORDER BY id')]}


def pooled_totals(output, manifest):
    rows = [r for j in manifest['jobs']
            for r in (ledger_snapshot(Path(output)/(j['id']+'.sqlite')) or {}).get('attempts', [])]
    return {'attempts': len(rows), 'successful_attempts': sum(r['state']=='success' for r in rows),
            'confirmed_microusd': sum(r['charged'] for r in rows if r['state']=='success'),
            'unknown_reserved_microusd': sum(r['charged'] for r in rows if r['state']!='success'),
            'charged_or_reserved_microusd': sum(r['charged'] for r in rows),
            'overrun': any(r['state']=='overrun' for r in rows)}


def verify_prefix(output, amendment):
    for job, old in amendment['original_ledgers'].items():
        new = ledger_snapshot(Path(output)/(job+'.sqlite'))
        if new is None or new['metadata'] != old['metadata'] or new['attempts'][:len(old['attempts'])] != old['attempts']:
            raise ValueError('original ledger rows/metadata changed')


def retryable_failure(path, payload):
    if not path.exists(): return False
    value = read_json(path)
    if value.get('payload_sha256') != digest(payload): raise ValueError('failure payload drift')
    status = value.get('http_status')
    return (status in (408, 429, 500, 502, 503, 504) or
            (status is None and value.get('exception_type') in (
                'URLError', 'TimeoutError', 'ConnectionError', 'ConnectionResetError',
                'RemoteDisconnected', 'IncompleteRead', 'SSLError')))


class RecoveryJournal(RequestJournal):
    """Preserves v1 declarations as history; amendment supplies pooled limits.

    Old failures are immutable. Each new attempt gets its own evidence directory.
    A response persisted before interruption is settled locally, never reissued.
    """
    def __init__(self, output, manifest, job, amendment, *, pause=time.sleep, transport_factory=QwenTransport):
        self.output, self.manifest, self.job, self.amendment = Path(output), manifest, job, amendment
        self.pause, self.transport_factory = pause, transport_factory
        super().__init__(self.output/(job['id']+'.sqlite'),
                         contract={'manifest_sha256':digest(manifest),'job':job['id']},
                         max_attempts=job['max_requests'], max_cost_microusd=job['max_microusd'],
                         reservation_microusd=RESERVATION, retries=0)

    def __enter__(self):
        with self.budget_lock():
            super().__enter__()
        try:
            verify_prefix(self.output,self.amendment)
            return self
        except BaseException:
            self.close()
            raise

    @contextmanager
    def budget_lock(self):
        with (self.output/'.recovery-budget.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def evidence(self, attempt_id):
        return self.output/'recovery_attempts'/self.job['id']/str(attempt_id)

    def stop(self, reason):
        durable_json(self.output/'recovery_stop.json', {'job':self.job['id'],'reason':reason})

    def settle(self, row, payload, transport):
        receipt = self.evidence(row['id'])/f'{digest(payload)}.json'
        reply = decode_reply(read_json(receipt))
        canonical = transport.directory/f'{digest(payload)}.json'
        raw = read_json(receipt)
        if canonical.exists() and read_json(canonical) != raw:
            raise ValueError('canonical provider receipt differs')
        durable_json(canonical, raw)
        with self.db:
            self.db.execute('UPDATE attempts SET state=?,charged=?,response=? WHERE id=?',
                            ('success', reply.cost_microusd, canonical_json(asdict(reply)), row['id']))
        return reply.text

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
                    try: return self.settle(last,payload,transport)
                    except Exception:
                        self.stop('invalid saved provider receipt'); raise
                if last['state']=='pending':
                    raise AmbiguousRequestError('pending attempt has no saved receipt; diagnose before retry')
                if canonical.exists(): raise AmbiguousRequestError('unexpected canonical receipt; no POST')
                old_count = len(self.amendment['original_ledgers'].get(self.job['id'],{}).get('attempts',[]))
                failure = (transport.directory if last['id'] <= old_count else evidence)/f'{digest(payload)}.failure.json'
                if last['state']!='fatal' or not retryable_failure(failure,payload):
                    self.stop('failure is not an eligible network error')
                    raise TransportFailure('non-network failure is not authorized for retry')
                if len(rows)>=3:
                    self.stop('two supplemental attempts exhausted')
                    raise TransportFailure('two supplemental attempts exhausted; reservations retained')
                self.pause((5,15)[len(rows)-1])
            elif canonical.exists():
                raise AmbiguousRequestError('orphan provider receipt; no POST')
            with self.budget_lock():
                if (self.output/'recovery_stop.json').exists(): raise TransportFailure('recovery is stopped')
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
                if not retryable_failure(evidence/f'{digest(payload)}.failure.json',payload):
                    self.stop('non-network/provider validation failure; reservation retained')
                    raise TransportFailure('non-network/provider validation failure') from None
                continue
            with self.db:
                self.db.execute('UPDATE attempts SET elapsed_seconds=? WHERE id=?',(time.monotonic()-started,attempt_id))
            return answer
