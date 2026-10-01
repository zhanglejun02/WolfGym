"""Durable, bounded requests for the combination host; no network implementation.

A transport performs exactly one attempt. Its frozen token limits/pricing must
justify the per-attempt reservation before this can bound real provider spend.
Unknown charges retain the full reservation. Uncommitted replies are ambiguous,
never silently reissued after process death. One journal owner at a time.
"""
from dataclasses import dataclass, field
import copy
import fcntl
import json
from pathlib import Path
import sqlite3
import time
import uuid

from wolfgym.evaluation.combination_public import canonical_json, digest


class RequestLimitError(RuntimeError):
    pass


class AmbiguousRequestError(RuntimeError):
    pass


class RetryableTransportError(RuntimeError):
    """The transport explicitly classifies a transient failure, not model JSON."""


class TransportFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Reply:
    text: str
    cost_microusd: int | None = None
    usage: dict = field(default_factory=dict)


def nonnegative_integer(value):
    return type(value) is int and value >= 0


class RequestJournal:
    def __init__(self, path, *, contract, max_attempts, max_cost_microusd,
                 reservation_microusd, retries=2):
        if not all(nonnegative_integer(x) for x in
                   (max_attempts, max_cost_microusd, reservation_microusd, retries)):
            raise ValueError('limits must be nonnegative integers')
        if reservation_microusd == 0 or retries > 2:
            raise ValueError('positive reservation and at most two retries required')
        self.path = Path(path)
        self.limits = dict(max_attempts=max_attempts, max_cost_microusd=max_cost_microusd,
                           reservation_microusd=reservation_microusd, retries=retries)
        self.declaration = {'version': 'combination-request-journal-v1',
                            'contract': copy.deepcopy(contract), **self.limits}
        self.db = self.lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(self.path.suffix+'.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.db = sqlite3.connect(self.path)
            self.db.row_factory = sqlite3.Row
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, declaration TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, request_key TEXT NOT NULL, payload TEXT NOT NULL,
                    state TEXT NOT NULL, charged INTEGER NOT NULL, response TEXT,
                    error_type TEXT, elapsed_seconds REAL, created_unix REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS request_lookup ON attempts(request_key, id);
            ''')
            declaration = canonical_json(self.declaration)
            old = self.db.execute('SELECT declaration FROM metadata WHERE id=1').fetchone()
            if old and old[0] != declaration:
                raise ValueError('request contract or limits changed')
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO metadata VALUES(1,?)', (declaration,))
                self.db.execute('INSERT OR IGNORE INTO metadata VALUES(2,?)', (uuid.uuid4().hex,))
            self.identity = self.db.execute('SELECT declaration FROM metadata WHERE id=2').fetchone()[0]
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def __exit__(self, *args):
        self.close()

    def totals(self):
        row = self.db.execute('SELECT COUNT(*), COALESCE(SUM(charged),0) FROM attempts').fetchone()
        return {'attempts': row[0], 'charged_or_reserved_microusd': row[1]}

    def request(self, scope, day, payload, transport):
        if not isinstance(scope, str) or not scope or type(day) is not int or not 0 <= day < 30:
            raise ValueError('request requires an episode/controller scope and decision day')
        key, encoded = digest([scope, day]), canonical_json(payload)
        while True:
            rows = self.db.execute('SELECT * FROM attempts WHERE request_key=? ORDER BY id', (key,)).fetchall()
            if any(row['payload'] != encoded for row in rows):
                raise ValueError('request input changed during replay')
            if rows:
                last = rows[-1]
                if last['state'] == 'success':
                    return json.loads(last['response'])['text']
                if last['state'] == 'pending':
                    raise AmbiguousRequestError('attempt has no durable outcome; reconcile before retrying')
                if last['state'] != 'retryable' or len(rows) > self.limits['retries']:
                    raise TransportFailure('request failed; recorded attempts cannot be silently reset')
            total = self.totals()
            if self.db.execute("SELECT 1 FROM attempts WHERE state='overrun' LIMIT 1").fetchone():
                raise RequestLimitError('provider bound was violated; new requests are blocked')
            reservation = self.limits['reservation_microusd']
            if (total['attempts'] >= self.limits['max_attempts'] or
                    total['charged_or_reserved_microusd']+reservation > self.limits['max_cost_microusd']):
                raise RequestLimitError('global request count or cost reservation limit reached')
            with self.db:
                cursor = self.db.execute(
                    'INSERT INTO attempts(request_key,payload,state,charged,created_unix) VALUES(?,?,?,?,?)',
                    (key, encoded, 'pending', reservation, time.time()))
                attempt_id = cursor.lastrowid
            started = time.monotonic()
            try:
                # No identity, private score, journal or environment goes to transport.
                reply = transport(copy.deepcopy(payload))
                if type(reply) is not Reply or type(reply.text) is not str:
                    raise TypeError('transport must return a Reply with text')
                if reply.cost_microusd is not None and not nonnegative_integer(reply.cost_microusd):
                    raise ValueError('invalid reported provider charge')
                if type(reply.usage) is not dict or any(
                    k not in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_input_tokens')
                    or not nonnegative_integer(v) for k, v in reply.usage.items()
                ):
                    raise ValueError('usage must contain only nonnegative token counts')
                response = canonical_json({'text': reply.text, 'cost_microusd': reply.cost_microusd,
                                           'usage': reply.usage})
            except Exception as exc:
                state = 'retryable' if isinstance(exc, RetryableTransportError) else 'fatal'
                with self.db:
                    self.db.execute('UPDATE attempts SET state=?,error_type=?,elapsed_seconds=? WHERE id=?',
                                    (state, type(exc).__name__, time.monotonic()-started, attempt_id))
                if state == 'retryable':
                    continue
                raise TransportFailure('non-retryable transport failure; reservation retained') from exc
            charge = reply.cost_microusd if reply.cost_microusd is not None else reservation
            state = 'overrun' if charge > reservation else 'success'
            with self.db:
                self.db.execute('UPDATE attempts SET state=?,charged=?,response=?,elapsed_seconds=? WHERE id=?',
                                (state, charge, response, time.monotonic()-started, attempt_id))
            if state == 'overrun':
                raise RequestLimitError('provider exceeded its declared attempt bound; recorded and stopped')
            return reply.text
