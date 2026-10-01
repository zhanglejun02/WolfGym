import sqlite3

import pytest

from wolfgym.evaluation.combination_requests import (
    RequestJournal, Reply, RequestLimitError, AmbiguousRequestError,
    RetryableTransportError, TransportFailure,
)


def journal(path, **kwargs):
    return RequestJournal(path, contract={'model': 'offline-mock-v1'},
                          **(dict(max_attempts=8, max_cost_microusd=800, reservation_microusd=100) | kwargs))


def test_committed_reply_replay_identity_input_drift_and_contract(tmp_path):
    path = tmp_path/'requests.sqlite'
    calls = []
    def transport(payload):
        calls.append(payload)
        return Reply('malformed model answer', 25, {'input_tokens': 10, 'output_tokens': 3})
    with journal(path) as j:
        identity = j.identity
        assert j.request('normal', 0, {'public': 1}, transport) == 'malformed model answer'
    with journal(path) as j:
        assert j.identity == identity
        assert j.request('normal', 0, {'public': 1}, transport) == 'malformed model answer'
        assert len(calls) == 1  # Invalid model JSON is not a transport retry.
        j.request('attack', 0, {'public': 1}, transport)
        assert len(calls) == 2  # Same public input, distinct episode scope.
        assert j.totals() == {'attempts': 2, 'charged_or_reserved_microusd': 50}
        with pytest.raises(ValueError, match='input changed'):
            j.request('normal', 0, {'public': 2}, transport)
    with pytest.raises(ValueError, match='contract or limits'):
        with journal(path, max_attempts=9):
            pass


def test_two_retries_are_identical_and_unknown_charges_are_retained(tmp_path):
    calls = []
    def transport(payload):
        calls.append(payload)
        if len(calls) < 3:
            raise RetryableTransportError('transient')
        return Reply('{"action":"WAIT"}', 20)
    with journal(tmp_path/'requests.sqlite') as j:
        j.request('one', 0, {'a': 1}, transport)
        assert calls == [{'a': 1}]*3
        assert j.totals() == {'attempts': 3, 'charged_or_reserved_microusd': 220}


def test_exhausted_retries_do_not_reset_after_restart(tmp_path):
    path = tmp_path/'requests.sqlite'
    calls = []
    def transport(_):
        calls.append(1)
        raise RetryableTransportError('transient')
    for _ in range(2):
        with journal(path) as j:
            with pytest.raises(TransportFailure):
                j.request('one', 0, {}, transport)
    assert len(calls) == 3


@pytest.mark.parametrize('limits', [dict(max_attempts=1), dict(max_cost_microusd=199)])
def test_global_attempt_and_money_caps_apply_before_transport(tmp_path, limits):
    calls = []
    def transport(_):
        calls.append(1)
        return Reply('ok')  # Unknown actual charge consumes its reservation.
    with journal(tmp_path/'requests.sqlite', **limits) as j:
        j.request('a', 0, {}, transport)
        with pytest.raises(RequestLimitError):
            j.request('b', 0, {}, transport)
    assert len(calls) == 1


def test_process_interruption_is_ambiguous_and_not_reissued(tmp_path):
    path = tmp_path/'requests.sqlite'
    def killed(_):
        raise KeyboardInterrupt
    with journal(path) as j:
        with pytest.raises(KeyboardInterrupt):
            j.request('one', 0, {}, killed)
    with journal(path) as j:
        with pytest.raises(AmbiguousRequestError):
            j.request('one', 0, {}, lambda _: pytest.fail('must not reissue'))
        assert j.totals()['charged_or_reserved_microusd'] == 100


def test_nonretryable_errors_and_reservation_overrun_stop(tmp_path):
    path = tmp_path/'requests.sqlite'
    def fatal(_):
        raise ValueError('fatal detail must not enter ledger')
    with journal(path) as j:
        with pytest.raises(TransportFailure):
            j.request('one', 0, {}, fatal)
        with pytest.raises(TransportFailure):
            j.request('one', 0, {}, lambda _: pytest.fail('must not retry'))
        with pytest.raises(RequestLimitError, match='exceeded'):
            j.request('two', 0, {}, lambda _: Reply('ok', 150))
        with pytest.raises(RequestLimitError, match='violated'):
            j.request('three', 0, {}, lambda _: pytest.fail('must not send'))
        assert j.totals()['charged_or_reserved_microusd'] == 250
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT error_type FROM attempts WHERE id=1').fetchone()[0] == 'ValueError'


def test_exclusive_owner_and_invalid_limits(tmp_path):
    path = tmp_path/'requests.sqlite'
    with journal(path):
        with pytest.raises(BlockingIOError):
            with journal(path):
                pass
    for bad in ({'retries': 3}, {'reservation_microusd': 0}, {'max_attempts': True}):
        with pytest.raises(ValueError):
            journal(path, **bad)
