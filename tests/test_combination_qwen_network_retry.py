import json
from types import SimpleNamespace
from urllib.error import URLError
import pytest
import test_combination_qwen_incomplete_recovery as fixtures
from wolfgym.evaluation.combination_qwen_dev32 import DevelopmentJournal
from wolfgym.evaluation.combination_qwen_network_retry import NetworkRetryJournal
from wolfgym.evaluation.combination_qwen_network_diagnostic import safe_error_chain,DiagnosticTransport
from wolfgym.evaluation.combination_qwen_recovery import ledger_snapshot,verify_prefix
from wolfgym.evaluation.combination_host import file_hash,read_json
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_requests import TransportFailure,RequestLimitError


def case(p):
    m={'jobs':[{'id':'LL_0'}],'max_requests_total':2048,'max_cost_microusd_total':60000000}
    durable_json(p/'manifest.json',m);target=SimpleNamespace(key='mock',directory=p/'responses');calls=[]
    with DevelopmentJournal(p,m,m['jobs'][0],pause=lambda _:None,transport_factory=fixtures.factory(['empty']*3,calls)) as j:
        with pytest.raises(TransportFailure):j.request('scope',0,fixtures.base.PAYLOAD,target)
    s={'acknowledged_stop_sha256':file_hash(p/'incomplete_recovery_stop.json'),
       'original_ledgers':{'LL_0':ledger_snapshot(p/'LL_0.sqlite')},
       'one_extra_attempt_keys':{'LL_0':[digest(['scope',0])]}}
    return m,s,target


def test_success_cached_prefix_preserved(tmp_path):
    m,s,t=case(tmp_path);calls=[]
    with NetworkRetryJournal(tmp_path,m,m['jobs'][0],s,pause=lambda _:None,transport_factory=fixtures.factory(['ok'],calls)) as j:
        assert j.request('scope',0,fixtures.base.PAYLOAD,t)
        assert j.request('scope',0,fixtures.base.PAYLOAD,t)
        assert j.request_cap('other')==3
    assert len(calls)==1;verify_prefix(tmp_path,s)


def test_no_fifth_attempt(tmp_path):
    m,s,t=case(tmp_path);calls=[]
    with NetworkRetryJournal(tmp_path,m,m['jobs'][0],s,pause=lambda _:None,transport_factory=fixtures.factory(['empty'],calls)) as j:
        with pytest.raises(TransportFailure,match='exhausted'):j.request('scope',0,fixtures.base.PAYLOAD,t)
        with pytest.raises(TransportFailure):j.request('scope',0,fixtures.base.PAYLOAD,t)
    assert len(calls)==1;verify_prefix(tmp_path,s)


def test_unlisted_request_cannot_retry(tmp_path):
    m,s,t=case(tmp_path);s['one_extra_attempt_keys']={}
    with NetworkRetryJournal(tmp_path,m,m['jobs'][0],s,pause=lambda _:None,transport_factory=lambda *a:pytest.fail('unauthorized')) as j:
        with pytest.raises(TransportFailure,match='exhausted'):j.request('scope',0,fixtures.base.PAYLOAD,t)


def test_diagnostic_redacts_cause_and_posts_once(tmp_path):
    secret='private-token';error=URLError(TimeoutError('Bearer '+secret+' https://user:password@example.com'))
    chain=safe_error_chain(error,secret);encoded=json.dumps(chain)
    assert secret not in encoded and 'password' not in encoded and len(chain)==2
    t=DiagnosticTransport(secret,tmp_path);calls=[]
    def fail(req,timeout):calls.append(req);raise error
    t.opener=SimpleNamespace(open=fail)
    payload={**fixtures.base.PAYLOAD,'version':'combination-qwen-pilot-v1'}
    with pytest.raises(TransportFailure):t(payload)
    r=read_json(tmp_path/f'{digest(payload)}.failure.json')
    assert len(calls)==1 and r['error_chain'][1]['type']=='TimeoutError'
    assert secret not in json.dumps(r)
