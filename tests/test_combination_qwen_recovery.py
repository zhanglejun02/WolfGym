import copy
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from wolfgym.evaluation.combination_host import read_json
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest,canonical_json
from wolfgym.evaluation.combination_qwen_pilot import CANONICAL,RESERVATION,decode_reply
from wolfgym.evaluation.combination_qwen_recovery import RecoveryJournal,ledger_snapshot,pooled_totals,verify_prefix
from wolfgym.evaluation.combination_requests import RequestJournal,TransportFailure,RequestLimitError,AmbiguousRequestError

PAYLOAD={'version':'mock','system':'unchanged','user':'public'}
RAW={'model':CANONICAL,'provider':'Alibaba','usage':{'prompt_tokens':100,'completion_tokens':10,'cost':.0001},
     'choices':[{'message':{'content':'{"action":"WAIT"}'}}]}


def make_case(tmp_path,old_failed=False,limit=1008,cost=30000000):
    jobs=[{'id':f'LL_{i}','mode':'LL','shard':i,'max_requests':120,'max_microusd':3250000} for i in range(2)]
    manifest={'jobs':jobs,'max_requests_total':limit,'max_cost_microusd_total':cost}
    target=SimpleNamespace(key='never-a-real-key',directory=tmp_path/'responses')
    if old_failed:
        with RequestJournal(tmp_path/'LL_0.sqlite',contract={'manifest_sha256':digest(manifest),'job':'LL_0'},
                            max_attempts=120,max_cost_microusd=3250000,reservation_microusd=RESERVATION,retries=0) as journal:
            def fail(p):
                durable_json(target.directory/f'{digest(p)}.failure.json',{'exception_type':'URLError','http_status':None,'payload_sha256':digest(p)})
                raise TransportFailure('offline')
            with pytest.raises(TransportFailure):journal.request('scope',0,PAYLOAD,fail)
    amendment={'original_ledgers':{'LL_0':ledger_snapshot(tmp_path/'LL_0.sqlite')} if old_failed else {}}
    return manifest,jobs,amendment,target


def transport_factory(events,calls):
    def factory(key,directory):
        def call(payload):
            calls.append((str(directory),copy.deepcopy(payload)))
            event=events.pop(0)
            if event!='ok':
                durable_json(Path(directory)/f'{digest(payload)}.failure.json',{'exception_type':'HTTPError' if event==401 else 'URLError',
                    'http_status':401 if event==401 else None,'payload_sha256':digest(payload)})
                raise TransportFailure('offline')
            durable_json(Path(directory)/f'{digest(payload)}.json',RAW)
            return decode_reply(RAW)
        return call
    return factory


def test_old_failure_preserved_success_never_reissued(tmp_path):
    manifest,jobs,amendment,target=make_case(tmp_path,True);calls=[]
    with RecoveryJournal(tmp_path,manifest,jobs[0],amendment,pause=lambda _:None,
                         transport_factory=transport_factory(['ok'],calls)) as j:
        result=j.request('scope',0,PAYLOAD,target)
        assert result==decode_reply(RAW).text
        assert j.request('scope',0,PAYLOAD,target)==result
        assert j.totals()=={'attempts':2,'charged_or_reserved_microusd':RESERVATION+100}
        with pytest.raises(ValueError,match='input changed'):j.request('scope',0,{**PAYLOAD,'user':'changed'},target)
    verify_prefix(tmp_path,amendment)
    assert len(calls)==1
    with RecoveryJournal(tmp_path,manifest,jobs[0],amendment,transport_factory=lambda *a:pytest.fail('reissued')) as j:
        assert j.request('scope',0,PAYLOAD,target)==result


def test_max_two_supplementals_survives_restart(tmp_path):
    manifest,jobs,amendment,target=make_case(tmp_path,True);calls=[]
    with RecoveryJournal(tmp_path,manifest,jobs[0],amendment,pause=lambda _:None,
                         transport_factory=transport_factory(['fail','fail'],calls)) as j:
        with pytest.raises(TransportFailure,match='exhausted'):j.request('scope',0,PAYLOAD,target)
        assert j.totals()['attempts']==3 and j.totals()['charged_or_reserved_microusd']==3*RESERVATION
    with RecoveryJournal(tmp_path,manifest,jobs[0],amendment,pause=lambda _:None,
                         transport_factory=lambda *a:pytest.fail('fourth attempt')) as j:
        with pytest.raises(TransportFailure,match='exhausted'):j.request('scope',0,PAYLOAD,target)
    assert len(calls)==2;verify_prefix(tmp_path,amendment)


@pytest.mark.parametrize('limit,cost',[(1,30000000),(1008,RESERVATION)])
def test_original_unknowns_consume_shared_limits(tmp_path,limit,cost):
    manifest,jobs,amendment,target=make_case(tmp_path,True,limit,cost)
    with RecoveryJournal(tmp_path,manifest,jobs[1],amendment,transport_factory=lambda *a:pytest.fail('over budget')) as j:
        with pytest.raises(RequestLimitError):j.request('different',0,PAYLOAD,target)
    assert pooled_totals(tmp_path,manifest)['attempts']==1


def test_global_reservation_race(tmp_path):
    manifest,jobs,amendment,_=make_case(tmp_path,limit=1);calls=[]
    def worker(job):
        target=SimpleNamespace(key='mock',directory=tmp_path/job['id']/'responses')
        try:
            with RecoveryJournal(tmp_path,manifest,job,amendment,transport_factory=transport_factory(['ok'],calls)) as j:
                j.request(job['id'],0,PAYLOAD,target)
            return True
        except RequestLimitError:return False
    with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(worker,jobs))
    assert sorted(results)==[False,True] and len(calls)==1
    assert pooled_totals(tmp_path,manifest)['attempts']==1


def test_non_network_failure_stops_without_retry(tmp_path):
    manifest,jobs,amendment,target=make_case(tmp_path);calls=[]
    with RecoveryJournal(tmp_path,manifest,jobs[0],amendment,pause=lambda _:None,
                         transport_factory=transport_factory([401],calls)) as j:
        with pytest.raises(TransportFailure,match='non-network'):j.request('scope',0,PAYLOAD,target)
    assert len(calls)==1 and (tmp_path/'recovery_stop.json').exists()


@pytest.mark.parametrize('has_receipt',[True,False])
def test_pending_response_reconcile_or_stop(tmp_path,has_receipt):
    manifest,jobs,amendment,target=make_case(tmp_path)
    with RecoveryJournal(tmp_path,manifest,jobs[0],amendment,transport_factory=lambda *a:pytest.fail('reissued')) as j:
        with j.db:
            j.db.execute('INSERT INTO attempts(request_key,payload,state,charged,created_unix) VALUES(?,?,?,?,?)',
                         (digest(['scope',0]),canonical_json(PAYLOAD),'pending',RESERVATION,0))
        if has_receipt:
            durable_json(j.evidence(1)/f'{digest(PAYLOAD)}.json',RAW)
            assert j.request('scope',0,PAYLOAD,target)==decode_reply(RAW).text
            assert j.totals()['attempts']==1 and j.totals()['charged_or_reserved_microusd']==100
        else:
            with pytest.raises(AmbiguousRequestError):j.request('scope',0,PAYLOAD,target)
            assert j.totals()['charged_or_reserved_microusd']==RESERVATION


def test_invalid_provider_receipt_blocks_new_requests(tmp_path):
    manifest,jobs,amendment,target=make_case(tmp_path)
    def factory(key,d):
        def call(p):
            durable_json(Path(d)/f'{digest(p)}.json',{**RAW,'provider':'unexpected'})
            raise ValueError('provider drift')
        return call
    with RecoveryJournal(tmp_path,manifest,jobs[0],amendment,transport_factory=factory) as j:
        with pytest.raises(TransportFailure):j.request('scope',0,PAYLOAD,target)
    with RecoveryJournal(tmp_path,manifest,jobs[1],amendment,transport_factory=lambda *a:pytest.fail('call after stop')) as j:
        with pytest.raises(TransportFailure,match='stopped'):j.request('other',0,PAYLOAD,target)


def test_recovery_audit_keeps_unknowns_separate(tmp_path,monkeypatch):
    import report_combination_qwen_recovery as report
    manifest,jobs,amendment,_=make_case(tmp_path)
    identity={'episode_id':'fixture','group_id':'one','condition':'normal'}
    manifest['episodes']=[identity];manifest['jobs']=[jobs[0]]
    target=SimpleNamespace(key='mock',directory=tmp_path/'LL/episodes/fixture/responses')
    scope=digest([digest(manifest),'LL',identity]);calls=[]
    with RecoveryJournal(tmp_path,manifest,jobs[0],amendment,pause=lambda _:None,
                         transport_factory=transport_factory(['fail','ok'],calls)) as j:
        answer=j.request(scope,0,PAYLOAD,target)
    for day in range(30):
        durable_json(tmp_path/'LL/episodes/fixture/decisions'/f'{day:02}.json',
                     {'request':PAYLOAD if day==0 else None,'response':answer if day==0 else None})
    monkeypatch.setattr(report,'wire_body',lambda p:None)
    monkeypatch.setattr(report,'selected_episodes',lambda m,j:[identity])
    usage,_,totals=report.audit_ledgers(tmp_path,manifest,amendment,('LL',))
    assert usage['LL_0']['confirmed_usd']==.0001 and usage['LL_0']['unknown_reserved_usd']==.11
    assert totals['attempts']==2 and totals['charged_or_reserved_microusd']==110100
    import sqlite3
    with sqlite3.connect(tmp_path/'LL_0.sqlite') as db:db.execute('UPDATE attempts SET charged=0 WHERE id=1')
    with pytest.raises(ValueError,match='reservation'):report.audit_ledgers(tmp_path,manifest,amendment,('LL',))
