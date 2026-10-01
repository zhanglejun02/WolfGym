import copy
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import test_combination_qwen_recovery as base
from wolfgym.evaluation.combination_host import read_json,file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest,canonical_json
from wolfgym.evaluation.combination_qwen_pilot import decode_reply,RESERVATION
from wolfgym.evaluation.combination_qwen_recovery import RecoveryJournal,ledger_snapshot,verify_prefix,pooled_totals
from wolfgym.evaluation.combination_qwen_incomplete_recovery import IncompleteRecoveryJournal,incomplete_receipt
from wolfgym.evaluation.combination_requests import TransportFailure,RequestLimitError,AmbiguousRequestError

EMPTY={**base.RAW,'id':'gen-offline','usage':None,'choices':[{'finish_reason':'stop','message':{'content':None,'refusal':None}}]}


def supplement(out,jobs,bind=False):
    durable_json(out/'recovery_stop.json',{'job':'LL_0','reason':'diagnosed test incident'})
    return {'acknowledged_stop_sha256':file_hash(out/'recovery_stop.json'),
        'original_ledgers':{j['id']:ledger_snapshot(out/(j['id']+'.sqlite')) for j in jobs if (out/(j['id']+'.sqlite')).exists()} if bind else {}}


def factory(events,calls):
    def create(key,directory):
        def send(payload):
            calls.append(digest(payload));event=events.pop(0)
            raw=base.RAW if event=='ok' else copy.deepcopy(EMPTY)
            durable_json(Path(directory)/f'{digest(payload)}.json',raw)
            return decode_reply(raw)
        return send
    return create


def test_diagnosed_old_failure_preserved_and_success_reused(tmp_path):
    m,jobs,a,target=base.make_case(tmp_path);calls=[]
    with RecoveryJournal(tmp_path,m,jobs[0],a,pause=lambda _:None,transport_factory=factory(['empty'],calls)) as j:
        with pytest.raises(TransportFailure):j.request('scope',0,base.PAYLOAD,target)
    old_stop=file_hash(tmp_path/'recovery_stop.json')
    s={'acknowledged_stop_sha256':old_stop,'original_ledgers':{'LL_0':ledger_snapshot(tmp_path/'LL_0.sqlite')}}
    with IncompleteRecoveryJournal(tmp_path,m,jobs[0],a,s,pause=lambda _:None,transport_factory=factory(['ok'],calls)) as j:
        answer=j.request('scope',0,base.PAYLOAD,target)
        assert j.request('scope',0,base.PAYLOAD,target)==answer
        assert j.totals()=={'attempts':2,'charged_or_reserved_microusd':110100}
    assert len(calls)==2 and file_hash(tmp_path/'recovery_stop.json')==old_stop
    verify_prefix(tmp_path,s)
    with IncompleteRecoveryJournal(tmp_path,m,jobs[0],a,s,transport_factory=lambda *a:pytest.fail('repeat success')) as j:
        assert j.request('scope',0,base.PAYLOAD,target)==answer
        with pytest.raises(ValueError,match='input changed'):j.request('scope',0,{**base.PAYLOAD,'user':'changed'},target)


@pytest.mark.parametrize('change',[
    {'model':'other'}, {'provider':'other'}, {'id':None}, {'usage':{}},
    {'choices':[{'finish_reason':'stop','message':{'content':'an existing answer','refusal':None}}]},
    {'choices':[{'finish_reason':'stop','message':{'content':None,'refusal':'refused'}}]},
    {'choices':[{'finish_reason':'length','message':{'content':None}}]},
    {'usage':base.RAW['usage']}, {'choices':[]},
])
def test_narrow_empty_response_classifier(tmp_path,change):
    p=tmp_path/'receipt.json';durable_json(p,EMPTY);assert incomplete_receipt(p)
    durable_json(p,{**EMPTY,**change});assert not incomplete_receipt(p)


def test_network_and_empty_share_two_supplemental_limit(tmp_path):
    m,jobs,a,target=base.make_case(tmp_path,old_failed=True);s=supplement(tmp_path,jobs,True);calls=[]
    with IncompleteRecoveryJournal(tmp_path,m,jobs[0],a,s,pause=lambda _:None,transport_factory=factory(['empty','empty'],calls)) as j:
        with pytest.raises(TransportFailure,match='exhausted'):j.request('scope',0,base.PAYLOAD,target)
        assert j.totals()['attempts']==3 and j.totals()['charged_or_reserved_microusd']==3*RESERVATION
    with IncompleteRecoveryJournal(tmp_path,m,jobs[0],a,s,pause=lambda _:None,transport_factory=lambda *a:pytest.fail('fourth')) as j:
        with pytest.raises(TransportFailure,match='exhausted'):j.request('scope',0,base.PAYLOAD,target)
    assert len(calls)==2;verify_prefix(tmp_path,s)


@pytest.mark.parametrize('limit,cost',[(1,30000000),(1008,110000)])
def test_global_limits_still_include_old_unknowns(tmp_path,limit,cost):
    m,jobs,a,target=base.make_case(tmp_path,True,limit,cost);s=supplement(tmp_path,jobs,True)
    with IncompleteRecoveryJournal(tmp_path,m,jobs[1],a,s,transport_factory=lambda *a:pytest.fail('over cap')) as j:
        with pytest.raises(RequestLimitError):j.request('other',0,base.PAYLOAD,target)


def test_concurrent_reservation_remains_atomic(tmp_path):
    m,jobs,a,_=base.make_case(tmp_path,limit=1);s=supplement(tmp_path,jobs);calls=[]
    def run(job):
        target=SimpleNamespace(key='mock',directory=tmp_path/job['id']/'responses')
        try:
            with IncompleteRecoveryJournal(tmp_path,m,job,a,s,transport_factory=factory(['ok'],calls)) as j:
                j.request(job['id'],0,base.PAYLOAD,target)
            return True
        except RequestLimitError:return False
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
    assert sorted(results)==[False,True] and len(calls)==1


@pytest.mark.parametrize('kind',['empty','valid','missing'])
def test_crash_receipt_is_classified_without_resetting_count(tmp_path,kind):
    m,jobs,a,target=base.make_case(tmp_path);s=supplement(tmp_path,jobs);calls=[]
    with IncompleteRecoveryJournal(tmp_path,m,jobs[0],a,s,pause=lambda _:None,transport_factory=factory(['ok'],calls)) as j:
        with j.db:j.db.execute('INSERT INTO attempts(request_key,payload,state,charged,created_unix) VALUES(?,?,?,?,?)',
            (digest(['scope',0]),canonical_json(base.PAYLOAD),'pending',RESERVATION,0))
        if kind!='missing':durable_json(j.evidence(1)/f'{digest(base.PAYLOAD)}.json',EMPTY if kind=='empty' else base.RAW)
        if kind=='missing':
            with pytest.raises(AmbiguousRequestError):j.request('scope',0,base.PAYLOAD,target)
            assert len(calls)==0
        else:
            assert j.request('scope',0,base.PAYLOAD,target)==decode_reply(base.RAW).text
            assert j.totals()['attempts']==(2 if kind=='empty' else 1)
            assert len(calls)==(1 if kind=='empty' else 0)


def test_changed_original_stop_and_new_stop_block(tmp_path):
    m,jobs,a,target=base.make_case(tmp_path);s=supplement(tmp_path,jobs)
    durable_json(tmp_path/'incomplete_recovery_stop.json',{'reason':'new incident'})
    with IncompleteRecoveryJournal(tmp_path,m,jobs[0],a,s,transport_factory=lambda *a:pytest.fail('stop bypass')) as j:
        with pytest.raises(TransportFailure,match='stopped'):j.request('scope',0,base.PAYLOAD,target)
    durable_json(tmp_path/'recovery_stop.json',{'changed':True})
    with pytest.raises(ValueError,match='stop marker changed'):
        with IncompleteRecoveryJournal(tmp_path,m,jobs[0],a,s):pass


def test_audit_accepts_empty_failure_preserves_fee_and_rejects_tamper(tmp_path,monkeypatch):
    import report_combination_qwen_incomplete_recovery as report
    m,jobs,a,_=base.make_case(tmp_path);identity={'episode_id':'fixture','group_id':'one','condition':'normal'}
    m['episodes']=[identity];m['jobs']=[jobs[0]];s=supplement(tmp_path,jobs)
    target=SimpleNamespace(key='mock',directory=tmp_path/'LL/episodes/fixture/responses');calls=[]
    scope=digest([digest(m),'LL',identity])
    with IncompleteRecoveryJournal(tmp_path,m,jobs[0],a,s,pause=lambda _:None,transport_factory=factory(['empty','ok'],calls)) as j:
        answer=j.request(scope,0,base.PAYLOAD,target)
    for day in range(30):durable_json(tmp_path/'LL/episodes/fixture/decisions'/f'{day:02}.json',
        {'request':base.PAYLOAD if day==0 else None,'response':answer if day==0 else None})
    monkeypatch.setattr(report,'wire_body',lambda p:None);monkeypatch.setattr(report,'selected_episodes',lambda m,j:[identity])
    usage,_,totals=report.audit_ledgers(tmp_path,m,a,('LL',))
    assert usage['LL_0']['unknown_reserved_usd']==.11 and totals['attempts']==2
    p=tmp_path/'recovery_attempts/LL_0/1'/f'{digest(base.PAYLOAD)}.json'
    durable_json(p,{**EMPTY,'provider':'other'})
    with pytest.raises(ValueError,match='evidence'):report.audit_ledgers(tmp_path,m,a,('LL',))
