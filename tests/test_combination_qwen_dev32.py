import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_combination_qwen_dev32 as runner
import report_combination_qwen_dev32 as reporter
import test_combination_qwen_incomplete_recovery as fixtures
from wolfgym.benchmark.combination import episode_plan
from wolfgym.evaluation.combination_qwen_dev32 import DevelopmentJournal
from wolfgym.evaluation.combination_host import read_json
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_pilot import CANONICAL,wire_body,decode_reply
from wolfgym.evaluation.combination_requests import RequestLimitError,TransportFailure


def small_manifest(identity):
    return {'episodes':[identity],'jobs':[{'id':'LL_0','mode':'LL','shard':0}],
            'max_requests_total':2048,'max_cost_microusd_total':60000000}


def integration_episodes():
    if not (runner.OLD/'manifest.json').exists():
        pytest.skip('private prior-Qwen sampling manifest is unavailable')
    rows=runner.episode_selection()
    if any(not (runner.baseline_directory(e)/'complete.json').exists() for e in rows):
        pytest.skip('private paired dev32 baseline artifacts are unavailable')
    return rows


def test_new_qwen_sample_complete_disjoint_baselines_fixed():
    rows=integration_episodes();old=read_json(runner.OLD/'manifest.json')['episodes']
    assert len(rows)==64 and len({e['group_id'] for e in rows})==32
    assert not {e['group_id'] for e in rows}&{e['group_id'] for e in old}
    assert {e['split'] for e in rows}=={'train'}
    assert runner.POLICIES==('wait','date_8_slot_1','alarm_refresh')
    for e in rows:
        d=runner.baseline_directory(e)
        assert read_json(d/'complete.json')['episode']==e
    jobs=[{'id':f'LL_{i}','mode':'LL','shard':i} for i in range(16)]
    manifest={'episodes':rows,'jobs':jobs}
    selected=[e for j in jobs for e in runner.selected_episodes(manifest,j)]
    assert len(selected)==len({e['episode_id'] for e in selected})==64


def test_budget_count_and_manifest_drift_stop(tmp_path):
    identity=episode_plan('train')[0].to_dict();m=small_manifest(identity);m['max_requests_total']=1
    durable_json(tmp_path/'manifest.json',m)
    target=SimpleNamespace(key='mock',directory=tmp_path/'responses');calls=[]
    with DevelopmentJournal(tmp_path,m,m['jobs'][0],pause=lambda _:None,transport_factory=fixtures.factory(['ok'],calls)) as j:
        assert j.request('scope',0,fixtures.base.PAYLOAD,target)
        assert j.request('scope',0,fixtures.base.PAYLOAD,target)
        with pytest.raises(RequestLimitError):j.request('scope',1,{**fixtures.base.PAYLOAD,'user':'day1'},target)
        declaration=json.loads(j.db.execute('SELECT declaration FROM metadata WHERE id=1').fetchone()[0])
        assert declaration['retries']==2 and declaration['max_cost_microusd']==60000000
    assert len(calls)==1
    changed={**m,'max_requests_total':2049};durable_json(tmp_path/'manifest.json',changed)
    with DevelopmentJournal(tmp_path,m,m['jobs'][0]) as j:
        with pytest.raises(ValueError,match='manifest drift'):j.request('scope',1,{**fixtures.base.PAYLOAD,'user':'day1'},target)


def test_empty_response_bounded_and_unknown_reserved(tmp_path):
    m=small_manifest(episode_plan('train')[0].to_dict());durable_json(tmp_path/'manifest.json',m);calls=[]
    target=SimpleNamespace(key='mock',directory=tmp_path/'responses')
    with DevelopmentJournal(tmp_path,m,m['jobs'][0],pause=lambda _:None,transport_factory=fixtures.factory(['empty']*3,calls)) as j:
        with pytest.raises(TransportFailure,match='exhausted'):j.request('scope',0,fixtures.base.PAYLOAD,target)
        assert j.totals()['attempts']==3 and j.totals()['charged_or_reserved_microusd']==330000
    assert len(calls)==3 and (tmp_path/'incomplete_recovery_stop.json').exists()


@pytest.mark.parametrize('condition',['normal','attack'])
def test_full_host_resume_receipts_accounts_baseline_pair_and_audit(tmp_path,condition):
    identity=next(e for e in integration_episodes() if e['condition']==condition)
    m=small_manifest(identity);durable_json(tmp_path/'manifest.json',m);job=m['jobs'][0];calls=[];failed=set()
    def factory(key,directory):
        def respond(payload):
            wire_body(payload);v=json.loads(payload['user']);f=v['observations'][-1];calls.append(f['day'])
            if f['day']==9 and 9 not in failed:
                failed.add(9);durable_json(Path(directory)/f'{digest(payload)}.json',fixtures.EMPTY)
                return decode_reply(fixtures.EMPTY)
            action={'action':'FREEZE','target_id':f['action_targets'][1]} if f['day'] in (8,13,18) else {'action':'WAIT'}
            raw={'id':f'gen-offline-{f["day"]}','model':CANONICAL,'provider':'Alibaba','usage':{'prompt_tokens':100,'completion_tokens':10,'cost':.0001},
                'choices':[{'finish_reason':'stop','message':{'content':json.dumps(action)}}]}
            durable_json(Path(directory)/f'{digest(payload)}.json',raw);return decode_reply(raw)
        return respond
    with DevelopmentJournal(tmp_path,m,job,pause=lambda _:None,transport_factory=factory) as j:
        assert runner.run_one(tmp_path,identity,m,job,j,'mock',stop_after=7) is None
    before=list(calls)
    with DevelopmentJournal(tmp_path,m,job,pause=lambda _:None,transport_factory=factory) as j:
        runner.run_one(tmp_path,identity,m,job,j,'mock')
        total=j.totals()['attempts']
    assert calls[:len(before)]==before and calls.count(0)==1 and calls.count(9)==2
    row=reporter.audit_episode((str(tmp_path),m,'LL',identity));assert row['mode']=='LL'
    baseline=reporter.audit_baselines(identity);assert {x['mode'] for x in baseline}==set(runner.POLICIES)
    assert row['benefit_pp']==next(x['benefit_pp'] for x in baseline if x['mode']=='date_8_slot_1')
    usage,_,totals=reporter.audit_ledgers(tmp_path,m)
    assert totals['attempts']==total and usage['LL_0']['unknown_reserved_usd']==.11
    # A changed public receipt cannot silently pass account/model replay.
    p=next((tmp_path/'LL/episodes'/identity['episode_id'].replace(':','_')/'decisions').glob('*.json'))
    x=read_json(p);x['action']=99;durable_json(p,x)
    with pytest.raises(ValueError):reporter.audit_episode((str(tmp_path),m,'LL',identity))


def test_statistics_pair_societies_not_episodes_and_reject_partial_grid():
    rows=[]
    for i in range(32):
        for condition in ('normal','attack'):
            for mode in ('LL',*runner.POLICIES):
                rows.append({'group_id':f'g{i:02}','trajectory_seed':30000000+i*3,'condition':condition,'mode':mode,
                             'benefit_pp':(2 if condition=='attack' else -1) if mode=='LL' else 0})
    r=reporter.paired_statistics(rows)
    for value in r.values():
        assert value['balanced_pp']['mean']==.5 and value['balanced_pp']['ci95']==[.5,.5]
        assert value['lag13_chain_sensitivity_ci95_pp']==[.5,.5]
        assert len(value['society_differences_pp'])==32 and not value['coverage_certified']
    with pytest.raises(ValueError,match='complete'):reporter.paired_statistics(rows[:-1])
