import copy
import json
from pathlib import Path
import sys

import pytest

from combination_qwen_fixtures import reference_wait_artifact

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_combination_qwen_calibration as runner
import report_combination_qwen_calibration as reporter
from wolfgym.evaluation.combination_qwen_calibration import CalibratedAdapter, calibrated_request, DICTIONARY
from wolfgym.evaluation.combination_qwen_pilot import pilot_request, wire_body, decode_reply, CANONICAL
from wolfgym.evaluation.combination_host import read_json, load_detector
from wolfgym.evaluation.combination_public import public_frame, PublicHistory, public_target, digest
from wolfgym.evaluation.combination_decision import CombinationDecisionAdapter
from wolfgym.evaluation.combination_requests import RequestJournal, TransportFailure
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination_fast import FastCombinationGymEnv


@pytest.fixture(scope='module')
def initial():
    env=FastCombinationGymEnv(CombinationEpisode('engineering',20260925,'normal'),message_mode='indexed')
    try: return env.reset()[1]
    finally: env.close()


def test_ll_changes_only_system_definitions(initial):
    view=PublicHistory().observe(public_frame(initial))
    old=pilot_request(view); new=calibrated_request(view,'LL')
    assert new['system']==old['system']+'\n\n'+DICTIONARY
    assert new['user']==old['user'] and new['response_schema']==old['response_schema']
    before=json.loads(wire_body(old)); after=json.loads(wire_body(new))
    after['messages'][0]['content']=before['messages'][0]['content']
    assert before==after
    assert '0.673909' not in DICTIONARY and '30000' not in DICTIONARY
    for mode in ('LR','RL'):
        p=calibrated_request(view,mode)
        assert 'suppresses associated messages' in p['system']
        assert p['system'].count(DICTIONARY)==1
        assert p['user']==old['user']


def test_lr_selects_current_rule_target_without_alarm(initial,monkeypatch):
    monkeypatch.setattr('wolfgym.evaluation.combination_public.detector_score',lambda *args:0.)
    adapter=CalibratedAdapter('LR',load_detector(runner.DETECTOR),lambda p:'{"action":"FREEZE"}')
    f=public_frame(initial)
    assert adapter(initial)==f['action_targets'].index(public_target(f))+1
    assert 'A public rule selects the asset after your decision.' in adapter.log[0]['request']['system']
    assert json.loads(adapter.log[0]['request']['user'])==PublicHistory().observe(f)


def test_rl_own_alarm_and_budget(initial,monkeypatch):
    monkeypatch.setattr('wolfgym.evaluation.combination_public.detector_score',lambda features,detector:1.)
    calls=[]
    def response(p):
        calls.append(p)
        target=json.loads(p['user'])['observations'][-1]['action_targets'][2]
        return json.dumps({'target_id':target})
    adapter=CalibratedAdapter('RL',load_detector(runner.DETECTOR),response)
    remaining=12
    for day in range(30):
        info=copy.deepcopy(initial);info['public_observation']['day']=day
        info['public_observation']['remaining_budget']=remaining;info['action_mask']=[1]+[int(remaining>=4)]*3
        action=adapter(info)
        assert action==(3 if day in (0,5,10) else 0)
        remaining-=4*bool(action)
    assert len(calls)==3 and remaining==0
    assert len(json.loads(calls[-1]['user'])['interventions'])==2


def test_transport_failure_does_not_advance(initial):
    def fail(payload): raise TransportFailure('mock offline failure')
    adapter=CalibratedAdapter('LL',load_detector(runner.DETECTOR),fail)
    with pytest.raises(TransportFailure):adapter(initial)
    assert not adapter.log and not adapter.inner.history.frames


@pytest.mark.parametrize('condition',['normal','attack'])
def test_rr_unchanged_on_real_engineering_trajectory(condition):
    env=FastCombinationGymEnv(CombinationEpisode('engineering',20260925,condition),message_mode='indexed')
    detector=load_detector(runner.DETECTOR)
    old=CombinationDecisionAdapter('RR',detector);new=CalibratedAdapter('RR',detector)
    try:
        _,info=env.reset()
        for day in range(30):
            action=new(info); assert action==old(info) and new.log[-1]==old.log[-1]
            _,_,_,_,info=env.step(action)
    finally:env.close()


def test_hard_limits_and_disjoint_job_coverage():
    manifest={'episodes':runner.episodes(),'jobs':runner.JOBS}
    assert sum(x['max_microusd'] for x in runner.JOBS)==30000000
    assert sum(x['max_requests'] for x in runner.JOBS)==1008
    for mode in ('LL','LR','RL'):
        selected=[e['episode_id'] for j in runner.JOBS if j['mode']==mode for e in runner.selected_episodes(manifest,j)]
        assert len(selected)==len(set(selected))==16
    invalid=copy.deepcopy(manifest['episodes'][0]);invalid['split']='test'
    with pytest.raises(ValueError,match='frozen train'):
        runner.run_one(Path('/unused'),invalid,manifest,runner.JOBS[0],None,None)


@pytest.mark.parametrize('mode,condition',[('LL','normal'),('LR','attack'),('RL','attack')])
def test_host_resume_full_account_and_action_audit(tmp_path,monkeypatch,mode,condition):
    identity=next(e for e in runner.episodes() if e['condition']==condition)
    baseline = tmp_path/'baselines'
    monkeypatch.setattr(reporter, 'BASE', baseline)
    durable_json(baseline/identity['episode_id'].replace(':','_')/'reference_wait.json.gz',
                 reference_wait_artifact(identity))
    job=next(j for j in runner.JOBS if j['mode']==mode)
    manifest={'episodes':[identity],'jobs':[job]};calls=[]
    def factory(key,directory):
        def transport(payload):
            calls.append(payload);view=json.loads(payload['user']);f=view['observations'][-1]
            wire_body(payload)
            if mode=='RL':response={'target_id':f['action_targets'][1]}
            elif f['day'] in (5,10,15):
                response={'action':'FREEZE'}
                if mode=='LL':response['target_id']=f['action_targets'][1]
            else: response={'action':'WAIT'}
            raw={'model':CANONICAL,'provider':'Alibaba','usage':{'prompt_tokens':100,'completion_tokens':10,'cost':.0001},
                 'choices':[{'message':{'content':json.dumps(response)}}],'engineering_mock':True}
            durable_json(Path(directory)/f'{digest(payload)}.json',raw)
            return decode_reply(raw)
        return transport
    monkeypatch.setattr(runner,'QwenTransport',factory)
    def ledger():return RequestJournal(tmp_path/(job['id']+'.sqlite'),contract={'test':mode},max_attempts=30,
             max_cost_microusd=4000000,reservation_microusd=110000,retries=0)
    with ledger() as journal:runner.run_one(tmp_path,identity,manifest,job,journal,'mock',stop_after=7)
    before=len(calls)
    with ledger() as journal:
        runner.run_one(tmp_path,identity,manifest,job,journal,'mock')
        assert journal.totals()['attempts']==len(calls)
    assert len(calls)>=before
    row=reporter.audit_episode((str(tmp_path),manifest,mode,identity))
    assert row['mode']==mode and row['cost']<=12
    outcome=tmp_path/mode/'episodes'/identity['episode_id'].replace(':','_')/'outcome.json.gz'
    value=read_json(outcome)
    assert sum(x['request'] is not None for x in value['decisions'])==len(calls)
    value['accounts']['policy'][0]['accounts'][next(iter(value['accounts']['policy'][0]['accounts']))]['cash']+=1
    durable_json(outcome,value)
    with pytest.raises(ValueError,match='artifact drift'):reporter.audit_episode((str(tmp_path),manifest,mode,identity))


def test_complete_paired_summary_rejects_missing():
    rows=[]
    for identity in runner.episodes():
        for mode in ('RR', 'old_LL'):
            benefit = (2. if identity['condition'] == 'attack' else -1.) if mode == 'old_LL' else 0.
            rows.append({**identity, 'mode':mode, 'benefit_pp':benefit,
                         'action_days':[5,10,15] if mode == 'old_LL' else [],
                         'cost':12. if mode == 'old_LL' else 0.,
                         'event_wait':identity['condition'] == 'attack',
                         'event_policy':False,
                         'financial_delta':{'executed_notional':-100. if mode == 'old_LL' else 0.}})
    summary=reporter.summarize(rows)
    assert summary['old_LL']['balanced_pp']==.5 and summary['RR']['balanced_pp']==0.
    assert summary['old_LL']['normal']['benefit_pp']==-1.
    assert summary['old_LL']['attack']['benefit_pp']==2.
    assert summary['old_LL']['attack']['prevented']==8
    assert summary['old_LL']['normal']['mean_admin_cost']==12.
    assert summary['old_LL']['normal']['freezes']==24
    with pytest.raises(ValueError,match='incomplete'):reporter.summarize(rows[:-1])
    unpaired=copy.deepcopy(rows)
    for row in unpaired:
        if row['condition']=='normal':row['condition']='attack'
    with pytest.raises(ValueError,match='unpaired'):reporter.summarize(unpaired)


def test_saved_paired_summary_matches_accepted_results():
    rows=[]
    for identity in runner.episodes():
        name=identity['episode_id'].replace(':','_')
        for mode,path in [('RR',reporter.BASE/name/'rule.json.gz'),('old_LL',reporter.OLD/'episodes'/name/'outcome.json.gz')]:
            if not path.exists():
                pytest.skip('private saved Qwen and rule comparison artifacts are unavailable')
            x=read_json(path)
            rows.append(reporter.result_row(identity,mode,x['result'],[z['action'] for z in x['decisions']]))
    summary=reporter.summarize(rows)
    assert summary['old_LL']['balanced_pp']==pytest.approx(.00932730730059254)
    assert summary['RR']['balanced_pp']==pytest.approx(.03052295161231046)
    with pytest.raises(ValueError,match='incomplete'):reporter.summarize(rows[:-1])
