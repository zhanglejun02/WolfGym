import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_combination_qwen_pilot as runner
from wolfgym.evaluation.combination_qwen_pilot import *
from wolfgym.evaluation.combination_requests import RequestJournal, Reply
from wolfgym.evaluation.combination_host import read_json
from wolfgym.benchmark.combination import CombinationEpisode


def response():
    return {'model':CANONICAL,'provider':'Alibaba','usage':{'prompt_tokens':100,'completion_tokens':12,'cost':0.000272},
            'choices':[{'message':{'content':'{"action":"WAIT"}'}}]}


def test_provider_receipt_and_bounds():
    assert decode_reply(response()).cost_microusd == 272
    for field,value in [('provider','Other'),('model','other')]:
        x=response(); x[field]=value
        with pytest.raises(ValueError): decode_reply(x)
    for field,value in [('cost',1),('cost',float('nan')),('prompt_tokens',INPUT_BOUND+1),('completion_tokens',MAX_OUTPUT+1)]:
        x=response(); x['usage'][field]=value
        with pytest.raises(ValueError): decode_reply(x)
    x=response(); del x['usage']['cost']
    with pytest.raises(KeyError): decode_reply(x)


def test_wire_size_and_provider_contract():
    x={'version':VERSION,'system':'test','user':'{}'}
    b=json.loads(wire_body(x)); assert b['provider']['only']==['alibaba'] and not b['provider']['allow_fallbacks']
    x['user']='x'*MAX_BODY_BYTES
    with pytest.raises(ValueError,match='input bound'): wire_body(x)


def test_train_grid_and_gate(tmp_path):
    xs=runner.episodes(); assert len(xs)==16 and len({x['group_id'] for x in xs})==8
    assert all(x['split']=='train' for x in xs)
    x=copy.deepcopy(xs[0]); x['split']='test'
    with pytest.raises(ValueError,match='16 train'): runner.run_one(tmp_path,x,{'episodes':xs},None,None)


@pytest.mark.parametrize('condition',['normal','attack'])
def test_real_host_replay_and_accounts(tmp_path,monkeypatch,condition):
    identity=next(x for x in runner.episodes() if x['condition']==condition)
    manifest={'episodes':[identity]}; calls=[]
    def factory(key,directory):
        def transport(payload):
            assert 'suppresses associated messages' in payload['system']
            view=json.loads(payload['user']); calls.append(payload)
            return Reply(json.dumps({'action':'FREEZE','target_id':view['observations'][-1]['action_targets'][0]}),50)
        return transport
    monkeypatch.setattr(runner,'QwenTransport',factory)
    def journal(): return RequestJournal(tmp_path/'requests.sqlite',contract={'test':1},max_attempts=30,
        max_cost_microusd=10000,reservation_microusd=100,retries=0)
    with journal() as j:
        assert runner.run_one(tmp_path,identity,manifest,j,'mock',stop_after=1) is None
    with journal() as j:
        r=runner.run_one(tmp_path,identity,manifest,j,'mock')
        assert j.totals()['attempts']==3 and len(calls)==3
        assert r['cost']==12 and r['invalid_actions']==0
    d=tmp_path/'episodes'/identity['episode_id'].replace(':','_')
    a=read_json(d/'outcome.json.gz')
    e=CombinationEpisode(identity['split'],identity['trajectory_seed'],condition)
    env=runner.FastCombinationGymEnv(e,message_mode='indexed')
    try:
        env.reset()
        for day in range(30): env.step(1 if day<3 else 0)
        assert a['accounts']=={k:[asdict(x) for x in v] for k,v in env._snapshots.items()}
        assert a['financial_rows']==env._rows
    finally: env.close()
    (d/'decisions/00.json').write_text('{}')
    with journal() as j:
        with pytest.raises(ValueError,match='drift'): runner.run_one(tmp_path,identity,manifest,j,'mock')
