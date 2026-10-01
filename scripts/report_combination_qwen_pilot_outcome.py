"""Read-only analysis including a budget-limited incomplete pilot; no API calls."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import numpy as np
from report_combination_baselines import wealth_table,independent_score,check_score
from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination_fast import FastCombinationGymEnv
from wolfgym.env.combination_contract import primary_contract
from wolfgym.evaluation.combination_host import read_json,file_hash,aggregate_financial
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import public_frame,digest
from wolfgym.evaluation.combination_qwen_pilot import PilotAdapter,decode_reply,wire_body
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/combination_qwen_pilot_20260928'
BASE=ROOT/'data/combination_action_diagnostic_20260928'


def audit_episode(identity):
    m=read_json(OUT/'manifest.json')
    d=OUT/'episodes'/identity['episode_id'].replace(':','_')
    logs=[read_json(p) for p in sorted((d/'decisions').glob('*.json'))]
    if not logs:return {'identity':identity,'complete':False,'recorded_days':0}
    def reply(payload):
        assert wire_body(payload)
        return decode_reply(read_json(d/'responses'/f'{digest(payload)}.json')).text
    adapter=PilotAdapter(reply)
    e=CombinationEpisode(identity['split'],identity['trajectory_seed'],identity['condition'])
    env=FastCombinationGymEnv(e,message_mode='indexed')
    try:
        _,info=env.reset()
        for log in logs:
            action=adapter(info)
            assert adapter.log[-1]==log,'public input/response/action replay mismatch'
            _,reward,done,truncated,info=env.step(action)
            assert reward==0 and not truncated and done==(log['day']==29)
        completed=(d/'complete.json').exists()
        row={'identity':identity,'complete':completed,'recorded_days':len(logs),
             'freezes':[{'decision_day':x['day'],'action_index':x['action'],
                        'target':x['frame']['action_targets'][x['action']-1]} for x in logs if x['action']],
             'invalid_actions':sum(x['invalid_action'] for x in logs)}
        if not completed:return row
        assert len(logs)==30
        marker=read_json(d/'complete.json')
        assert marker['scope']==digest([digest(m),identity])
        for p,sha in marker['sha256'].items():assert file_hash(d/p)==sha
        a=read_json(d/'outcome.json.gz');r=a['result']
        assert a['decisions']==logs
        assert all(r[k]==v for k,v in identity.items())
        assert all(r[k]==v for k,v in env.evaluation_result().items())
        assert a['accounts']=={k:[asdict(x) for x in v] for k,v in env._snapshots.items()}
        assert a['financial_rows']==env._rows
        assert r['invalid_actions']==row['invalid_actions']
        assert r['model_decisions']==sum(x['request'] is not None for x in logs)
        assert r['cost']==4*len(row['freezes'])
        b=BASE/identity['episode_id'].replace(':','_');bm=read_json(b/'complete.json')
        for name in ('reference_wait.json.gz','rule.json.gz'):assert file_hash(b/name)==bm['sha256'][name]
        old=read_json(b/'reference_wait.json.gz')
        assert all(a['accounts'][k]==old['accounts'][k] and a['financial_rows'][k]==old['financial_rows'][k] for k in ('reference','wait'))
        ref,initial=wealth_table(a['accounts']['reference'],a['accounts']['reference'])
        scores={}
        for world in ('wait','policy'):
            actual,_=wealth_table(a['accounts'][world],a['accounts']['reference'])
            scores[world]=independent_score(ref,actual,initial,a['financial_rows']['reference'],a['financial_rows'][world],primary_contract()['harm_threshold'])
            check_score(scores[world],r[world])
        benefit=float(scores['wait']['H'][-1]-scores['policy']['H'][-1]);assert np.isclose(benefit,r['H_reduction'],rtol=1e-10,atol=1e-12)
        f={k:aggregate_financial(v) for k,v in a['financial_rows'].items()};assert f==r['financial_outcomes']
        teacher=read_json(b/'rule.json.gz')['result']
        row.update({'benefit_pp':100*benefit,'teacher_pp':100*teacher['H_reduction'],'cost':r['cost'],
                    'H_wait_pp':100*float(scores['wait']['H'][-1]),'H_policy_pp':100*float(scores['policy']['H'][-1]),
                    'event_wait':scores['wait']['event'],'event_policy':scores['policy']['event'],
                    'financial_delta':{k:f['policy'][k]-f['wait'][k] for k in f['policy']}})
        return row
    finally:env.close()


def main():
    import fcntl
    with (OUT/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB) # never audit unstable running output
        m=read_json(OUT/'manifest.json')
        for p,sha in m['source_sha256'].items():assert file_hash(p)==sha
        ba=read_json(BASE/'acceptance.json')
        for p,sha in ba['outputs_sha256'].items():assert file_hash(BASE/p)==sha
        with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(audit_episode,m['episodes']))
        receipts={};caps=[];declarations=[];all_attempts=[];cost=0;tokens={'input':0,'output':0,'reasoning':0};latency=[];states={};attempts=0
        for i in range(4):
            db=sqlite3.connect(f'file:{OUT}/requests_{i}.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
            xs=[dict(x) for x in db.execute('select * from attempts order by id')]
            declaration=json.loads(db.execute('select declaration from metadata where id=1').fetchone()[0]);db.close()
            assert declaration['contract']=={'manifest_sha256':digest(m),'shard':i}
            assert declaration['max_attempts']==120 and declaration['reservation_microusd']==110000 and declaration['retries']==0
            caps.append(declaration['max_cost_microusd']);declarations.append(declaration);all_attempts.append(xs)
            assert len(xs)<=120 and sum(x['charged'] for x in xs)<=caps[-1]
            for x in xs:
                attempts+=1;states[x['state']]=states.get(x['state'],0)+1
                assert x['state']=='success','unknown or failed request requires separate reconciliation'
                payload=json.loads(x['payload']); saved=json.loads(x['response']); matches=[]
                for identity in m['episodes']:
                    scope=digest([digest(m),identity])
                    for day in range(30):
                        if x['request_key']==digest([scope,day]):matches.append((identity,day))
                assert len(matches)==1
                identity,day=matches[0];d=OUT/'episodes'/identity['episode_id'].replace(':','_')
                path=d/'responses'/f'{digest(payload)}.json'; reply=decode_reply(read_json(path))
                assert reply.text==saved['text'] and reply.cost_microusd==saved['cost_microusd']==x['charged'] and reply.usage==saved['usage']
                log=read_json(d/'decisions'/f'{day:02}.json');assert log['request']==payload and log['response']==reply.text
                receipts[str(path.relative_to(OUT))]=file_hash(path);cost+=x['charged'];latency.append(x['elapsed_seconds'])
                tokens['input']+=reply.usage['input_tokens'];tokens['output']+=reply.usage['output_tokens'];tokens['reasoning']+=reply.usage.get('reasoning_tokens',0)
        assert sum(caps)==10000000
        budget_receipts={}
        if (OUT/'budget_settlement.json').exists():
            amendment=read_json(OUT/'budget_settlement.json')
            assert amendment['original_manifest_sha256']==file_hash(OUT/'manifest.json')
            for p,sha in amendment['source_sha256'].items():assert file_hash(p)==sha
            stages=sorted(OUT.glob('budget_settlement_[0-3].json'))
            assert stages
            previous=None
            for stage in stages:
                v=read_json(stage);assert v['applied']
                if previous is not None:assert [x['declaration'] for x in v['before']]==previous
                assert sum(x['max_cost_microusd'] for x in v['after_declarations'])==10000000
                for i,b in enumerate(v['before']):assert digest(all_attempts[i][:b['attempts']])==b['attempts_sha256']
                previous=v['after_declarations']
            assert declarations==previous
            budget_receipts={str(p.relative_to(OUT)):file_hash(p) for p in OUT.glob('budget_settlement*.json')}
        else:assert caps==[2500000]*4
        assert attempts==len(receipts)==sum(x['request'] is not None for p in (OUT/'episodes').glob('*/decisions/*.json') for x in [read_json(p)])
        assert cost<=10000000
        completed=[r for r in rows if r['complete']]
        groups={r['identity']['group_id'] for r in completed if sum(s['complete'] and s['identity']['group_id']==r['identity']['group_id'] for s in rows)==2}
        paired=[r for r in completed if r['identity']['group_id'] in groups]
        summary={'complete_pairs':len(groups),'balanced_benefit_pp':float(np.mean([r['benefit_pp'] for r in paired])) if paired else None,
                 'teacher_balanced_pp':float(np.mean([r['teacher_pp'] for r in paired])) if paired else None}
        for condition in ('normal','attack'):
            cr=[r for r in paired if r['identity']['condition']==condition]
            summary[condition]={'n':len(cr),'benefit_pp':float(np.mean([r['benefit_pp'] for r in cr])) if cr else None,
                'teacher_pp':float(np.mean([r['teacher_pp'] for r in cr])) if cr else None,
                'events_wait':sum(r['event_wait'] for r in cr),'events_policy':sum(r['event_policy'] for r in cr),
                'prevented':sum(r['event_wait'] and not r['event_policy'] for r in cr),
                'added':sum(not r['event_wait'] and r['event_policy'] for r in cr),
                'intervened':sum(bool(r['freezes']) for r in cr),'mean_cost':float(np.mean([r['cost'] for r in cr])) if cr else None,
                'mean_financial_delta':{k:float(np.mean([r['financial_delta'][k] for r in cr])) for k in cr[0]['financial_delta']} if cr else {}}
        report={'scope':'Descriptive train pilot; complete pairs only in comparison; incomplete traces retained, never scored as WAIT.',
                'full_grid_complete':len(completed)==16,'completed_episodes':len(completed),'planned_episodes':16,
                'summary':summary,'rows':rows,'usage':{'attempts':attempts,'states':states,'usd':cost/1e6,'tokens':tokens,
                    'latency_median':float(np.median(latency)),'latency_p95':float(np.quantile(latency,.95))},
                'audit':{'public_frames':sum(r['recorded_days'] for r in rows),'complete_account_closes':len(completed)*3*30*490,
                         'api_requests_during_audit':0,'method':'recreate markets, replay saved provider actions, exact public frames and all accounts; independent H/G/OR scoring'}}
        durable_json(OUT/'outcome_review.json',report)
        durable_json(OUT/'outcome_review_receipt.json',{'full_grid_complete':report['full_grid_complete'],
                     'script_sha256':file_hash(Path(__file__)),'manifest_sha256':file_hash(OUT/'manifest.json'),
                     'report_sha256':file_hash(OUT/'outcome_review.json'),'provider_responses_sha256':receipts,'budget_settlement_sha256':budget_receipts})
        print(json.dumps({k:report[k] for k in ('full_grid_complete','completed_episodes','summary','usage','audit')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
