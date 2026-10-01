"""Full final BC validation audit and paired, society-level development estimates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import math
from pathlib import Path

import numpy as np
import torch

from audit_combination_seed_dependencies import chain_sensitivity
from report_combination_baselines import bootstrap
from report_combination_bc_pilot import audit_one
from run_combination_bc_final import declaration, SEEDS
from run_combination_bc_pilot import load_samples
from wolfgym.benchmark.combination import validate_episode_manifest
from wolfgym.env.combination_contract import validate_primary_scenario
from wolfgym.evaluation.combination_bc import BCPolicy, masked_logits, classification_metrics
from wolfgym.evaluation.combination_host import read_json, file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'data/combination_baselines_20260926'
PPO=ROOT/'data/combination_learning_20260926'
POLICIES=tuple(f'bc_seed_{s}' for s in SEEDS)
IDENTITY=('split','trajectory_seed','condition','group_id','episode_id')


def verify_historical(directory):
    for name,sha in read_json(directory/'acceptance.json')['outputs_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if file_hash(path)!=sha:raise ValueError('historical accepted artifact drift')


def historical_pair(job):
    identity,expected_hash=job
    directory=BASELINE/'validation'/str(identity['trajectory_seed'])/identity['condition']
    marker=read_json(directory/'complete.json')
    cache=read_json(directory/'reference_wait.json.gz')
    if (marker['episode']!=identity or cache['episode']!=identity or cache['execution_id']!=marker['execution_id']
            or file_hash(directory/'reference_wait.json.gz')!=marker['sha256']['reference_wait.json.gz']):
        raise ValueError('historical raw pairing integrity drift')
    # The older host also records detector/action diagnostics in each row. Compare
    # the complete accounts and the shared day/financial schema, not log metadata.
    actual=digest({world:{'accounts':cache[world],
                         'rows':[{'day':r['day'],'financial_outcomes':r['financial_outcomes']}
                                 for r in cache[world+'_rows']]}
                   for world in ('reference','wait')})
    if actual!=expected_hash:raise ValueError('current and historical reference/WAIT differ')
    return identity['episode_id']


def paired_arrays(records,episodes,policies):
    validate_episode_manifest(episodes,split='validation')
    expected={(p,e['episode_id']) for p in policies for e in episodes}
    actual=[(r['policy_id'],r['episode_id']) for r in records]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('incomplete/duplicate seed episode grid')
    identities={e['episode_id']:e for e in episodes}
    by_key={}
    for r in records:
        if {k:r[k] for k in IDENTITY}!=identities[r['episode_id']]:raise ValueError('foreign episode identity')
        if not math.isfinite(r['benefit']):raise ValueError('nonfinite benefit')
        by_key[r['policy_id'],r['trajectory_seed'],r['condition']]=r
    roots=sorted({e['trajectory_seed'] for e in episodes})
    arrays={p:{c:np.array([by_key[p,s,c]['benefit'] for s in roots])*100 for c in ('normal','attack')} for p in policies}
    for p in arrays:arrays[p]['balanced']=.5*(arrays[p]['normal']+arrays[p]['attack'])
    for episode in identities:
        if len({r['reference_wait_sha256'] for r in records if r['episode_id']==episode})!=1:
            raise ValueError('unpaired current seeds')
    return roots,arrays


def interval(values,roots):
    values=np.asarray(values,dtype=float)
    indices=np.random.default_rng(20260926).integers(0,len(roots),(20000,len(roots)))
    primary=bootstrap(values,indices)
    return {'mean_pp':primary['mean'],'society_bootstrap_ci95_pp':primary['ci95'],
            'dependence_sensitivity':chain_sensitivity(roots,values)}


def outcome_summary(records,policies):
    result={}
    for label,ps in [(p,[p]) for p in policies]+[('three_seed_mean',list(policies))]:
        result[label]={}
        for condition in ('normal','attack'):
            rs=[r for r in records if r['policy_id'] in ps and r['condition']==condition]
            result[label][condition]={
                'event_wait_mean_count':sum(r['event_wait'] for r in rs)/len(ps),
                'event_policy_mean_count':sum(r['event_policy'] for r in rs)/len(ps),
                'prevented_mean_count':sum(r['prevented'] for r in rs)/len(ps),
                'new_event_mean_count':sum(r['new_event'] for r in rs)/len(ps),
                'mean_remaining_H_pp':100*float(np.mean([r['H_policy'] for r in rs])),
                'mean_cost':float(np.mean([r['cost'] for r in rs])),
                'mean_action_count':float(np.mean([len(r['action_days']) for r in rs])),
                'intervened_mean_count':sum(bool(r['action_days']) for r in rs)/len(ps),
                'mean_financial_delta':{k:float(np.mean([r['financial_delta'][k] for r in rs])) for k in rs[0]['financial_delta']}}
    return result


def reference_arrays(episodes):
    identities={e['episode_id']:e for e in episodes};roots=sorted({e['trajectory_seed'] for e in episodes})
    teacher=[r for r in read_json(BASELINE/'audited_rows.json.gz') if r['split']=='validation' and r['policy_id']=='alarm_refresh']
    validate_episode_manifest([{k:r[k] for k in IDENTITY} for r in teacher],split='validation')
    sources={'alarm_refresh':teacher}
    ppo=read_json(PPO/'selected_validation_results.json.gz')
    if len(ppo)!=3:raise ValueError('old PPO seed selection incomplete')
    for seed,rows in zip(SEEDS,ppo):
        validate_episode_manifest([{k:r[k] for k in IDENTITY} for r in rows],split='validation')
        sources[f'old_ppo_{seed}']=[{**r,'benefit':r['H_reduction']} for r in rows]
    arrays={};wait_H={}
    for name,rows in sources.items():
        if any({k:r[k] for k in IDENTITY}!=identities[r['episode_id']] for r in rows):raise ValueError('reference identity drift')
        lookup={(r['trajectory_seed'],r['condition']):r for r in rows}
        arrays[name]={c:np.array([lookup[s,c]['benefit'] for s in roots])*100 for c in ('normal','attack')}
        arrays[name]['balanced']=.5*(arrays[name]['normal']+arrays[name]['attack'])
        for key,r in lookup.items():
            value=r['H_wait'] if name=='alarm_refresh' else r['wait']['damage']['H'][-1]
            if key in wait_H and not math.isclose(wait_H[key],value,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('historical PPO WAIT score mismatch')
            wait_H[key]=value
    arrays['old_ppo_mean3']={c:np.mean([arrays[f'old_ppo_{s}'][c] for s in SEEDS],axis=0) for c in ('normal','attack','balanced')}
    return {name:arrays[name] for name in ('alarm_refresh','old_ppo_mean3')},wait_H


def main(output,workers):
    with (output/'.audit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=read_json(output/'progress.json')
        if not progress['complete'] or progress['completed']!=768:raise ValueError('final validation incomplete')
        manifest=read_json(output/'manifest.json');execution=digest(manifest)
        if manifest!=declaration(manifest['workers']):raise ValueError('final configuration drift')
        validate_primary_scenario();verify_historical(BASELINE);verify_historical(PPO)
        torch.set_num_threads(1);x,y,masks=load_samples()
        for p in POLICIES:
            receipt=read_json(output/p/'training_complete.json')
            if receipt['execution_id']!=execution or receipt['total_updates']!=2000:raise ValueError('training provenance drift')
            for name,sha in receipt['sha256'].items():
                if file_hash(output/p/name)!=sha:raise ValueError('training model/curve drift')
            state=torch.load(output/p/'model_1000.pt',weights_only=True,map_location='cpu')
            if state['update']!=2000 or state['execution_id']!=execution:raise ValueError('checkpoint step/identity drift')
            curve=read_json(output/p/'curve.json')
            if [r['update'] for r in curve]!=list(range(0,2001,25)):raise ValueError('curve schedule incomplete')
            with torch.no_grad():metrics=classification_metrics(masked_logits(BCPolicy(output/p/'model_1000.pt').model,x,masks),y)
            if any(v!=curve[-1][k] for k,v in metrics.items()):raise ValueError('final training curve replay mismatch')
        jobs=[(str(output),ep,p,execution) for p in POLICIES for ep in manifest['validation']]
        records=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i,r in enumerate(pool.map(audit_one,jobs),1):
                records.append(r)
                if i%64==0:print(f'audited {i}/768',flush=True)
        roots,arrays=paired_arrays(records,manifest['validation'],POLICIES)
        hashes={r['episode_id']:r['reference_wait_sha256'] for r in records}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(historical_pair,[(e,hashes[e['episode_id']]) for e in manifest['validation']]))
        references,wait_H=reference_arrays(manifest['validation'])
        if any(not math.isclose(r['H_wait'],wait_H[r['trajectory_seed'],r['condition']],rel_tol=1e-10,abs_tol=1e-12) for r in records):
            raise ValueError('current/historical WAIT outcome mismatch')
        means={c:np.mean([arrays[p][c] for p in POLICIES],axis=0) for c in ('normal','attack','balanced')}
        report={'execution_id':execution,'scope':'fixed-method validation; not independent formal test',
            'per_seed':{p:{c:interval(v,roots) for c,v in a.items()} for p,a in arrays.items()},
            'three_seed_mean':{c:interval(v,roots) for c,v in means.items()},
            'paired_vs_references':{name:{c:interval(means[c]-v,roots) for c,v in a.items()} for name,a in references.items()},
            'reference_means_pp':{name:{c:float(v.mean()) for c,v in a.items()} for name,a in references.items()},
            'outcomes':outcome_summary(records,POLICIES),
            'audit':{'policy_episodes':768,'societies':128,'ordinary_account_closes_checked':768*3*30*490,
                     'model_actions_replayed':768*30,'historical_raw_reference_WAIT_pairs_verified':256,
                     'curve_endpoints_recomputed':3,'market_resimulation':False},
            'uncertainty':'Pointwise development intervals conditional on three trained models; seed11 used in selection; validation used by prior research. 13-chain coverage not certified. Not formal K4/8 family inference.',
            'P_star':'alarm_refresh','further_pilot_rounds':0,'formal_test_episodes':0,'api_calls':0}
        durable_json(output/'audited_rows.json.gz',records);durable_json(output/'report.json',report)
        lines=['# 固定BC方法：三个seed结项验证','',report['scope'],'',
               '| 方法 | 等权减损 pp | 正常 pp | 攻击 pp |','| --- | ---: | ---: | ---: |']
        for name,s in [*report['per_seed'].items(),('three_seed_mean',report['three_seed_mean'])]:
            lines.append(f"| {name} | {s['balanced']['mean_pp']:.5f} | {s['normal']['mean_pp']:.5f} | {s['attack']['mean_pp']:.5f} |")
        for name,s in report['reference_means_pp'].items():
            lines.append(f"| {name} | {s['balanced']:.5f} | {s['normal']:.5f} | {s['attack']:.5f} |")
        lines+=['',f"三seed等权区间/依赖敏感性：{report['three_seed_mean']['balanced']}",
                '',f"正常影响：{report['outcomes']['three_seed_mean']['normal']}",
                '',f"攻击影响：{report['outcomes']['three_seed_mean']['attack']}",
                '',report['uncertainty'],
                '','所有seed先在同一社会内平均，再社会级配对；768条不是768独立社会。旧PPO检查点由validation选择，新方法seed11参与内部开发，两者选择机制不同。',
                '完整曲线、两轮负结果保留。P*=alarm_refresh不变；不再追加训练轮或改市场/攻击/collapse，正式test仍关闭。']
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        names=['report.json','audited_rows.json.gz','REPORT.md','final_three_seed_curves.png','final_three_seed_curves.pdf']
        sources=[Path(__file__),ROOT/'scripts/report_combination_bc_pilot.py',ROOT/'scripts/report_combination_baselines.py',
                 ROOT/'scripts/audit_combination_seed_dependencies.py']
        durable_json(output/'acceptance.json',{'execution_id':execution,'audit_complete':True,
            'analysis_sources_sha256':{str(p.resolve()):file_hash(p) for p in sources},
            'historical_acceptance_sha256':{str(d/'acceptance.json'):file_hash(d/'acceptance.json') for d in (BASELINE,PPO)},
            'outputs_sha256':{n:file_hash(output/n) for n in names}})
        durable_json(output/'progress.json',{**progress,'stage':'final_bc_validation_audited','audit_complete':True})
        print('Final three-seed validation audit complete; formal test remains closed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.input.resolve(),a.workers)
