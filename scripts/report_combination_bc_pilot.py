"""Audit every first-round BC monitor artifact before descriptive selection.

Does not simulate markets or access validation/test; reports train development only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import math
from pathlib import Path

import numpy as np
import torch

from report_combination_baselines import wealth_table, independent_score, check_score
from wolfgym.env.combination_contract import primary_contract
from wolfgym.evaluation.combination_bc import BCPolicy
from wolfgym.evaluation.combination_host import read_json, file_hash, load_detector
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import BaselinePolicy, PublicHistory, digest

ROOT=Path(__file__).resolve().parents[1]
POLICIES=('bc_unweighted','bc_balanced_wait_freeze','alarm_refresh')


def audit_one(job):
    output,identity,policy_id,execution_id=job
    torch.set_num_threads(1)
    output=Path(output)
    directory=output/'monitor'/policy_id/identity['episode_id'].replace(':','_')
    marker=read_json(directory/'complete.json')
    sha=file_hash(directory/'artifact.json.gz')
    if marker!={'execution_id':execution_id,'sha256':sha}:raise ValueError('artifact hash/identity changed')
    a=read_json(directory/'artifact.json.gz');r=a['result']
    if a['execution_id']!=execution_id or a['policy_id']!=policy_id or any(r[k]!=v for k,v in identity.items()):
        raise ValueError('artifact provenance mismatch')
    accounts=a['accounts'];rows=a['financial_rows']
    if set(accounts)!= {'reference','wait','policy'} or set(rows)!=set(accounts):raise ValueError('missing world')
    ref,initial=wealth_table(accounts['reference'],accounts['reference'])
    scores={}
    for world in ('wait','policy'):
        values,_=wealth_table(accounts[world],accounts['reference'])
        scores[world]=independent_score(ref,values,initial,rows['reference'],rows[world],primary_contract()['harm_threshold'])
        check_score(scores[world],r[world])
    benefit=float(scores['wait']['H'][-1]-scores['policy']['H'][-1])
    if not math.isclose(benefit,r['H_reduction'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('benefit mismatch')
    detector=load_detector(ROOT/'data/task_validation_v4_20260923/detector.json')
    spec={'id':'alarm_refresh','family':'alarm_refresh'}
    policy=BaselinePolicy(spec,detector) if policy_id=='alarm_refresh' else BCPolicy(output/policy_id/'model_1000.pt')
    teacher=BaselinePolicy(spec,detector)
    history=PublicHistory();remaining=12.;days=[];conf={'tp':0,'fp':0,'fn':0,'tn':0,'correct_target':0}
    if len(a['decisions'])!=30:raise ValueError('incomplete decisions')
    for day,log in enumerate(a['decisions']):
        frame=log['frame']
        if log['day']!=day or frame['day']!=day or frame['remaining_budget']!=remaining:
            raise ValueError('day/budget trace mismatch')
        if frame['action_mask']!=[1,*([int(remaining>=4)]*3)] or len(frame['action_targets'])!=3:
            raise ValueError('mask/asset coverage mismatch')
        view=history.observe(frame);action=policy(view);label=teacher(view)
        if action!=log['action'] or digest(view)!=log['input_sha256'] or not frame['action_mask'][action]:
            raise ValueError('model/rule action/input replay mismatch')
        if remaining>=4:
            key='tp' if action and label else 'fp' if action else 'fn' if label else 'tn'
            conf[key]+=1
            conf['correct_target']+=int(action==label and label!=0)
        history.record(frame,action)
        if action:remaining-=4;days.append(day)
    if r['cost']!=12-remaining:raise ValueError('cost mismatch')
    financial={}
    for world in rows:
        if len(rows[world])!=30:raise ValueError('incomplete financial trace')
        financial[world]={k:math.fsum(x['financial_outcomes'][k] for x in rows[world])
                          for k in r['financial_outcomes'][world]}
        if financial[world]!=r['financial_outcomes'][world]:raise ValueError('financial total mismatch')
    return {**identity,'policy_id':policy_id,'benefit':benefit,'H_wait':float(scores['wait']['H'][-1]),
        'H_policy':float(scores['policy']['H'][-1]),'event_wait':scores['wait']['event'],
        'event_policy':scores['policy']['event'],'prevented':scores['wait']['event'] and not scores['policy']['event'],
        'new_event':scores['policy']['event'] and not scores['wait']['event'],'cost':r['cost'],
        'action_days':days,'teacher_agreement_on_student_states':conf,
        'financial_delta':{k:financial['policy'][k]-financial['wait'][k] for k in financial['policy']},
        'reference_wait_sha256':digest({k:{'accounts':accounts[k],'rows':rows[k]} for k in ('reference','wait')}),
        'artifact_sha256':sha}


def summarize(records):
    expected={(p,r['episode_id']) for p in POLICIES for r in records if r['policy_id']==POLICIES[0]}
    actual=[(r['policy_id'],r['episode_id']) for r in records]
    if len(records)!=384 or len(set(actual))!=384 or set(actual)!=expected:raise ValueError('incomplete monitor grid')
    groups={r['group_id'] for r in records}
    if len(groups)!=64 or any({r['condition'] for r in records if r['group_id']==g}!={'normal','attack'} for g in groups):
        raise ValueError('incomplete society pairs')
    for episode in {r['episode_id'] for r in records}:
        if len({r['reference_wait_sha256'] for r in records if r['episode_id']==episode})!=1:
            raise ValueError('policies do not share matched reference/WAIT')
    result={}
    for p in POLICIES:
        subset=[r for r in records if r['policy_id']==p]
        summary={'balanced_benefit_pp':100*float(np.mean([r['benefit'] for r in subset])),
                 'mean_cost':float(np.mean([r['cost'] for r in subset]))}
        for condition in ('normal','attack'):
            rs=[r for r in subset if r['condition']==condition]
            summary[condition]={'benefit_pp':100*float(np.mean([r['benefit'] for r in rs])),
                'remaining_H_pp':100*float(np.mean([r['H_policy'] for r in rs])),
                'WAIT_events':sum(r['event_wait'] for r in rs),'policy_events':sum(r['event_policy'] for r in rs),
                'prevented':sum(r['prevented'] for r in rs),'new_events':sum(r['new_event'] for r in rs),
                'intervened_episodes':sum(bool(r['action_days']) for r in rs),
                'mean_action_count':float(np.mean([len(r['action_days']) for r in rs])),
                'mean_cost':float(np.mean([r['cost'] for r in rs])),
                'action_day_counts':{str(d):sum(d in r['action_days'] for r in rs) for d in range(30)},
                'mean_financial_delta':{k:float(np.mean([r['financial_delta'][k] for r in rs])) for k in rs[0]['financial_delta']}}
        c={k:sum(r['teacher_agreement_on_student_states'][k] for r in subset) for k in ('tp','fp','fn','tn','correct_target')}
        summary['student_state_teacher_agreement']={**c,
            'freeze_precision':c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None,
            'freeze_recall':c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None,
            'correct_target_recall':c['correct_target']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None}
        result[p]=summary
    selected=min(POLICIES[:2],key=lambda p:(-result[p]['balanced_benefit_pp'],result[p]['mean_cost'],p))
    return result,selected


def main(output,workers):
    with (output/'.audit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=read_json(output/'progress.json')
        if not progress['complete'] or progress['completed']!=384:raise ValueError('monitor still incomplete')
        manifest=read_json(output/'manifest.json');execution_id=digest(manifest)
        for p,h in manifest['source_sha256'].items():
            if file_hash(p)!=h:raise ValueError('source drift')
        prep=ROOT/'data/combination_imitation_preparation_20260927'
        if file_hash(prep/'acceptance.json')!=manifest['data_receipt']:raise ValueError('data receipt changed')
        for p,h in read_json(prep/'acceptance.json')['outputs_sha256'].items():
            if file_hash(ROOT/p)!=h:raise ValueError('prepared data drift')
        for p in POLICIES[:2]:
            receipt=read_json(output/p/'training_complete.json')
            if receipt['execution_id']!=execution_id or receipt['updates']!=1000:raise ValueError('training receipt drift')
            for n,h in receipt['sha256'].items():
                if file_hash(output/p/n)!=h:raise ValueError('model/curve drift')
            if [r['update'] for r in read_json(output/p/'curve.json')]!=list(range(0,1001,25)):
                raise ValueError('curve schedule incomplete')
        jobs=[(str(output),ep,p,execution_id) for p in POLICIES for ep in manifest['monitor']]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records=[]
            for i,r in enumerate(pool.map(audit_one,jobs),1):
                records.append(r)
                if i%32==0:print(f'audited {i}/384',flush=True)
        results,selected=summarize(records)
        report={'execution_id':execution_id,'scope':'train-development-only; selection is not independent confirmation',
            'audit':{'episodes':384,'societies':64,'ordinary_account_closes_checked':384*3*30*490,
                     'public_actions_replayed':384*30,'market_resimulation':False},
            'results':results,'selected_first_round_student':selected,'P_star':'alarm_refresh',
            'validation_test_episodes':0,'remote_api_calls':0,
            'uncertainty':'descriptive train monitor means only; no independent generalization claim or IID interval'}
        durable_json(output/'audited_rows.json.gz',records);durable_json(output/'report.json',report)
        lines=['# 第一轮 BC：完整内部监测与审计','',report['scope'],'',
            '| 策略 | 等权收益 pp | 正常 pp | 攻击 pp | 正常新增事件 | 正常行动轨迹/64 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for p,s in results.items():
            lines.append(f"| {p} | {s['balanced_benefit_pp']:.5f} | {s['normal']['benefit_pp']:.5f} | {s['attack']['benefit_pp']:.5f} | {s['normal']['new_events']} | {s['normal']['intervened_episodes']} |")
        lines+=['',f'后续学生：{selected}。P*=alarm_refresh保持；不根据训练召回选择。',
                '','384条、64社会完整网格及三世界账户/金融结果重算通过，11520个模型/规则动作重放通过。',
                '公开输入来自已保存frame和history，本审计没有重新模拟市场以验证frame；原始账户独立核算与动作重放不能冒称全环境重放。',
                '学生状态上的教师一致性受各自策略预算/轨迹影响，不能把跨策略分类指标当同一固定测试集准确率。',
                '本轮只有开发seed11和内部train监测，不是三seed或独立validation/test；没有付费API调用。']
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        receipt={'execution_id':execution_id,'analysis_sources_sha256':{str(Path(__file__).resolve()):file_hash(__file__),
            str(ROOT/'scripts/report_combination_baselines.py'):file_hash(ROOT/'scripts/report_combination_baselines.py')},
            'outputs_sha256':{n:file_hash(output/n) for n in ('report.json','audited_rows.json.gz','REPORT.md','round1_training_curves.png','round1_training_curves.pdf')}}
        durable_json(output/'acceptance.json',receipt)
        durable_json(output/'progress.json',{**progress,'audit_complete':True,'stage':'round1_audited'})
        print(f'Complete audit; selected {selected}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.input.resolve(),a.workers)
