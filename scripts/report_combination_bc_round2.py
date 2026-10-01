"""Audit the complete second-round monitor and compare all fixed pilot candidates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path

import numpy as np

from report_combination_bc_pilot import audit_one
from train_combination_bc_round2 import BRANCHES, declaration, verify_receipt
from wolfgym.evaluation.combination_host import read_json, file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest

ROOT=Path(__file__).resolve().parents[1]
FIRST=ROOT/'data/combination_bc_round1_20260927'


def summarize(rows,episodes,policies):
    expected={(p,e['episode_id']) for p in policies for e in episodes}
    actual=[(r['policy_id'],r['episode_id']) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('incomplete/duplicate comparison grid')
    for ep in episodes:
        matched=[r for r in rows if r['episode_id']==ep['episode_id']]
        if any(any(r[k]!=v for k,v in ep.items()) for r in matched):raise ValueError('episode identity drift')
        if len({r['reference_wait_sha256'] for r in matched})!=1:raise ValueError('unpaired reference/WAIT')
    results={}
    for policy in policies:
        subset=[r for r in rows if r['policy_id']==policy]
        s={'balanced_benefit_pp':100*float(np.mean([r['benefit'] for r in subset])),
           'mean_cost':float(np.mean([r['cost'] for r in subset]))}
        for condition in ('normal','attack'):
            rs=[r for r in subset if r['condition']==condition]
            s[condition]={'benefit_pp':100*float(np.mean([r['benefit'] for r in rs])),
                'remaining_H_pp':100*float(np.mean([r['H_policy'] for r in rs])),
                'WAIT_events':sum(r['event_wait'] for r in rs),'policy_events':sum(r['event_policy'] for r in rs),
                'prevented':sum(r['prevented'] for r in rs),'new_events':sum(r['new_event'] for r in rs),
                'intervened_episodes':sum(bool(r['action_days']) for r in rs),
                'mean_action_count':float(np.mean([len(r['action_days']) for r in rs])),
                'mean_cost':float(np.mean([r['cost'] for r in rs])),
                'action_day_counts':{str(day):sum(day in r['action_days'] for r in rs) for day in range(30)},
                'mean_financial_delta':{k:float(np.mean([r['financial_delta'][k] for r in rs])) for k in rs[0]['financial_delta']}}
        conf={k:sum(r['teacher_agreement_on_student_states'][k] for r in subset) for k in ('tp','fp','fn','tn','correct_target')}
        s['student_state_teacher_agreement']={**conf,
            'freeze_precision':conf['tp']/(conf['tp']+conf['fp']) if conf['tp']+conf['fp'] else None,
            'freeze_recall':conf['tp']/(conf['tp']+conf['fn']) if conf['tp']+conf['fn'] else None,
            'correct_target_recall':conf['correct_target']/(conf['tp']+conf['fn']) if conf['tp']+conf['fn'] else None}
        results[policy]=s
    groups=sorted({e['group_id'] for e in episodes});differences=[]
    for group in groups:
        values={p:np.mean([r['benefit'] for r in rows if r['policy_id']==p and r['group_id']==group]) for p in BRANCHES}
        differences.append({'group_id':group,'aggregated_minus_fixed_pp':100*float(values[BRANCHES[1]]-values[BRANCHES[0]])})
    return results,differences


def main(output,workers):
    with (output/'.training_audit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=read_json(output/'training_progress.json')
        if not progress['complete'] or progress['completed']!=256:raise ValueError('monitor still incomplete')
        manifest=read_json(output/'training_manifest.json');execution=digest(manifest)
        if manifest!=declaration(output,manifest['workers']):raise ValueError('training declaration drift')
        for branch in BRANCHES:
            r=read_json(output/branch/'training_complete.json')
            if r['execution_id']!=execution or r['additional_updates']!=1000 or r['total_updates']!=2000:
                raise ValueError('training update count mismatch')
            for n,h in r['sha256'].items():
                if file_hash(output/branch/n)!=h:raise ValueError('model/curve drift')
            curve=read_json(output/branch/'curve.json')
            if [x['additional_update'] for x in curve]!=list(range(0,1001,25)):raise ValueError('curve schedule incomplete')
        curves=[read_json(output/b/'curve.json') for b in BRANCHES]
        if any(curves[0][0][k]!=curves[1][0][k] for k in ('original','student')):
            raise ValueError('unequal initial predictions on common probes')
        jobs=[(str(output),ep,b,execution) for b in BRANCHES for ep in manifest['monitor']]
        records=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i,r in enumerate(pool.map(audit_one,jobs),1):
                records.append(r)
                if i%32==0:print(f'audited {i}/256',flush=True)
        verify_receipt(FIRST,'acceptance.json')
        cached=read_json(FIRST/'audited_rows.json.gz')
        policies=['bc_unweighted','bc_balanced_wait_freeze','alarm_refresh',*BRANCHES]
        results,differences=summarize([*cached,*records],manifest['monitor'],policies)
        report={'execution_id':execution,'scope':'train-development only; no independent validation/test',
            'audit':{'new_policy_episodes':256,'societies':64,'ordinary_account_closes_checked':256*3*30*490,
                     'public_actions_replayed':256*30,'cached_first_round_episodes':384,'market_resimulation':False},
            'results':results,'paired_society_aggregated_minus_fixed_pp':differences,
            'mean_aggregated_minus_fixed_pp':float(np.mean([r['aggregated_minus_fixed_pp'] for r in differences])),
            'uncertainty':'descriptive train monitor; no independent confirmation or IID interval',
            'P_star':'alarm_refresh','further_development_rounds':0,'validation_test_episodes':0,'api_calls':0}
        durable_json(output/'training_audited_rows.json.gz',records);durable_json(output/'training_report.json',report)
        lines=['# 第二轮：等更新次数对照与学生状态聚合','',report['scope'],'',
               '| 策略 | 等权收益 pp | 正常 pp | 攻击 pp | 正常干预/64 | 正常新增事件 |',
               '| --- | ---: | ---: | ---: | ---: | ---: |']
        for p,s in results.items():
            lines.append(f"| {p} | {s['balanced_benefit_pp']:.5f} | {s['normal']['benefit_pp']:.5f} | {s['attack']['benefit_pp']:.5f} | {s['normal']['intervened_episodes']} | {s['normal']['new_events']} |")
        lines+=['',f"聚合减固定数据平均差：{report['mean_aggregated_minus_fixed_pp']:.6f} pp。",
            '','两个分支均从同一模型/优化器/RNG出发，各追加1000更新、权重全1；差别是监督数据来源。',
            '新采集256条/7680状态，仅聚合分支用于优化；原数据可选择帧2891，学生帧6529，聚合9420。相同更新不等于相同数据/仿真预算。',
            '完整曲线每25更新，包括两个共同诊断集合。学生集合对聚合分支是训练集，对固定分支是train开发诊断，不称独立验证。',
            '所有结果都保留；不加第三轮，不改市场/攻击/collapse，P*=alarm_refresh保持。最终三seed与独立test尚未完成。',
            '原始账户独立重算及落盘公开输入模型重放通过；没有重新模拟市场验证frame，不冒称全环境重放。']
        (output/'TRAINING_REPORT.md').write_text('\n'.join(lines)+'\n')
        names=['training_report.json','training_audited_rows.json.gz','TRAINING_REPORT.md','round2_training_curves.png','round2_training_curves.pdf']
        durable_json(output/'training_acceptance.json',{'execution_id':execution,
            'analysis_sources_sha256':{str(Path(__file__).resolve()):file_hash(__file__),
                str(ROOT/'scripts/report_combination_bc_pilot.py'):file_hash(ROOT/'scripts/report_combination_bc_pilot.py'),
                str(ROOT/'scripts/report_combination_baselines.py'):file_hash(ROOT/'scripts/report_combination_baselines.py')},
            'outputs_sha256':{n:file_hash(output/n) for n in names}})
        durable_json(output/'training_progress.json',{**progress,'audit_complete':True,'stage':'round2_audited'})
        print('Second-round complete grid audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.input.resolve(),a.workers)
