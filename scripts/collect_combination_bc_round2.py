"""Collect the fixed first-round student states, then audit and label every day."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
from pathlib import Path
import os
import shutil
import time

import torch

from run_combination_bc_pilot import monitor_job
from report_combination_bc_pilot import audit_one
from wolfgym.evaluation.combination_host import read_json, file_hash, load_detector
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import BaselinePolicy, PublicHistory, digest

ROOT=Path(__file__).resolve().parents[1]
FIRST=ROOT/'data/combination_bc_round1_20260927'
PREP=ROOT/'data/combination_imitation_preparation_20260927'


def label_student_history(artifact):
    """Teacher sees student states and actual action history; never changes execution."""
    teacher=BaselinePolicy({'id':'alarm_refresh','family':'alarm_refresh'},
                          load_detector(ROOT/'data/task_validation_v4_20260923/detector.json'))
    history=PublicHistory();samples=[];provenance=[];remaining=12.
    if len(artifact['decisions'])!=30:raise ValueError('incomplete student episode')
    identity={k:artifact['result'][k] for k in ('split','trajectory_seed','condition','group_id','episode_id')}
    if identity['split']!='train':raise ValueError('collection must use train only')
    for day,row in enumerate(artifact['decisions']):
        frame=row['frame'];actual=row['action']
        if row['day']!=day or frame['day']!=day or frame['remaining_budget']!=remaining:
            raise ValueError('student day/budget mismatch')
        view=history.observe(frame)
        if digest(view)!=row['input_sha256']:raise ValueError('student input changed')
        label=teacher(view)
        if not frame['action_mask'][actual] or not frame['action_mask'][label]:raise ValueError('illegal label or action')
        sample_id=digest(['student_collection',identity['episode_id'],day])
        samples.append({'sample_id':sample_id,'view':view,'action':label})
        provenance.append({'sample_id':sample_id,'episode':identity,'day':day,
                           'executed_student_action':actual,'teacher_action':label})
        history.record(frame,actual)
        remaining-=4*bool(actual)
    return samples,provenance


def declaration(workers):
    receipt=read_json(FIRST/'acceptance.json')
    for name,sha in receipt['outputs_sha256'].items():
        if file_hash(FIRST/name)!=sha:raise ValueError('first-round audit output drift')
    report=read_json(FIRST/'report.json');selected=report['selected_first_round_student']
    if report['results'][selected]['balanced_benefit_pp']>=report['results']['alarm_refresh']['balanced_benefit_pp']:
        raise ValueError('declared second-round gate not met')
    if selected!='bc_unweighted':raise ValueError('second-round contract selected student drift')
    first_manifest=read_json(FIRST/'manifest.json')
    if digest(first_manifest)!=receipt['execution_id']:raise ValueError('first-round execution mismatch')
    for path,sha in first_manifest['source_sha256'].items():
        if file_hash(path)!=sha:raise ValueError('first-round source drift')
    for name,sha in read_json(FIRST/selected/'training_complete.json')['sha256'].items():
        if file_hash(FIRST/selected/name)!=sha:raise ValueError('first-round model/optimizer drift')
    for path,sha in read_json(PREP/'acceptance.json')['outputs_sha256'].items():
        if file_hash(ROOT/path)!=sha:raise ValueError('prepared partition/data drift')
    partitions=read_json(PREP/'partitions.json');episodes=partitions['student_collection']
    roots={e['trajectory_seed'] for e in episodes}
    if len(episodes)!=256 or len(roots)!=128 or len({e['episode_id'] for e in episodes})!=256:
        raise ValueError('collection size drift')
    if any(e['split']!='train' for e in episodes):raise ValueError('non-train collection')
    for root in roots:
        if {e['condition'] for e in episodes if e['trajectory_seed']==root}!={'normal','attack'}:
            raise ValueError('unpaired collection')
    for name in ('demonstrations','development_monitor'):
        if roots & {e['trajectory_seed'] for e in partitions[name]}:raise ValueError('partition overlap')
    sources=[Path(__file__),ROOT/'scripts/run_combination_bc_pilot.py',ROOT/'scripts/report_combination_bc_pilot.py',
             ROOT/'scripts/report_combination_baselines.py',ROOT/'src/wolfgym/evaluation/combination_bc.py',
             ROOT/'src/wolfgym/env/combination.py',ROOT/'src/wolfgym/env/combination_v51.json',
             ROOT/'src/wolfgym/env/combination_learning.py',ROOT/'src/wolfgym/evaluation/combination_public.py',
             ROOT/'docs/EXPERIMENT_PROTOCOL.md']
    return {'version':'combination-bc-round2-collection-v1','episodes':episodes,'selected_student':selected,
        'source_sha256':{str(p.resolve()):file_hash(p) for p in sources},'workers':workers,
        'parent_acceptance_sha256':file_hash(FIRST/'acceptance.json'),
        'parent_model_sha256':file_hash(FIRST/selected/'model_1000.pt'),
        'parent_optimizer_state_sha256':file_hash(FIRST/selected/'latest.pt'),
        'data_acceptance_sha256':file_hash(PREP/'acceptance.json'),
        'planned_branches':{'fixed_data':1000,'aggregated_data':1000},'class_weights':[1.,1.,1.,1.],
        'batch_size':128,'api_allowed':False,'test_allowed':False}


def main(output,workers):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=declaration(workers);execution=digest(manifest);path=output/'collection_manifest.json'
        if path.exists() and read_json(path)!=manifest:raise ValueError('collection configuration drift')
        durable_json(path,manifest)
        if (output/'collection_acceptance.json').exists():
            receipt=read_json(output/'collection_acceptance.json')
            if receipt['execution_id']!=execution:raise ValueError('foreign collection receipt')
            for name,sha in receipt['outputs_sha256'].items():
                if file_hash(output/name)!=sha:raise ValueError('collection output drift')
            print('Collection already audited; no rerun',flush=True);return
        selected=manifest['selected_student'];dest=output/selected/'model_1000.pt';dest.parent.mkdir(exist_ok=True)
        if dest.exists() and file_hash(dest)!=manifest['parent_model_sha256']:raise ValueError('student checkpoint drift')
        if not dest.exists():shutil.copyfile(FIRST/selected/'model_1000.pt',dest)
        jobs=[(ep,selected,str(dest),str(output),execution) for ep in manifest['episodes']]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(monitor_job,*job) for job in jobs]
            for count,f in enumerate(as_completed(futures),1):
                f.result()
                durable_json(output/'progress.json',{'stage':'collect_student_states','completed':count,'total':256,
                    'complete':False,'pid':os.getpid(),'updated_unix':time.time()})
                if count%16==0:print(f'collected {count}/256',flush=True)
        durable_json(output/'progress.json',{'stage':'audit_student_states','completed':256,'total':256,
            'complete':False,'pid':os.getpid(),'updated_unix':time.time()})
        audit_jobs=[(str(output),e,selected,execution) for e in manifest['episodes']]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records=[]
            for count,r in enumerate(pool.map(audit_one,audit_jobs),1):
                records.append(r)
                if count%32==0:print(f'audited {count}/256',flush=True)
        samples=[];provenance=[]
        for ep in manifest['episodes']:
            artifact=read_json(output/'monitor'/selected/ep['episode_id'].replace(':','_')/'artifact.json.gz')
            s,p=label_student_history(artifact);samples.extend(s);provenance.extend(p)
        if len(samples)!=7680 or len({s['sample_id'] for s in samples})!=7680:raise ValueError('incomplete/duplicate samples')
        eligible=[s for s in samples if s['view']['observations'][-1]['remaining_budget']>=4]
        statistics={'all_frames':len(samples),'eligible_frames':len(eligible),
                    'all_teacher_action_counts':{str(k):sum(s['action']==k for s in samples) for k in range(4)},
                    'eligible_teacher_action_counts':{str(k):sum(s['action']==k for s in eligible) for k in range(4)},
                    'action_disagreements':sum(p['teacher_action']!=p['executed_student_action'] for p in provenance),
                    'policy_episodes':256,'public_actions_replayed':256*30,'account_closes_checked':256*3*30*490,
                    'optimization_updates':0,'validation_test_episodes':0,'api_calls':0}
        for name,value in [('student_demonstrations.json.gz',samples),('student_provenance.json.gz',provenance),
                           ('collection_audited_rows.json.gz',records),('collection_statistics.json',statistics)]:
            durable_json(output/name,value)
        names=['student_demonstrations.json.gz','student_provenance.json.gz','collection_audited_rows.json.gz','collection_statistics.json']
        durable_json(output/'collection_acceptance.json',{'execution_id':execution,
            'outputs_sha256':{n:file_hash(output/n) for n in names},'audit_complete':True})
        durable_json(output/'progress.json',{'stage':'collection_audited_training_pending','complete':True,
            'collection_audit_complete':True,'completed':256,'total':256,'optimization_updates':0})
        print('Student collection audited; both continuation training branches remain pending',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.output.resolve(),a.workers)
