"""Reuse accepted train demonstrations and diagnose committed PPO logs only.

No optimization, simulator rollout, provider call, or test-outcome access.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from wolfgym.benchmark.combination import episode_plan
from wolfgym.env.combination_learning import encode_public_view
from wolfgym.evaluation.combination_host import file_hash, read_json, load_detector
from wolfgym.evaluation.combination_public import BaselinePolicy, PublicHistory, canonical_json, digest

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    results = {}
    for condition in ('normal', 'attack'):
        selected = [row for row in rows if row['result']['condition'] == condition]
        if not selected:
            raise ValueError('missing condition')
        counts = Counter()
        first, exhausted = [], []
        zero = 0
        for row in selected:
            actions = row['actions']
            if len(actions) != 30 or any(type(a) is not int or a not in range(4) for a in actions):
                raise ValueError('invalid action log')
            days = [d for d,a in enumerate(actions) if a]
            if len(days) > 3 or 4*len(days) != row['result']['cost']:
                raise ValueError('budget mismatch')
            counts.update(actions)
            zero += not days
            if days:first.append(days[0])
            if len(days)==3:exhausted.append(days[-1])
        benefits = np.array([r['result']['H_reduction'] for r in selected])
        if not np.all(np.isfinite(benefits)):raise ValueError('nonfinite reward')
        results[condition] = {'episodes':len(selected),'action_counts':{str(a):counts[a] for a in range(4)},
            'all_wait_episode_fraction':zero/len(selected),
            'three_freezes_fraction':len(exhausted)/len(selected),
            'mean_first_freeze_day_if_any':float(np.mean(first)) if first else None,
            'mean_budget_exhausted_day_if_any':float(np.mean(exhausted)) if exhausted else None,
            'freeze_days_total':counts[1]+counts[2]+counts[3],
            'benefit_mean_pp':float(100*benefits.mean()),
            'benefit_pp_quantiles_0_25_50_75_100':np.quantile(100*benefits,[0,.25,.5,.75,1]).tolist(),
            'terminal_training_reward_scale1000_std':float(np.std(1000*benefits)),
            'negative_benefit_fraction':float(np.mean(benefits<0)),
            'zero_benefit_fraction':float(np.mean(benefits==0))}
    return results


def committed_logs(learning, seed, checkpoint, kind, execution_id, sources):
    path = learning/f'seed_{seed}'/f'checkpoint_{checkpoint}'/f'{kind}_complete.json'
    receipt = read_json(path)
    sources[str(path.relative_to(ROOT))] = file_hash(path)
    if receipt['execution_id'] != execution_id:raise ValueError('foreign checkpoint')
    rows=[]
    for relative,expected in receipt['sha256'].items():
        if Path(relative).name.startswith('worker_') and relative.endswith('.jsonl'):
            p=path.parent/relative
            if file_hash(p)!=expected:raise ValueError('committed worker log changed')
            sources[str(p.relative_to(ROOT))]=expected
            rows.extend(json.loads(line) for line in p.read_text().splitlines())
    if len(rows)!=(500 if kind=='training' else 256):raise ValueError('incomplete committed segment')
    ids=[r['result']['episode_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate episode in segment')
    expected_split='train' if kind=='training' else 'validation'
    if any(r['result']['split']!=expected_split for r in rows):raise ValueError('foreign split')
    return rows


def prepare(output):
    if output.exists():raise ValueError('new immutable preparation directory required')
    baseline=ROOT/'data/combination_baselines_20260926'
    learning=ROOT/'data/combination_learning_20260926'
    sources={}
    for folder in (baseline,learning):
        acceptance=read_json(folder/'acceptance.json')
        for relative,expected in acceptance['outputs_sha256'].items():
            if file_hash(ROOT/relative)!=expected:raise ValueError('accepted input changed')
        sources[str((folder/'acceptance.json').relative_to(ROOT))]=file_hash(folder/'acceptance.json')
    plan=[e.to_dict() for e in episode_plan('train')]
    groups=list(dict.fromkeys(e['group_id'] for e in plan))
    partitions={'demonstrations':groups[:64], 'development_monitor':groups[64:128], 'student_collection':groups[128:256]}
    manifests={name:[e for e in plan if e['group_id'] in group] for name,group in partitions.items()}
    declared=[e for e in read_json(baseline/'manifest.json')['episodes'] if e['split']=='train']
    if declared!=manifests['demonstrations']:raise ValueError('baseline subset differs from proposed demonstrations')
    detector=load_detector(ROOT/'data/task_validation_v4_20260923/detector.json')
    samples, provenance, vectors=[],[],[]
    teacher_rows=[]
    counts=Counter(); eligible=Counter()
    for episode in manifests['demonstrations']:
        directory=baseline/'train'/str(episode['trajectory_seed'])/episode['condition']
        marker=read_json(directory/'complete.json')
        artifact_path=directory/'alarm_refresh.json.gz'
        if marker['episode']!=episode or file_hash(artifact_path)!=marker['sha256']['alarm_refresh.json.gz']:
            raise ValueError('teacher trajectory fingerprint/identity mismatch')
        sources[str(artifact_path.relative_to(ROOT))]=file_hash(artifact_path)
        artifact=read_json(artifact_path)
        if len(artifact['log'])!=30:raise ValueError('incomplete demonstration')
        history=PublicHistory()
        teacher=BaselinePolicy({'id':'alarm_refresh','family':'alarm_refresh'},detector)
        actions=[]
        for day,row in enumerate(artifact['log']):
            frame=row['frame']; action=row['action']
            if frame['day']!=day:raise ValueError('day mismatch')
            view=history.observe(frame)
            if digest(view)!=row['input_sha256'] or teacher(view)!=action:
                raise ValueError('teacher public history/action replay mismatch')
            if row['invalid_action']:raise ValueError('invalid teacher action')
            sample_id=len(samples)
            samples.append({'sample_id':sample_id,'view':view,'action':action})
            provenance.append({'sample_id':sample_id,'episode':episode,'day':day,'source':str(artifact_path.relative_to(ROOT))})
            vectors.append(encode_public_view(view))
            counts[action]+=1
            if frame['remaining_budget']>=4:eligible[action]+=1
            history.record(frame,action);actions.append(action)
        teacher_rows.append({'result':artifact['result'],'actions':actions})
    matrix=np.stack(vectors)
    if matrix.shape!=(3840,1895) or not np.all(np.isfinite(matrix)):raise ValueError('encoded demo shape/finite check failed')
    diagnostics={}
    declaration=read_json(learning/'manifest.json')
    for seed in (11,22,33):
        diagnostics[str(seed)]={}
        train_seen=[]
        for cp in (500,1000,1500,2000):
            train=committed_logs(learning,seed,cp,'training',declaration['execution_id'],sources)
            validation=committed_logs(learning,seed,cp,'validation',declaration['execution_id'],sources)
            train_seen.extend(row['result']['episode_id'] for row in train)
            diagnostics[str(seed)][str(cp)]={'training_segment':summarize(train),'validation':summarize(validation)}
        if len(set(train_seen))!=2000 or set(train_seen)!={e['episode_id'] for e in plan}:
            raise ValueError('committed seed did not visit complete train list once')
    output.mkdir(parents=True)
    with gzip.open(output/'demonstrations.jsonl.gz','wt',encoding='utf-8') as stream:
        for sample in samples:stream.write(canonical_json(sample)+'\n')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'partitions.json').write_text(json.dumps(manifests,indent=2)+'\n')
    report={'demonstration_societies':64,'demonstration_episodes':128,'samples':len(samples),
        'teacher':'alarm_refresh_public_rule_not_optimal_labels','action_counts':dict(counts),
        'action_counts_when_budget_allows_freeze':dict(eligible),
        'always_wait_sample_accuracy':counts[0]/len(samples),
        'always_wait_accuracy_on_budget_eligible_frames':eligible[0]/sum(eligible.values()),
        'encoded_shape':list(matrix.shape),'encoded_abs_quantiles_50_95_99_100':np.quantile(np.abs(matrix),[.5,.95,.99,1]).tolist(),
        'constant_dimensions_on_this_subset':int(np.sum(np.ptp(matrix,axis=0)==0)),
        'teacher_trajectory_summary':summarize(teacher_rows),'ppo_logs':diagnostics,
        'missing_diagnostics':['training action probabilities/entropy and optimizer losses not persisted; no causal diagnosis from terminal/action logs'],
        'market_days_run':0,'optimizer_updates':0,'remote_api_calls':0,'test_outcomes_read':False,
        'source_sha256':sources}
    (output/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    receipt={'status':'demonstrations_prepared_no_training','source_sha256':{str(Path(__file__).relative_to(ROOT)):file_hash(__file__)},
             'outputs_sha256':{p.name:file_hash(p) for p in output.iterdir() if p.is_file()}}
    (output/'preparation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('samples','action_counts','action_counts_when_budget_allows_freeze',
        'always_wait_sample_accuracy','always_wait_accuracy_on_budget_eligible_frames','encoded_abs_quantiles_50_95_99_100','constant_dimensions_on_this_subset')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
