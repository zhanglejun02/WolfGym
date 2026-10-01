#!/usr/bin/env python
"""Independent validation-account audit and deterministic learned-action replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path

import numpy as np

from report_combination_baselines import wealth_table, independent_score, check_score, bootstrap
from train_combination_policy import check_receipt, read_logs
from wolfgym.benchmark.combination import validate_episode_manifest
from wolfgym.env.combination_contract import validate_primary_scenario
from wolfgym.env.combination_learning import encode_public_view
from wolfgym.evaluation.combination_host import read_json, write_json, file_hash, aggregate_financial
from wolfgym.evaluation.combination_public import PublicHistory, digest
from wolfgym.observations.serialization import assert_no_hidden_fields

ROOT = Path(__file__).resolve().parents[1]
MODEL = None


def load_model(path):
    from sb3_contrib import MaskablePPO
    import torch
    global MODEL
    torch.set_num_threads(1)
    MODEL = MaskablePPO.load(path, device='cpu')


def audit_policy(job):
    path, cache_directory, threshold, expected = job
    artifact = read_json(path)
    result = artifact['result']
    if any(result[k] != v for k, v in expected.items()):
        raise ValueError('validation identity mismatch')
    cache_directory = Path(cache_directory)
    marker = read_json(cache_directory/'complete.json')
    cache_path = cache_directory/'reference_wait.json.gz'
    if file_hash(cache_path) != marker['sha256']['reference_wait.json.gz']:
        raise ValueError('shared reference/WAIT audit source changed')
    cache = read_json(cache_path)
    if cache['episode'] != expected:
        raise ValueError('shared reference belongs to another episode')
    reference, initial = wealth_table(cache['reference'], cache['reference'])
    wait, _ = wealth_table(cache['wait'], cache['reference'])
    actual, _ = wealth_table(artifact['policy_accounts'], cache['reference'])
    ws = independent_score(reference, wait, initial, cache['reference_rows'], cache['wait_rows'], threshold)
    ps = independent_score(reference, actual, initial, cache['reference_rows'], artifact['policy_rows'], threshold)
    check_score(ws, result['wait'])
    check_score(ps, result['policy'])
    benefit = float(ws['H'][-1]-ps['H'][-1])
    if not np.isclose(benefit, result['H_reduction'], rtol=1e-10, atol=1e-12):
        raise ValueError('validation benefit mismatch')
    history = PublicHistory()
    budget = 12.
    if len(artifact['public_log']) != 30 or len(artifact['actions']) != 30:
        raise ValueError('incomplete decision trace')
    for day, row in enumerate(artifact['public_log']):
        frame = row['frame']
        assert_no_hidden_fields(frame)
        view = history.observe(frame)
        vector = encode_public_view(view)
        if (frame['day'] != day or frame['remaining_budget'] != budget
                or frame['action_mask'] != [1,*([int(budget>=4)]*3)]
                or digest(view) != row['input_sha256']
                or hashlib.sha256(vector.tobytes()).hexdigest() != row['vector_sha256']):
            raise ValueError('public history, encoding or mask mismatch')
        predicted, _ = MODEL.predict(vector, deterministic=True, action_masks=np.array(frame['action_mask'], dtype=bool))
        action = int(predicted)
        if action != row['action'] or action != artifact['actions'][day]:
            raise ValueError('saved model does not reproduce the executed action')
        history.record(frame, action)
        budget -= 4.*bool(action)
    if result['cost'] != 12-budget:
        raise ValueError('invalid intervention cost')
    for key, rows in [('reference',cache['reference_rows']),('wait',cache['wait_rows']),('policy',artifact['policy_rows'])]:
        if aggregate_financial(rows) != result['financial_outcomes'][key]:
            raise ValueError('financial result mismatch')
    return result


def pair_values(results):
    groups = {}
    for row in results:
        pair = groups.setdefault(row['group_id'], {})
        if row['condition'] in pair:
            raise ValueError('duplicate result')
        pair[row['condition']] = row
    if any(set(pair) != {'normal','attack'} for pair in groups.values()):
        raise ValueError('incomplete group')
    ordered = sorted(groups)
    arrays = {c:np.array([groups[g][c]['H_reduction'] for g in ordered]) for c in ('normal','attack')}
    return ordered, arrays, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--baselines', type=Path, default=ROOT/'data/combination_baselines_20260926')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest = read_json(args.input/'manifest.json')
    if manifest['engineering_smoke'] or not read_json(args.input/'progress.json')['complete']:
        raise ValueError('complete scientific training and validation required')
    validate_primary_scenario()
    for path, sha in manifest['source_sha256'].items():
        if file_hash(path) != sha:
            raise ValueError(f'training source drift: {path}')
    baseline_acceptance = read_json(args.baselines/'acceptance.json')
    for path, sha in baseline_acceptance['outputs_sha256'].items():
        p = Path(path) if Path(path).is_absolute() else ROOT/path
        if file_hash(p) != sha:
            raise ValueError('accepted baseline changed')
    threshold = read_json(ROOT/'src/wolfgym/env/combination_v51.json')['harm_threshold']
    selected_results = []
    curves = {}
    audit_count = 0
    for seed in manifest['config']['seeds']:
        directory = args.input/f'seed_{seed}'
        selected = read_json(directory/'selection.json')
        train = [manifest['train'][int(i)] for i in np.random.default_rng(seed).permutation(len(manifest['train']))]
        start = 0
        curve = []
        for checkpoint in manifest['config']['checkpoints']:
            cp = directory/f'checkpoint_{checkpoint}'
            training = check_receipt(cp/'training_complete.json',manifest['execution_id'])
            logs = read_logs(directory/training['logs'],train[start:checkpoint])
            if training['seed_timesteps'] != 30*checkpoint:
                raise ValueError('training budget mismatch')
            validation = check_receipt(cp/'validation_complete.json',manifest['execution_id'])
            model_path = directory/training['model']
            if file_hash(model_path) != validation['model_sha256']:
                raise ValueError('validation checkpoint mismatch')
            results = read_json(cp/validation['results'])
            validate_episode_manifest([{k:r[k] for k in ('split','trajectory_seed','condition','group_id','episode_id')}
                                       for r in results],split='validation')
            jobs = [(str(cp/validation['raw']/f"{e['trajectory_seed']}_{e['condition']}.json.gz"),
                     str(args.baselines/e['split']/str(e['trajectory_seed'])/e['condition']),threshold,e)
                    for e in manifest['validation']]
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=load_model,initargs=(str(model_path),)) as pool:
                audited = list(pool.map(audit_policy,jobs))
            by_id = {r['episode_id']:r for r in results}
            if any(r != by_id[r['episode_id']] for r in audited):
                raise ValueError('validation summary differs from audited raw outcomes')
            groups, values, _ = pair_values(audited)
            balanced = .5*(values['normal']+values['attack'])
            if not np.isclose(balanced.mean(),validation['balanced_mean'],rtol=1e-10,atol=1e-12):
                raise ValueError('checkpoint selection metric mismatch')
            curve.append({'checkpoint':checkpoint,'balanced_mean':float(balanced.mean()),
                          'normal_mean':float(values['normal'].mean()),'attack_mean':float(values['attack'].mean()),
                          'training_segment_mean_reward':float(np.mean([r['result']['H_reduction']*1000 for r in logs]))})
            if checkpoint == selected['chosen']['checkpoint']:
                selected_results.append(audited)
            audit_count += len(audited)
            start = checkpoint
            print(f'audited seed={seed} checkpoint={checkpoint} ({audit_count}/3072)',flush=True)
        best = min(curve,key=lambda r:(-r['balanced_mean'],r['checkpoint']))
        if best['checkpoint'] != selected['chosen']['checkpoint']:
            raise ValueError('selected checkpoint violates the frozen rule')
        curves[str(seed)] = curve
    chosen_arrays = []
    seeds = {}
    for seed, results in zip(manifest['config']['seeds'],selected_results):
        groups, values, records = pair_values(results)
        indices = np.random.default_rng(20260926).integers(0,len(groups),(20000,len(groups)))
        balanced = .5*(values['normal']+values['attack'])
        chosen_arrays.append(balanced)
        seeds[str(seed)] = {'balanced':bootstrap(balanced,indices),
                           'normal':bootstrap(values['normal'],indices),'attack':bootstrap(values['attack'],indices),
                           'condition_outcomes':{}}
        for c in ('normal','attack'):
            rows = [records[g][c] for g in groups]
            seeds[str(seed)]['condition_outcomes'][c] = {
                'event_wait':sum(r['wait']['event'] for r in rows),
                'event_policy':sum(r['policy']['event'] for r in rows),
                'prevented':sum(r['wait']['event'] and not r['policy']['event'] for r in rows),
                'new':sum(not r['wait']['event'] and r['policy']['event'] for r in rows),
                'mean_cost':float(np.mean([r['cost'] for r in rows])),
                'mean_remaining_H':float(np.mean([r['policy']['damage']['H'][-1] for r in rows])),
                'financial_delta':{k:float(np.mean([r['financial_outcomes']['policy'][k]-r['financial_outcomes']['wait'][k]
                                                  for r in rows])) for k in rows[0]['financial_outcomes']['policy']}}
    mean3 = np.mean(chosen_arrays,axis=0)
    baseline_rows = read_json(args.baselines/'audited_rows.json.gz')
    comparisons = {}
    for policy in ('alarm_refresh','alarm_lock','date_8_slot_1','early_lock_8','wait'):
        lookup = {(r['group_id'],r['condition']):r['benefit'] for r in baseline_rows
                  if r['split']=='validation' and r['policy_id']==policy}
        baseline = np.array([.5*(lookup[g,'normal']+lookup[g,'attack']) for g in groups])
        comparisons[policy] = bootstrap(mean3-baseline,indices)
    baseline_table = read_json(args.baselines/'report.json')['results']['validation']['strategies']
    candidates = {name:{'mean':row['balanced']['mean'],'cost':row['mean_cost']}
                  for name,row in baseline_table.items()}
    candidates['learned_mean3'] = {'mean':float(mean3.mean()),
                                  'cost':float(np.mean([r['cost'] for results in selected_results for r in results]))}
    selected_baseline = min(candidates,key=lambda name:(-candidates[name]['mean'],candidates[name]['cost'],name))
    report = {'execution_id':manifest['execution_id'],'analysis_sha256':file_hash(__file__),
              'stage':'training_validation_complete_test_pending','learning_curves':curves,'selected_seeds':seeds,
              'three_seed_mean':bootstrap(mean3,indices),'paired_vs_baselines':comparisons,
              'selected_non_llm_P_star':selected_baseline,'P_star_candidates':candidates,
              'audit':{'validation_policy_episodes':audit_count,'account_closes':audit_count*30*490,
                       'model_decisions_replayed':audit_count*30,'training_episodes':6000,
                       'source_and_checkpoint_integrity':True},
              'uncertainty':'validation-selected checkpoints; not independent test confirmation; group CI conditional on three training instances',
              'formal_test_episodes':0,'llm_calls':0}
    write_json(args.input/'report.json',report)
    write_json(args.input/'selected_validation_results.json.gz',selected_results)
    lines = ['# 第 2 项：轻量学习训练与验证报告','',
             '检查点在验证集选择；以下不是独立测试成绩。收益单位为百分点。','',
             '| seed | episode | 验证等权收益 | 正常收益 | 攻击收益 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for seed, curve in curves.items():
        for row in curve:
            lines.append(f"| {seed} | {row['checkpoint']} | {100*row['balanced_mean']:.5f} | {100*row['normal_mean']:.5f} | {100*row['attack_mean']:.5f} |")
    lines += ['',f"三 seed 所选策略均值：{report['three_seed_mean']}（原始 H 比例）。",
              f"配对基线差值：{comparisons}（原始 H 比例）。",'',
              f"validation 选择的最强非 LLM 基线 P*：{selected_baseline}。",'',
              '正常事件、新增/防止事件、剩余损害、预算及交易功能见 report.json。',
              '不能根据本次结果额外增加训练预算、挑最好 seed 或改动损害定义。',
              '正式 test 等待主表和诊断共同冻结；后续按同一社会报告三个训练实例的平均。']
    (args.input/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('training/validation audit complete; independent test still pending',flush=True)


if __name__ == '__main__':
    main()
