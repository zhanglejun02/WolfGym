"""Bounded train-only privileged schedule search, not a policy leaderboard."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import math
import os
from pathlib import Path

import numpy as np

from report_combination_baselines import wealth_table, independent_score, check_score
from wolfgym.benchmark.combination import CombinationEpisode, episode_plan
from wolfgym.env.combination_fast import FastCombinationGymEnv
from wolfgym.env.combination_contract import primary_contract
from wolfgym.evaluation.combination_host import read_json, write_json, file_hash, load_detector, aggregate_financial
from wolfgym.evaluation.combination_public import BaselinePolicy, PublicHistory, public_frame, digest

ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT/'data/task_validation_v4_20260923/detector.json'


def selected_episodes():
    groups = list(dict.fromkeys(e.group_id for e in episode_plan('train')))[:8]
    return [e.to_dict() for g in groups for e in episode_plan('train') if e.group_id == g]


def random_plans():
    rng = np.random.default_rng(20260928)
    return {f'random_{i:02}': sorted(zip(map(int, rng.choice(30, 3, replace=False)),
                                       map(int, rng.integers(1, 4, 3)))) for i in range(12)}


def validate_plan(plan):
    if len(plan) > 3 or len({d for d, _ in plan}) != len(plan):
        raise ValueError('duplicate day or excessive budget')
    if any(type(d) is not int or type(a) is not int or not 0 <= d < 30 or not 1 <= a <= 3 for d, a in plan):
        raise ValueError('illegal day or target')


def candidate_plans(rule_plan):
    validate_plan(rule_plan)
    plans = {'rule': list(rule_plan), 'wait': []}
    for shift in (-2, -1, 1, 2):
        plans[f'shift_{shift:+d}'] = [(d+shift, a) for d, a in rule_plan if 0 <= d+shift < 30]
    for slot in (1, 2, 3):
        plans[f'target_{slot}'] = [(d, slot) for d, _ in rule_plan]
    for index in range(3):
        plans[f'drop_{index+1}'] = [x for i, x in enumerate(rule_plan) if i != index]
    plans.update(random_plans())
    for plan in plans.values():
        validate_plan(plan)
    return plans


def simulate(identity, plan):
    episode = CombinationEpisode(identity['split'], identity['trajectory_seed'], identity['condition'])
    if episode.split != 'train':
        raise ValueError('diagnostic is train-only')
    env = FastCombinationGymEnv(episode, message_mode='indexed')
    rule = BaselinePolicy({'id': 'alarm_refresh', 'family': 'alarm_refresh'}, load_detector(DETECTOR))
    history = PublicHistory(); logs = []
    try:
        _, info = env.reset()
        targets = tuple(info['action_targets'])
        schedule = dict(plan or [])
        for day in range(30):
            frame = public_frame(info); view = history.observe(frame)
            if tuple(frame['action_targets']) != targets:
                raise ValueError('public target slots changed inside episode')
            action = rule(view) if plan is None else schedule.get(day, 0)
            logs.append({'day': day, 'frame': frame, 'input_sha256': digest(view), 'action': action})
            history.record(frame, action)
            _, reward, done, truncated, info = env.step(action)
            if reward != 0 or truncated or done != (day == 29):
                raise ValueError('evaluation feedback changed')
        return {'episode': identity, 'result': env.evaluation_result(), 'decisions': logs,
                'accounts': {k: [asdict(x) for x in v] for k, v in env._snapshots.items()},
                'financial_rows': env._rows}
    finally:
        env.close()


def audit(artifact, common, identity, name, plan):
    r = artifact['result']; accounts = common['accounts']; rows = common['financial_rows']
    if artifact['episode'] != identity or any(r[k] != v for k, v in identity.items()):
        raise ValueError('episode drift')
    ref, initial = wealth_table(accounts['reference'], accounts['reference'])
    scores = {}
    actual_accounts = {'wait': accounts['wait'], 'policy': artifact['accounts']}
    actual_rows = {'wait': rows['wait'], 'policy': artifact['financial_rows']}
    for world in actual_accounts:
        values, _ = wealth_table(actual_accounts[world], accounts['reference'])
        scores[world] = independent_score(ref, values, initial, rows['reference'], actual_rows[world], primary_contract()['harm_threshold'])
        check_score(scores[world], r[world])
    benefit = float(scores['wait']['H'][-1] - scores['policy']['H'][-1])
    if not math.isclose(benefit, r['H_reduction'], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('benefit drift')
    history = PublicHistory(); remaining = 12; executed = []; targets = None
    rule = BaselinePolicy({'id': 'alarm_refresh', 'family': 'alarm_refresh'}, load_detector(DETECTOR))
    if len(artifact['decisions']) != 30:
        raise ValueError('incomplete decisions')
    for day, log in enumerate(artifact['decisions']):
        f = log['frame']; view = history.observe(f)
        targets = targets or f['action_targets']
        action = rule(view) if name == 'rule' else dict(plan).get(day, 0)
        if (f['day'] != day or log['day'] != day or f['remaining_budget'] != remaining
                or f['action_mask'] != [1, *([int(remaining >= 4)]*3)]
                or f['action_targets'] != targets or len(set(targets)) != 3
                or action != log['action'] or digest(view) != log['input_sha256'] or not f['action_mask'][action]):
            raise ValueError('action/input/budget replay failed')
        history.record(f, action)
        if action:
            remaining -= 4; executed.append((day, action))
    if executed != [tuple(x) for x in plan] or r['cost'] != 12-remaining:
        raise ValueError('schedule or cost mismatch')
    financial = {'reference': aggregate_financial(rows['reference']), 'wait': aggregate_financial(rows['wait']),
                 'policy': aggregate_financial(artifact['financial_rows'])}
    if financial != r['financial_outcomes']:
        raise ValueError('financial totals drift')
    if name == 'wait' and (artifact['accounts'] != accounts['wait'] or benefit != 0 or artifact['financial_rows'] != rows['wait']):
        raise ValueError('WAIT does not reproduce paired control')
    return {**identity, 'candidate': name, 'benefit_pp': 100*benefit, 'cost': r['cost'],
            'H_wait_pp': 100*float(scores['wait']['H'][-1]), 'H_policy_pp': 100*float(scores['policy']['H'][-1]),
            'event_wait': scores['wait']['event'], 'event_policy': scores['policy']['event'],
            'components': scores['policy']['components'], 'first_days': scores['policy']['first_days'],
            'plan': plan, 'financial_delta': {k: financial['policy'][k]-financial['wait'][k] for k in financial['policy']}}


def run_episode(job):
    directory, identity, execution = job
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'complete.json'
    if marker.exists():
        complete = read_json(marker)
        if complete['execution_id'] != execution or complete['episode'] != identity:
            raise ValueError('completed identity mismatch')
        for name, sha in complete['sha256'].items():
            if file_hash(directory/name) != sha:
                raise ValueError('completed artifact drift')
        return read_json(directory/'audited.json')
    common_path = directory/'reference_wait.json.gz'
    def obtain(name, plan):
        path = directory/f'{name}.json.gz'
        if not path.exists():
            a = simulate(identity, plan)
            common = {'accounts': {k: a['accounts'][k] for k in ('reference', 'wait')},
                      'financial_rows': {k: a['financial_rows'][k] for k in ('reference', 'wait')}}
            if common_path.exists():
                if digest(read_json(common_path)) != digest(common):
                    raise ValueError('reference/WAIT branches differ across candidates')
            else:
                write_json(common_path, common)
            a['accounts'] = a['accounts']['policy']; a['financial_rows'] = a['financial_rows']['policy']
            a['execution_id'] = execution; a['reference_wait_sha256'] = file_hash(common_path)
            write_json(path, a)
        a = read_json(path)
        if a['execution_id'] != execution or a['reference_wait_sha256'] != file_hash(common_path):
            raise ValueError('uncommitted provenance mismatch')
        return a
    rule = obtain('rule', None)
    rule_plan = [(x['day'], x['action']) for x in rule['decisions'] if x['action']]
    plans = candidate_plans(rule_plan); common = read_json(common_path)
    by_schedule = {}; results = []; artifacts = [common_path]
    for name, plan in plans.items():
        key = tuple(plan)
        if key in by_schedule:
            alias = by_schedule[key]
            a = read_json(directory/f'{alias}.json.gz')
        else:
            alias = name; a = rule if name == 'rule' else obtain(name, plan)
            by_schedule[key] = name; artifacts.append(directory/f'{name}.json.gz')
        row = audit(a, common, identity, name, plan)
        row['artifact_candidate'] = alias; results.append(row)
    write_json(directory/'audited.json', results); artifacts.append(directory/'audited.json')
    write_json(marker, {'execution_id': execution, 'episode': identity,
                       'unique_rollouts': len(by_schedule), 'sha256': {p.name: file_hash(p) for p in artifacts}})
    return results


def summarize(records):
    expected = {(e['episode_id'], name) for e in selected_episodes() for name in candidate_plans([])}
    actual = [(r['episode_id'], r['candidate']) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('incomplete or duplicate diagnostic grid')
    def selection_key(r):
        return (-r['benefit_pp'], r['cost'], r['candidate'])
    grouped = {e['episode_id']: [r for r in records if r['episode_id'] == e['episode_id']] for e in selected_episodes()}
    best = [min(rs, key=selection_key) for rs in grouped.values()]
    policies = {name: [r for r in records if r['candidate'] == name] for name in candidate_plans([])}
    policies['hindsight_best'] = best
    summaries = {}
    for name, rs in policies.items():
        s = {'balanced_benefit_pp': float(np.mean([r['benefit_pp'] for r in rs]))}
        for condition in ('normal', 'attack'):
            cr = [r for r in rs if r['condition'] == condition]
            s[condition] = {'benefit_pp': float(np.mean([r['benefit_pp'] for r in cr])),
                'remaining_H_pp': float(np.mean([r['H_policy_pp'] for r in cr])),
                'events_wait': sum(r['event_wait'] for r in cr), 'events_policy': sum(r['event_policy'] for r in cr),
                'prevented': sum(r['event_wait'] and not r['event_policy'] for r in cr),
                'added': sum(not r['event_wait'] and r['event_policy'] for r in cr),
                'mean_cost': float(np.mean([r['cost'] for r in cr])),
                'mean_financial_delta': {k: float(np.mean([r['financial_delta'][k] for r in cr])) for k in cr[0]['financial_delta']}}
        summaries[name] = s
    detail = []
    for episode, rs in grouped.items():
        b = min(rs, key=selection_key); rule = next(r for r in rs if r['candidate'] == 'rule')
        detail.append({'episode_id': episode, 'condition': b['condition'], 'group_id': b['group_id'],
                       'best_candidate': b['candidate'], 'best_plan': b['plan'], 'rule_plan': rule['plan'],
                       'headroom_pp': b['benefit_pp']-rule['benefit_pp'],
                       'any_candidate_without_event': any(not r['event_policy'] for r in rs),
                       'rule_event': rule['event_policy'], 'best_H_event': b['event_policy']})
    return {'summaries': summaries, 'episode_details': detail,
            'balanced_headroom_pp': summaries['hindsight_best']['balanced_benefit_pp']-summaries['rule']['balanced_benefit_pp'],
            'scope': '8 train societies only; privileged in-sample selection; neither deployable policy nor true optimum; no significance/generalization claim',
            'predeclared_descriptive_headroom_scale_pp': .02, 'formal_test_or_api_calls': 0}


def main(output, workers):
    if workers != 4:
        raise ValueError('bounded run fixes four workers')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [Path(__file__), ROOT/'docs/EXPERIMENT_PROTOCOL.md',
                   ROOT/'scripts/report_combination_baselines.py', ROOT/'tests/test_action_diagnostic.py', DETECTOR]
        sources += [ROOT/'src/wolfgym'/p for p in ['env/combination_fast.py', 'observations/fast_messages.py',
            'env/combination.py', 'env/combination_contract.py', 'env/combination_v51.json',
            'benchmark/combination.py', 'benchmark/combination_splits_v1.json',
            'evaluation/combination_public.py', 'evaluation/combination_host.py', 'evaluation/combination_task.py']]
        sources += [ROOT/'src/wolfgym'/p for p in primary_contract()['source_files']]
        manifest = {'version': 'action-diagnostic-v1', 'episodes': selected_episodes(), 'max_rollouts': 384,
                    'random_plans': random_plans(), 'workers': workers,
                    'source_sha256': {str(p.resolve()): file_hash(p) for p in sources}}
        path = output/'manifest.json'
        if path.exists() and digest(read_json(path)) != digest(manifest):
            raise ValueError('frozen diagnostic manifest/source drift')
        if not path.exists(): write_json(path, manifest)
        execution = digest(manifest)
        if (output/'acceptance.json').exists():
            raise ValueError('diagnostic already accepted; do not rerun')
        jobs = [(str(output/e['episode_id'].replace(':', '_')), e, execution) for e in manifest['episodes']]
        records = []; write_json(output/'progress.json', {'pid': os.getpid(), 'completed_pairs': 0, 'total_pairs': 16, 'complete': False})
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_episode, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                records.extend(future.result())
                write_json(output/'progress.json', {'pid': os.getpid(), 'completed_pairs': i, 'total_pairs': 16, 'complete': False})
                print(f'complete and audited {i}/16 episode candidate sets', flush=True)
        records.sort(key=lambda r: (r['episode_id'], r['candidate']))
        report = summarize(records)
        unique = sum(read_json(Path(job[0])/'complete.json')['unique_rollouts'] for job in jobs)
        report['audit'] = {'candidate_rows': len(records), 'unique_market_rollouts': unique,
                           'unique_raw_account_closes': unique*30*490+16*2*30*490,
                           'candidate_actions_replayed': len(records)*30, 'full_market_resimulation_audit': False}
        write_json(output/'audited_rows.json', records); write_json(output/'report.json', report)
        lines = ['# 现有监管动作：小型诊断', '', report['scope'], '',
                 '| 方案 | 等权收益 pp | 正常 pp | 攻击 pp | 正常新增事件 | 攻击事件/8 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, s in report['summaries'].items():
            lines.append(f"| {name} | {s['balanced_benefit_pp']:.5f} | {s['normal']['benefit_pp']:.5f} | {s['attack']['benefit_pp']:.5f} | {s['normal']['added']} | {s['attack']['events_policy']} |")
        lines += ['', f"逐轨迹事后选择相对规则的等权headroom: {report['balanced_headroom_pp']:.5f} pp。",
                  '事后最佳包含条件和未来结果特权，不是公开策略，也不是真正上限。8社会仅作诊断，不报独立显著性。',
                  '同样三次单日FREEZE，没有增加工具、权限、预算，没有改collapse或散户行为。原始账户独立重算及动作重放通过，不等于全环境重模拟审计。']
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        outputs = {n: file_hash(output/n) for n in ('manifest.json', 'audited_rows.json', 'report.json', 'REPORT.md')}
        outputs.update({str((Path(j[0])/'complete.json').relative_to(output)): file_hash(Path(j[0])/'complete.json') for j in jobs})
        write_json(output/'acceptance.json', {'execution_id': execution, 'outputs_sha256': outputs, 'audit': report['audit']})
        write_json(output/'progress.json', {'completed_pairs': 16, 'total_pairs': 16, 'complete': True, 'audit_complete': True})
        print('Diagnostic complete and audited', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4); args = p.parse_args(); main(args.output.resolve(), args.workers)
