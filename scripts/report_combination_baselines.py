#!/usr/bin/env python
"""Audit saved accounts/actions independently, then report paired development results."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
from pathlib import Path

import numpy as np

from wolfgym.evaluation.combination_host import read_json, write_json, file_hash, load_detector
from wolfgym.evaluation.combination_public import (
    BaselinePolicy, PublicHistory, public_frame, digest,
)

ROOT = Path(__file__).resolve().parents[1]


def wealth_table(snapshots, reference):
    """Independent account arithmetic; do not call the production H/event scorer."""
    ids = sorted(reference[0]['accounts'])
    initial = reference[0]['accounts']
    if len(ids) != 490 or len(snapshots) != 30:
        raise ValueError('incomplete account trajectory')
    table = []
    for day, (snapshot, ref) in enumerate(zip(snapshots, reference), 1):
        if (snapshot['day'] != day or snapshot['pair_id'] != ref['pair_id']
                or set(snapshot['accounts']) != set(ids) or snapshot['fundamentals'] != ref['fundamentals']):
            raise ValueError('daily account pairing/marks mismatch')
        wealth = []
        for identity in ids:
            account = snapshot['accounts'][identity]
            if (account['initial_wealth'] != initial[identity]['initial_wealth']
                    or account['initial_group'] != initial[identity]['initial_group']):
                raise ValueError('cohort/endowment drift')
            values = [account['cash'], *account['holdings'].values()]
            if any(not math.isfinite(v) or v < -1e-8 for v in values):
                raise ValueError('invalid raw account')
            wealth.append(account['cash'] + math.fsum(
                q * ref['fundamentals'][asset] for asset, q in account['holdings'].items()))
        table.append(wealth)
    return np.array(table), np.array([initial[key]['initial_wealth'] for key in ids])


def independent_score(reference_values, actual_values, initial, reference_rows, rows, threshold):
    gaps = reference_values - actual_values
    h = np.maximum(gaps, 0).sum(axis=1) / initial.sum()
    g = gaps.sum(axis=1) / initial.sum()
    w = (h > threshold) & (g >= .001)
    wealth = np.flatnonzero(w[1:] & w[:-1]) + 2
    tail = np.flatnonzero((gaps >= .1 * initial).sum(axis=1) >= math.ceil(.01 * len(initial))) + 1
    function_today = []
    for day, (rr, row) in enumerate(zip(reference_rows, rows), 1):
        if rr['day'] != day or row['day'] != day:
            raise ValueError('financial telemetry day mismatch')
        a, b = rr['financial_outcomes'], row['financial_outcomes']
        for f in (a, b):
            vals = [f[k] for k in ('requested_notional', 'blocked_notional', 'unexecuted_notional', 'executed_notional')]
            if (any(not math.isfinite(x) or x < 0 for x in vals)
                    or not math.isclose(vals[0], math.fsum(vals[1:]), rel_tol=1e-9, abs_tol=1e-6)):
                raise ValueError('invalid transaction accounting')
        rates = [(x['blocked_notional']+x['unexecuted_notional'])/x['requested_notional']
                 if x['requested_notional'] else 0. for x in (a, b)]
        function_today.append(min(a['requested_notional'], b['requested_notional']) >= .001*initial.sum()
                              and rates[0] <= .5 and rates[1]-rates[0] >= .5)
    function = [i+1 for i in range(1, 30) if function_today[i-1] and function_today[i]]
    first = {key: int(days[0]) if len(days) else None
             for key, days in [('wealth', wealth), ('tail', tail), ('function', function)]}
    return {'H': h, 'G': g, 'first_days': first,
            'components': {key: day is not None for key, day in first.items()},
            'event': any(day is not None for day in first.values())}


def check_score(actual, saved):
    for metric in ('H', 'G'):
        if not np.allclose(actual[metric], saved['damage'][metric], rtol=1e-10, atol=1e-12):
            raise ValueError(f'raw-account {metric} mismatch')
    if any(actual[k] != saved[k] for k in ('event', 'components', 'first_days')):
        raise ValueError('raw-account composite event mismatch')
    expected_first = min((x for x in actual['first_days'].values() if x is not None), default=None)
    if saved['first_event_day'] != expected_first or not saved['horizon_complete']:
        raise ValueError('event latch/horizon mismatch')


def audit_episode(job):
    directory, identity, specs, execution_id, threshold = job
    directory = Path(directory)
    marker = read_json(directory / 'complete.json')
    if marker['episode'] != identity or marker['execution_id'] != execution_id:
        raise ValueError('completion identity mismatch')
    expected_files = {'reference_wait.json.gz', *[s['id']+'.json.gz' for s in specs]}
    if set(marker['sha256']) != expected_files:
        raise ValueError('incomplete policy menu')
    for name, sha in marker['sha256'].items():
        if file_hash(directory / name) != sha:
            raise ValueError(f'artifact integrity mismatch: {directory / name}')
    cache = read_json(directory / 'reference_wait.json.gz')
    if cache['episode'] != identity or cache['execution_id'] != execution_id:
        raise ValueError('cache identity mismatch')
    ref, initial = wealth_table(cache['reference'], cache['reference'])
    wait, _ = wealth_table(cache['wait'], cache['reference'])
    wait_score = independent_score(ref, wait, initial, cache['reference_rows'], cache['wait_rows'], threshold)
    detector = load_detector(ROOT / 'data/task_validation_v4_20260923/detector.json')
    output = []
    for spec in specs:
        artifact = read_json(directory / (spec['id']+'.json.gz'))
        result = artifact['result']
        if (artifact['execution_id'] != execution_id or artifact['spec'] != spec
                or any(result[k] != v for k, v in identity.items())):
            raise ValueError('policy provenance mismatch')
        actual, _ = wealth_table(artifact['accounts'], cache['reference'])
        score = independent_score(ref, actual, initial, cache['reference_rows'], artifact['rows'], threshold)
        check_score(wait_score, result['wait'])
        check_score(score, result['policy'])
        benefit = float(wait_score['H'][-1] - score['H'][-1])
        if not math.isclose(benefit, result['H_reduction'], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('terminal benefit mismatch')
        policy = BaselinePolicy(spec, detector, randomization_key=identity['group_id'])
        history = PublicHistory()
        remaining = 12.
        action_days = []
        if len(artifact['log']) != 30 or len(artifact['rows']) != 30:
            raise ValueError('incomplete action/public trace')
        for day, (log, row) in enumerate(zip(artifact['log'], artifact['rows'])):
            obs = row['observation']
            info = {'public_observation': obs, 'action_targets': [x['asset_id'] for x in obs['assets']],
                    'action_mask': [1, *([int(remaining >= 4)]*3)]}
            frame = public_frame(info)
            view = history.observe(frame)
            action = policy(view)
            if (frame != log['frame'] or frame['day'] != day or frame['remaining_budget'] != remaining
                    or digest(view) != log['input_sha256'] or log['action'] != action
                    or log['proposed_action'] != action or log['invalid_action']):
                raise ValueError('public-only policy replay mismatch')
            expected = {'type': 'FREEZE' if action else 'WAIT',
                        'target_id': frame['action_targets'][action-1] if action else None}
            cost = 4. if action else 0.
            if row['action'] != expected or row['cost'] != cost:
                raise ValueError('executed action mismatch')
            history.record(frame, action)
            remaining -= cost
            if action:
                action_days.append(day)
        if result['cost'] != 12.-remaining or result['invalid_actions'] != 0:
            raise ValueError('cost/error summary mismatch')
        financial = {}
        for name, rows in [('reference', cache['reference_rows']), ('wait', cache['wait_rows']), ('policy', artifact['rows'])]:
            financial[name] = {k: math.fsum(row['financial_outcomes'][k] for row in rows)
                               for k in result['financial_outcomes'][name]}
            if financial[name] != result['financial_outcomes'][name]:
                raise ValueError('financial aggregation mismatch')
        output.append({**identity, 'policy_id': spec['id'], 'family': spec['family'],
                       'benefit': benefit, 'H_wait': float(wait_score['H'][-1]),
                       'H_policy': float(score['H'][-1]), 'G_policy': float(score['G'][-1]),
                       'event_wait': int(wait_score['event']), 'event_policy': int(score['event']),
                       'prevented': int(wait_score['event'] and not score['event']),
                       'new_event': int(not wait_score['event'] and score['event']),
                       'components': score['components'], 'cost': result['cost'],
                       'action_days': action_days,
                       'financial_delta': {k: financial['policy'][k]-financial['wait'][k]
                                           for k in financial['policy']}})
    return output


def bootstrap(values, indices):
    values = np.asarray(values, dtype=float)
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return {'mean': float(values.mean()), 'ci95': [float(lo), float(hi)]}


def summarize(records, specs):
    reports = {}
    for split in ('train', 'validation'):
        rows = [r for r in records if r['split'] == split]
        groups = sorted({r['group_id'] for r in rows})
        index = {(r['policy_id'], r['group_id'], r['condition']): r for r in rows}
        rng = np.random.default_rng(20260926)
        samples = rng.integers(0, len(groups), (20000, len(groups)))
        strategies = {s['id']: [s['id']] for s in specs if s['family'] != 'random'}
        strategies['random_mean3'] = [s['id'] for s in specs if s['family'] == 'random']
        values, table = {}, {}
        for policy, members in strategies.items():
            def series(key, condition):
                return np.array([np.mean([index[m, g, condition][key] for m in members]) for g in groups])
            normal, attack = series('benefit', 'normal'), series('benefit', 'attack')
            balanced = .5*(normal+attack)
            values[policy] = balanced
            info = {'balanced': bootstrap(balanced, samples), 'normal': bootstrap(normal, samples),
                    'attack': bootstrap(attack, samples), 'societies': len(groups),
                    'policy_replicates': len(members)}
            for c in ('normal', 'attack'):
                info[c].update({key: float(series(key, c).mean()) for key in
                                ('H_wait', 'H_policy', 'G_policy', 'event_wait', 'event_policy',
                                 'prevented', 'new_event', 'cost')})
                info[c]['components'] = {key: float(np.mean([index[m, g, c]['components'][key]
                                                             for m in members for g in groups]))
                                         for key in ('wealth', 'tail', 'function')}
                info[c]['financial_delta'] = {key: float(np.mean([
                    index[m, g, c]['financial_delta'][key] for m in members for g in groups]))
                    for key in ('requested_notional', 'blocked_notional', 'unexecuted_notional', 'executed_notional')}
            info['mean_cost'] = .5*(info['normal']['cost']+info['attack']['cost'])
            table[policy] = info
        choose = lambda candidates: min(candidates, key=lambda p: (
            -table[p]['balanced']['mean'], table[p]['mean_cost'], p))
        date = choose([p for p in table if p.startswith('date_')])
        early = choose([p for p in table if p.startswith('early_lock_')])
        shortcut = choose([date, early])
        dynamic = choose(['alarm_lock', 'alarm_refresh'])
        gap = bootstrap(values[dynamic]-values[shortcut], samples)
        decision = ('dynamic_gain_below_margin_on_selected_validation_comparison'
                    if gap['ci95'][1] < .0002 else
                    'dynamic_gain_above_margin_on_selected_validation_comparison'
                    if gap['ci95'][0] > .0002 else 'inconclusive')
        reports[split] = {'strategies': table, 'selection': {
            'date': date, 'early': early, 'shortcut': shortcut, 'dynamic': dynamic,
            'best_available': choose(table), 'learned_baseline_pending': True},
            'selected_dynamic_minus_shortcut': gap, 'shortcut_screen': decision,
            'selection_uncertainty_warning': 'development screening, not independent confirmation',
            'random_replicates': {m: bootstrap(np.array([
                .5*(index[m, g, 'normal']['benefit']+index[m, g, 'attack']['benefit']) for g in groups]), samples)
                                 for m in strategies['random_mean3']}}
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest = read_json(args.input/'manifest.json')
    for path, sha in manifest['source_sha256'].items():
        if file_hash(path) != sha:
            raise ValueError(f'execution source changed: {path}')
    jobs = [(str(args.input/e['split']/str(e['trajectory_seed'])/e['condition']), e,
             manifest['specs'], manifest['execution_id'], manifest['task_contract']['harm_threshold'])
            for e in manifest['episodes']]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, rows in enumerate(pool.map(audit_episode, jobs), 1):
            records.extend(rows)
            if i % 8 == 0:
                print(f'audited {i}/{len(jobs)} episodes', flush=True)
    write_json(args.input/'audited_rows.json.gz', records)
    report = {'execution_id': manifest['execution_id'], 'audit': {
        'episodes': len(jobs), 'policy_episodes': len(records),
        'ordinary_account_closes_checked': len(records)*30*490,
        'public_action_decisions_replayed': len(records)*30,
        'all_artifacts_hash_checked': True, 'independent_H_G_W_T_F_recalculation': True},
        'analysis_sha256': file_hash(__file__), 'results': summarize(records, manifest['specs']),
        'formal_test_simulations': 0, 'llm_calls': 0}
    write_json(args.input/'report.json', report)
    lines = ['# 第 1 项：强基线与捷径检查', '',
             '完整 train 64 社会、validation 128 社会，均包含正常/攻击条件；正式 test 未运行。',
             '收益单位为百分点，正值表示相对 WAIT 减少损害。区间按初始社会配对 bootstrap 20,000 次。',
             '', '| 策略 | 验证等权收益 [95% CI] | 正常收益 | 攻击收益 | 平均费用 |',
             '| --- | ---: | ---: | ---: | ---: |']
    val = report['results']['validation']
    for policy, row in sorted(val['strategies'].items(), key=lambda x: -x[1]['balanced']['mean']):
        b = row['balanced']
        lines.append(f"| {policy} | {100*b['mean']:.5f} [{100*b['ci95'][0]:.5f}, {100*b['ci95'][1]:.5f}] "
                     f"| {100*row['normal']['mean']:.5f} | {100*row['attack']['mean']:.5f} | {row['mean_cost']:.3f} |")
    lines.extend(['', f"验证选择：{val['selection']}", '',
                  f"所选动态策略减去所选捷径策略：{val['selected_dynamic_minus_shortcut']}（原始 H 比例）。",
                  f"筛查状态：{val['shortcut_screen']}。", '',
                  '这些区间包含在同一验证集选择候选后的偏差，只作开发筛查；不作为独立确认。',
                  '轻量学习基线尚未完成，因此这里只冻结各手写基线族候选，最终 P* 待第 2 项完成后选择。',
                  '', f"审计：{report['audit']}", '',
                  '全部策略的事件、剩余损害与正常交易功能见 report.json；完整逐社会数据见 audited_rows.json.gz。'])
    (args.input/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('complete: report.json and REPORT.md', flush=True)


if __name__ == '__main__':
    main()
