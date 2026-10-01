"""Offline completeness and paired statistics, not an execution/audit release gate.

Input terminal scores must separately pass account/action audits. This module
checks their internal consistency and complete declared grid; it cannot certify
raw accounts, provider charges, or that selections occurred before test.
"""
import hashlib
import gzip
import json
import math
from pathlib import Path

import numpy as np

from wolfgym.benchmark.combination import CombinationEpisode, validate_episode_manifest

VERSION = 'combination-paired-results-v1'
IDENTITY = ('split', 'trajectory_seed', 'condition', 'group_id', 'episode_id')
FINANCIAL = ('requested_notional', 'blocked_notional', 'unexecuted_notional', 'executed_notional')
COMPONENTS = ('wealth', 'tail', 'function')
BOOTSTRAPS = 20000
RNG_SEED = 20260926


def completed_engineering_result(directory):
    """Check a committed engineering-host artifact, without rerunning a world.

    Hash integrity and decision/result consistency are not an independent
    recomputation of wealth/events or a provider billing audit.
    """
    directory = Path(directory)
    receipt = json.loads((directory/'complete.json').read_text())
    required = {'manifest.json', 'outcome.json.gz'} | {f'decisions/{day:02d}.json' for day in range(30)}
    if set(receipt['sha256']) != required:
        raise ValueError('completion receipt must bind all 30 decisions and outcome')
    for path, digest in receipt['sha256'].items():
        if hashlib.sha256((directory/path).read_bytes()).hexdigest() != digest:
            raise ValueError('completed file fingerprint changed')
    manifest = json.loads((directory/'manifest.json').read_text())
    scope = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (manifest['version'] != 'combination-model-engineering-v1'
            or manifest['episode']['split'] != 'engineering'
            or manifest['formal_test_allowed'] is not False):
        raise ValueError('reader only supports engineering host artifacts')
    with gzip.open(directory/'outcome.json.gz', 'rt') as stream:
        outcome = json.load(stream)
    if receipt['scope'] != scope or outcome['scope'] != scope:
        raise ValueError('completion scope mismatch')
    result = outcome['result']
    episode = manifest['episode']
    if episode != CombinationEpisode(episode['split'], episode['trajectory_seed'], episode['condition']).to_dict():
        raise ValueError('noncanonical engineering identity')
    if {k: result[k] for k in IDENTITY} != episode:
        raise ValueError('manifest/outcome identity mismatch')
    if len(outcome['decisions']) != 30:
        raise ValueError('incomplete embedded decision log')
    for day, row in enumerate(outcome['decisions']):
        if row != json.loads((directory/f'decisions/{day:02d}.json').read_text()) or row['day'] != day:
            raise ValueError('decision log mismatch')
        if type(row['action']) is not int or row['action'] not in range(4) or type(row['invalid_action']) is not bool:
            raise ValueError('invalid logged action')
        if row['invalid_action'] and (row['action'] != 0 or row['request'] is None):
            raise ValueError('invalid response must have requested model and executed WAIT')
    decisions = outcome['decisions']
    if (result['cost'] != 4*sum(row['action'] != 0 for row in decisions)
            or result['invalid_actions'] != sum(row['invalid_action'] for row in decisions)
            or result['model_decisions'] != sum(row['request'] is not None for row in decisions)):
        raise ValueError('decision counts disagree with terminal result')
    _metrics(result)
    return result


def _number(value, name, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name}: finite numeric value required')
    if nonnegative and value < 0:
        raise ValueError(f'{name}: negative value')
    return float(value)


def _score(score):
    if score['horizon_complete'] is not True:
        raise ValueError('incomplete 30-day horizon')
    for metric in ('H', 'G'):
        values = score['damage'][metric]
        if len(values) != 30:
            raise ValueError('damage must contain 30 closes')
        for value in values:
            _number(value, metric, nonnegative=(metric == 'H'))
    if set(score['components']) != set(COMPONENTS):
        raise ValueError('missing event component')
    if any(type(v) is not bool for v in [score['event'], *score['components'].values()]):
        raise ValueError('event flags must be booleans')
    if score['event'] != any(score['components'].values()):
        raise ValueError('C must be W OR T OR F')


def _metrics(result):
    for branch in ('wait', 'policy'):
        _score(result[branch])
    wait, policy = result['wait'], result['policy']
    benefit = _number(result['H_reduction'], 'benefit')
    if not math.isclose(benefit, wait['damage']['H'][-1] - policy['damage']['H'][-1], abs_tol=1e-12, rel_tol=0):
        raise ValueError('benefit does not match terminal H')
    cost = _number(result['cost'], 'cost', True)
    if cost not in (0, 4, 8, 12):
        raise ValueError('invalid one-day intervention budget')
    invalid, decisions = result['invalid_actions'], result['model_decisions']
    if any(type(v) is not int for v in (invalid, decisions)) or not 0 <= invalid <= decisions <= 30:
        raise ValueError('invalid model decision counts')
    financial = result['financial_outcomes']
    for branch in ('reference', 'wait', 'policy'):
        for metric in FINANCIAL:
            _number(financial[branch][metric], metric, True)
        values = financial[branch]
        if not math.isclose(values['requested_notional'], sum(values[k] for k in FINANCIAL[1:]), rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('financial accounting does not balance')
    values = {'benefit_pp': 100*benefit, 'H_wait_pp': 100*wait['damage']['H'][-1],
              'H_policy_pp': 100*policy['damage']['H'][-1],
              'G_policy_pp': 100*policy['damage']['G'][-1],
              'event_wait': int(wait['event']), 'event_policy': int(policy['event']),
              'prevented': int(wait['event'] and not policy['event']),
              'new_event': int(policy['event'] and not wait['event']),
              'cost': cost, 'action_days': cost/4,
              'invalid_actions': invalid, 'model_decisions': decisions}
    for component in COMPONENTS:
        a, b = wait['components'][component], policy['components'][component]
        values.update({f'{component}_wait': int(a), f'{component}_policy': int(b),
                       f'{component}_prevented': int(a and not b), f'{component}_new': int(b and not a)})
    for metric in FINANCIAL:
        values[f'{metric}_delta'] = financial['policy'][metric] - financial['wait'][metric]
    return values


def _population(episodes, population):
    if population not in ('full_split', 'development_subset') or not episodes:
        raise ValueError('explicit nonempty population required')
    splits, pairs, seen = set(), {}, set()
    for row in episodes:
        canonical = CombinationEpisode(row['split'], row['trajectory_seed'], row['condition']).to_dict()
        if row != canonical or row['episode_id'] in seen:
            raise ValueError('noncanonical or duplicate expected episode')
        seen.add(row['episode_id'])
        splits.add(row['split'])
        pairs.setdefault(row['group_id'], set()).add(row['condition'])
    if len(splits) != 1 or any(c != {'normal', 'attack'} for c in pairs.values()):
        raise ValueError('population requires paired conditions in one split')
    split = next(iter(splits))
    if population == 'full_split':
        validate_episode_manifest(episodes, split=split)
    elif split not in ('train', 'engineering'):
        raise ValueError('validation/test subsets cannot produce reports')
    return split, sorted(pairs)


def _interval(values, indices, confidence=.95):
    samples = np.mean(values[indices], axis=1)
    alpha = (1-confidence)/2
    return {'mean': float(np.mean(values)), 'confidence': confidence,
            'interval': [float(x) for x in np.quantile(samples, [alpha, 1-alpha])]}


def paired_results(rows, *, expected_episodes, policy_members, population,
                   comparison_families=None):
    """Require every policy/member/episode; average members inside each society.

    Families are explicit lists of {id,left,right}, exactly 4 main / 8 ability.
    The caller must bind these selections to the eventual joint freeze. No
    result-based winner selection or source-file audit is performed here.
    """
    split, groups = _population(expected_episodes, population)
    if not policy_members:
        raise ValueError('no declared policies')
    for policy, members in policy_members.items():
        if not isinstance(policy, str) or not policy or not isinstance(members, list) or not members:
            raise ValueError('policy requires declared members')
        if any(not isinstance(m, str) or not m for m in members) or len(set(members)) != len(members):
            raise ValueError('duplicate or invalid member')
    expected = {r['episode_id']: r for r in expected_episodes}
    grid = {(p, m, e) for p, members in policy_members.items() for m in members for e in expected}
    metrics, common = {}, {}
    for row in rows:
        result = row['result']
        key = row['policy_id'], row['member_id'], result['episode_id']
        if key not in grid or key in metrics:
            raise ValueError('duplicate or undeclared result')
        if {k: result[k] for k in IDENTITY} != expected[key[2]]:
            raise ValueError('result identity differs from population')
        metrics[key] = _metrics(result)
        baseline = {'wait': result['wait'], 'reference_financial': result['financial_outcomes']['reference'],
                    'wait_financial': result['financial_outcomes']['wait']}
        if key[2] in common and common[key[2]] != baseline:
            raise ValueError('unmatched WAIT/reference across policies or members')
        common[key[2]] = baseline
    if set(metrics) != grid:
        raise ValueError(f'incomplete result grid: {len(metrics)}/{len(grid)}; no subset ranking')
    families = comparison_families or {}
    for family, comparisons in families.items():
        if family not in ('main', 'ability') or len(comparisons) != {'main': 4, 'ability': 8}[family]:
            raise ValueError('comparison family must keep predeclared K=4/8')
        identifiers = set()
        directions = set()
        for c in comparisons:
            if set(c) != {'id', 'left', 'right'} or not isinstance(c['id'], str) or not c['id']:
                raise ValueError('invalid comparison declaration')
            direction = c['left'], c['right']
            if c['id'] in identifiers or direction in directions or c['left'] == c['right'] or any(p not in policy_members for p in direction):
                raise ValueError('duplicate, self or unknown comparison')
            identifiers.add(c['id'])
            directions.add(direction)
    indices = np.random.default_rng(RNG_SEED).integers(0, len(groups), size=(BOOTSTRAPS, len(groups)))
    episode_ids = {(e['group_id'], e['condition']): e['episode_id'] for e in expected_episodes}
    metric_names = list(next(iter(metrics.values())))
    benefit_vectors, policies = {}, {}
    for policy, members in policy_members.items():
        member_arrays = {member: {condition: {
            metric: np.array([metrics[policy, member, episode_ids[g, condition]][metric] for g in groups])
            for metric in metric_names} for condition in ('normal', 'attack')} for member in members}
        means = {condition: {metric: np.mean([a[condition][metric] for a in member_arrays.values()], axis=0)
                             for metric in metric_names} for condition in ('normal', 'attack')}
        means['balanced'] = {metric: .5*(means['normal'][metric]+means['attack'][metric]) for metric in metric_names}
        benefit_vectors[policy] = means['balanced']['benefit_pp']
        summary = {}
        for condition, vectors in means.items():
            summary[condition] = {metric: _interval(values, indices) for metric, values in vectors.items()}
            denominator = float(np.sum(vectors['model_decisions']))
            summary[condition]['invalid_action_rate'] = (float(np.sum(vectors['invalid_actions']))/denominator if denominator else None)
        policies[policy] = {'member_ids': members, 'summary': summary, 'members': {
            m: {c: _interval(a[c]['benefit_pp'], indices) for c in ('normal', 'attack')}
            | {'balanced': _interval(.5*(a['normal']['benefit_pp']+a['attack']['benefit_pp']), indices)}
            for m, a in member_arrays.items()}}
    comparisons = {}
    for family, declarations in families.items():
        k = len(declarations)
        comparisons[family] = {c['id']: {**c, 'family_size': k,
            'raw': _interval(benefit_vectors[c['left']]-benefit_vectors[c['right']], indices),
            'bonferroni': _interval(benefit_vectors[c['left']]-benefit_vectors[c['right']], indices, 1-.05/k)}
            for c in declarations}
    binding = {'episodes': sorted(expected_episodes, key=lambda e: e['episode_id']),
               'policy_members': policy_members, 'comparison_families': families, 'population': population}
    return {'version': VERSION, 'split': split, 'population': population,
            'declaration_sha256': hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'complete_result_grid': True, 'result_count': len(metrics), 'independent_societies': len(groups),
            'bootstrap_replicates': BOOTSTRAPS, 'bootstrap_seed': RNG_SEED,
            'policies': policies, 'comparisons': comparisons,
            'official_release_authorized': False,
            'limitations': ['Terminal-score checks do not replace raw-account/action audits.',
                           'Provider cost/token ledger reconciliation is not performed.',
                           'Joint freeze and pre-test selection evidence require external verification.',
                           'Intervals condition on the declared policy instances; bootstrap coverage is approximate.',
                           'Separate comparison families do not give one joint 95% guarantee.']}
