"""Pre-simulation extension contracts and conservative seed namespace checks.

No market runner is provided. Frozen primary scenarios are never overwritten.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import yaml

from wolfgym.benchmark.combination import benchmark_contract
from wolfgym.env.combination_contract import primary_contract, validate_primary_scenario

VERSION = 'combination-extensions-plan-v1'
# Current direct integer entropy inputs; 23017 conservatively reserves the
# optional social-game initializer even though the primary task does not use it.
OFFSETS = (0, 7, 13, 23, 23017, 1000000, 2000000, 3000000)
BLOCKS = {
    'liquidity': {'development': (100000000, 16), 'confirmation': (110000000, 128)},
    'benign_spread': {'development': (120000000, 16), 'confirmation': (130000000, 128)},
}
STRIDE = 1000
ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extension_scenario(name):
    """Only construct the declared scenario; does not bypass primary Gym locks."""
    if name not in BLOCKS:
        raise ValueError('unknown extension')
    scenario = deepcopy(validate_primary_scenario())
    if name == 'liquidity':
        if scenario.target_asset != 'asset_2' or scenario.asset('asset_2').daily_execution_liquidity != 12000:
            raise ValueError('primary liquidity assumption changed')
        scenario.asset('asset_2').daily_execution_liquidity = 9000.
    else:
        if scenario.retail.get('post_probability') != .1 or scenario.retail.get('post_intensity') != .2:
            raise ValueError('primary posting assumptions changed')
        scenario.retail['post_probability'] = .3
    return scenario


def extension_contract(name):
    scenario = extension_scenario(name)
    return {'version': VERSION, 'extension': name, 'primary_task': primary_contract(),
            'scenario': asdict(scenario),
            'change': ({'path': 'assets.asset_2.daily_execution_liquidity', 'from': 12000, 'to': 9000}
                       if name == 'liquidity' else {'path': 'retail.post_probability', 'from': .1, 'to': .3}),
            'conditions': ['normal', 'attack'] if name == 'liquidity' else ['normal'],
            'condition_weights': {'normal': .5, 'attack': .5} if name == 'liquidity' else {'normal': 1.},
            'paired_worlds_same_extension': True,
            'normal_definition': 'attack_off_same_cohort_ordinary_and_social_rng_replica',
            'primary_score_inclusion': False, 'regulatory_budget': 12, 'freeze_cost': 4,
            'horizon_days': 30, 'seed_stride': STRIDE,
            'seed_blocks': {s: {'start': start, 'count': count} for s, (start,count) in BLOCKS[name].items()},
            'policies': ['wait', 'date_8_slot_1', 'alarm_refresh', 'learned_seed_11', 'learned_seed_22',
                         'learned_seed_33', 'M1', 'M3'],
            'learned_checkpoints': {'11': 1000, '22': 2000, '33': 1000},
            'selection_source': 'validation_only_before_extension_confirmation',
            'analysis': {'unit': 'initial_society', 'bootstrap_replicates': 20000, 'rng_seed': 20260926,
                         'benefit': 'H_WAIT_30_minus_H_policy_30_signed',
                         'members': 'all_three_learned_instances_mean_within_society',
                         'normal_event_interval': 'one_sided_95_binomial_upper_no_inherited_guarantee'},
            'development_use': 'engineering_only_no_parameter_or_model_selection',
            'confirmation_requires_joint_freeze': True, 'runtime_release': 'closed'}


def extension_manifest(name):
    contract = extension_contract(name)
    rows = []
    for stage, (start, count) in BLOCKS[name].items():
        for index in range(count):
            seed = start + STRIDE*index
            group = 'extension_' + hashlib.sha256(f'{VERSION}:{name}:{seed}'.encode()).hexdigest()[:24]
            for condition in contract['conditions']:
                rows.append({'extension': name, 'stage': stage, 'trajectory_seed': seed,
                             'condition': condition, 'group_id': group,
                             'episode_id': f'{group}:{condition}'})
    return sorted(rows, key=lambda r: hashlib.sha256(r['episode_id'].encode()).hexdigest())


def check_seed_namespaces(block_roots, reserved_intervals):
    """Check cross-role integer entropy reuse, not mathematical RNG independence.

    SeedSequence descendants have unique spawn keys within their parent. We
    reserve each parent entropy value across roles; child random integer draws
    and hash-derived event/alias keys are not claimed collision-free by this check.
    """
    used = {}
    for block, roots in block_roots.items():
        for root in roots:
            if type(root) is not int or root < 0:
                raise ValueError('invalid root seed')
            for offset in OFFSETS:
                value = root + offset
                if value in used:
                    raise ValueError(f'derived seed collision: {block}:{root}+{offset} vs {used[value]}')
                if any(lo <= value <= hi for lo, hi in reserved_intervals):
                    raise ValueError(f'seed collides with reserved historical namespace: {value}')
                used[value] = (block, root, offset)
    return {'root_count': sum(len(v) for v in block_roots.values()),
            'direct_entropy_values_checked': len(used), 'offsets': list(OFFSETS), 'conflicts': []}


def historical_inventory(root=ROOT):
    """Metadata-only audit: config/manifest seed fields and raw file names.

    Broad interval reserves intentionally include unused seed values. We do not
    load compressed account payloads or inspect confirmation outcome values.
    """
    root = Path(root)
    numbers, inputs, raw_names = set(), {}, []
    def collect(value, in_seed=False):
        if isinstance(value, dict):
            for key, item in value.items():
                collect(item, in_seed or 'seed' in str(key).lower())
        elif isinstance(value, list):
            for item in value:
                collect(item, in_seed)
        elif in_seed and type(value) is int and value >= 0:
            numbers.add(value)
    paths = sorted(set((root/'configs').rglob('*.yaml')) | set((root/'data').glob('*/manifest.json')))
    for p in paths:
        data = json.loads(p.read_text()) if p.suffix == '.json' else yaml.safe_load(p.read_text())
        collect(data)
        inputs[str(p.relative_to(root))] = sha(p)
    for p in sorted((root/'data').rglob('*.json.gz')):
        match = re.match(r'^(\d+)_', p.name)
        if match:
            numbers.add(int(match.group(1)))
            raw_names.append(str(p.relative_to(root)))
    base = benchmark_contract()
    # Retire the entire legacy range, not just observed file names.
    intervals = [[0, base['reserved_below']-1+max(OFFSETS)]]
    for block in base['splits'].values():
        intervals.append([block['start'], block['start']+block['count']-1+max(OFFSETS)])
    # A broad envelope also covers undisplayed interior values in seed ranges.
    if numbers:
        intervals.append([min(numbers), max(numbers)+max(OFFSETS)])
    return {'reserved_intervals_inclusive': intervals, 'seed_numeric_values': sorted(numbers),
            'metadata_sha256': inputs, 'raw_seed_filename_count': len(raw_names),
            'raw_seed_filenames_sha256': hashlib.sha256(json.dumps(raw_names).encode()).hexdigest(),
            'coverage': 'local config and top-level study manifests, numeric raw trajectory names, entire legacy and primary ranges',
            'limitations': 'not an exhaustive RNG independence proof or inspection of undocumented external studies'}


def plan_extensions(root=ROOT):
    root = Path(root)
    inventory = historical_inventory(root)
    roots = {f'{name}/{stage}': [start+STRIDE*i for i in range(count)]
             for name, blocks in BLOCKS.items() for stage,(start,count) in blocks.items()}
    check = check_seed_namespaces(roots, inventory['reserved_intervals_inclusive'])
    return {'version': VERSION, 'contracts': {name: extension_contract(name) for name in BLOCKS},
            'manifests': {name: extension_manifest(name) for name in BLOCKS},
            'historical_inventory': inventory, 'seed_check': check,
            'market_episodes_run': 0, 'runtime_release': 'closed'}
