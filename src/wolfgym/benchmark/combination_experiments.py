"""Outcome-independent experiment inventory and offline freeze preflight.

This module inventories requirements; it does not grant API spending or unlock
the engineering-only host. Provider evidence and the research runner are still
required before a candidate configuration can become an executable freeze.
"""
import copy
import hashlib
import math
from pathlib import Path

from wolfgym.benchmark.combination import episode_plan
from wolfgym.evaluation.combination_public import digest


REPO = Path(__file__).resolve().parents[3]
SLOTS = ('M1', 'M2', 'M3', 'M4')
FOCAL = ('M1', 'M3')
ADDITIONAL_DIAGNOSTICS = ('none', 'rename_reorder', 'liquidity', 'benign_spread')


def experiment_cells(additional_diagnostic=None):
    if additional_diagnostic is not None and (
        type(additional_diagnostic) is not str or additional_diagnostic not in ADDITIONAL_DIAGNOSTICS
    ):
        raise ValueError('additional_diagnostic must be one of: ' + ', '.join(ADDITIONAL_DIAGNOSTICS))
    cells = []
    def add(slot, experiment, mode, window, dataset, episodes, presentation='original'):
        cells.append({'id': f'{slot}:{experiment}', 'model_slot': slot,
                      'experiment': experiment, 'mode': mode, 'window': window,
                      'dataset': dataset, 'episodes': episodes, 'presentation': presentation,
                      'max_regular_requests': episodes*(3 if mode == 'RL' else 30)})
    for slot in SLOTS:
        add(slot, 'main', 'LL', 5, 'test', 512)
    for slot in FOCAL:
        add(slot, 'rule_timing', 'RL', 5, 'test', 512)
        add(slot, 'rule_target', 'LR', 5, 'test', 512)
        add(slot, 'current_only', 'LL', 1, 'test', 512)
        if additional_diagnostic == 'rename_reorder':
            add(slot, 'rename_reorder', 'LL', 5, 'test', 512, 'consistent_bijection_and_permutation')
        elif additional_diagnostic == 'liquidity':
            add(slot, 'liquidity', 'LL', 5, 'liquidity_confirmation', 256)
        elif additional_diagnostic == 'benign_spread':
            add(slot, 'benign_spread', 'LL', 5, 'benign_confirmation', 128)
    return cells


def cost_pilot_episodes():
    """First eight complete societies in the existing immutable train order."""
    plan = episode_plan('train')
    groups = list(dict.fromkeys(e.group_id for e in plan))[:8]
    return [e.to_dict() for e in plan if e.group_id in groups]


def bound_sources():
    return [REPO/'docs/EXPERIMENT_PROTOCOL.md', Path(__file__),
            REPO/'src/wolfgym/benchmark/combination.py',
            REPO/'src/wolfgym/benchmark/combination_splits_v1.json',
            REPO/'src/wolfgym/evaluation/combination_decision.py',
            REPO/'src/wolfgym/evaluation/combination_requests.py',
            REPO/'src/wolfgym/evaluation/combination_model_host.py',
            REPO/'src/wolfgym/evaluation/combination_presentation.py',
            REPO/'scripts/plan_combination_experiments.py',
            REPO/'data/combination_learning_20260926/acceptance.json']


def source_fingerprints():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound_sources()}


def make_draft(additional_diagnostic=None):
    cells = experiment_cells(additional_diagnostic)
    extensions = {
        'liquidity': {'parameter': 'daily_execution_liquidity', 'asset': 'asset_2',
                      'from': 12000, 'to': 9000, 'development_societies': 16,
                      'confirmation_societies': 128, 'conditions': ['normal','attack'],
                      'contract_path': None, 'manifest_path': None},
        'benign_spread': {'parameter': 'retail.post_probability', 'from': .1, 'to': .3,
                          'post_intensity': .2, 'development_societies': 16,
                          'confirmation_societies': 128, 'conditions': ['normal'],
                          'contract_path': None, 'manifest_path': None},
    }
    models = {slot: {'role': role, 'profile': None, 'allocation': None} for slot, role in zip(SLOTS, (
        'strong_closed', 'different_family', 'open_large', 'same_open_family_small'))}
    return {'version': 'combination-joint-experiments-draft-v2', 'status': 'draft',
            'runtime_release': 'engineering_only', 'focal_models': list(FOCAL),
            'additional_diagnostic': additional_diagnostic,
            'models': models, 'cells': cells, 'project_budget_microusd': None,
            'pilot': {'episode_manifest': cost_pilot_episodes(), 'model_slots': [],
                      'max_models': 2, 'max_prompt_versions': 2,
                      'max_regular_requests_per_model': 480, 'allocations': {},
                      'scope_note': 'All prompt versions share one per-model call allowance.'},
            'analysis': {'bootstrap_replicates': 20000, 'rng_seed': 20260926,
                         'unit': 'initial_society', 'condition_weights': {'normal': .5, 'attack': .5},
                         'confirmatory_families': {'main_vs_P_star': 4, 'module_and_history': 8},
                         'P_star': 'alarm_refresh', 'learned_aggregation': 'within_society_mean_of_all_three_seeds'},
            'reuse': {'RR': 'alarm_refresh', 'LL': 'main'},
            'extensions': {name: extension for name, extension in extensions.items()
                           if name == additional_diagnostic},
            'source_sha256': source_fingerprints(),
            'test_manifest_sha256': digest([e.to_dict() for e in episode_plan('test')])}


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def preflight(candidate):
    """Fail closed on omissions/drift; structural completeness is not launch approval."""
    if type(candidate) is not dict:
        raise ValueError('candidate must be a JSON object')
    plan = copy.deepcopy(candidate)
    blockers = []
    additional_diagnostic = plan.get('additional_diagnostic')
    if type(additional_diagnostic) is not str or additional_diagnostic not in ADDITIONAL_DIAGNOSTICS:
        blockers.append('additional_diagnostic: explicitly select one of: ' + ', '.join(ADDITIONAL_DIAGNOSTICS))
        additional_diagnostic = None
    expected = make_draft(additional_diagnostic)
    if set(plan) != set(expected):
        blockers.append('top-level fields: missing or unexpected entries')
    for key in ('version', 'runtime_release', 'focal_models', 'cells', 'analysis', 'reuse',
                'source_sha256', 'test_manifest_sha256'):
        if plan.get(key) != expected[key]:
            blockers.append(f'{key}: differs from the current declared protocol/source')
    if plan.get('status') != 'draft':
        blockers.append('status: only draft candidates are accepted by this preflight')
    models = plan.get('models', {})
    if type(models) is not dict:
        models = {}
    if set(models) != set(SLOTS):
        blockers.append('models: all four declared slots are required')
    regular = {s: sum(c['max_regular_requests'] for c in expected['cells'] if c['model_slot']==s) for s in SLOTS}
    allocations = []
    profiles = {}
    for slot in SLOTS:
        row = models.get(slot, {})
        if type(row) is not dict:
            row = {}
        if set(row) != {'role', 'profile', 'allocation'}:
            blockers.append(f'{slot}: invalid model entry fields')
        if row.get('role') != expected['models'][slot]['role']:
            blockers.append(f'{slot}: model role changed')
        profile = row.get('profile')
        required = {'provider', 'model_id', 'revision', 'family', 'transport_source_sha256',
                    'temperature', 'reasoning_budget', 'max_input_tokens', 'max_output_tokens',
                    'max_billable_output_tokens', 'timeout_seconds', 'internal_retries',
                    'mutable_routing', 'reservation_microusd', 'cost_bound_evidence_path'}
        if not isinstance(profile, dict) or set(profile) != required:
            blockers.append(f'{slot}: exact model/decode/transport/cost-bound profile missing')
        else:
            profiles[slot] = profile
            for k in ('provider','model_id','revision','family','transport_source_sha256','cost_bound_evidence_path'):
                if not isinstance(profile[k], str) or not profile[k].strip():
                    blockers.append(f'{slot}.{k}: required')
            fingerprint = profile['transport_source_sha256']
            if not isinstance(fingerprint,str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
                blockers.append(f'{slot}: transport fingerprint must be a SHA-256 hex digest')
            for k in ('max_input_tokens','max_output_tokens','max_billable_output_tokens','reservation_microusd'):
                if not _integer(profile[k], 1):
                    blockers.append(f'{slot}.{k}: positive integer required')
            if (_integer(profile['max_billable_output_tokens']) and _integer(profile['max_output_tokens'])
                    and profile['max_billable_output_tokens'] < profile['max_output_tokens']):
                blockers.append(f'{slot}: billable output bound cannot omit visible output')
            for k in ('temperature','timeout_seconds'):
                v = profile[k]
                if type(v) not in (int,float) or not math.isfinite(v) or v < (0 if k=='temperature' else 1):
                    blockers.append(f'{slot}.{k}: invalid numeric value')
            if profile['mutable_routing'] is not False or type(profile['internal_retries']) is not int or profile['internal_retries'] != 0:
                blockers.append(f'{slot}: automatic model routing/hidden retries forbidden')
        allocation = row.get('allocation')
        if not isinstance(allocation, dict) or set(allocation) != {'max_attempts','max_cost_microusd'} or not all(_integer(v,1) for v in allocation.values()):
            blockers.append(f'{slot}: positive request and money allocation required')
        else:
            allocations.append(allocation['max_cost_microusd'])
            if allocation['max_attempts'] < regular[slot]:
                blockers.append(f'{slot}: call allocation cannot cover the regular-request upper bound')
    if all(s in profiles for s in SLOTS):
        if profiles['M1']['family'] == profiles['M2']['family']:
            blockers.append('M2: another model family required')
        if profiles['M3']['family'] != profiles['M4']['family'] or profiles['M3']['model_id'] == profiles['M4']['model_id']:
            blockers.append('M3/M4: distinct sizes within the same open family required')
    pilot = plan.get('pilot', {})
    if type(pilot) is not dict:
        pilot = {}
    for k in ('episode_manifest','max_models','max_prompt_versions','max_regular_requests_per_model','scope_note'):
        if pilot.get(k) != expected['pilot'][k]:
            blockers.append(f'pilot.{k}: changed fixed cost-pilot scope')
    slots = pilot.get('model_slots', [])
    if not isinstance(slots,list) or not 1 <= len(slots) <= 2 or any(s not in SLOTS for s in slots) or len(set(slots)) != len(slots):
        blockers.append('pilot: select one or two distinct model slots')
        slots = []
    budgets = pilot.get('allocations', {})
    if type(budgets) is not dict:
        budgets = {}
    if set(budgets) != set(slots):
        blockers.append('pilot: every pilot model needs a separate budget allocation')
    for slot, budget in budgets.items():
        if (not isinstance(budget,dict) or set(budget) != {'max_attempts','max_cost_microusd'}
                or not all(_integer(v,1) for v in budget.values()) or budget['max_attempts'] < 480):
            blockers.append(f'pilot.{slot}: invalid allocation')
        else:
            allocations.append(budget['max_cost_microusd'])
    total = plan.get('project_budget_microusd')
    if not _integer(total,1):
        blockers.append('project: explicit total budget is missing')
    elif sum(allocations) > total:
        blockers.append('project: summed model and pilot allocations exceed total budget')
    extensions = plan.get('extensions', {})
    if type(extensions) is not dict:
        extensions = {}
    if set(extensions) != set(expected['extensions']):
        blockers.append('extensions: required extension set changed')
    for name, extension in expected['extensions'].items():
        provided = extensions.get(name, {})
        if type(provided) is not dict:
            provided = {}
        if set(provided) != set(extension):
            blockers.append(f'{name}: missing or undeclared extension fields')
        if any(provided.get(k) != v for k,v in extension.items() if k not in ('contract_path','manifest_path')):
            blockers.append(f'{name}: extension mechanism or population scope changed')
        for key in ('contract_path','manifest_path'):
            if not isinstance(provided.get(key),str) or not provided[key]:
                blockers.append(f'{name}.{key}: independent extension declaration missing')
    return {'candidate_sha256': digest(plan), 'configuration_complete': not blockers,
            'additional_diagnostic': additional_diagnostic,
            'blockers': blockers, 'regular_requests_by_model': regular,
            'total_regular_research_requests': sum(regular.values()),
            'allocated_cost_microusd': sum(allocations), 'runtime_release': 'engineering_only',
            'ready_for_paid_or_test_execution': False,
            'remaining_release_requirements': ['Verify referenced provider/extension evidence and adapter behavior',
                                               'Bind one shared budget ledger per allocation in the research runner',
                                               'Seal joint model/prompt/diagnostic configuration before any test execution']}
