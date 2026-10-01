import copy
import json
import runpy
import sys

import pytest

from wolfgym.benchmark.combination import episode_plan
from wolfgym.benchmark.combination_experiments import REPO, make_draft, preflight, experiment_cells, cost_pilot_episodes


def filled_example(additional_diagnostic='none'):
    """Synthetic structural example, never a real provider freeze or spend grant."""
    p = make_draft(additional_diagnostic)
    for s, row in p['models'].items():
        row['profile'] = dict(provider='mock', model_id=s, revision='mock-immutable-1',
            family='open-mock' if s in ('M3','M4') else s,
            transport_source_sha256='a'*64, temperature=0, reasoning_budget='disabled',
            max_input_tokens=8000, max_output_tokens=128, max_billable_output_tokens=128,
            timeout_seconds=30, internal_retries=0, mutable_routing=False,
            reservation_microusd=1000, cost_bound_evidence_path='synthetic-not-verified.json')
        row['allocation'] = {'max_attempts': 250000, 'max_cost_microusd': 100000000}
    p['project_budget_microusd'] = 402000000
    p['pilot']['model_slots'] = ['M1','M3']
    p['pilot']['allocations'] = {s: {'max_attempts':1440, 'max_cost_microusd':1000000} for s in ('M1','M3')}
    for row in p['extensions'].values():
        row['contract_path'] = 'synthetic-contract.json'
        row['manifest_path'] = 'synthetic-manifest.json'
    return p


def test_inventory_counts_reuse_and_complete_pilot_pairs():
    cells = experiment_cells()
    assert len(cells) == len({c['id'] for c in cells}) == 10
    assert sum(c['max_regular_requests'] for c in cells) == 125952
    assert sum(c['max_regular_requests'] for c in cells if c['experiment']=='main') == 61440
    assert all(c['mode'] != 'RR' for c in cells)
    pilot = cost_pilot_episodes()
    ordered_groups = list(dict.fromkeys(e.group_id for e in episode_plan('train')))[:8]
    assert len(pilot) == 16 and {r['group_id'] for r in pilot} == set(ordered_groups)
    for group in ordered_groups:
        assert {r['condition'] for r in pilot if r['group_id']==group} == {'normal','attack'}
    assert all(r['split']=='train' for r in pilot)


def test_draft_is_explicitly_unfunded_and_not_runnable():
    draft = make_draft()
    assert draft['additional_diagnostic'] is None and draft['extensions'] == {}
    result = preflight(draft)
    assert not result['configuration_complete']
    assert not result['ready_for_paid_or_test_execution']
    assert any('total budget' in b for b in result['blockers'])
    assert any(b.startswith('additional_diagnostic:') for b in result['blockers'])
    assert result['regular_requests_by_model'] == {'M1':47616,'M2':15360,'M3':47616,'M4':15360}


@pytest.mark.parametrize('choice,focal_requests,total,extension', [
    ('none', 47616, 125952, None),
    ('rename_reorder', 62976, 156672, None),
    ('liquidity', 55296, 141312, 'liquidity'),
    ('benign_spread', 51456, 133632, 'benign_spread'),
])
def test_selected_diagnostic_counts_and_completeness_do_not_unlock_runtime(choice, focal_requests, total, extension):
    draft = filled_example(choice)
    result = preflight(draft)
    assert len(draft['cells']) == (10 if choice == 'none' else 12)
    assert set(draft['extensions']) == ({extension} if extension else set())
    assert result['additional_diagnostic'] == choice
    assert result['regular_requests_by_model'] == {
        'M1':focal_requests,'M2':15360,'M3':focal_requests,'M4':15360}
    assert result['total_regular_research_requests'] == total
    assert draft['analysis']['confirmatory_families'] == {'main_vs_P_star':4,'module_and_history':8}
    assert result['configuration_complete'] and not result['blockers']
    assert result['runtime_release'] == 'engineering_only'
    assert not result['ready_for_paid_or_test_execution']
    assert result['remaining_release_requirements']


@pytest.mark.parametrize('choice', [None, 'unknown', True, ['liquidity'], ['liquidity','benign_spread']])
def test_diagnostic_requires_one_explicit_valid_choice(choice):
    draft = filled_example()
    draft['additional_diagnostic'] = choice
    result = preflight(draft)
    assert not result['configuration_complete']
    assert any(b.startswith('additional_diagnostic:') for b in result['blockers'])
    if choice is not None:
        with pytest.raises(ValueError, match='additional_diagnostic'):
            make_draft(choice)
        with pytest.raises(ValueError, match='additional_diagnostic'):
            experiment_cells(choice)


def test_diagnostic_cannot_be_missing_or_drift_from_cells_and_extensions():
    draft = filled_example()
    del draft['additional_diagnostic']
    assert not preflight(draft)['configuration_complete']
    draft = filled_example('liquidity')
    draft['additional_diagnostic'] = 'benign_spread'
    result = preflight(draft)
    assert any(b.startswith('cells:') for b in result['blockers'])
    assert any(b.startswith('extensions:') for b in result['blockers'])


@pytest.mark.parametrize('choice', ['liquidity', 'benign_spread'])
def test_only_selected_extension_requires_declared_evidence(choice):
    draft = filled_example(choice)
    draft['extensions'][choice]['contract_path'] = None
    assert any(b.startswith(choice+'.contract_path:') for b in preflight(draft)['blockers'])
    draft = filled_example('none')
    draft['extensions'] = filled_example(choice)['extensions']
    assert any(b.startswith('extensions:') for b in preflight(draft)['blockers'])


@pytest.mark.parametrize('choice', [None, 'none', 'rename_reorder', 'liquidity', 'benign_spread'])
def test_offline_cli_creates_matching_selection_cells_and_extensions(tmp_path, monkeypatch, choice):
    main = runpy.run_path(str(REPO/'scripts/plan_combination_experiments.py'))['main']
    arguments = ['plan_combination_experiments.py', '--output', str(tmp_path)]
    if choice is not None:
        arguments += ['--additional-diagnostic', choice]
    monkeypatch.setattr(sys, 'argv', arguments)
    main()
    draft = json.loads((tmp_path/'candidate.json').read_text())
    expected = make_draft(choice)
    assert draft['additional_diagnostic'] == choice
    assert draft['cells'] == expected['cells'] and draft['extensions'] == expected['extensions']
    assert not json.loads((tmp_path/'preflight.json').read_text())['ready_for_paid_or_test_execution']


def test_offline_cli_cannot_silently_change_an_existing_candidate(tmp_path, monkeypatch):
    (tmp_path/'candidate.json').write_text(json.dumps(make_draft('none')))
    main = runpy.run_path(str(REPO/'scripts/plan_combination_experiments.py'))['main']
    monkeypatch.setattr(sys, 'argv', ['plan_combination_experiments.py', '--output', str(tmp_path),
                                   '--additional-diagnostic', 'liquidity'])
    with pytest.raises(ValueError, match='existing candidate differs'):
        main()
    assert json.loads((tmp_path/'candidate.json').read_text())['additional_diagnostic'] == 'none'


@pytest.mark.parametrize('key', ['cells','analysis','source_sha256','test_manifest_sha256','focal_models'])
def test_joint_configuration_cannot_drop_conditions_or_change_statistics(key):
    p = filled_example()
    p[key] = [] if isinstance(p[key],list) else {}
    result = preflight(p)
    assert any(b.startswith(key+':') for b in result['blockers'])


def test_project_budget_includes_pilot_allocations_and_request_coverage():
    p = filled_example()
    p['project_budget_microusd'] -= 1
    p['models']['M1']['allocation']['max_attempts'] = 47615
    result = preflight(p)
    assert any('exceed total' in b for b in result['blockers'])
    assert any('regular-request upper bound' in b for b in result['blockers'])


def test_pilot_cannot_select_attack_only_or_add_models_and_prompt_budgets():
    p = filled_example()
    p['pilot']['episode_manifest'] = [r for r in p['pilot']['episode_manifest'] if r['condition']=='attack']
    p['pilot']['model_slots'] = ['M1','M2','M3']
    p['pilot']['max_prompt_versions'] = 3
    result = preflight(p)
    assert any('episode_manifest' in b for b in result['blockers'])
    assert any('distinct model slots' in b for b in result['blockers'])
    assert any('max_prompt_versions' in b for b in result['blockers'])


def test_model_and_extension_contract_drift_rejected():
    p = filled_example('liquidity')
    p['models']['M1']['profile']['internal_retries'] = 1
    p['models']['M2']['profile']['family'] = 'M1'
    p['models']['M3']['profile']['max_billable_output_tokens'] = 64
    p['extensions']['liquidity']['to'] = 6000
    p['extensions']['liquidity']['new_hidden_parameter'] = 2
    result = preflight(p)
    assert any('hidden retries' in b for b in result['blockers'])
    assert any('another model family' in b for b in result['blockers'])
    assert any('visible output' in b for b in result['blockers'])
    assert any('mechanism' in b for b in result['blockers'])
    assert any('undeclared extension fields' in b for b in result['blockers'])


def test_malformed_sections_and_boolean_limits_fail_closed():
    p = filled_example('liquidity')
    p['models']['M1']['allocation']['max_cost_microusd'] = True
    p['models']['M2'] = None
    p['pilot']['model_slots'] = [{}]
    p['extensions']['liquidity'] = None
    assert not preflight(p)['configuration_complete']
    with pytest.raises(ValueError):
        preflight([])
