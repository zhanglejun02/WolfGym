import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_combination_actions import candidate_plans, selected_episodes, random_plans, validate_plan, summarize


def test_candidate_budget_boundaries_and_pairing():
    for base in ([], [(0, 1), (1, 2), (29, 3)], [(8, 2), (13, 3), (18, 1)]):
        plans = candidate_plans(base)
        assert len(plans) == 24
        for plan in plans.values(): validate_plan(plan)
        assert {k: v for k, v in plans.items() if k.startswith('random')} == random_plans()
    assert candidate_plans([(0, 1), (29, 2)])['shift_-2'] == [(27, 2)]
    assert candidate_plans([(0, 1), (29, 2)])['shift_+2'] == [(2, 1)]
    eps = selected_episodes()
    assert len(eps) == 16 and all(e['split'] == 'train' for e in eps)
    assert len({e['group_id'] for e in eps}) == 8
    for g in {e['group_id'] for e in eps}:
        assert {e['condition'] for e in eps if e['group_id'] == g} == {'normal', 'attack'}


@pytest.mark.parametrize('plan', [[(0, 1), (0, 2)], [(0, 0)], [(30, 1)], [(1, True)], [(i, 1) for i in range(4)]])
def test_reject_illegal_plans(plan):
    with pytest.raises(ValueError): validate_plan(plan)


def test_complete_grid_hindsight_is_not_global_policy():
    rows = []
    for e in selected_episodes():
        for name in candidate_plans([]):
            benefit = (0.1 if name == 'rule' else 0.2 if name == 'random_00' else 0.) if e['condition'] == 'attack' else (-0.1 if name != 'wait' else 0.)
            rows.append({**e, 'candidate': name, 'benefit_pp': benefit, 'cost': 0 if name == 'wait' else 12,
                         'H_wait_pp': 1., 'H_policy_pp': 1.-benefit, 'event_wait': True,
                         'event_policy': True, 'plan': [], 'financial_delta': {'executed_notional': 0.}})
    result = summarize(rows)
    assert result['balanced_headroom_pp'] == pytest.approx(.1)
    assert result['summaries']['hindsight_best']['normal']['benefit_pp'] == 0
    assert result['summaries']['random_00']['normal']['benefit_pp'] == -.1
    for invalid in (rows[:-1], rows+[rows[0]]):
        with pytest.raises(ValueError): summarize(invalid)
