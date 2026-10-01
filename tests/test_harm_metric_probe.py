from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.stock_market import load_stock_market
from wolfgym.evaluation.harm_metric_probe import choose_action, run_branch, summarize_branch
from wolfgym.observations import ObservationBuilder
from wolfgym.regulation import RegulatoryActionType


def config():
    return yaml.safe_load((Path(__file__).parents[1] / "tests/fixtures/harm_probe.yaml").read_text())


def test_public_schedule_targets_only_current_public_assets_and_respects_budget():
    sim = WolfSocietySimulator(load_stock_market(), 20, 0., seed=31)
    observation = ObservationBuilder().build(sim, sim.get_state(), remaining_budget=12)
    observation = replace(observation, day=5)
    action = choose_action("scheduled_public_freeze", observation, config())
    assert action.action_type is RegulatoryActionType.FREEZE
    assert action.target_id in {a.asset_id for a in observation.assets}
    assert choose_action("scheduled_public_freeze", replace(observation, day=4), config()).action_type is RegulatoryActionType.WAIT
    assert choose_action("scheduled_public_freeze", replace(observation, remaining_budget=3), config()).action_type is RegulatoryActionType.WAIT


@pytest.mark.parametrize("family", [None, "pump_and_dump", "finfluencer"])
def test_probe_pairs_full_histories_and_single_day_budgeted_controls(family):
    template = WolfSocietySimulator(load_stock_market(family), 20, 0. if family is None else .1, seed=31)
    initial = template.get_state()
    arguments = dict(pair_id="fixture", continuation_seed=1031, config=config())
    reference, ref_rows = run_branch(template, initial, kind="reference", **arguments)
    wait, wait_rows = run_branch(template, initial, kind="wait", **arguments)
    policy, rows = run_branch(template, initial, kind="scheduled_public_freeze", **arguments)
    summary = summarize_branch(reference, wait, policy, rows)
    assert summary["intervention_cost"] == 12
    assert summary["collapse"]["horizon_complete"] is True
    assert [r["day"] for r in rows if r["cost"]] == [6, 11, 16]
    assert all(r["financial_outcomes"]["blocked_notional"] == 0 for r in rows if not r["cost"])
    assert len(reference) == len(wait) == len(policy) == 30
    assert template._next_day == 0
    if family is None:
        assert reference == wait
        assert ref_rows == wait_rows
        assert summary["welfare"]["shortfall_reduction"] <= 0
