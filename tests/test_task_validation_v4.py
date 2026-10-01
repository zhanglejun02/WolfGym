import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from wolfgym.core.agents.switching import install_switching_cohort, prepare_tactic
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.stock_market import load_stock_market
from wolfgym.evaluation.task_validation import (
    public_observation, public_features, fit_detector, detector_score,
    damage_series, damage_event, sustained_damage_score, exact_rate, reseed_normal_decisions, run_validation_branch,
)
from wolfgym.observations import ObservationBuilder


def config():
    return yaml.safe_load((Path(__file__).parents[1] / "tests/fixtures/task_validation.yaml").read_text())


def fixture():
    return install_switching_cohort(WolfSocietySimulator(load_stock_market("finfluencer"), 20, .1, seed=31))


def test_public_fast_view_equals_existing_contract_and_does_not_mutate():
    sim = fixture()
    for _ in range(7):
        prepare_tactic(sim, mode="S2", selection_seed=31)
        sim.step_day()
    before = sim.fork_from_state(sim.get_state())
    fast = public_observation(sim, 12)
    assert fast == ObservationBuilder().build(sim, sim.get_state(), remaining_budget=12)
    changed = replace(fast, day=100, remaining_budget=0,
                      assets=tuple(replace(a, asset_id="changed") for a in reversed(fast.assets)),
                      entities=tuple(replace(e, entity_id="changed") for e in reversed(fast.entities)))
    assert public_features(fast) == public_features(changed)
    for _ in range(3):
        assert prepare_tactic(sim, mode="S2", selection_seed=31) == prepare_tactic(before, mode="S2", selection_seed=31)
        assert sim.step_day() == before.step_day()
    assert [a.portfolio for a in sim.society.all_agents] == [a.portfolio for a in before.society.all_agents]


def test_normal_replica_changes_only_normal_decision_streams():
    sim = fixture()
    before_market = copy.deepcopy(sim.market.rng.bit_generator.state)
    before_attack = copy.deepcopy(sim.society.attackers[0].rng.bit_generator.state)
    before_retail = copy.deepcopy(sim.society.retail[0].rng.bit_generator.state)
    reseed_normal_decisions(sim, 2000031)
    assert sim.market.rng.bit_generator.state == before_market
    assert sim.society.attackers[0].rng.bit_generator.state == before_attack
    assert sim.society.retail[0].rng.bit_generator.state != before_retail


def test_collapse_needs_both_damage_axes_and_persistence_without_lowering_floor():
    assert not damage_event([.003], [.002], .0025)["event"]
    assert not damage_event([.0025, .0025], [.002, .002], .0025)["event"]
    assert not damage_event([.01, .01], [.002, -.002], .0025)["event"]
    assert damage_event([.003, .003, 0.], [.001, .001, 0.], .0025)["first_day"] == 2
    assert sustained_damage_score([.003, .004, .01], [.001, .001, 0.]) == .003
    with pytest.raises(ValueError):
        damage_event([.01, .01], [.002, .002], .001)
    with pytest.raises(ValueError):
        sustained_damage_score([.003], [])


def test_exact_bound_does_not_treat_zero_as_certified_zero_risk():
    assert exact_rate(0, 64)["upper_95_one_sided"] == pytest.approx(1 - .05 ** (1 / 64))
    assert exact_rate(64, 64)["lower_95_one_sided"] < 1.


def test_detector_schema_and_deterministic_fit():
    features = [{"a": float(i), "constant": 4.} for i in range(10)]
    labels = [0] * 5 + [1] * 5
    model = fit_detector(features, labels, config()["detector"])
    assert model == fit_detector(features, labels, config()["detector"])
    assert detector_score(features[-1], model) > detector_score(features[0], model)
    assert json.loads(json.dumps(model)) == model
    with pytest.raises(ValueError):
        detector_score({"a": 1.}, model)


def test_all_branches_pair_and_replay_with_budget_and_serializable_outputs():
    sim = fixture()
    cfg = config()
    initial = sim.get_state()
    names = sorted(public_features(public_observation(sim, 12)))
    # Always-alert fixture exercises six single-day throttles without a learned model.
    model = {"features": names, "center": [0.] * len(names), "scale": [1.] * len(names),
             "coefficient": [0.] * len(names), "intercept": 10.}
    args = dict(seed=31, config=cfg, pair_id="fixture", model=model, frozen={"alarm_threshold": .5, "harm_threshold": .0025})
    reference, ref_rows = run_validation_branch(sim, initial, name="reference", **args)
    for name in cfg["validation"]["branches"]:
        snapshots, rows = run_validation_branch(sim, initial, name=name, **args)
        series = damage_series(reference, snapshots)
        json.dumps(series, allow_nan=False)
        assert all(r.fundamentals == a.fundamentals for r, a in zip(reference, snapshots))
        assert sum(row["cost"] for row in rows) <= 12
        if name.endswith("public"):
            assert sum(row["cost"] > 0 for row in rows) == 6
            assert all(row["financial_outcomes"]["blocked_notional"] == 0 for row in rows if not row["cost"])
        if name == "normal_replica":
            assert snapshots != reference
        if name == "attack":
            assert (snapshots, rows) == run_validation_branch(sim, initial, name=name, **args)
    assert sim._next_day == 0
