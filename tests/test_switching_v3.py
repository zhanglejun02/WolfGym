from dataclasses import replace
from pathlib import Path
import copy
import json
import numpy as np

import pytest
import yaml

from wolfgym.core.agents.switching import TACTICS, install_switching_cohort, select_tactic, prepare_tactic
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.stock_market import load_stock_market
from wolfgym.evaluation.switching_probe import run_switching_branch, check_wallets, branch_specs
from wolfgym.evaluation.composite_harm import score_composite_harm
from wolfgym.evaluation.counterfactual_welfare import InvestorAccount, WelfareSnapshot


def config():
    return yaml.safe_load((Path(__file__).parents[1] / "tests/fixtures/switching_probe.yaml").read_text())


def fixture():
    return install_switching_cohort(WolfSocietySimulator(load_stock_market("finfluencer"), 20, .1, seed=31))


def test_install_preserves_wealth_and_shared_wallets_and_counterparties():
    sim = fixture()
    check_wallets(sim)
    ids = {a.agent_id for a in sim.society.attackers}
    for a in sim.society.attackers:
        assert a.portfolio.cash == .8 * a.portfolio.initial_wealth
        assert a.portfolio.mark_to_market({k: v.price for k, v in sim.market.assets.items()}) == a.portfolio.initial_wealth
        assert a.modules["S4"].counterparty_id in ids - {a.agent_id}
    restored = sim.fork_from_state(sim.get_state())
    restored.set_continuation_seed(55)
    check_wallets(restored)
    assert restored.society.attackers[0].portfolio is not sim.society.attackers[0].portfolio
    with pytest.raises(ValueError):
        install_switching_cohort(sim)


def test_random_ablation_keeps_opportunities_and_never_reallocates():
    args = dict(start_delay=2, public_signals={}, inventory_share=.2, selection_seed=90)
    original = [select_tactic("random", day=d, **args) for d in range(30)]
    assert original == [select_tactic("random", day=d, **args) for d in range(30)]
    assert set(a for a, b in original) == {*TACTICS, "WAIT"}
    for tactic in TACTICS:
        for day, (planned, actual) in enumerate(original):
            assert select_tactic("random", day=day, disabled=tactic, **args) == (planned, "WAIT" if planned == tactic else actual)
    with pytest.raises(ValueError):
        select_tactic("unknown", day=-1, **args)


@pytest.mark.parametrize("signals,share,wanted", [
    ({"return_3d": .04, "volume_ratio": .5, "depth_imbalance": 0.}, .2, "S2"),
    ({"return_3d": .04, "volume_ratio": .5, "depth_imbalance": 0.}, .05, "S4"),
    ({"return_3d": 0., "volume_ratio": 2., "depth_imbalance": 0.}, .2, "S3"),
    ({"return_3d": 0., "volume_ratio": 2., "depth_imbalance": .4}, .2, "S1"),
])
def test_adaptive_public_priority(signals, share, wanted):
    assert select_tactic("adaptive", day=10, start_delay=0, public_signals=signals,
                        inventory_share=share, selection_seed=1) == (wanted, wanted)


def test_cap_and_public_whitelist_and_no_capital_refresh():
    sim = fixture()
    a = sim.society.attackers[0]
    a.selected_tactic = "S4"
    view = {"prices": {k: v.price for k, v in sim.market.assets.items()},
            "market": {k: {"price": v.price, "volume": 1e10, "fundamental": 9999} for k, v in sim.market.assets.items()}}
    other = copy.deepcopy(a)
    changed = copy.deepcopy(view)
    changed["social_env"] = object()
    for entry in changed["market"].values():
        entry["fundamental"] = -999999
    before = copy.deepcopy(a.portfolio)
    output = a.decide(10, view)
    assert output == other.decide(10, changed)
    assert a.portfolio == before
    assert a.last_activity["submitted_notional"] <= a.portfolio.initial_wealth + 1e-7
    assert a.last_activity["orders"] > 0


def test_mid_episode_restore_retains_tactic_rng_and_wallet_state():
    sim = fixture()
    sim.set_continuation_seed(1031)
    for _ in range(12):
        prepare_tactic(sim, mode="random", selection_seed=1031)
        sim.step_day()
    restored = sim.fork_from_state(sim.get_state())
    for _ in range(18):
        assert prepare_tactic(sim, mode="random", selection_seed=1031) == prepare_tactic(restored, mode="random", selection_seed=1031)
        assert sim.step_day() == restored.step_day()
        check_wallets(restored)
    assert [a.portfolio for a in sim.society.all_agents] == [a.portfolio for a in restored.society.all_agents]


@pytest.mark.parametrize("name", ["reference", "random", "adaptive", "single_S3", "minus_S2",
                                  "adaptive_scheduled_public_freeze", "off_blanket_freeze"])
def test_full_small_branches_replay_and_single_day_controls(name):
    sim = fixture()
    state = sim.get_state()
    args = dict(pair_id="fixture", continuation_seed=1031, spec=branch_specs()[name], config=config())
    snapshots, rows = run_switching_branch(sim, state, **args)
    assert (snapshots, rows) == run_switching_branch(sim, state, **args)
    assert len(snapshots) == 30 and sim._next_day == 0
    if name == "adaptive_scheduled_public_freeze":
        assert sum(r["cost"] for r in rows) == 12
        assert [r["day"] for r in rows if r["cost"]] == [6, 11, 16]
        assert all(r["financial_outcomes"]["blocked_notional"] == 0 for r in rows if not r["cost"])
    elif name == "off_blanket_freeze":
        assert sum(r["cost"] for r in rows) == 360
        assert all(r["financial_outcomes"]["executed_notional"] == 0 for r in rows)


def history(cash_days):
    return [WelfareSnapshot("pair", day, {"stock": 1.},
                            {str(i): InvestorAccount(100., cash, {}, "retail") for i, cash in enumerate(values)})
            for day, values in enumerate(cash_days, 1)]


def telemetry(days, *, unmet=0., demand=100.):
    return [{"day": d, "financial_outcomes": {"requested_notional": demand, "blocked_notional": unmet * demand,
                                              "unexecuted_notional": 0., "executed_notional": (1 - unmet) * demand}}
            for d in range(1, days + 1)]


def test_tail_exact_threshold_and_recovery_latched_without_broad():
    ref = history([[100.] * 100] * 2)
    actual = history([[90.] + [100.] * 99, [100.] * 100])
    result = score_composite_harm(ref, actual, telemetry(2), telemetry(2))
    assert result["components"] == {"broad": False, "tail": True, "function": False}
    assert result["first_event_day"] == 1
    assert not result["daily"][1]["tail"]
    assert not score_composite_harm(ref, history([[90.01] + [100.] * 99] * 2), telemetry(2), telemetry(2))["event"]


def test_composite_serializes_numpy_account_scalars():
    ref = history([[np.float64(100.)] * 100] * 2)
    actual = history([[np.float64(90.)] + [np.float64(100.)] * 99] * 2)
    result = score_composite_harm(ref, actual, telemetry(2), telemetry(2))
    assert json.loads(json.dumps(result)) == result


def test_broad_keeps_v2_same_account_persistence():
    ref = history([[100.] * 100] * 2)
    actual = history([[95.] * 10 + [100.] * 90] * 2)
    result = score_composite_harm(ref, actual, telemetry(2), telemetry(2))
    assert result["components"] == {"broad": True, "tail": False, "function": False}
    assert result["first_event_day"] == 2


def test_function_exact_boundary_needs_two_days_and_material_demand():
    ref = history([[100.] * 100] * 2)
    result = score_composite_harm(ref, ref, telemetry(2, unmet=.5, demand=10.), telemetry(2, unmet=1., demand=10.))
    assert result["components"] == {"broad": False, "tail": False, "function": True}
    assert result["first_event_day"] == 2
    assert not score_composite_harm(ref[:1], ref[:1], telemetry(1), telemetry(1, unmet=1.))["event"]
    assert not score_composite_harm(ref, ref, telemetry(2, demand=9.99), telemetry(2, unmet=1., demand=9.99))["event"]
    assert not score_composite_harm(ref, ref, telemetry(2, unmet=.51), telemetry(2, unmet=1.))["event"]


def test_bad_telemetry_and_pairing_rejected():
    ref = history([[100.] * 100] * 2)
    bad = telemetry(2)
    bad[0]["financial_outcomes"]["unexecuted_notional"] = 1.
    with pytest.raises(ValueError, match="inconsistent"):
        score_composite_harm(ref, ref, telemetry(2), bad)
    with pytest.raises(ValueError):
        score_composite_harm(ref, [replace(ref[0], fundamentals={"stock": 2.}), ref[1]], telemetry(2), telemetry(2))
