from dataclasses import replace
from types import SimpleNamespace

import pytest

from wolfgym.evaluation.counterfactual_welfare import (
    InvestorAccount, WelfareSnapshot, capture_welfare_snapshot, score_welfare_triplet,
    COLLAPSE_DEFINITION_VERSION, score_collapse_history,
)


def snapshot(cash, *, initial=100., pair_id="same-initial-state-and-seed"):
    return WelfareSnapshot(pair_id, 30, {"stock": 10.}, {
        str(i): InvestorAccount(initial, float(value), {}, "retail") for i, value in enumerate(cash)
    })


def test_harm_can_exist_while_everyone_is_above_initial_wealth():
    result = score_welfare_triplet(snapshot([115, 115]), snapshot([110, 110]), snapshot([113, 113]))
    assert result["wait"]["positive_shortfall_rate"] == pytest.approx(.05)
    assert result["shortfall_reduction"] == pytest.approx(.03)


def test_other_investors_gains_do_not_cancel_a_victims_loss():
    reference = snapshot([100, 100])
    wait = snapshot([80, 120])
    result = score_welfare_triplet(reference, wait, reference)
    assert result["wait"]["signed_net_gap_rate"] == 0
    assert result["wait"]["positive_shortfall_rate"] == pytest.approx(.1)
    assert result["wait"]["worst_decile_mean_shortfall_rate"] == pytest.approx(.2)


def test_excess_profit_is_not_an_extra_protection_bonus():
    reference = snapshot([100, 100])
    wait = snapshot([90, 90])
    exact = score_welfare_triplet(reference, wait, reference)
    excess = score_welfare_triplet(reference, wait, snapshot([200, 200]))
    assert exact["shortfall_reduction"] == excess["shortfall_reduction"] == .1
    assert excess["net_wealth_gain"] > exact["net_wealth_gain"]


def test_clean_intervention_damage_is_negative_and_zero_baseline_is_valid():
    reference = snapshot([100, 100])
    result = score_welfare_triplet(reference, reference, snapshot([95, 95]))
    assert result["wait"]["positive_shortfall_rate"] == 0
    assert result["shortfall_reduction"] == pytest.approx(-.05)
    assert result["release_eligible"] is False


def test_inflating_terminal_market_price_cannot_improve_score():
    asset = SimpleNamespace(price=10., fundamental=10.)
    agent = SimpleNamespace(agent_id="a", sub_role="retail", portfolio=SimpleNamespace(
        cash=0., initial_wealth=100., holdings={"stock": 10.}))
    simulator = SimpleNamespace(_next_day=30, market=SimpleNamespace(assets={"stock": asset}),
                                society=SimpleNamespace(retail=[agent]))
    before = capture_welfare_snapshot(simulator, pair_id="pair")
    asset.price = 1000.
    after = capture_welfare_snapshot(simulator, pair_id="pair")
    assert score_welfare_triplet(before, before, after)["shortfall_reduction"] == 0


@pytest.mark.parametrize("change, message", [
    ({"pair_id": "different"}, "pairing identities"),
    ({"day": 29}, "day-30"),
    ({"fundamentals": {"stock": 11.}}, "fundamentals diverged"),
    ({"accounts": {"another": InvestorAccount(100., 100., {}, "retail")}}, "protected cohort"),
    ({"accounts": {"0": InvestorAccount(90., 90., {}, "retail")}}, "endowment/group"),
])
def test_incompatible_branches_are_rejected(change, message):
    reference = snapshot([100])
    with pytest.raises(ValueError, match=message):
        score_welfare_triplet(reference, reference, replace(reference, **change))


def test_missing_valuation_is_rejected_instead_of_silently_zeroing_holdings():
    reference = snapshot([100])
    bad = replace(reference, accounts={"0": InvestorAccount(100., 0., {"missing": 10.}, "retail")})
    with pytest.raises(ValueError, match="no common terminal mark"):
        score_welfare_triplet(reference, reference, bad)


def test_account_order_does_not_change_scores_and_tail_uses_full_cohort():
    reference = snapshot([100] * 10)
    wait = snapshot([90] + [100] * 9)
    result = score_welfare_triplet(reference, wait, reference)
    reordered = replace(wait, accounts=dict(reversed(list(wait.accounts.items()))))
    assert result == score_welfare_triplet(reference, reordered, reference)
    assert result["wait"]["positive_shortfall_rate"] == pytest.approx(.01)
    assert result["wait"]["worst_decile_mean_shortfall_rate"] == pytest.approx(.1)


def histories(daily_cash, *, reference_cash=None):
    reference_cash = reference_cash or [100.] * len(daily_cash[0])
    return (
        [replace(snapshot(reference_cash), day=day) for day in range(1, len(daily_cash) + 1)],
        [replace(snapshot(cash), day=day) for day, cash in enumerate(daily_cash, start=1)],
    )


def test_collapse_includes_exact_thresholds_and_survives_recovery():
    reference, branch = histories([[95.] + [100.] * 9] * 2 + [[100.] * 10])
    result = score_collapse_history(reference, branch)
    assert result["collapse"] is True
    assert result["first_collapse_day"] == 2
    assert result["required_harmed_accounts"] == 1
    assert result["daily"][-1]["collapse_at_close"] is False
    assert result["horizon_complete"] is False
    assert result["definition"] == COLLAPSE_DEFINITION_VERSION


@pytest.mark.parametrize("n, harmed, expected", [(490, 48, False), (490, 49, True),
                                                (11, 1, False), (11, 2, True)])
def test_collapse_requires_ceil_ten_percent_of_initial_retail(n, harmed, expected):
    result = score_collapse_history(*histories([[95.] * harmed + [100.] * (n - harmed)] * 2))
    assert result["collapse"] is expected


def test_collapse_requires_same_accounts_not_two_different_victim_groups():
    result = score_collapse_history(*histories([[95., 100.] + [100.] * 8,
                                                [100., 95.] + [100.] * 8]))
    assert result["collapse"] is False
    assert [row["harmed_accounts"] for row in result["daily"]] == [1, 1]


@pytest.mark.parametrize("cash", [[[95.], [100.], [95.]], [[95.000000001]] * 2, [[95.]]])
def test_spikes_subthreshold_losses_and_single_close_do_not_collapse(cash):
    assert score_collapse_history(*histories(cash))["collapse"] is False


def test_collapse_uses_counterfactual_loss_and_own_initial_wealth():
    # Still profitable against initial 100; shortfall 5 / 100, not 5 / 115.
    assert score_collapse_history(*histories([[110.], [110.]], reference_cash=[115.]))["collapse"]


def test_collapse_values_each_close_at_its_common_contemporaneous_fundamentals():
    reference, branch = histories([[100.], [100.]])
    for index, mark in enumerate((10., 20.)):
        reference[index] = replace(reference[index], fundamentals={"stock": mark},
                                   accounts={"0": InvestorAccount(100., 0., {"stock": 10.}, "retail")})
        branch[index] = replace(branch[index], fundamentals={"stock": mark},
                                accounts={"0": InvestorAccount(100., 0., {"stock": 9.5}, "retail")})
    assert score_collapse_history(reference, branch)["collapse"]


@pytest.mark.parametrize("change, message", [
    ({"day": 3}, "every close"),
    ({"pair_id": "other"}, "pairing identities"),
    ({"accounts": {"other": InvestorAccount(100., 95., {}, "retail")}}, "protected cohort"),
    ({"fundamentals": {"stock": 11.}}, "fundamentals diverged"),
])
def test_collapse_rejects_invalid_paired_history(change, message):
    reference, branch = histories([[95.], [95.]])
    branch[1] = replace(branch[1], **change)
    with pytest.raises(ValueError, match=message):
        score_collapse_history(reference, branch)


def test_collapse_rejects_endowment_changes_even_when_both_branches_change():
    reference, branch = histories([[95.], [95.]])
    for series in (reference, branch):
        series[1] = replace(series[1], accounts={"0": InvestorAccount(90., 90., {}, "retail")})
    with pytest.raises(ValueError, match="endowment/group"):
        score_collapse_history(reference, branch)


def test_clean_wait_has_no_event_but_clean_intervention_can_collapse():
    reference, policy = histories([[95.]] * 30)
    wait_result = score_collapse_history(reference, reference)
    assert wait_result["collapse"] is False
    assert wait_result["horizon_complete"] is True
    assert score_collapse_history(reference, policy)["collapse"] is True


def test_collapse_config_matches_the_implemented_definition():
    from pathlib import Path
    import yaml
    config = yaml.safe_load((Path(__file__).parents[1] / "tests/fixtures/welfare_protocol.yaml").read_text())
    result = score_collapse_history(*histories([[95.], [95.]]))
    for key in ("definition", "individual_shortfall_threshold", "cohort_fraction_threshold", "persistence_days"):
        assert config["collapse"][key] == result[key]


@pytest.mark.parametrize("family", [None, "pump_and_dump", "finfluencer"])
def test_small_simulator_fixture_preserves_common_fundamentals_and_cohort(family):
    from wolfgym.core.env.environment import WolfSocietySimulator
    from wolfgym.core.scenarios.stock_market import load_stock_market
    from wolfgym.regulation import RegulatoryAction, RegulatoryActionType
    from wolfgym.regulation.executor import RegulatoryActionExecutor

    original = WolfSocietySimulator(load_stock_market(family), 20, 0. if family is None else .1, seed=17)
    initial = original.get_state()
    executor = RegulatoryActionExecutor()
    snapshots = []
    for kind in ("reference", "wait", "policy"):
        branch = original.fork_from_state(initial)
        branch.set_continuation_seed(23)
        if kind == "reference":
            for agent in branch.society.attackers:
                executor.apply(branch, RegulatoryAction(RegulatoryActionType.FREEZE,
                                                        branch.public_entity_id(agent.agent_id)), 0)
        target = branch.public_asset_id(branch.target_asset)
        for day in range(30):
            action = RegulatoryAction(RegulatoryActionType.FREEZE, target)
            branch.step_day(pre_step=(lambda sim, t: executor.apply(sim, action, t))
                            if kind == "policy" and day == 5 else None)
            if kind == "policy" and day == 5:
                executor.release(branch, target)
        snapshots.append(capture_welfare_snapshot(branch, pair_id="fixture:17:23"))
    result = score_welfare_triplet(*snapshots)
    assert result["protected_investors"] == len(original.society.retail)
    assert original._next_day == 0
    if family is None:
        assert result["wait"]["positive_shortfall_rate"] == 0
