from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.env.market import Order
from wolfgym.core.env.social import Message
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.counterfactual import CounterfactualEvaluator, ProportionateOracle
from wolfgym.counterfactual.evaluator import ActionOutcome, CounterfactualEvaluation
from wolfgym.counterfactual.failure import (
    FailureCriterionIdentity,
    FailureEvent,
    failure_criterion_for_scenario,
)
from wolfgym.regulation import (
    RegulatoryAction,
    RegulatoryActionExecutor,
    RegulatoryActionType,
)


DIRECT_ACTIONS = [
    RegulatoryAction(RegulatoryActionType.WAIT),
    RegulatoryAction(RegulatoryActionType.THROTTLE),
    RegulatoryAction(RegulatoryActionType.FREEZE),
]


def _env(scenario: str = "s1", seed: int = 7) -> WolfSocietySimulator:
    alpha = 0.0 if scenario == "s0" else 0.10
    return WolfSocietySimulator(load_scenario(scenario), n_society=40, alpha=alpha, seed=seed)


def test_incremental_step_is_identical_to_legacy_run() -> None:
    direct = _env().run()
    incremental = _env()
    while incremental._next_day < incremental.scenario.horizon_days:
        incremental.step_day()
    stepped = incremental.run()

    assert stepped.daily_log == direct.daily_log
    assert asdict(stepped.metrics) == asdict(direct.metrics)


def test_snapshot_restore_reproduces_same_future_without_aliasing() -> None:
    env = _env("s2")
    for _ in range(6):
        env.step_day()
    snapshot = env.get_state()

    first = [env.step_day() for _ in range(4)]
    env.set_state(snapshot)
    second = [env.step_day() for _ in range(4)]

    assert first == second
    assert snapshot.next_day == 6
    assert env._next_day == 10


def test_forks_have_independent_mutable_object_graphs() -> None:
    env = _env("s1")
    for _ in range(3):
        env.step_day()
    snapshot = env.get_state()
    left = env.fork_from_state(snapshot)
    right = env.fork_from_state(snapshot)

    assert left.society is not right.society
    assert left.market is not right.market
    assert left.social is not right.social
    assert left.graph is not right.graph
    assert left.society.all_agents[0] is not right.society.all_agents[0]
    assert left.society.all_agents[0].portfolio is not right.society.all_agents[0].portfolio

    left.society.all_agents[0].portfolio.cash = -123.0
    left.market.assets[left.target_asset].history["price"].append(-456.0)
    restored = env.fork_from_state(snapshot)
    assert right.society.all_agents[0].portfolio.cash != -123.0
    assert restored.society.all_agents[0].portfolio.cash != -123.0
    assert -456.0 not in right.market.assets[right.target_asset].history["price"]
    assert -456.0 not in restored.market.assets[restored.target_asset].history["price"]


def test_snapshot_metadata_must_match_hidden_payload() -> None:
    env = _env("s1")
    snapshot = env.get_state()
    inconsistent = replace(snapshot, next_day=snapshot.next_day + 1)
    with pytest.raises(ValueError, match="hidden day"):
        env.fork_from_state(inconsistent)


def test_32_way_matched_counterfactuals_do_not_mutate_original() -> None:
    env = _env("s3")
    for _ in range(4):
        env.step_day()
    snapshot = env.get_state()
    before = env.clone_state()

    result = CounterfactualEvaluator().evaluate_state(
        simulator=env,
        state=snapshot,
        candidate_actions=DIRECT_ACTIONS,
        seeds=range(32),
        horizon=5,
    )

    assert result.seeds == tuple(range(32))
    assert set(result.actions) == {"WAIT", "THROTTLE", "FREEZE"}
    assert all(len(outcome.failure_outcomes) == 32 for outcome in result.actions.values())
    assert all(
        len(outcome.generic_collapse_outcomes) == 32
        for outcome in result.actions.values()
    )
    assert result.failure_criterion == FailureCriterionIdentity(
        "spoof_liquidity_failure", "v1"
    )
    assert env._next_day == before.next_day == 4

    expected = env.clone()
    expected.set_state(before)
    assert env.step_day() == expected.step_day()


def test_matched_rollout_is_repeatable() -> None:
    env = _env("s1")
    for _ in range(8):
        env.step_day()
    state = env.get_state()
    kwargs = dict(
        simulator=env,
        state=state,
        candidate_actions=DIRECT_ACTIONS,
        seeds=[101, 102, 103],
        horizon=4,
    )
    evaluator = CounterfactualEvaluator()
    assert evaluator.evaluate_state(**kwargs).to_dict() == evaluator.evaluate_state(**kwargs).to_dict()


def test_identical_wait_rollouts_match_every_seed_and_trajectory_entry() -> None:
    env = _env("s2")
    for _ in range(5):
        env.step_day()
    state = env.get_state()
    evaluator = CounterfactualEvaluator()
    wait = RegulatoryAction(RegulatoryActionType.WAIT)

    for seed in range(8):
        first = evaluator._rollout_once(env, state, wait, seed, horizon=5)
        second = evaluator._rollout_once(env, state, wait, seed, horizon=5)
        assert first == second


def test_targeted_actions_have_distinct_identity_and_preserve_action_semantics() -> None:
    env = _env("s1")
    assets = list(env.market.assets)
    actions = [
        RegulatoryAction(RegulatoryActionType.THROTTLE, assets[0]),
        RegulatoryAction(RegulatoryActionType.THROTTLE, assets[1]),
    ]
    result = CounterfactualEvaluator().evaluate_state(
        simulator=env,
        state=env.get_state(),
        candidate_actions=actions,
        seeds=[1],
        horizon=1,
    )
    assert set(result.actions) == {
        f"THROTTLE:{assets[0]}",
        f"THROTTLE:{assets[1]}",
    }
    assert {outcome.action for outcome in result.actions.values()} == set(actions)


def test_regulatory_controls_are_explicit_and_persistent() -> None:
    env = _env("s1")
    executor = RegulatoryActionExecutor()
    target = env.target_asset
    executor.apply(env, RegulatoryAction(RegulatoryActionType.THROTTLE, target), day=0)
    expected = {
        "message_filter": 0.5,
        "trade_throttle": 0.5,
        "spoof_filter": 0.5,
        "wash_filter": 0.5,
    }
    assert env._regulatory_asset_controls[target] == expected
    env.step_day()
    env.step_day()
    assert env._regulatory_asset_controls[target] == expected

    orders, messages = env._apply_public_market_controls(
        [Order("some_agent", target, "buy", 10.0)],
        [Message("some_agent", target, 1.0, 2.0, text="Public message")],
    )
    assert orders[0].quantity == 5.0
    assert messages[0].intensity == 1.0
    assert messages[0].text == "Public message"


def test_targeted_freeze_is_agent_scoped() -> None:
    env = _env("s3")
    internal_target = env.society.attackers[0].agent_id
    target = env.public_entity_id(internal_target)
    RegulatoryActionExecutor().apply(
        env,
        RegulatoryAction(RegulatoryActionType.FREEZE, target),
        day=0,
    )
    assert env._regulatory_agent_controls[internal_target] == {
        "message_filter": 1.0,
        "trade_throttle": 1.0,
        "spoof_filter": 1.0,
        "wash_filter": 1.0,
    }
    other = env.society.retail[0].agent_id
    asset = env.target_asset
    orders, messages = env._apply_public_market_controls(
        [
            Order(internal_target, asset, "buy", 10.0, is_harmful=True),
            Order(other, asset, "buy", 10.0),
        ],
        [
            Message(internal_target, asset, 1.0, 2.0, is_harmful=True),
            Message(other, asset, 1.0, 2.0),
        ],
    )
    assert [order.agent_id for order in orders] == [other]
    assert [message.sender_id for message in messages] == [other]


def test_rollout_stops_immediately_after_system_failure() -> None:
    env = _env("s1")
    evaluator = CounterfactualEvaluator()

    class AlwaysFail:
        identity = FailureCriterionIdentity("test_failure", "v1")

        def evaluate(self, simulator, entry):
            return FailureEvent(True, "test_failure")

    rollout = evaluator._rollout_once(
        env,
        env.get_state(),
        RegulatoryAction(RegulatoryActionType.WAIT),
        continuation_seed=1,
        horizon=10,
        criterion=AlwaysFail(),
    )
    assert rollout.system_failure is True
    assert rollout.failure_type == "test_failure"
    assert rollout.steps == len(rollout.trace) == 1


def test_registered_failure_type_is_scenario_specific_and_serialized() -> None:
    expected = {
        "s1": "pump_dislocation_failure",
        "s2": "finfluencer_dislocation_failure",
        "s3": "spoof_liquidity_failure",
        "s4": "fake_liquidity_failure",
    }
    for scenario, failure_type in expected.items():
        assert failure_criterion_for_scenario(scenario).identity.to_dict() == {
            "id": failure_type,
            "version": "v1",
        }
    with pytest.raises(KeyError, match="no audited"):
        failure_criterion_for_scenario("unregistered")


def test_s3_rollout_records_primary_and_legacy_events_separately() -> None:
    env = WolfSocietySimulator(load_scenario("s3"), n_society=100, alpha=0.50, seed=7)
    result = CounterfactualEvaluator().evaluate_state(
        env,
        env.get_state(),
        [RegulatoryAction(RegulatoryActionType.WAIT)],
        seeds=range(4),
        horizon=5,
    )
    outcome = result.actions["WAIT"]
    payload = outcome.to_dict()
    assert len(outcome.failure_outcomes) == len(outcome.generic_collapse_outcomes) == 4
    assert set(outcome.failure_types) == {"spoof_liquidity_failure"}
    assert payload["failure_probability"] == 1.0
    assert payload["generic_collapse_probability"] == 0.0
    assert "collapse_probability" not in payload


def test_investigate_is_excluded_from_direct_oracle() -> None:
    env = _env()
    with pytest.raises(ValueError, match="value-of-information"):
        CounterfactualEvaluator().evaluate_state(
            simulator=env,
            state=env.get_state(),
            candidate_actions=[RegulatoryAction(RegulatoryActionType.INVESTIGATE)],
            seeds=[1],
            horizon=1,
        )


def test_proportionate_oracle_chooses_lowest_cost_safe_action() -> None:
    result = CounterfactualEvaluation(
        state_day=5,
        horizon=3,
        seeds=(1,),
        failure_criterion=FailureCriterionIdentity("test_failure", "v1"),
        actions={
            "WAIT": ActionOutcome(
                action=RegulatoryAction(RegulatoryActionType.WAIT),
                failure_probability=0.7,
                cost=0.0,
                failure_outcomes=(True,),
                failure_types=("test_failure",),
                generic_collapse_probability=0.0,
                generic_collapse_outcomes=(False,),
                rollout_steps=(1,),
            ),
            "THROTTLE": ActionOutcome(
                action=RegulatoryAction(RegulatoryActionType.THROTTLE),
                failure_probability=0.2,
                cost=2.0,
                failure_outcomes=(False,),
                failure_types=("test_failure",),
                generic_collapse_probability=0.0,
                generic_collapse_outcomes=(False,),
                rollout_steps=(3,),
            ),
            "FREEZE": ActionOutcome(
                action=RegulatoryAction(RegulatoryActionType.FREEZE),
                failure_probability=0.0,
                cost=4.0,
                failure_outcomes=(False,),
                failure_types=("test_failure",),
                generic_collapse_probability=0.0,
                generic_collapse_outcomes=(False,),
                rollout_steps=(3,),
            ),
        },
    )
    decision = ProportionateOracle(safety_threshold=0.2).choose(result)
    assert decision.primary_action == RegulatoryAction(RegulatoryActionType.THROTTLE)
    assert decision.valid_actions == (RegulatoryAction(RegulatoryActionType.THROTTLE),)
    assert (decision.risk, decision.cost) == (0.2, 2.0)
