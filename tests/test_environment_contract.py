"""Regression checks for shared environment action timing and permissions."""

import pytest

from wolfgym.core import WolfSocietySimulator
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.env import WolfGymEnv, WolfGymEnvConfig
from wolfgym.evaluation import InteractiveEvaluator
from wolfgym.observations import ObservationConfig
from wolfgym.regulation import RegulatoryAction, RegulatoryActionType


def test_interactive_action_timing_budget_controls_release_and_trace(monkeypatch) -> None:
    from types import SimpleNamespace
    from wolfgym.counterfactual.failure import FailureEvent, FailureCriterionIdentity
    # This test exercises action timing, not whether one stochastic market
    # happens to stay below an old collapse threshold. Failure has its own tests.
    monkeypatch.setattr("wolfgym.env.core.failure_criterion_for_scenario",
                        lambda _: SimpleNamespace(identity=FailureCriterionIdentity("test", "v1"), evaluate=lambda *args: FailureEvent(False, "test")))
    simulator = WolfSocietySimulator(load_scenario("s0"), 30, 0.0, seed=9)
    for _ in range(3):
        simulator.step_day()
    initial = simulator.get_state()
    public_asset = simulator.public_asset_id(simulator.target_asset)
    env = WolfGymEnv(
        simulator,
        initial,
        config=WolfGymEnvConfig(
            initial_budget=8.0,
            max_interactions=10,
            observation=ObservationConfig(top_k_entities=3),
        ),
    )
    observation, _ = env.reset(seed=123)
    initial_day = observation.day

    investigated, reward, terminated, truncated, info = env.step(
        RegulatoryAction(RegulatoryActionType.INVESTIGATE, public_asset)
    )
    assert investigated.day == initial_day
    assert investigated.investigation is not None
    assert investigated.remaining_budget == 7.0
    assert reward == 0.0 and not terminated and not truncated
    assert info["advanced_day"] is False

    throttled, _, terminated, truncated, info = env.step(
        RegulatoryAction(RegulatoryActionType.THROTTLE, public_asset)
    )
    assert not terminated and not truncated
    assert throttled.day == initial_day + 1
    assert throttled.active_controls[0].action_type == "throttle"
    assert info["reward_components"]["intervention_cost"] == 2.0

    released, _, terminated, truncated, _ = env.step(
        RegulatoryAction(RegulatoryActionType.RELEASE, public_asset)
    )
    assert not terminated and not truncated
    assert released.day == initial_day + 2
    assert released.active_controls == ()
    assert len(env.trace.transitions) == 3
    evaluation = InteractiveEvaluator().evaluate(env.trace)
    assert evaluation.interactions == 3
    assert evaluation.elapsed_days == 2
    assert evaluation.investigation_cost == 1.0
    assert evaluation.intervention_cost == 2.0


def test_release_is_interactive_only() -> None:
    simulator = WolfSocietySimulator(load_scenario("s0"), 20, 0.0, seed=1)
    with pytest.raises(ValueError, match="interactive"):
        from wolfgym.counterfactual import CounterfactualEvaluator
        CounterfactualEvaluator().evaluate_state(
            simulator,
            simulator.get_state(),
            [RegulatoryAction(RegulatoryActionType.RELEASE, simulator.public_asset_id(
                simulator.target_asset
            ))],
            [1],
            1,
        )
