from __future__ import annotations

from wolfgym.counterfactual.evaluator import ActionOutcome, CounterfactualEvaluation
from wolfgym.counterfactual.failure import FailureCriterionIdentity
from wolfgym.counterfactual.oracle import ProportionateOracle
from wolfgym.regulation.actions import RegulatoryAction, RegulatoryActionType


WAIT = RegulatoryAction(RegulatoryActionType.WAIT)
THROTTLE = RegulatoryAction(RegulatoryActionType.THROTTLE)
FREEZE = RegulatoryAction(RegulatoryActionType.FREEZE)


def _outcome(action: RegulatoryAction, risk: float, cost: float) -> ActionOutcome:
    return ActionOutcome(
        action=action,
        failure_probability=risk,
        cost=cost,
        failure_outcomes=(),
        failure_types=(),
        generic_collapse_probability=0.0,
        generic_collapse_outcomes=(),
        rollout_steps=(),
    )


def _result(*outcomes: ActionOutcome) -> CounterfactualEvaluation:
    return CounterfactualEvaluation(
        state_day=5,
        horizon=10,
        seeds=(),
        failure_criterion=FailureCriterionIdentity("test_failure", "v1"),
        actions={outcome.action.label: outcome for outcome in outcomes},
    )


def test_wait_is_primary_when_already_safely_below_threshold() -> None:
    decision = ProportionateOracle().choose(_result(
        _outcome(WAIT, 0.10, 0.0),
        _outcome(THROTTLE, 0.03, 2.0),
        _outcome(FREEZE, 0.00, 4.0),
    ))
    assert decision.primary_action == WAIT
    assert decision.valid_actions == (WAIT,)
    assert decision.ambiguous is False


def test_same_cost_safe_targeted_actions_are_all_valid_and_order_independent() -> None:
    throttle_a = RegulatoryAction(RegulatoryActionType.THROTTLE, "cluster_a")
    throttle_b = RegulatoryAction(RegulatoryActionType.THROTTLE, "cluster_b")
    candidates = (
        _outcome(WAIT, 0.70, 0.0),
        _outcome(throttle_a, 0.15625, 2.0),
        _outcome(throttle_b, 0.18750, 2.0),
        _outcome(FREEZE, 0.01, 4.0),
    )
    oracle = ProportionateOracle()
    forward = oracle.choose(_result(*candidates))
    reverse = oracle.choose(_result(*reversed(candidates)))

    assert forward.primary_action == reverse.primary_action == throttle_a
    assert forward.valid_actions == reverse.valid_actions == (throttle_a, throttle_b)
    assert forward.accepts(throttle_a)
    assert forward.accepts(throttle_b)


def test_no_safe_action_uses_risk_margin_then_minimum_cost() -> None:
    decision = ProportionateOracle(risk_margin=0.05).choose(_result(
        _outcome(WAIT, 0.80, 0.0),
        _outcome(THROTTLE, 0.43, 2.0),
        _outcome(FREEZE, 0.40, 4.0),
    ))
    assert decision.primary_action == THROTTLE
    assert decision.valid_actions == (THROTTLE,)
    assert decision.risk == 0.43


def test_no_safe_near_equivalent_same_cost_actions_are_all_valid() -> None:
    throttle_a = RegulatoryAction(RegulatoryActionType.THROTTLE, "agent_a")
    throttle_b = RegulatoryAction(RegulatoryActionType.THROTTLE, "agent_b")
    decision = ProportionateOracle(risk_margin=0.05).choose(_result(
        _outcome(WAIT, 0.80, 0.0),
        _outcome(throttle_a, 0.40, 2.0),
        _outcome(throttle_b, 0.43, 2.0),
        _outcome(FREEZE, 0.60, 4.0),
    ))
    assert decision.primary_action == throttle_a
    assert decision.valid_actions == (throttle_a, throttle_b)


def test_near_threshold_action_is_ambiguous_only_if_crossing_changes_decision() -> None:
    consequential = ProportionateOracle(ambiguity_margin=0.05).choose(_result(
        _outcome(WAIT, 0.19, 0.0),
        _outcome(THROTTLE, 0.03, 2.0),
        _outcome(FREEZE, 0.00, 4.0),
    ))
    assert consequential.primary_action == WAIT
    assert consequential.ambiguous is True

    irrelevant = ProportionateOracle(ambiguity_margin=0.05).choose(_result(
        _outcome(WAIT, 0.80, 0.0),
        _outcome(THROTTLE, 0.10, 2.0),
        _outcome(FREEZE, 0.19, 4.0),
    ))
    assert irrelevant.primary_action == THROTTLE
    assert irrelevant.ambiguous is False


def test_oracle_serializes_structured_actions_without_string_parsing() -> None:
    targeted = RegulatoryAction(RegulatoryActionType.THROTTLE, "asset_1")
    decision = ProportionateOracle().choose(_result(
        _outcome(WAIT, 0.70, 0.0),
        _outcome(targeted, 0.10, 2.0),
        _outcome(FREEZE, 0.00, 4.0),
    ))
    assert decision.to_dict() == {
        "primary_action": {"type": "THROTTLE", "target_id": "asset_1"},
        "valid_actions": [{"type": "THROTTLE", "target_id": "asset_1"}],
        "oracle_risk": 0.10,
        "oracle_cost": 2.0,
        "ambiguous": False,
    }
