"""Cost-proportionate intervention oracle with set-valued labels."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from wolfgym.counterfactual.evaluator import ActionOutcome, CounterfactualEvaluation
from wolfgym.regulation.actions import RegulatoryAction, RegulatoryActionType


ORACLE_VERSION = "proportionate-oracle-v2"


_ACTION_ORDER = {
    action_type: index for index, action_type in enumerate(RegulatoryActionType)
}


def _action_sort_key(action: RegulatoryAction) -> tuple[int, str]:
    return _ACTION_ORDER[action.action_type], action.target_id or ""


@dataclass(frozen=True)
class OracleDecision:
    primary_action: RegulatoryAction | None
    valid_actions: tuple[RegulatoryAction, ...]
    risk: float | None
    cost: float | None
    ambiguous: bool

    def accepts(self, action: RegulatoryAction) -> bool:
        """Return whether a model action is a valid proportionate decision."""
        return action in self.valid_actions

    def to_dict(self) -> dict:
        return {
            "primary_action": None if self.primary_action is None else self.primary_action.to_dict(),
            "valid_actions": [action.to_dict() for action in self.valid_actions],
            "oracle_risk": self.risk,
            "oracle_cost": self.cost,
            "ambiguous": self.ambiguous,
        }


@dataclass(frozen=True)
class _Selection:
    primary: ActionOutcome | None
    valid: tuple[ActionOutcome, ...]


class ProportionateOracle:
    """Legacy v1/v2 diagnostic rule; use EffectiveInterventionOracle for IS.

    If one or more actions meet ``safety_threshold``, every safe action at the
    minimum safe cost is valid. If none is safe, actions within ``risk_margin``
    of the empirical minimum risk are treated as Monte Carlo-equivalent, then
    the minimum-cost members of that set are valid.
    """

    def __init__(
        self,
        safety_threshold: float = 0.20,
        ambiguity_margin: float = 0.05,
        risk_margin: float = 0.05,
    ) -> None:
        if not math.isfinite(float(safety_threshold)) or not 0.0 <= safety_threshold <= 1.0:
            raise ValueError("safety_threshold must be finite and in [0, 1]")
        if not math.isfinite(float(ambiguity_margin)) or ambiguity_margin < 0.0:
            raise ValueError("ambiguity_margin must be finite and non-negative")
        if not math.isfinite(float(risk_margin)) or risk_margin < 0.0:
            raise ValueError("risk_margin must be finite and non-negative")
        self.safety_threshold = float(safety_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.risk_margin = float(risk_margin)

    def choose(self, result: CounterfactualEvaluation) -> OracleDecision:
        candidates = tuple(result.actions.values())
        self._validate_candidates(candidates)
        selection = self._select(candidates)
        primary = selection.primary
        return OracleDecision(
            primary_action=primary.action,
            valid_actions=tuple(outcome.action for outcome in selection.valid),
            risk=primary.failure_probability,
            cost=primary.cost,
            ambiguous=self._is_decision_ambiguous(candidates),
        )

    def _select(
        self,
        candidates: Sequence[ActionOutcome],
        risk_overrides: Mapping[RegulatoryAction, float] | None = None,
    ) -> _Selection:
        overrides = risk_overrides or {}

        def risk(outcome: ActionOutcome) -> float:
            return float(overrides.get(outcome.action, outcome.failure_probability))

        safe = [outcome for outcome in candidates if risk(outcome) <= self.safety_threshold]
        if safe:
            min_cost = min(outcome.cost for outcome in safe)
            valid = [
                outcome for outcome in safe
                if math.isclose(outcome.cost, min_cost, rel_tol=0.0, abs_tol=1e-12)
            ]
        else:
            min_risk = min(risk(outcome) for outcome in candidates)
            near_minimum = [
                outcome for outcome in candidates
                if risk(outcome) <= min_risk + self.risk_margin + 1e-12
            ]
            min_cost = min(outcome.cost for outcome in near_minimum)
            valid = [
                outcome for outcome in near_minimum
                if math.isclose(outcome.cost, min_cost, rel_tol=0.0, abs_tol=1e-12)
            ]

        ordered = tuple(sorted(valid, key=lambda outcome: _action_sort_key(outcome.action)))
        primary = min(
            ordered,
            key=lambda outcome: (risk(outcome), _action_sort_key(outcome.action)),
        )
        return _Selection(primary=primary, valid=ordered)

    def _is_decision_ambiguous(self, candidates: Sequence[ActionOutcome]) -> bool:
        """Flag only threshold uncertainty that changes the valid-action set."""
        if self.ambiguity_margin == 0.0:
            return False
        for candidate in candidates:
            risk = candidate.failure_probability
            if abs(risk - self.safety_threshold) >= self.ambiguity_margin:
                continue
            safe_side = self._select(
                candidates,
                risk_overrides={candidate.action: self.safety_threshold},
            )
            unsafe_side = self._select(
                candidates,
                risk_overrides={
                    candidate.action: math.nextafter(self.safety_threshold, math.inf),
                },
            )
            safe_valid = {outcome.action for outcome in safe_side.valid}
            unsafe_valid = {outcome.action for outcome in unsafe_side.valid}
            if safe_valid != unsafe_valid:
                return True
        return False

    @staticmethod
    def _validate_candidates(candidates: Sequence[ActionOutcome]) -> None:
        if not candidates:
            raise ValueError("counterfactual result contains no actions")
        actions = [outcome.action for outcome in candidates]
        if len(set(actions)) != len(actions):
            raise ValueError("counterfactual result contains duplicate actions")
        for outcome in candidates:
            risk = float(outcome.failure_probability)
            cost = float(outcome.cost)
            if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
                raise ValueError("failure probabilities must be finite and in [0, 1]")
            if not math.isfinite(cost) or cost < 0.0:
                raise ValueError("action costs must be finite and non-negative")


EFFECTIVE_ORACLE_VERSION = "effective-intervention-oracle-v3"


class EffectiveInterventionOracle(ProportionateOracle):
    """IS labels the cheapest safe intervention that is also affordable.

    Infeasible states retain their RF/AE outcomes, but have no primary or valid
    IS action. Threshold perturbations also test changes in feasibility. A
    missing budget keeps legacy behavior in which every candidate is affordable.
    """

    def _select(self, candidates, risk_overrides=None) -> _Selection:
        overrides = risk_overrides or {}
        safe = tuple(
            outcome for outcome in candidates
            if overrides.get(outcome.action, outcome.failure_probability) <= self.safety_threshold
        )
        if not safe:
            return _Selection(None, ())
        return super()._select(safe, risk_overrides)

    @staticmethod
    def _affordable(
        candidates: Sequence[ActionOutcome],
        remaining_budget: float | None,
    ) -> tuple[ActionOutcome, ...]:
        if remaining_budget is None:
            return tuple(candidates)
        budget = float(remaining_budget)
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError("remaining_budget must be finite and non-negative")
        return tuple(item for item in candidates if item.cost <= budget + 1e-12)

    def choose(
        self,
        result: CounterfactualEvaluation,
        remaining_budget: float | None = None,
    ) -> OracleDecision:
        candidates = tuple(result.actions.values())
        self._validate_candidates(candidates)
        affordable = self._affordable(candidates, remaining_budget)
        selection = self._select(affordable)
        primary = selection.primary
        return OracleDecision(
            primary_action=None if primary is None else primary.action,
            valid_actions=tuple(item.action for item in selection.valid),
            risk=None if primary is None else primary.failure_probability,
            cost=None if primary is None else primary.cost,
            ambiguous=self._is_decision_ambiguous(affordable),
        )

    def label_outcomes(
        self,
        outcomes: Mapping[str, Mapping],
        remaining_budget: float | None = None,
    ) -> dict:
        """Label serialized rollout outcomes without trusting a stored oracle."""
        candidates = []
        for item in outcomes.values():
            action = item["action"]
            candidates.append(ActionOutcome(
                RegulatoryAction(RegulatoryActionType[str(action["type"]).upper()],
                                 action.get("target_id")),
                float(item["failure_probability"]), float(item["cost"]),
                (), (), 0.0, (), (),
            ))
        self._validate_candidates(candidates)
        affordable = self._affordable(candidates, remaining_budget)
        selection = self._select(affordable)
        primary = selection.primary
        return {
            "oracle_version": EFFECTIVE_ORACLE_VERSION,
            "safety_threshold": self.safety_threshold,
            "feasible": primary is not None,
            "primary_action": None if primary is None else primary.action.to_dict(),
            "valid_actions": [item.action.to_dict() for item in selection.valid],
            "oracle_risk": None if primary is None else primary.failure_probability,
            "oracle_cost": None if primary is None else primary.cost,
            "ambiguous": self._is_decision_ambiguous(affordable),
            "remaining_budget": remaining_budget,
        }
