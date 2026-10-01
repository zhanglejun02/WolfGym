"""Matched-seed counterfactual rollouts from an exact hidden state."""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.env.state import SimulatorState
from wolfgym.core.metrics.collapse import collapse_triggered
from wolfgym.counterfactual.failure import (
    FailureCriterionIdentity,
    SystemFailureCriterion,
    failure_criterion_for_scenario,
)
from wolfgym.regulation.actions import (
    DEFAULT_ACTION_COSTS,
    RegulatoryAction,
    RegulatoryActionType,
)
from wolfgym.regulation.executor import RegulatoryActionExecutor


@dataclass(frozen=True)
class ActionOutcome:
    action: RegulatoryAction
    failure_probability: float
    cost: float
    failure_outcomes: tuple[bool, ...]
    failure_types: tuple[str, ...]
    generic_collapse_probability: float
    generic_collapse_outcomes: tuple[bool, ...]
    rollout_steps: tuple[int, ...]

    @property
    def collapse_probability(self) -> float:
        """Deprecated compatibility alias; new code must use failure_probability."""
        return self.failure_probability

    @property
    def collapse_outcomes(self) -> tuple[bool, ...]:
        """Deprecated compatibility alias; new code must use failure_outcomes."""
        return self.failure_outcomes

    def to_dict(self) -> dict:
        return {
            "action": self.action.to_dict(),
            "failure_probability": self.failure_probability,
            "cost": self.cost,
            "failure_outcomes": list(self.failure_outcomes),
            "failure_types": list(self.failure_types),
            "generic_collapse_probability": self.generic_collapse_probability,
            "generic_collapse_outcomes": list(self.generic_collapse_outcomes),
            "rollout_steps": list(self.rollout_steps),
        }


@dataclass(frozen=True)
class CounterfactualEvaluation:
    state_day: int
    horizon: int
    seeds: tuple[int, ...]
    failure_criterion: FailureCriterionIdentity
    actions: Mapping[str, ActionOutcome]

    @property
    def noop_failure_probability(self) -> float | None:
        for outcome in self.actions.values():
            if outcome.action.action_type is RegulatoryActionType.WAIT:
                return outcome.failure_probability
        return None

    @property
    def noop_collapse_probability(self) -> float | None:
        """Deprecated compatibility alias; new code must use noop_failure_probability."""
        return self.noop_failure_probability

    def to_dict(self) -> dict:
        return {
            "state_day": self.state_day,
            "horizon": self.horizon,
            "seeds": list(self.seeds),
            "failure_criterion": self.failure_criterion.to_dict(),
            "noop_failure_probability": self.noop_failure_probability,
            "actions": {name: value.to_dict() for name, value in self.actions.items()},
        }


def merge_evaluations(
    parts: Sequence[CounterfactualEvaluation],
) -> CounterfactualEvaluation:
    """Merge disjoint matched-seed batches without rerunning earlier rollouts."""
    if not parts:
        raise ValueError("cannot merge an empty evaluation sequence")
    first = parts[0]
    labels = tuple(first.actions)
    if any(tuple(part.actions) != labels for part in parts):
        raise ValueError("evaluation batches have different candidate interventions")
    if any(part.failure_criterion != first.failure_criterion for part in parts):
        raise ValueError("evaluation batches have different failure criteria")
    seeds = sum((part.seeds for part in parts), ())
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation batches contain overlapping seeds")
    outcomes: dict[str, ActionOutcome] = {}
    for label in labels:
        values = [part.actions[label] for part in parts]
        failures = sum((value.failure_outcomes for value in values), ())
        failure_types = sum((value.failure_types for value in values), ())
        generic = sum((value.generic_collapse_outcomes for value in values), ())
        steps = sum((value.rollout_steps for value in values), ())
        outcomes[label] = ActionOutcome(
            action=values[0].action,
            failure_probability=sum(failures) / len(failures),
            cost=values[0].cost,
            failure_outcomes=failures,
            failure_types=failure_types,
            generic_collapse_probability=sum(generic) / len(generic),
            generic_collapse_outcomes=generic,
            rollout_steps=steps,
        )
    return CounterfactualEvaluation(
        state_day=first.state_day,
        horizon=first.horizon,
        seeds=seeds,
        failure_criterion=first.failure_criterion,
        actions=outcomes,
    )


@dataclass(frozen=True)
class RolloutResult:
    system_failure: bool
    failure_type: str
    generic_collapse: bool
    steps: int
    trace: tuple[dict, ...]


class CounterfactualEvaluator:
    """Evaluate direct interventions using common continuation seeds."""

    def __init__(
        self,
        executor: RegulatoryActionExecutor | None = None,
        costs: Mapping[RegulatoryActionType, float] | None = None,
    ) -> None:
        self.executor = executor or RegulatoryActionExecutor()
        self.costs = dict(DEFAULT_ACTION_COSTS if costs is None else costs)
        missing_costs = set(RegulatoryActionType) - set(self.costs)
        if missing_costs:
            names = ", ".join(sorted(action.name for action in missing_costs))
            raise ValueError(f"missing regulatory costs for: {names}")
        if any(
            not math.isfinite(float(cost)) or float(cost) < 0.0
            for cost in self.costs.values()
        ):
            raise ValueError("regulatory costs must be finite and non-negative")

    @staticmethod
    def failure_identity_for_scenario(scenario_id: str) -> FailureCriterionIdentity:
        return failure_criterion_for_scenario(scenario_id).identity

    def evaluate_state(
        self,
        simulator: WolfSocietySimulator,
        state: SimulatorState,
        candidate_actions: Sequence[RegulatoryAction],
        seeds: Iterable[int],
        horizon: int,
    ) -> CounterfactualEvaluation:
        if isinstance(horizon, bool) or int(horizon) != horizon or horizon <= 0:
            raise ValueError("horizon must be positive")
        requested_horizon = int(horizon)
        try:
            matched_seeds = tuple(operator.index(seed) for seed in seeds)
        except TypeError as exc:
            raise ValueError("continuation seeds must be integers") from exc
        if not matched_seeds:
            raise ValueError("at least one continuation seed is required")
        if any(seed < 0 for seed in matched_seeds):
            raise ValueError("continuation seeds must be non-negative")
        if len(set(matched_seeds)) != len(matched_seeds):
            raise ValueError("continuation seeds must be unique")
        if not candidate_actions:
            raise ValueError("at least one candidate action is required")
        interactive_only = {
            RegulatoryActionType.INVESTIGATE,
            RegulatoryActionType.RELEASE,
        }
        if any(a.action_type in interactive_only for a in candidate_actions):
            raise ValueError(
                "INVESTIGATE is a value-of-information action and RELEASE is a "
                "control-lifecycle action; both are interactive and excluded "
                "from the one-step direct-intervention oracle"
            )
        action_keys = [action.key for action in candidate_actions]
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("candidate actions must be unique by type and target_id")

        contract_probe = simulator.fork_from_state(state)
        contract_probe.assert_counterfactual_rng_contract()

        remaining = state.horizon_days - state.next_day
        effective_horizon = min(requested_horizon, remaining)
        if effective_horizon <= 0:
            raise ValueError("snapshot is already at the simulation horizon")

        criterion = failure_criterion_for_scenario(state.scenario_id)
        outcomes: dict[str, ActionOutcome] = {}
        for action in candidate_actions:
            failed_trials: list[bool] = []
            failure_types: list[str] = []
            generic_trials: list[bool] = []
            rollout_steps: list[int] = []
            for continuation_seed in matched_seeds:
                rollout = self._rollout_once(
                    simulator=simulator,
                    state=state,
                    action=action,
                    continuation_seed=continuation_seed,
                    horizon=effective_horizon,
                    criterion=criterion,
                )
                failed_trials.append(rollout.system_failure)
                failure_types.append(rollout.failure_type)
                generic_trials.append(rollout.generic_collapse)
                rollout_steps.append(rollout.steps)
            probability = sum(failed_trials) / len(failed_trials)
            outcomes[action.label] = ActionOutcome(
                action=action,
                failure_probability=float(probability),
                cost=float(self.costs[action.action_type]),
                failure_outcomes=tuple(failed_trials),
                failure_types=tuple(failure_types),
                generic_collapse_probability=(
                    sum(generic_trials) / len(generic_trials)
                ),
                generic_collapse_outcomes=tuple(generic_trials),
                rollout_steps=tuple(rollout_steps),
            )

        return CounterfactualEvaluation(
            state_day=state.next_day,
            horizon=effective_horizon,
            seeds=matched_seeds,
            failure_criterion=criterion.identity,
            actions=outcomes,
        )

    def _rollout_once(
        self,
        simulator: WolfSocietySimulator,
        state: SimulatorState,
        action: RegulatoryAction,
        continuation_seed: int,
        horizon: int,
        criterion: SystemFailureCriterion | None = None,
    ) -> RolloutResult:
        """Run one independent branch and return its exact daily trace.

        The trace is primarily a reproducibility diagnostic. Production
        evaluation discards it after extracting both event diagnostics and
        executed steps. Generic collapse is observed over the executed prefix;
        the scenario-aligned system event remains the stopping condition.
        """
        branch = simulator.fork_from_state(state)
        branch.set_continuation_seed(continuation_seed)
        active_criterion = criterion or failure_criterion_for_scenario(state.scenario_id)
        trace: list[dict] = []
        generic_collapsed = False

        def apply_once(env: WolfSocietySimulator, day: int) -> None:
            self.executor.apply(env, action, day)

        for offset in range(horizon):
            entry = branch.step_day(pre_step=apply_once if offset == 0 else None)
            trace.append(entry)
            generic_collapsed = generic_collapsed or collapse_triggered(entry["components"])
            event = active_criterion.evaluate(branch, entry)
            if event.triggered:
                return RolloutResult(
                    system_failure=True,
                    failure_type=event.failure_type,
                    generic_collapse=generic_collapsed,
                    steps=offset + 1,
                    trace=tuple(trace),
                )
        return RolloutResult(
            system_failure=False,
            failure_type=active_criterion.identity.criterion_id,
            generic_collapse=generic_collapsed,
            steps=horizon,
            trace=tuple(trace),
        )
