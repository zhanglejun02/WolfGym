"""Gym-compatible partially observable regulatory environment."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.env.state import SimulatorState
from wolfgym.counterfactual.failure import (
    CandidateSustainedLossCriterion, SystemFailureCriterion, failure_criterion_for_scenario,
)
from wolfgym.observations import (
    ActiveControlObservation,
    InformationLevel,
    InteractionHistoryObservation,
    ObservationBuilder,
    ObservationConfig,
    RegulatoryObservation,
)
from wolfgym.regulation import (
    DEFAULT_ACTION_COSTS,
    RegulatoryAction,
    RegulatoryActionExecutor,
    RegulatoryActionType,
)
from wolfgym.env.rewards import REWARD_DIRECTIONS, RewardComponents
from wolfgym.env.trace import EpisodeTrace, EpisodeTransition


@dataclass(frozen=True)
class WolfGymEnvConfig:
    initial_budget: float = 12.0
    max_interactions: int = 64
    observation: ObservationConfig = ObservationConfig()
    intervention_duration_steps: int | None = None
    allow_candidate_criterion: bool = False
    terminate_on_failure: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_budget) or self.initial_budget < 0.0:
            raise ValueError("initial_budget must be finite and non-negative")
        if self.max_interactions <= 0:
            raise ValueError("max_interactions must be positive")
        if self.intervention_duration_steps not in {None, 1}:
            raise ValueError("intervention_duration_steps must be 1 or None")


class WolfGymEnv:
    """Wrap one exact simulator snapshot as a POMDP-style control episode.

    ``INVESTIGATE`` consumes budget but not simulator time. All other actions
    advance one day. The scalar reward is intentionally fixed at zero; raw,
    unweighted outcome axes are returned in ``info['reward_components']``.
    """

    def __init__(
        self,
        simulator: WolfSocietySimulator,
        initial_state: SimulatorState,
        *,
        config: WolfGymEnvConfig | None = None,
        executor: RegulatoryActionExecutor | None = None,
        costs: Mapping[RegulatoryActionType, float] | None = None,
        episode_id: str = "interactive_episode",
        trajectory_id: str = "interactive_trajectory",
        failure_criterion: SystemFailureCriterion | None = None,
    ) -> None:
        self._template = simulator
        self._initial_state = initial_state.clone()
        self.config = config or WolfGymEnvConfig()
        # Resolve before reset/step can mutate any simulator state.
        self.failure_criterion = failure_criterion or failure_criterion_for_scenario(initial_state.scenario_id)
        if isinstance(self.failure_criterion, CandidateSustainedLossCriterion):
            if not self.config.allow_candidate_criterion or initial_state.scenario_id != "stock_market_v2":
                raise ValueError("candidate criterion requires explicit unified-market pilot opt-in")
        elif self.config.allow_candidate_criterion or not self.config.terminate_on_failure:
            raise ValueError("nonterminating diagnostics require an explicit candidate criterion")
        self.executor = executor or RegulatoryActionExecutor()
        self.costs = dict(DEFAULT_ACTION_COSTS if costs is None else costs)
        missing = set(RegulatoryActionType) - set(self.costs)
        if missing:
            raise ValueError(f"missing regulatory action costs: {sorted(a.name for a in missing)}")
        if any(not math.isfinite(float(value)) or value < 0 for value in self.costs.values()):
            raise ValueError("action costs must be finite and non-negative")
        self.episode_id = str(episode_id)
        self.trajectory_id = str(trajectory_id)
        self.simulator: WolfSocietySimulator
        self.remaining_budget: float
        self._history: list[InteractionHistoryObservation]
        self._interaction_index: int
        self._terminated: bool
        self._truncated: bool
        self.trace: EpisodeTrace
        self.reset()

    @property
    def horizon_days(self) -> int:
        return self._initial_state.horizon_days

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[RegulatoryObservation, dict[str, Any]]:
        del options
        self.simulator = self._template.fork_from_state(self._initial_state)
        if seed is not None:
            self.simulator.set_continuation_seed(int(seed))
        if isinstance(self.failure_criterion, CandidateSustainedLossCriterion):
            self.failure_criterion.evaluate(self.simulator, {})
        self.remaining_budget = float(self.config.initial_budget)
        self._history = []
        self._interaction_index = 0
        self._terminated = False
        self._truncated = False
        observation = self._observation()
        self.trace = EpisodeTrace(
            episode_id=self.episode_id,
            trajectory_id=self.trajectory_id,
            initial_observation=observation.to_dict(),
        )
        return observation, self._base_info()

    def step(
        self,
        action: RegulatoryAction,
    ) -> tuple[RegulatoryObservation, float, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("episode is finished; call reset()")
        if not isinstance(action, RegulatoryAction):
            raise TypeError("action must be a RegulatoryAction")
        cost = float(self.costs[action.action_type])
        if cost > self.remaining_budget + 1e-12:
            raise ValueError("action exceeds remaining regulatory budget")
        if action.action_type is not RegulatoryActionType.WAIT:
            if self.simulator.scenario.id == "stock_market_v2" and action.target_id is None:
                raise ValueError("unified-market actions require an explicit public target")
            if self.simulator.scenario.id == "stock_market_v2" and action.target_id is not None:
                public_ids = set(self.simulator._public_asset_ids.values()) | set(self.simulator._public_entity_ids.values())
                if action.target_id not in public_ids:
                    raise ValueError("unified-market actions require a public target identifier")
            if action.target_id is not None:
                scope, internal_id = self.executor._resolve_target(self.simulator, action.target_id)
                if action.action_type is RegulatoryActionType.RELEASE:
                    if internal_id not in self.simulator.persistent_regulatory_controls()[scope]:
                        raise ValueError("no active regulatory control for target")

        previous = self._observation()
        action_day = self.simulator.get_state().next_day
        investigation = None
        event = None
        entry = None
        advanced_day = action.action_type is not RegulatoryActionType.INVESTIGATE
        if action.action_type is RegulatoryActionType.INVESTIGATE:
            if action.target_id is None:
                raise ValueError("INVESTIGATE requires a target")
            investigation = ObservationBuilder(self.config.observation).build(
                self.simulator,
                self.simulator.get_state(),
                investigation_target=action.target_id,
                information_level=InformationLevel.INVESTIGATION,
            ).investigation
        else:
            entry = self.simulator.step_day(
                pre_step=lambda env, day: self.executor.apply(env, action, day)
            )
            event = self.failure_criterion.evaluate(self.simulator, entry)
            self._terminated = event.triggered and self.config.terminate_on_failure
            if (
                self.config.intervention_duration_steps == 1
                and action.action_type in {
                    RegulatoryActionType.THROTTLE,
                    RegulatoryActionType.FREEZE,
                }
            ):
                self.executor.release(self.simulator, action.target_id)

        self.remaining_budget -= cost
        self._interaction_index += 1
        self._history.append(InteractionHistoryObservation(
            interaction_index=self._interaction_index,
            day=action_day,
            action_type=action.action_type.value,
            target_id=action.target_id,
            advanced_day=advanced_day,
            cost=cost,
            investigation_result=investigation,
        ))
        at_horizon = self.simulator.get_state().next_day >= self._initial_state.horizon_days
        interaction_limit = self._interaction_index >= self.config.max_interactions
        self._truncated = not self._terminated and (at_horizon or interaction_limit)

        observation = self._observation(
            investigation_target=action.target_id if investigation is not None else None,
        )
        intervention_cost = cost if action.action_type in {
            RegulatoryActionType.THROTTLE,
            RegulatoryActionType.FREEZE,
        } else 0.0
        investigation_cost = cost if action.action_type is RegulatoryActionType.INVESTIGATE else 0.0
        components = RewardComponents(
            system_failure=1.0 if event is not None and event.triggered else 0.0,
            intervention_cost=intervention_cost,
            investigation_cost=investigation_cost,
            # A paired WAIT reference is needed; action fees are not welfare.
            clean_utility_loss=None,
            risk_reduction=None,
        )
        info = {
            **self._base_info(),
            "reward_components": components.to_dict(),
            "reward_directions": dict(REWARD_DIRECTIONS),
            "failure_event": None if event is None else {
                "triggered": event.triggered,
                "type": event.failure_type,
            },
            "advanced_day": advanced_day,
            "intervention_duration_steps": self.config.intervention_duration_steps,
            "intervention_cost_basis": "per_issued_action",
            "financial_outcomes": None if entry is None else entry.get("retail_outcomes"),
            "failure_criterion": self.failure_criterion.identity.to_dict(),
            "candidate_only": isinstance(self.failure_criterion, CandidateSustainedLossCriterion),
            "terminate_on_failure": self.config.terminate_on_failure,
        }
        self.trace.transitions.append(EpisodeTransition(
            interaction_index=self._interaction_index,
            observation=previous.to_dict(),
            action=action.to_dict(),
            next_observation=observation.to_dict(),
            terminated=self._terminated,
            truncated=self._truncated,
            info=info,
        ))
        return observation, 0.0, self._terminated, self._truncated, info

    def _observation(self, investigation_target: str | None = None) -> RegulatoryObservation:
        level = (
            InformationLevel.INVESTIGATION
            if investigation_target is not None
            else self.config.observation.information_level
        )
        return ObservationBuilder(self.config.observation).build(
            self.simulator,
            self.simulator.get_state(),
            investigation_target=investigation_target,
            information_level=level,
            remaining_budget=self.remaining_budget,
            active_controls=self._active_controls(),
            history=tuple(self._history),
        )

    def _active_controls(self) -> tuple[ActiveControlObservation, ...]:
        controls = self.simulator.persistent_regulatory_controls()
        result: list[ActiveControlObservation] = []
        for scope, targets in controls.items():
            for internal_id, values in targets.items():
                public_id = (
                    self.simulator.public_asset_id(internal_id)
                    if scope == "asset"
                    else self.simulator.public_entity_id(internal_id)
                )
                maximum = max(values.values(), default=0.0)
                kind = "freeze" if maximum >= 1.0 - 1e-12 else "throttle"
                result.append(ActiveControlObservation(kind, public_id))
        return tuple(sorted(result, key=lambda item: (item.target_id, item.action_type)))

    def _base_info(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "day": self.simulator._next_day,
            "remaining_budget": self.remaining_budget,
            "interaction_index": self._interaction_index,
        }
