"""Vector-valued evaluation for closed-loop regulatory traces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from wolfgym.env.trace import EpisodeTrace


@dataclass(frozen=True)
class InteractiveEvaluation:
    system_failure: float
    intervention_cost: float
    investigation_cost: float
    clean_utility_loss: float | None
    oracle_regret: float | None
    interactions: int
    elapsed_days: int

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


class InteractiveEvaluator:
    """Aggregate raw axes without imposing a benchmark-wide scalar reward."""

    def evaluate(
        self,
        trace: EpisodeTrace,
        oracle_cost_by_interaction: Mapping[int, float] | None = None,
    ) -> InteractiveEvaluation:
        system_failure = 0.0
        intervention_cost = 0.0
        investigation_cost = 0.0
        clean_loss = 0.0
        actual_cost_by_interaction: dict[int, float] = {}
        elapsed_days = 0
        for transition in trace.transitions:
            components = transition.info["reward_components"]
            system_failure = max(system_failure, float(components["system_failure"]))
            intervention_cost += float(components["intervention_cost"])
            investigation_cost += float(components["investigation_cost"])
            if components["clean_utility_loss"] is None:
                clean_loss = None
            elif clean_loss is not None:
                clean_loss += float(components["clean_utility_loss"])
            actual_cost_by_interaction[transition.interaction_index] = (
                float(components["intervention_cost"])
                + float(components["investigation_cost"])
            )
            elapsed_days += int(bool(transition.info["advanced_day"]))
        regret = None
        if oracle_cost_by_interaction is not None:
            regret = sum(
                max(0.0, actual_cost_by_interaction.get(index, 0.0) - float(cost))
                for index, cost in oracle_cost_by_interaction.items()
            )
        return InteractiveEvaluation(
            system_failure=system_failure,
            intervention_cost=intervention_cost,
            investigation_cost=investigation_cost,
            clean_utility_loss=clean_loss,
            oracle_regret=regret,
            interactions=len(trace.transitions),
            elapsed_days=elapsed_days,
        )
