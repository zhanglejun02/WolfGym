"""Unweighted outcome components for interactive WolfGym episodes."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardComponents:
    """Raw evaluation axes; WolfGym deliberately does not scalarize them."""

    system_failure: float
    intervention_cost: float
    investigation_cost: float
    clean_utility_loss: float | None
    risk_reduction: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


REWARD_DIRECTIONS = {
    "system_failure": "minimize",
    "intervention_cost": "minimize",
    "investigation_cost": "minimize",
    "clean_utility_loss": "minimize",
    "risk_reduction": "maximize",
}
