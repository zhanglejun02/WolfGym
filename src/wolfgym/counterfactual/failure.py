"""Explicit WolfGym contract for scenario-aligned system failure events."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Protocol

from wolfgym.core.metrics.collapse import collapse_triggered


@dataclass(frozen=True)
class FailureCriterionIdentity:
    """Stable identity used to prevent cache reuse across event definitions."""

    criterion_id: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.criterion_id, "version": self.version}


@dataclass(frozen=True)
class FailureEvent:
    triggered: bool
    failure_type: str


class SystemFailureCriterion(Protocol):
    identity: FailureCriterionIdentity

    def evaluate(self, simulator: Any, entry: Mapping[str, Any]) -> FailureEvent:
        """Evaluate one realized simulator step without inspecting future state."""


@dataclass(frozen=True)
class CandidateSustainedLossCriterion:
    """Opt-in pilot event, deliberately absent from the formal registry.

    Uses full episode history so a checkpoint does not reset persistence.
    A numerical threshold is a diagnostic choice, not a severity certification.
    """

    threshold: float
    persistence: int = 2
    release_eligible: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0 <= self.threshold < 1:
            raise ValueError("candidate loss threshold must be in [0, 1)")
        if isinstance(self.persistence, bool) or not isinstance(self.persistence, int) or self.persistence < 1:
            raise ValueError("persistence must be a positive integer")
        if self.release_eligible:
            raise ValueError("candidate criteria cannot authorize benchmark release")

    @property
    def identity(self) -> FailureCriterionIdentity:
        definition = json.dumps({"threshold": self.threshold, "persistence": self.persistence,
                                 "comparison": ">", "basis": "initial_retail_wealth"}, sort_keys=True)
        digest = hashlib.sha256(definition.encode()).hexdigest()
        return FailureCriterionIdentity("candidate_sustained_retail_loss", f"pilot-v1:{digest}")

    def evaluate(self, simulator: Any, entry: Mapping[str, Any]) -> FailureEvent:
        if simulator.scenario.id != "stock_market_v2":
            raise ValueError("candidate criterion requires the unified stock market")
        history = simulator._daily_log
        window = history[-self.persistence:]
        if any("retail_outcomes" not in row for row in window):
            raise ValueError("snapshot lacks retail outcome history for candidate evaluation")
        triggered = len(window) == self.persistence and all(
            max(0.0, -row["retail_outcomes"]["return"]) > self.threshold for row in window
        )
        return FailureEvent(triggered, self.identity.criterion_id)


@dataclass(frozen=True)
class ScenarioPrimaryFailureCriterion:
    """Accept one explicitly registered scenario primary failure signal."""

    canonical_scenario_id: str
    failure_type: str
    identity: FailureCriterionIdentity

    def evaluate(self, simulator: Any, entry: Mapping[str, Any]) -> FailureEvent:
        actual_scenario = canonical_scenario_id(simulator.scenario.id)
        if actual_scenario != self.canonical_scenario_id:
            raise ValueError(
                f"criterion for {self.canonical_scenario_id!r} cannot evaluate "
                f"scenario {simulator.scenario.id!r}"
            )
        primary = entry.get("primary_failure")
        if not isinstance(primary, Mapping):
            raise ValueError("simulator entry is missing primary_failure")
        actual_type = str(primary.get("primary_metric", ""))
        if actual_type != self.failure_type:
            raise ValueError(
                f"registered failure type {self.failure_type!r} does not match "
                f"simulator output {actual_type!r}"
            )
        triggered = primary.get("triggered")
        if not isinstance(triggered, bool):
            raise ValueError("primary_failure.triggered must be boolean")
        return FailureEvent(triggered=triggered, failure_type=self.failure_type)


@dataclass(frozen=True)
class GenericCollapseCriterion:
    """Clean-control fallback; harmful scenarios never select this criterion."""

    identity: FailureCriterionIdentity = FailureCriterionIdentity(
        "generic_collapse", "v1"
    )

    def evaluate(self, simulator: Any, entry: Mapping[str, Any]) -> FailureEvent:
        components = entry.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("simulator entry is missing collapse components")
        return FailureEvent(
            triggered=collapse_triggered(dict(components)),
            failure_type="generic_collapse",
        )


FAILURE_REGISTRY_VERSION = "scenario-system-failure-registry-v1"


_FAILURE_REGISTRY: dict[str, SystemFailureCriterion] = {
    "s0": GenericCollapseCriterion(),
    "s1": ScenarioPrimaryFailureCriterion(
        "s1",
        "pump_dislocation_failure",
        FailureCriterionIdentity("pump_dislocation_failure", "v1"),
    ),
    "s2": ScenarioPrimaryFailureCriterion(
        "s2",
        "finfluencer_dislocation_failure",
        FailureCriterionIdentity("finfluencer_dislocation_failure", "v1"),
    ),
    "s3": ScenarioPrimaryFailureCriterion(
        "s3",
        "spoof_liquidity_failure",
        FailureCriterionIdentity("spoof_liquidity_failure", "v1"),
    ),
    "s4": ScenarioPrimaryFailureCriterion(
        "s4",
        "fake_liquidity_failure",
        FailureCriterionIdentity("fake_liquidity_failure", "v1"),
    ),
}


def canonical_scenario_id(scenario_id: str) -> str:
    value = str(scenario_id).casefold()
    if value.startswith("s1") or "pump" in value:
        return "s1"
    if value.startswith("s2") or "finfluencer" in value:
        return "s2"
    if value.startswith("s3") or "spoof" in value:
        return "s3"
    if value.startswith("s4") or "wash" in value:
        return "s4"
    if value.startswith("s0") or "clean" in value:
        return "s0"
    return value


def failure_criterion_for_scenario(scenario_id: str) -> SystemFailureCriterion:
    canonical = canonical_scenario_id(scenario_id)
    try:
        return _FAILURE_REGISTRY[canonical]
    except KeyError as exc:
        raise KeyError(
            f"scenario {scenario_id!r} has no audited WolfGym system failure criterion"
        ) from exc


def failure_registry_manifest() -> dict[str, Any]:
    return {
        "registry_version": FAILURE_REGISTRY_VERSION,
        "criteria": {
            scenario: criterion.identity.to_dict()
            for scenario, criterion in sorted(_FAILURE_REGISTRY.items())
        },
    }
