"""Canonical, simulator-grounded benchmark records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping


class RiskRegime(str, Enum):
    STABLE = "stable"
    SUBCRITICAL = "subcritical"
    NEAR_CRITICAL = "near-critical"
    SUPERCRITICAL = "supercritical"


@dataclass(frozen=True)
class RiskThresholds:
    stable_upper: float = 0.10
    near_critical_lower: float = 0.40
    supercritical_lower: float = 0.60

    def __post_init__(self) -> None:
        values = (
            self.stable_upper,
            self.near_critical_lower,
            self.supercritical_lower,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("risk thresholds must be finite")
        if not 0.0 <= self.stable_upper <= self.near_critical_lower \
                <= self.supercritical_lower <= 1.0:
            raise ValueError("risk thresholds must be ordered within [0, 1]")

    def classify(self, probability: float) -> RiskRegime:
        risk = float(probability)
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError("risk probability must be finite and in [0, 1]")
        if risk < self.stable_upper:
            return RiskRegime.STABLE
        if risk < self.near_critical_lower:
            return RiskRegime.SUBCRITICAL
        if risk <= self.supercritical_lower:
            return RiskRegime.NEAR_CRITICAL
        return RiskRegime.SUPERCRITICAL


def stable_instance_id(identity: Mapping[str, Any]) -> str:
    """Hash only state identity/configuration, never object addresses."""
    canonical = json.dumps(identity, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return "wg_" + hashlib.blake2s(canonical.encode(), digest_size=12).hexdigest()


@dataclass(frozen=True)
class WolfGymRecord:
    """Single source of truth used by every static and interactive view."""

    schema_version: str
    instance_id: str
    trajectory_id: str
    split: str
    scenario_id: str
    checkpoint_day: int
    hidden_state_ref: str
    observation: Mapping[str, Any]
    observation_text: str
    risk_probability: float
    risk_regime: RiskRegime
    mechanism: str
    society_size: int
    counterfactuals: Mapping[str, Any]
    failure_criterion: Mapping[str, Any]
    wait_outcomes: Mapping[str, Any]
    oracle: Mapping[str, Any]
    task_eligibility: tuple[str, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.instance_id.startswith("wg_"):
            raise ValueError("instance_id must be a stable WolfGym id")
        if not math.isfinite(float(self.risk_probability)) \
                or not 0.0 <= self.risk_probability <= 1.0:
            raise ValueError("risk_probability must be finite and in [0, 1]")
        if self.society_size <= 0:
            raise ValueError("society_size must be positive")
        if self.checkpoint_day < 0:
            raise ValueError("checkpoint_day must be non-negative")
        if not self.trajectory_id.startswith("traj_"):
            raise ValueError("trajectory_id must be stable and trajectory-scoped")
        if not self.hidden_state_ref.startswith("state://"):
            raise ValueError("hidden_state_ref must be opaque")
        if not self.observation_text.strip():
            raise ValueError("observation_text must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_regime"] = self.risk_regime.value
        return json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WolfGymRecord":
        values = dict(payload)
        values["risk_regime"] = RiskRegime(values["risk_regime"])
        values["task_eligibility"] = tuple(values["task_eligibility"])
        return cls(**values)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
        )


# Transitional import alias: there remains exactly one serialized record type.
BenchmarkRecord = WolfGymRecord
