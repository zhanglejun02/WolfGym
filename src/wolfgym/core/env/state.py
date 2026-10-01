"""In-memory snapshots for deterministic WolfGym continuation rollouts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimulatorState:
    """A deep, opaque snapshot of every mutable simulator field.

    The payload is intentionally not exposed as a regulator observation or a
    JSON benchmark record.  It contains hidden labels and RNG states and is
    only for trusted simulator-side restoration and counterfactual rollout.
    """

    scenario_id: str
    n_society: int
    next_day: int
    horizon_days: int
    _payload: dict[str, Any] = field(repr=False, compare=False)
    schema_version: int = 8

    def clone(self) -> "SimulatorState":
        """Return an independent deep copy, never a shallow alias."""
        return deepcopy(self)

    def payload_copy(self) -> dict[str, Any]:
        """Return a defensive copy for restoration by the simulator."""
        return deepcopy(self._payload)
