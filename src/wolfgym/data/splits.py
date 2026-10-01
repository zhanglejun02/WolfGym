"""Trajectory-disjoint static/RL/OOD split manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any, Iterable


class SplitName(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    IID_TEST = "iid_test"
    RL_TRAIN = "rl_train"
    SCALE_OOD = "scale_ood"
    PARAMETER_OOD = "parameter_ood"


@dataclass(frozen=True)
class SplitEntry:
    trajectory_id: str
    split: SplitName
    trajectory_seed: int
    society_size: int
    parameter_profile: str = "iid"


@dataclass(frozen=True)
class SplitManifest:
    schema_version: str
    entries: tuple[SplitEntry, ...]

    def __post_init__(self) -> None:
        seen: dict[str, SplitName] = {}
        for entry in self.entries:
            prior = seen.setdefault(entry.trajectory_id, entry.split)
            if prior is not entry.split:
                raise ValueError("one trajectory cannot cross split boundaries")
            if entry.split is SplitName.SCALE_OOD and entry.society_size != 2000:
                raise ValueError("Scale OOD is frozen at N=2000")
            if entry.split is SplitName.PARAMETER_OOD and entry.parameter_profile == "iid":
                raise ValueError("Parameter OOD requires a non-IID parameter profile")

    @classmethod
    def from_entries(cls, entries: Iterable[SplitEntry]) -> "SplitManifest":
        return cls("wolfgym-splits-1.0", tuple(entries))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))
