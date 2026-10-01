"""Canonical interactive episode traces, separate from static records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class EpisodeTransition:
    interaction_index: int
    observation: Mapping[str, Any]
    action: Mapping[str, Any]
    next_observation: Mapping[str, Any]
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


@dataclass
class EpisodeTrace:
    episode_id: str
    trajectory_id: str
    initial_observation: Mapping[str, Any]
    transitions: list[EpisodeTransition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), allow_nan=False, sort_keys=True))
