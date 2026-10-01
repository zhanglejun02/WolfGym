"""Typed regulatory decisions and benchmark-owned costs."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


REGULATION_VERSION = "regulatory-actions-effects-costs-v2"


class RegulatoryActionType(str, Enum):
    WAIT = "wait"
    INVESTIGATE = "investigate"
    THROTTLE = "throttle"
    FREEZE = "freeze"
    RELEASE = "release"


DEFAULT_ACTION_COSTS: Mapping[RegulatoryActionType, float] = MappingProxyType({
    RegulatoryActionType.WAIT: 0.0,
    RegulatoryActionType.INVESTIGATE: 1.0,
    RegulatoryActionType.THROTTLE: 2.0,
    RegulatoryActionType.FREEZE: 4.0,
    RegulatoryActionType.RELEASE: 0.0,
})


@dataclass(frozen=True)
class RegulatoryAction:
    action_type: RegulatoryActionType
    target_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, RegulatoryActionType):
            raise TypeError("action_type must be a RegulatoryActionType")
        if self.target_id is not None:
            target = str(self.target_id).strip()
            if not target:
                raise ValueError("target_id must be non-empty when provided")
            object.__setattr__(self, "target_id", target)
        if self.action_type is RegulatoryActionType.WAIT and self.target_id is not None:
            raise ValueError("WAIT cannot have a target")
        if self.action_type is RegulatoryActionType.RELEASE and self.target_id is None:
            raise ValueError("RELEASE requires a target")

    @property
    def key(self) -> tuple[RegulatoryActionType, str | None]:
        """Semantic identity used to distinguish targeted candidates."""
        return self.action_type, self.target_id

    @property
    def label(self) -> str:
        if self.target_id is None:
            return self.action_type.name
        return f"{self.action_type.name}:{self.target_id}"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "type": self.action_type.name,
            "target_id": self.target_id,
        }
