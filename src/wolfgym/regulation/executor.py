"""Central translation from WolfGym actions to simulator controls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.regulation.actions import RegulatoryAction, RegulatoryActionType


@dataclass(frozen=True)
class RegulatoryEffectConfig:
    """Persistent intervention strengths applied until the rollout horizon."""

    throttle_fraction: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.throttle_fraction <= 1.0:
            raise ValueError("throttle_fraction must be in [0, 1]")


class RegulatoryActionExecutor:
    """Apply persistent public controls without changing scenario logic.

    THROTTLE scales the target's messages and all order types by the configured
    fraction. FREEZE suppresses them completely. An omitted target means the
    scenario target asset; an agent id scopes the control to that entity.
    Effects persist for the rest of the branch and are captured in snapshots.
    """

    def __init__(self, config: RegulatoryEffectConfig | None = None) -> None:
        self.config = config or RegulatoryEffectConfig()

    def apply(self, simulator: WolfSocietySimulator, action: RegulatoryAction, day: int) -> None:
        del day  # reserved for future duration/budget policies
        kind = action.action_type
        if kind in {RegulatoryActionType.WAIT, RegulatoryActionType.INVESTIGATE}:
            return

        scope, target_id = self._resolve_target(simulator, action.target_id)
        if kind is RegulatoryActionType.RELEASE:
            simulator.clear_persistent_regulatory_controls(
                scope=scope,
                target_id=target_id,
            )
            return
        strength = (
            self.config.throttle_fraction
            if kind is RegulatoryActionType.THROTTLE
            else 1.0
        )
        simulator.set_persistent_regulatory_controls(
            scope=scope,
            target_id=target_id,
            controls={
                "message_filter": strength,
                "trade_throttle": strength,
                "spoof_filter": strength,
                "wash_filter": strength,
            },
        )

    def release(self, simulator: WolfSocietySimulator, target_id: str | None) -> None:
        """Clear controls on one public target without constructing a new action."""
        scope, internal_id = self._resolve_target(simulator, target_id)
        simulator.clear_persistent_regulatory_controls(
            scope=scope,
            target_id=internal_id,
        )

    @staticmethod
    def _resolve_target(
        simulator: WolfSocietySimulator,
        target_id: str | None,
    ) -> tuple[Literal["asset", "agent"], str]:
        if target_id is None:
            return "asset", simulator.target_asset
        if target_id in simulator.market.assets:
            return "asset", target_id
        try:
            internal_asset = simulator.internal_asset_id(target_id)
        except ValueError:
            pass
        else:
            return "asset", internal_asset
        try:
            internal_id = simulator.internal_entity_id(target_id)
        except ValueError as exc:
            raise ValueError(f"unknown regulatory target {target_id!r}") from exc
        return "agent", internal_id
