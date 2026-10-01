"""RL-facing reward, fixed action vocabulary, and tensor observations."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from wolfgym.env.rewards import RewardComponents
from wolfgym.observations import RegulatoryObservation
from wolfgym.regulation import DEFAULT_ACTION_COSTS, RegulatoryAction, RegulatoryActionType


RL_INTERFACE_VERSION = "wolfgym-rl-interface-v2"


@dataclass(frozen=True)
class RLRewardConfig:
    """Scalarize realized outcomes only; no counterfactual oracle is queried."""

    profile: str
    failure_penalty: float
    intervention_cost_weight: float = 1.0
    investigation_cost_weight: float = 1.0
    clean_utility_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.failure_penalty,
            self.intervention_cost_weight,
            self.investigation_cost_weight,
            self.clean_utility_loss_weight,
        )
        if not self.profile.strip():
            raise ValueError("reward profile must be non-empty")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("reward weights must be finite and non-negative")

    def reward(self, components: RewardComponents | Mapping[str, float | None]) -> float:
        values = components.to_dict() if isinstance(components, RewardComponents) else components
        if self.clean_utility_loss_weight and values["clean_utility_loss"] is None:
            raise ValueError("clean utility loss requires a measured matched reference")
        return -float(
            self.failure_penalty * float(values["system_failure"] or 0.0)
            + self.intervention_cost_weight * float(values["intervention_cost"] or 0.0)
            + self.investigation_cost_weight * float(values["investigation_cost"] or 0.0)
            + self.clean_utility_loss_weight * float(values["clean_utility_loss"] or 0.0)
        )


REWARD_PROFILES: Mapping[str, RLRewardConfig] = MappingProxyType({
    "safety_first": RLRewardConfig("safety_first", failure_penalty=100.0),
    "balanced": RLRewardConfig("balanced", failure_penalty=25.0),
    "cost_sensitive": RLRewardConfig("cost_sensitive", failure_penalty=10.0),
})


def reward_profile(name: str = "balanced") -> RLRewardConfig:
    try:
        return REWARD_PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown RL reward profile {name!r}") from exc


@dataclass(frozen=True)
class FixedActionCodec:
    """Map dynamic public targets into a fixed masked Discrete vocabulary.

    Index 0 is WAIT. The next ``max_targets`` indices are THROTTLE slots,
    followed by ``max_targets`` FREEZE slots. Asset slots precede entity slots.
    """

    max_assets: int = 3
    max_entities: int = 5

    def __post_init__(self) -> None:
        if self.max_assets < 0 or self.max_entities < 0:
            raise ValueError("action target limits must be non-negative")
        if self.max_targets == 0:
            raise ValueError("at least one action target slot is required")

    @property
    def max_targets(self) -> int:
        return self.max_assets + self.max_entities

    @property
    def size(self) -> int:
        return 1 + 2 * self.max_targets

    def targets(self, observation: RegulatoryObservation) -> tuple[str, ...]:
        if len(observation.assets) > self.max_assets:
            raise ValueError("action slots cannot silently omit observed assets")
        assets = tuple(item.asset_id for item in observation.assets[:self.max_assets])
        entities = tuple(item.entity_id for item in observation.entities[:self.max_entities])
        return assets + entities

    def action_mask(
        self,
        observation: RegulatoryObservation,
        costs: Mapping[RegulatoryActionType, float] = DEFAULT_ACTION_COSTS,
    ) -> np.ndarray:
        mask = np.zeros(self.size, dtype=np.int8)
        mask[0] = 1
        budget = math.inf if observation.remaining_budget is None else observation.remaining_budget
        target_count = len(self.targets(observation))
        for kind_index, kind in enumerate((
            RegulatoryActionType.THROTTLE,
            RegulatoryActionType.FREEZE,
        )):
            if float(costs[kind]) <= float(budget) + 1e-12:
                start = 1 + kind_index * self.max_targets
                mask[start:start + target_count] = 1
        return mask

    def decode(
        self,
        index: int,
        observation: RegulatoryObservation,
        costs: Mapping[RegulatoryActionType, float] = DEFAULT_ACTION_COSTS,
    ) -> RegulatoryAction:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("RL action must be an integer index")
        value = int(index)
        if value < 0 or value >= self.size:
            raise ValueError(f"RL action index must be in [0, {self.size})")
        if not self.action_mask(observation, costs)[value]:
            raise ValueError("RL action is masked in the current observation")
        if value == 0:
            return RegulatoryAction(RegulatoryActionType.WAIT)
        offset = value - 1
        kind = (
            RegulatoryActionType.THROTTLE
            if offset < self.max_targets
            else RegulatoryActionType.FREEZE
        )
        slot = offset % self.max_targets
        return RegulatoryAction(kind, self.targets(observation)[slot])

    def encode(
        self,
        action: RegulatoryAction,
        observation: RegulatoryObservation,
    ) -> int:
        if action.action_type is RegulatoryActionType.WAIT:
            return 0
        if action.action_type not in {
            RegulatoryActionType.THROTTLE, RegulatoryActionType.FREEZE,
        }:
            raise ValueError("the fixed RL vocabulary supports WAIT/THROTTLE/FREEZE")
        try:
            slot = self.targets(observation).index(str(action.target_id))
        except ValueError as exc:
            raise ValueError("action target is not in the current fixed slots") from exc
        kind_offset = 0 if action.action_type is RegulatoryActionType.THROTTLE else 1
        return 1 + kind_offset * self.max_targets + slot

    def labels(self, observation: RegulatoryObservation) -> tuple[str, ...]:
        targets = self.targets(observation)
        padded = targets + ("<PAD>",) * (self.max_targets - len(targets))
        return ("WAIT", *(
            f"{kind.name}:slot_{slot}:{target}"
            for kind in (RegulatoryActionType.THROTTLE, RegulatoryActionType.FREEZE)
            for slot, target in enumerate(padded)
        ))


GLOBAL_FEATURE_NAMES = (
    "day_fraction",
    "log_society_size",
    "information_level",
    "remaining_budget_fraction",
    "market.return_1d",
    "market.return_3d",
    "market.volatility_3d",
    "market.volume_zscore",
    "market.liquidity",
    "market.order_imbalance",
    "market.cancellation_rate",
    "social.message_rate",
    "social.cascade_reach",
    "social.sentiment_concentration",
    "social.network_degree_concentration",
    "social.information_concentration",
    "social.consensus_concentration",
)

TARGET_FEATURE_NAMES = (
    "is_asset",
    "is_entity",
    "signal_1",
    "signal_2",
    "signal_3",
    "signal_4",
    "signal_5",
)


@dataclass(frozen=True)
class StructuredObservationEncoder:
    """Convert the public observation into fixed-shape float32 tensors."""

    action_codec: FixedActionCodec = FixedActionCodec()
    horizon_days: int = 30
    initial_budget: float = 12.0
    value_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if not math.isfinite(self.initial_budget) or self.initial_budget <= 0.0:
            raise ValueError("initial_budget must be finite and positive")
        if not math.isfinite(self.value_clip) or self.value_clip <= 0.0:
            raise ValueError("value_clip must be finite and positive")

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "global": (len(GLOBAL_FEATURE_NAMES),),
            "targets": (self.action_codec.max_targets, len(TARGET_FEATURE_NAMES)),
            "target_mask": (self.action_codec.max_targets,),
        }

    def encode(self, observation: RegulatoryObservation) -> dict[str, np.ndarray]:
        self.action_codec.targets(observation)  # validate asset coverage
        market = observation.market
        social = observation.social
        budget = (
            self.initial_budget
            if observation.remaining_budget is None
            else observation.remaining_budget
        )
        global_values = np.asarray((
            observation.day / self.horizon_days,
            math.log1p(observation.society_size) / math.log1p(2000),
            int(observation.information_level) / 3.0,
            budget / self.initial_budget,
            market.return_1d,
            market.return_3d,
            market.volatility_3d,
            market.volume_zscore,
            market.liquidity,
            market.order_imbalance,
            market.cancellation_rate,
            social.message_rate,
            social.cascade_reach,
            social.sentiment_concentration,
            social.network_degree_concentration,
            social.information_concentration,
            social.consensus_concentration,
        ), dtype=np.float32)
        global_values = np.nan_to_num(
            global_values, nan=0.0, posinf=self.value_clip, neginf=-self.value_clip,
        )
        global_values = np.clip(global_values, -self.value_clip, self.value_clip)

        target_values = np.zeros(self.shapes["targets"], dtype=np.float32)
        target_mask = np.zeros(self.shapes["target_mask"], dtype=np.int8)
        slot = 0
        for item in observation.assets[:self.action_codec.max_assets]:
            target_values[slot] = (
                1.0, 0.0, item.return_3d, item.volatility_3d,
                item.volume_zscore, item.order_imbalance, item.cancellation_rate,
            )
            target_mask[slot] = 1
            slot += 1
        for item in observation.entities[:self.action_codec.max_entities]:
            target_values[slot] = (
                0.0, 1.0, item.message_reach, item.trading_concentration,
                item.cancellation_rate, item.social_influence, item.activity_change,
            )
            target_mask[slot] = 1
            slot += 1
        target_values = np.nan_to_num(
            target_values, nan=0.0, posinf=self.value_clip, neginf=-self.value_clip,
        )
        target_values = np.clip(target_values, -self.value_clip, self.value_clip)
        return {
            "global": global_values.astype(np.float32, copy=False),
            "targets": target_values.astype(np.float32, copy=False),
            "target_mask": target_mask,
        }
