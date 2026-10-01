"""Small, explicitly regulator-visible observation schema."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


OBSERVATION_SCHEMA_VERSION = "regulatory-observation-v2"
LANGUAGE_OBSERVATION_SCHEMA_VERSION = "regulatory-observation-v3"


class InformationLevel(IntEnum):
    AGGREGATE = 1
    TOP_ENTITIES = 2
    INVESTIGATION = 3


@dataclass(frozen=True)
class MarketObservation:
    return_1d: float
    return_3d: float
    volatility_3d: float
    volume_zscore: float
    liquidity: float
    order_imbalance: float
    cancellation_rate: float


@dataclass(frozen=True)
class AssetObservation:
    asset_id: str
    return_3d: float
    volatility_3d: float
    volume_zscore: float
    order_imbalance: float
    cancellation_rate: float


@dataclass(frozen=True)
class SocialObservation:
    message_rate: float
    cascade_reach: float
    sentiment_concentration: float
    network_degree_concentration: float
    information_concentration: float
    consensus_concentration: float


@dataclass(frozen=True)
class PublicMessageObservation:
    """One realized public message with episode-local identities."""

    day: int
    sender_id: str
    asset_id: str
    kind: str
    text: str
    audience_reach: float


@dataclass(frozen=True)
class EntityObservation:
    entity_id: str
    message_reach: float
    trading_concentration: float
    cancellation_rate: float
    social_influence: float
    activity_change: float


@dataclass(frozen=True)
class InvestigationObservation:
    target_id: str
    target_type: str
    recent_message_pattern: tuple[float, ...]
    recent_order_pattern: tuple[float, ...]
    network_neighbors: tuple[str, ...]


@dataclass(frozen=True)
class ActiveControlObservation:
    action_type: str
    target_id: str


@dataclass(frozen=True)
class InteractionHistoryObservation:
    interaction_index: int
    day: int
    action_type: str
    target_id: str | None
    advanced_day: bool
    cost: float
    investigation_result: InvestigationObservation | None = None


@dataclass(frozen=True)
class RegulatoryObservation:
    day: int
    society_size: int
    information_level: InformationLevel
    market: MarketObservation
    social: SocialObservation
    assets: tuple[AssetObservation, ...] = ()
    entities: tuple[EntityObservation, ...] = ()
    investigation: InvestigationObservation | None = None
    remaining_budget: float | None = None
    active_controls: tuple[ActiveControlObservation, ...] = ()
    history: tuple[InteractionHistoryObservation, ...] = ()
    # ``None`` means the frozen v2 schema. An explicit tuple, including an
    # empty one, means the language-enabled v3 schema.
    messages: tuple[PublicMessageObservation, ...] | None = None

    @property
    def schema_version(self) -> str:
        return (
            OBSERVATION_SCHEMA_VERSION
            if self.messages is None
            else LANGUAGE_OBSERVATION_SCHEMA_VERSION
        )

    def to_dict(self) -> dict:
        from wolfgym.observations.serialization import observation_to_dict

        return observation_to_dict(self)

    def to_prompt(self) -> str:
        from wolfgym.observations.serialization import observation_to_prompt

        return observation_to_prompt(self)

    def to_text(self) -> str:
        from wolfgym.observations.serialization import observation_to_text

        return observation_to_text(self)
