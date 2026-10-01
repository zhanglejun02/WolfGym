"""Build deterministic, informative but non-oracular regulator observations."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.env.state import SimulatorState
from wolfgym.observations.schema import (
    ActiveControlObservation,
    AssetObservation,
    EntityObservation,
    InformationLevel,
    InvestigationObservation,
    InteractionHistoryObservation,
    MarketObservation,
    PublicMessageObservation,
    RegulatoryObservation,
    SocialObservation,
)


_IDENTIFIER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9_]+(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class ObservationConfig:
    window_days: int = 3
    top_k_entities: int = 5
    information_level: InformationLevel = InformationLevel.TOP_ENTITIES
    investigation_neighbor_limit: int = 10
    zscore_clip: float = 10.0
    activity_change_clip: float = 10.0
    include_messages: bool = False
    max_messages: int = 8

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError("window_days must be positive")
        if self.top_k_entities < 0:
            raise ValueError("top_k_entities must be non-negative")
        if self.investigation_neighbor_limit < 0:
            raise ValueError("investigation_neighbor_limit must be non-negative")
        if not np.isfinite(self.zscore_clip) or self.zscore_clip <= 0.0:
            raise ValueError("zscore_clip must be finite and positive")
        if not np.isfinite(self.activity_change_clip) or self.activity_change_clip <= 1.0:
            raise ValueError("activity_change_clip must be finite and greater than 1")
        if not isinstance(self.information_level, InformationLevel):
            raise TypeError("information_level must be an InformationLevel")
        if not isinstance(self.include_messages, bool):
            raise TypeError("include_messages must be a bool")
        if self.max_messages < 0:
            raise ValueError("max_messages must be non-negative")


class ObservationBuilder:
    """Create regulator-visible statistics from an exact hidden snapshot."""

    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config = config or ObservationConfig()

    def build(
        self,
        simulator: WolfSocietySimulator,
        state: SimulatorState,
        investigation_target: str | None = None,
        information_level: InformationLevel | None = None,
        remaining_budget: float | None = None,
        active_controls: tuple[ActiveControlObservation, ...] = (),
        history: tuple[InteractionHistoryObservation, ...] = (),
    ) -> RegulatoryObservation:
        level = information_level or self.config.information_level
        if investigation_target is not None and level is not InformationLevel.INVESTIGATION:
            raise ValueError("investigation_target requires information level 3")

        view = simulator.fork_from_state(state)
        entities: tuple[EntityObservation, ...] = ()
        if level >= InformationLevel.TOP_ENTITIES:
            entities = self._build_entities(view)

        investigation = None
        if level is InformationLevel.INVESTIGATION and investigation_target is not None:
            investigation = self._build_investigation(view, investigation_target)

        market, assets = self._build_market(view)
        messages = self._build_messages(view) if self.config.include_messages else None
        observation = RegulatoryObservation(
            day=state.next_day,
            society_size=state.n_society,
            information_level=level,
            market=market,
            social=self._build_social(view),
            assets=assets,
            entities=entities,
            investigation=investigation,
            remaining_budget=remaining_budget,
            active_controls=active_controls,
            history=history,
            messages=messages,
        )
        # Serialization is part of the contract, not a downstream best effort.
        observation.to_prompt()
        return observation

    def _build_messages(
        self,
        simulator: WolfSocietySimulator,
    ) -> tuple[PublicMessageObservation, ...]:
        """Select recent public messages without consulting hidden labels.

        Recency is the first ordering key. Within a day, messages with greater
        realized audience reach are shown first. Remaining ties use only
        episode-local public identifiers and the message id, making the view
        deterministic without depending on agent role or scenario metadata.
        """
        if self.config.max_messages == 0:
            return ()
        events = getattr(simulator.social, "message_events", ())
        first_day = simulator.get_state().next_day - self.config.window_days
        candidates: list[tuple[tuple, PublicMessageObservation]] = []
        public_ids = {
            internal_id: simulator.public_entity_id(internal_id)
            for internal_id in simulator.graph.id_to_node
        }
        public_ids.update({
            internal_id: simulator.public_asset_id(internal_id)
            for internal_id in simulator.market.assets
        })
        for event in events:
            day = int(event.get("day", -1))
            text = str(event.get("text", "")).strip()
            sender = str(event.get("sender_id", ""))
            asset = str(event.get("asset", ""))
            if day < first_day or not text or sender not in simulator.graph.id_to_node:
                continue
            if asset not in simulator.market.assets:
                continue
            # Agent-authored text is public evidence. Replace any raw simulator
            # identifiers before it reaches the benchmark-facing observation.
            text = _IDENTIFIER_TOKEN.sub(
                lambda match: public_ids.get(match.group(0), match.group(0)),
                text,
            )
            audience_reach = float(np.clip(
                float(event.get("audience_size", 0.0)) / max(simulator.n_society, 1),
                0.0,
                1.0,
            ))
            public_sender = simulator.public_entity_id(sender)
            public_asset = simulator.public_asset_id(asset)
            message = PublicMessageObservation(
                day=day,
                sender_id=public_sender,
                asset_id=public_asset,
                kind=str(event.get("kind", "post")),
                text=text,
                audience_reach=audience_reach,
            )
            order = (
                -day,
                -audience_reach,
                public_sender,
                str(event.get("message_id", "")),
            )
            candidates.append((order, message))
        candidates.sort(key=lambda item: item[0])
        return tuple(message for _, message in candidates[:self.config.max_messages])

    def _build_market(
        self, simulator: WolfSocietySimulator,
    ) -> tuple[MarketObservation, tuple[AssetObservation, ...]]:
        one_day_returns: list[float] = []
        three_day_returns: list[float] = []
        asset_volatilities: list[float] = []
        volume_zscores: list[float] = []
        liquidities: list[float] = []
        imbalances: list[float] = []
        cancellation_rates: list[float] = []
        asset_observations: list[AssetObservation] = []

        initial_prices = {asset.id: float(asset.fundamental) for asset in simulator.scenario.assets}
        window = self.config.window_days
        for asset_id, asset in simulator.market.assets.items():
            prices = [initial_prices[asset_id], *map(float, asset.history["price"])]
            returns = [
                (right - left) / max(abs(left), 1e-12)
                for left, right in zip(prices, prices[1:])
            ]
            one_day_returns.append(returns[-1] if returns else 0.0)
            start = max(0, len(prices) - 1 - window)
            three_day_returns.append(
                (prices[-1] - prices[start]) / max(abs(prices[start]), 1e-12)
                if len(prices) > 1 else 0.0
            )
            recent = returns[-window:]
            volatility = float(np.std(recent)) if recent else 0.0
            asset_volatilities.append(volatility)

            volumes = list(map(float, asset.history["volume"]))
            if len(volumes) >= 3:
                baseline = np.asarray(volumes[-(window + 1):-1], dtype=float)
                zscore = (
                    (volumes[-1] - float(baseline.mean()))
                    / (float(baseline.std()) + 1e-6)
                )
                volume_zscore = float(np.clip(
                    zscore, -self.config.zscore_clip, self.config.zscore_clip,
                ))
            else:
                volume_zscore = 0.0
            volume_zscores.append(volume_zscore)

            spread_ratio = asset.spread_bps / max(simulator.market.base_spread_bps, 1e-12)
            liquidities.append(1.0 / max(spread_ratio, 1.0))
            imbalance = self._last(asset.history["depth_imbalance"])
            cancellation_rate = self._last(asset.history["cancel_rate"])
            imbalances.append(imbalance)
            cancellation_rates.append(cancellation_rate)
            asset_observations.append(AssetObservation(
                asset_id=simulator.public_asset_id(asset_id),
                return_3d=three_day_returns[-1],
                volatility_3d=volatility,
                volume_zscore=volume_zscore,
                order_imbalance=imbalance,
                cancellation_rate=cancellation_rate,
            ))

        market = MarketObservation(
            return_1d=self._mean(one_day_returns),
            return_3d=self._mean(three_day_returns),
            volatility_3d=self._mean(asset_volatilities),
            volume_zscore=self._mean(volume_zscores),
            liquidity=self._mean(liquidities),
            order_imbalance=self._mean(imbalances),
            cancellation_rate=self._mean(cancellation_rates),
        )
        return market, tuple(sorted(asset_observations, key=lambda item: item.asset_id))

    def _build_social(self, simulator: WolfSocietySimulator) -> SocialObservation:
        volumes: list[float] = []
        sentiments: list[float] = []
        for asset_id in simulator.market.assets:
            history = simulator.social.state.history.get(asset_id, [])
            latest = history[-1] if history else {}
            volume = max(0.0, float(latest.get("msg_volume", 0.0)))
            volumes.append(volume)
            sentiments.append(float(latest.get("sentiment", 0.0)))

        total_volume = sum(volumes)
        sentiment_concentration = (
            sum(volume * abs(sentiment) for volume, sentiment in zip(volumes, sentiments))
            / total_volume if total_volume > 0.0 else 0.0
        )
        exposed = set().union(*simulator.social.state.cascade_size.values()) \
            if simulator.social.state.cascade_size else set()
        cascade_reach = len(exposed) / max(len(simulator.society.all_agents), 1)

        degrees = list(simulator.graph.degrees().values())
        exposure_by_agent: list[float] = []
        signed_exposure: list[float] = []
        for agent_id in simulator.graph.id_to_node:
            values = simulator.social.state.exposure.get(agent_id, {})
            signed = sum(float(value) for value in values.values())
            magnitude = sum(abs(float(value)) for value in values.values())
            signed_exposure.append(signed)
            exposure_by_agent.append(magnitude)
        total_exposure = sum(abs(value) for value in signed_exposure)
        consensus = (
            abs(sum(signed_exposure)) / total_exposure
            if total_exposure > 0.0 else 0.0
        )

        return SocialObservation(
            message_rate=float(total_volume / max(simulator.n_society, 1)),
            cascade_reach=float(np.clip(cascade_reach, 0.0, 1.0)),
            sentiment_concentration=float(np.clip(sentiment_concentration, 0.0, 1.0)),
            network_degree_concentration=self._normalized_hhi(degrees),
            information_concentration=self._normalized_hhi(exposure_by_agent),
            consensus_concentration=float(np.clip(consensus, 0.0, 1.0)),
        )

    def _build_entities(
        self,
        simulator: WolfSocietySimulator,
    ) -> tuple[EntityObservation, ...]:
        history = simulator.observable_entity_history()
        if not history or self.config.top_k_entities == 0:
            return ()
        current = history[-1]["entities"]
        total_reach = sum(float(stats["potential_message_reach"]) for stats in current.values())
        total_quantity = sum(float(stats["order_quantity"]) for stats in current.values())
        degrees = simulator.graph.degrees()
        max_degree = max(degrees.values(), default=1)
        observations: list[tuple[float, EntityObservation]] = []

        for internal_id, stats in current.items():
            message_reach = float(stats["potential_message_reach"]) / max(total_reach, 1e-12)
            trading = float(stats["order_quantity"]) / max(total_quantity, 1e-12)
            cancellations = float(stats["cancellation_count"]) / max(
                float(stats["order_count"]), 1.0,
            )
            influence = degrees.get(internal_id, 0) / max(max_degree, 1)
            current_activity = self._activity(stats)
            previous = [
                self._activity(entry["entities"].get(internal_id, {}))
                for entry in history[-(self.config.window_days + 1):-1]
            ]
            baseline = self._mean(previous)
            activity_change = self._activity_change(current_activity, baseline)
            entity = EntityObservation(
                entity_id=simulator.public_entity_id(internal_id),
                message_reach=float(np.clip(message_reach, 0.0, 1.0)),
                trading_concentration=float(np.clip(trading, 0.0, 1.0)),
                cancellation_rate=float(np.clip(cancellations, 0.0, 1.0)),
                social_influence=float(np.clip(influence, 0.0, 1.0)),
                activity_change=float(np.clip(
                    activity_change, 0.0, self.config.activity_change_clip,
                )),
            )
            rank_score = (
                entity.message_reach
                + entity.trading_concentration
                + entity.cancellation_rate
                + entity.social_influence
                + min(entity.activity_change / 2.0, 1.0)
            )
            observations.append((rank_score, entity))

        observations.sort(key=lambda item: (-item[0], item[1].entity_id))
        return tuple(entity for _, entity in observations[:self.config.top_k_entities])

    def _build_investigation(
        self,
        simulator: WolfSocietySimulator,
        public_target: str,
    ) -> InvestigationObservation:
        try:
            internal_id = simulator.internal_entity_id(public_target)
        except ValueError:
            return self._build_asset_investigation(simulator, public_target)
        history = simulator.observable_entity_history()[-self.config.window_days:]
        message_pattern = tuple(
            float(entry["entities"].get(internal_id, {}).get("message_intensity", 0.0))
            for entry in history
        )
        order_pattern = tuple(
            float(entry["entities"].get(internal_id, {}).get("order_quantity", 0.0))
            for entry in history
        )
        neighbours = sorted(
            simulator.public_entity_id(neighbour)
            for neighbour in simulator.graph.neighbours(internal_id)
        )[:self.config.investigation_neighbor_limit]
        return InvestigationObservation(
            target_id=public_target,
            target_type="entity",
            recent_message_pattern=message_pattern,
            recent_order_pattern=order_pattern,
            network_neighbors=tuple(neighbours),
        )

    def _build_asset_investigation(
        self,
        simulator: WolfSocietySimulator,
        public_target: str,
    ) -> InvestigationObservation:
        internal_id = simulator.internal_asset_id(public_target)
        asset = simulator.market.assets[internal_id]
        window = self.config.window_days
        order_pattern = tuple(map(float, asset.history["volume"][-window:]))
        message_history = simulator.social.state.history.get(internal_id, [])[-window:]
        message_pattern = tuple(
            float(entry.get("msg_volume", 0.0)) for entry in message_history
        )
        return InvestigationObservation(
            target_id=public_target,
            target_type="asset",
            recent_message_pattern=message_pattern,
            recent_order_pattern=order_pattern,
            network_neighbors=(),
        )

    @staticmethod
    def _activity(stats: dict) -> float:
        return max(0.0, float(stats.get("message_intensity", 0.0))) + max(
            0.0, float(stats.get("order_count", 0.0)),
        )

    def _activity_change(self, current: float, baseline: float) -> float:
        if baseline > 1e-12:
            return current / baseline
        return self.config.activity_change_clip if current > 0.0 else 1.0

    @staticmethod
    def _last(values: Iterable[float]) -> float:
        values = list(values)
        return float(values[-1]) if values else 0.0

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _normalized_hhi(values: Iterable[float]) -> float:
        nonnegative = np.asarray([max(0.0, float(value)) for value in values], dtype=float)
        n = len(nonnegative)
        total = float(nonnegative.sum())
        if n <= 1 or total <= 0.0:
            return 0.0
        shares = nonnegative / total
        hhi = float(np.square(shares).sum())
        return float(np.clip((hhi - 1.0 / n) / (1.0 - 1.0 / n), 0.0, 1.0))
