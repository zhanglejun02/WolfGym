"""WolfGym society environment: 30-day MAS episode loop."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
from typing import Any, Callable, Literal

import numpy as np

from wolfgym.core.agents.wolfguard import WolfGuardAgent
from wolfgym.core.env.market import MarketEnv, Order
from wolfgym.core.version import SIMULATOR_VERSION
from wolfgym.core.env.social import SocialEnv, SocialGraph, Message
from wolfgym.core.env.state import SimulatorState
from wolfgym.core.metrics.collapse import (
    EpisodeMetrics, compute_collapse_score, collapse_triggered,
    primary_failure_signal,
)
from wolfgym.core.scenarios.base import ScenarioConfig
from wolfgym.core.scenarios.society import Society, build_society
from wolfgym.core.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EpisodeResult:
    scenario_id: str
    n_society: int
    alpha: float
    seed: int
    metrics: EpisodeMetrics
    target_asset: str
    daily_log: list[dict[str, Any]] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorConfig:
    """Evaluator-owned intervention accounting parameters.

    These values must come from the benchmark harness, not from a submitted
    policy object, so submissions cannot alter costs or false-positive rules.
    """
    err_trade_exposure_threshold: float = 0.3
    intervention_cost_warning: float = 0.01
    intervention_cost_cooldown: float = 0.05
    intervention_cost_block: float = 0.10


VALID_INTERVENTIONS = {"none", "warning", "cooldown", "block"}


def _nonnegative_finite(value: float, default: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(val):
        return default
    return max(0.0, val)


def validate_interventions(actions: Any, valid_assets: set[str]) -> dict[str, dict]:
    """Return a sanitized per-asset intervention map.

    Unknown assets and malformed entries are ignored. Action names are limited
    to the official enum, risk is finite and clipped to [0, 1], and the dict key
    is treated as the authoritative asset id.
    """
    if not isinstance(actions, dict):
        return {}
    clean: dict[str, dict] = {}
    for asset, raw in actions.items():
        if asset not in valid_assets or not isinstance(raw, dict):
            continue
        action = raw.get("action", "none")
        if action not in VALID_INTERVENTIONS:
            action = "none"
        try:
            risk = float(raw.get("risk", 0.0))
        except (TypeError, ValueError):
            risk = 0.0
        if not np.isfinite(risk):
            risk = 0.0
        risk = float(np.clip(risk, 0.0, 1.0))
        clean[asset] = {
            "asset": asset,
            "action": action,
            "risk": risk,
            "reason": str(raw.get("reason", ""))[:256],
            "components": raw.get("components", {}) if isinstance(raw.get("components", {}), dict) else {},
        }
    return clean


class WolfSocietySimulator:
    """Main 30-day environment.

    Usage::
        env = WolfSocietySimulator(scenario, n_society=1000, alpha=0.02, seed=1)
        result = env.run()
    """

    def __init__(self, scenario: ScenarioConfig, n_society: int, alpha: float,
                 seed: int = 0,
                 wolfguard: WolfGuardAgent | None = None,
                 baseline: dict | None = None,
                 placement_override: str | None = None,
                 llm_backend=None,
                 n_llm_leaders: int = 0,
                 expose_oracle: bool = False,
                 evaluator_config: EvaluatorConfig | None = None,
                 record_trajectory: bool = False):
        self.scenario = scenario
        self.n_society = n_society
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.society: Society = build_society(
            scenario, n_society, alpha, self.rng,
            placement_override=placement_override,
            llm_backend=llm_backend,
            n_llm_leaders=n_llm_leaders,
        )
        self._public_entity_ids = self._build_public_entity_ids()
        self._internal_entity_ids = {
            public_id: internal_id
            for internal_id, public_id in self._public_entity_ids.items()
        }
        self._public_asset_ids = self._build_public_asset_ids()
        self._internal_asset_ids = {
            public_id: internal_id
            for internal_id, public_id in self._public_asset_ids.items()
        }
        self._observable_entity_history: list[dict[str, Any]] = []
        # Baseline liquidity grows sub-linearly with society size. Experiments
        # can override the exponent to separate population/network effects from
        # this market-depth design choice (q=0 fixed depth, q=1 per-capita).
        liquidity_exponent = float(scenario.market_makers.get("liquidity_exponent", 0.5))
        liquidity_reference_n = max(
            1.0, float(scenario.market_makers.get("liquidity_reference_n", 1000.0))
        )
        liquidity_scale = max(
            0.1, (n_society / liquidity_reference_n) ** liquidity_exponent
        )
        self.liquidity_exponent = liquidity_exponent
        self.liquidity_reference_n = liquidity_reference_n
        self.market = MarketEnv(scenario, np.random.default_rng(seed + 7),
                                accounts={agent.agent_id: agent.portfolio for agent in self.society.all_agents},
                                market_maker_ids=[agent.agent_id for agent in self.society.market_makers],
                                liquidity_scale=liquidity_scale)
        self.intraday_steps = max(
            1,
            int(scenario.market_makers.get("intraday_steps", 1)),
        )

        self.graph = SocialGraph(
            n_agents=len(self.society.all_agents),
            mean_degree=int(scenario.social.get("mean_degree", 8)),
            rng=np.random.default_rng(seed + 13),
            graph_type=str(scenario.social.get("graph", "scale_free")),
        )
        agent_ids = [a.agent_id for a in self.society.all_agents]
        # placement: optionally swap harmful agent_ids onto top-degree nodes
        if self.society.placement == "high_degree" and self.society.attackers:
            agent_ids = self._place_harmful_on_hubs(agent_ids)
        self.graph.assign_ids(agent_ids)

        self.social = SocialEnv(self.graph, scenario, np.random.default_rng(seed + 23))
        self.wolfguard = wolfguard
        self.expose_oracle = bool(expose_oracle)
        self.record_trajectory = bool(record_trajectory)
        cfg = evaluator_config or EvaluatorConfig()
        self.evaluator_config = EvaluatorConfig(
            err_trade_exposure_threshold=_nonnegative_finite(
                cfg.err_trade_exposure_threshold,
                EvaluatorConfig.err_trade_exposure_threshold,
            ),
            intervention_cost_warning=_nonnegative_finite(
                cfg.intervention_cost_warning,
                EvaluatorConfig.intervention_cost_warning,
            ),
            intervention_cost_cooldown=_nonnegative_finite(
                cfg.intervention_cost_cooldown,
                EvaluatorConfig.intervention_cost_cooldown,
            ),
            intervention_cost_block=_nonnegative_finite(
                cfg.intervention_cost_block,
                EvaluatorConfig.intervention_cost_block,
            ),
        )
        if self.wolfguard is not None and baseline is not None:
            self.wolfguard.fit_baseline(baseline)

        self.target_asset = self.society.target_asset
        self._initial_retail_wealth = sum(a.portfolio.initial_wealth for a in self.society.retail)
        self._initial_harmful_wealth = sum(a.portfolio.initial_wealth for a in self.society.attackers)
        self._intervention_cost = 0.0
        self._false_positive_count = 0
        self._false_positive_opportunities = 0
        self._intervention_count = 0
        self._utility_loss = 0.0
        self._oracle_suppressed_assets: set[str] = set()
        self._asset_controls: dict[str, dict[str, float]] = {}
        # WolfGym regulator controls are separate from the legacy defense
        # channel. They are persistent for the remainder of a counterfactual
        # continuation and are included in SimulatorState snapshots.
        self._regulatory_asset_controls: dict[str, dict[str, float]] = {}
        self._regulatory_agent_controls: dict[str, dict[str, float]] = {}
        self._control_decay = {
            "message_filter": 1.0,
            "trade_throttle": 0.85,
            "spoof_filter": 1.0,
            "wash_filter": 1.0,
        }
        self._next_day = 0
        self._metrics = EpisodeMetrics(
            horizon_days=self.scenario.horizon_days,
            target_asset=self.target_asset,
        )
        self._daily_log: list[dict[str, Any]] = []
        self._recent_returns: dict[str, list[float]] = defaultdict(list)
        self._primary_eval_start = self._primary_failure_eval_start_day()
        self._s4_fake_liquidity_score_max = 0.0
        self._last_pre_step_prices = {
            aid: state.price for aid, state in self.market.assets.items()
        }

    # -----------------------------------------------------------------

    def _build_public_entity_ids(self) -> dict[str, str]:
        """Create episode-local opaque ids independent of displayed roles."""
        mapping: dict[str, str] = {}
        used: set[str] = set()
        for agent in self.society.all_agents:
            nonce = 0
            while True:
                raw = (
                    f"{self.scenario.id}|{self.n_society}|{self.alpha:.17g}|"
                    f"{self.seed}|entity|{agent.agent_id}|{nonce}"
                ).encode()
                token = hashlib.blake2s(raw, digest_size=6).hexdigest()
                public_id = f"entity_{token}"
                if public_id not in used:
                    break
                nonce += 1
            mapping[agent.agent_id] = public_id
            used.add(public_id)
        return mapping

    def _build_public_asset_ids(self) -> dict[str, str]:
        """Create episode-local asset pseudonyms for regulator-facing data."""
        mapping: dict[str, str] = {}
        for asset in self.scenario.assets:
            raw = (
                f"{self.scenario.id}|{self.n_society}|{self.alpha:.17g}|"
                f"{self.seed}|asset|{asset.id}"
            ).encode()
            mapping[asset.id] = f"asset_{hashlib.blake2s(raw, digest_size=6).hexdigest()}"
        return mapping

    def public_entity_id(self, internal_id: str) -> str:
        try:
            return self._public_entity_ids[internal_id]
        except KeyError as exc:
            raise ValueError(f"unknown internal entity {internal_id!r}") from exc

    def internal_entity_id(self, public_id: str) -> str:
        try:
            return self._internal_entity_ids[public_id]
        except KeyError as exc:
            raise ValueError(f"unknown public entity {public_id!r}") from exc

    def public_asset_id(self, internal_id: str) -> str:
        try:
            return self._public_asset_ids[internal_id]
        except KeyError as exc:
            raise ValueError(f"unknown internal asset {internal_id!r}") from exc

    def internal_asset_id(self, public_id: str) -> str:
        try:
            return self._internal_asset_ids[public_id]
        except KeyError as exc:
            raise ValueError(f"unknown public asset {public_id!r}") from exc

    def observable_entity_history(self) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of regulator-visible activity telemetry."""
        return tuple(deepcopy(self._observable_entity_history))

    def _place_harmful_on_hubs(self, agent_ids: list[str]) -> list[str]:
        # find top-degree nodes; swap harmful ids into those positions
        n_harm = len(self.society.attackers)
        order = sorted(self.graph.g.degree, key=lambda x: x[1], reverse=True)
        hub_positions = [n for n, _ in order[:n_harm]]
        # node order in agent_ids matches self.graph.agent_nodes order
        node_index = {n: i for i, n in enumerate(self.graph.agent_nodes)}
        ids = list(agent_ids)
        harm_ids = [a.agent_id for a in self.society.attackers]
        # current indices of harmful ids
        current_idx = {ids[i]: i for i in range(len(ids))}
        for hid, hub_node in zip(harm_ids, hub_positions):
            target_pos = node_index[hub_node]
            cur_pos = current_idx[hid]
            if target_pos != cur_pos:
                ids[cur_pos], ids[target_pos] = ids[target_pos], ids[cur_pos]
                # update map
                current_idx[ids[cur_pos]] = cur_pos
                current_idx[ids[target_pos]] = target_pos
        return ids

    # -----------------------------------------------------------------

    def get_state(self) -> SimulatorState:
        """Capture all hidden state, including every component RNG state."""
        return SimulatorState(
            scenario_id=self.scenario.id,
            n_society=self.n_society,
            next_day=self._next_day,
            horizon_days=self.scenario.horizon_days,
            _payload=deepcopy(self.__dict__),
        )

    def set_state(self, state: SimulatorState) -> None:
        """Restore a snapshot without retaining aliases to its payload."""
        self._validate_state_compatibility(state)
        payload = self._validated_payload_copy(state)
        self.__dict__.clear()
        self.__dict__.update(payload)

    @staticmethod
    def _validated_payload_copy(state: SimulatorState) -> dict[str, Any]:
        payload = state.payload_copy()
        if int(payload.get("_next_day", -1)) != state.next_day:
            raise ValueError("snapshot metadata does not match its hidden day")
        payload_scenario = payload.get("scenario")
        if str(getattr(payload_scenario, "id", "")) != state.scenario_id:
            raise ValueError("snapshot metadata does not match its hidden scenario")
        if int(getattr(payload_scenario, "horizon_days", -1)) != state.horizon_days:
            raise ValueError("snapshot metadata does not match its hidden horizon")
        if int(payload.get("n_society", -1)) != state.n_society:
            raise ValueError("snapshot metadata does not match its hidden society size")
        return payload

    def _validate_state_compatibility(self, state: SimulatorState) -> None:
        if state.schema_version != 8:
            raise ValueError(f"unsupported snapshot schema {state.schema_version}")
        if state.scenario_id != self.scenario.id:
            raise ValueError(
                f"snapshot scenario {state.scenario_id!r} does not match "
                f"environment scenario {self.scenario.id!r}"
            )
        if state.n_society != self.n_society:
            raise ValueError(
                f"snapshot society size {state.n_society} does not match "
                f"environment society size {self.n_society}"
            )
        if state.horizon_days != self.scenario.horizon_days:
            raise ValueError(
                f"snapshot horizon {state.horizon_days} does not match "
                f"environment horizon {self.scenario.horizon_days}"
            )

    def clone_state(self) -> SimulatorState:
        return self.get_state().clone()

    def clone(self) -> "WolfSocietySimulator":
        """Return an independent simulator clone at the current day boundary."""
        return deepcopy(self)

    def fork_from_state(self, state: SimulatorState) -> "WolfSocietySimulator":
        """Create a branch whose mutable object graph comes only from ``state``.

        This avoids cloning the caller's current mutable state before restoring
        the requested snapshot. Each call obtains a fresh deep payload copy.
        """
        self._validate_state_compatibility(state)
        branch = type(self).__new__(type(self))
        branch.__dict__.update(self._validated_payload_copy(state))
        return branch

    def set_continuation_seed(self, seed: int) -> None:
        """Reset future stochastic streams for matched counterfactual trials.

        Each component and agent receives its own spawned stream, so suppressing
        one agent does not shift every other agent's random-number sequence.
        """
        agents = list(self.society.all_agents)
        streams = np.random.SeedSequence(int(seed)).spawn(4 + len(agents))
        self.rng = np.random.default_rng(streams[0])
        self.market.rng = np.random.default_rng(streams[1])
        self.social.set_rng(np.random.default_rng(streams[2]))
        self.graph.rng = np.random.default_rng(streams[3])
        for agent, stream in zip(agents, streams[4:]):
            if hasattr(agent, "rng"):
                generator = np.random.default_rng(stream)
                agent.rng = generator
                # Instrumented retail wrappers may delegate decisions to a
                # contained base agent. Keep that alias on the same stream.
                base = getattr(agent, "base", None)
                if base is not None and hasattr(base, "rng"):
                    base.rng = generator

    def assert_counterfactual_rng_contract(self) -> None:
        """Reject components whose future randomness is not simulator-local."""
        if self.wolfguard is not None:
            raise ValueError(
                "counterfactual label rollouts require wolfguard=None; "
                "candidate regulation is applied by RegulatoryActionExecutor"
            )
        from wolfgym.core.agents.llm import RuleFallbackBackend

        for agent in self.society.all_agents:
            backend = getattr(agent, "backend", None)
            if backend is not None and not isinstance(backend, RuleFallbackBackend):
                raise ValueError(
                    "counterfactual label rollouts cannot call a non-deterministic "
                    f"LLM backend (agent={agent.agent_id!r})"
                )
            rng = getattr(agent, "rng", None)
            if rng is not None and not isinstance(rng, np.random.Generator):
                raise TypeError(f"agent {agent.agent_id!r} has an unsupported RNG")

    def run(self) -> EpisodeResult:
        """Run from the current state to the configured episode horizon."""
        while self._next_day < self.scenario.horizon_days:
            self.step_day()
        return self._episode_result()

    def step_day(
        self,
        pre_step: Callable[["WolfSocietySimulator", int], None] | None = None,
    ) -> dict[str, Any]:
        """Advance exactly one day under the versioned market transition.

        ``pre_step`` runs after old controls decay and before ``begin_day`` and
        agent decisions.  WolfGym uses this narrow hook to apply a centralized
        regulatory action without adding action semantics to scenario code.
        """
        if self._next_day >= self.scenario.horizon_days:
            raise StopIteration("simulation horizon has been reached")

        day = self._next_day
        metrics = self._metrics
        recent_returns = self._recent_returns
        primary_eval_start_day = self._primary_eval_start

        self._oracle_suppressed_assets.clear()
        self._decay_asset_controls()
        prices = {aid: s.price for aid, s in self.market.assets.items()}
        self._last_pre_step_prices = dict(prices)
        recent_ret = {
            aid: float(np.mean(recent_returns[aid][-3:]))
            if recent_returns[aid] else 0.0
            for aid in self.market.assets
        }

        # --- WolfGuard chooses interventions from prior close signals ---
        actions: dict[str, dict] = {}
        public_summary: dict[str, Any] | None = None
        oracle_actions: dict[str, dict] = {}
        if self.record_trajectory or self.wolfguard is not None:
            public_summary = self._system_summary(day, recent_ret, include_oracle=False)
        if self.record_trajectory:
            oracle_actions = self._oracle_label_actions(day, public_summary or {})
        if self.wolfguard is not None:
            summary = public_summary or self._system_summary(day, recent_ret, include_oracle=False)
            if self.expose_oracle:
                summary = self._system_summary(day, recent_ret, include_oracle=True)
            actions = validate_interventions(
                self.wolfguard.decide(day, summary),
                set(self.market.assets),
            )
            self._apply_defense(day, actions)
        if pre_step is not None:
            pre_step(self, day)

        # Reset daily microstructure counters only after the defense has
        # observed the previous close. Otherwise spoof/wash signals are
        # zeroed before public policies can react to them.
        # Traders also observe the previous close, not counters reset to zero.
        observation = self._build_observation(day, prices, recent_ret)
        self.market.begin_day()
        self.market.prepare_normal_quotes(self._combined_controls)

        # --- agent decisions ---
        # Attack quotes are displayed before retail reacts, not observed only
        # on tomorrow's close after they have already been cancelled.
        attack_orders: list[Order] = []
        attack_messages: list[Message] = []
        for ag in self.society.attackers:
            if getattr(ag, "target_asset", None) in self._oracle_suppressed_assets:
                continue
            o, m = ag.decide(day, observation)
            attack_orders.extend(o)
            attack_messages.extend(m)
        attack_orders, attack_messages = self._apply_public_market_controls(attack_orders, attack_messages)
        depth = self.market.displayed_depth(attack_orders)
        for asset, imbalance in depth.items():
            observation["market"][asset]["depth_imbalance"] = imbalance
        orders: list[Order] = []
        messages: list[Message] = []
        for ag in [*self.society.retail, *self.society.market_makers]:
            if hasattr(ag, "decide"):
                o, m = ag.decide(day, observation)
                orders.extend(o)
                messages.extend(m)
        # Trusted accounting telemetry only; never an input to agent decisions.
        retail_ids = {agent.agent_id for agent in self.society.retail}
        requested_retail_notional = sum(
            order.quantity * prices[order.asset] for order in orders
            if order.agent_id in retail_ids
        )
        orders, messages = self._apply_public_market_controls(orders, messages)
        submitted_retail_notional = sum(
            order.quantity * prices[order.asset] for order in orders
            if order.agent_id in retail_ids
        )
        orders.extend(attack_orders)
        messages.extend(attack_messages)
        first_trade = len(self.market.trades)
        # --- market clearing ---
        for step in range(self.intraday_steps):
            step_orders = orders if self.intraday_steps == 1 else orders[step::self.intraday_steps]
            # The clock and exogenous price process advance even with no orders.
            # MarketEnv owns validation and both sides of settlement.
            self.market.submit_orders(day, step=step, orders=step_orders)
        # Record only behavior that has now entered the public market/social
        # channels. This deliberately occurs after order submission.

        # --- social propagation ---
        market_returns = {}
        for aid, s in self.market.assets.items():
            market_returns[aid] = (s.price - s.last_price) / max(s.last_price, 1e-6)
            recent_returns[aid].append(market_returns[aid])
        self.social.step(day, messages, market_returns)

        self.market.end_day(day)
        # Include the normal quotes' end-of-day expiration in public telemetry.
        self._record_observable_entity_activity(day, orders, messages)

        # --- metrics ---
        comp = self._collapse_components(day)
        mechanism = self._mechanism_components(day)
        score = compute_collapse_score(comp)
        raw_primary = primary_failure_signal(self.scenario.id, comp, mechanism)
        if self._is_s4_scenario():
            raw_primary = self._ordered_s4_primary_signal(
                raw_primary,
                mechanism,
                self._s4_fake_liquidity_score_max,
            )
            if day >= primary_eval_start_day:
                self._s4_fake_liquidity_score_max = float(
                    raw_primary.get(
                        "ordered_fake_liquidity_score_max",
                        self._s4_fake_liquidity_score_max,
                    )
                )
        primary = dict(raw_primary)
        primary["raw_primary_failure_score"] = float(raw_primary["primary_failure_score"])
        primary["evaluation_start_day"] = primary_eval_start_day
        if day < primary_eval_start_day:
            primary["triggered"] = False
            primary["primary_failure_score"] = 0.0
            primary["suppressed_by_grace_period"] = True
        metrics.daily_collapse_score.append(score)
        metrics.daily_components.append(comp)
        metrics.primary_metric = str(primary["primary_metric"])
        metrics.daily_primary_failure_score.append(float(primary["primary_failure_score"]))
        metrics.daily_primary_failure.append(1.0 if primary["triggered"] else 0.0)
        metrics.primary_failure_score_max = max(
            metrics.primary_failure_score_max,
            float(primary["primary_failure_score"]),
        )
        metrics.max_collapse_score = max(metrics.max_collapse_score, score)
        metrics.price_dislocation_max = max(metrics.price_dislocation_max,
                                            comp.get("price_dislocation", 0.0))
        metrics.liquidity_stress_max = max(metrics.liquidity_stress_max,
                                           comp.get("liquidity_stress", 0.0))
        metrics.social_cascade_peak = max(metrics.social_cascade_peak,
                                          comp.get("social_cascade", 0.0))
        metrics.wash_share_max = max(metrics.wash_share_max, mechanism["wash_share"])
        metrics.volume_distortion_max = max(
            metrics.volume_distortion_max,
            mechanism["volume_distortion"],
        )
        metrics.volume_signal_z_max = max(metrics.volume_signal_z_max, mechanism["volume_signal_z"])
        metrics.cancel_rate_max = max(metrics.cancel_rate_max, mechanism["cancel_rate"])
        metrics.spoof_depth_to_liquidity_max = max(
            metrics.spoof_depth_to_liquidity_max,
            mechanism["spoof_depth_to_liquidity"],
        )
        metrics.withdrawal_loss_max = max(
            metrics.withdrawal_loss_max,
            mechanism["withdrawal_loss"],
        )

        if metrics.collapse_day is None and collapse_triggered(comp):
            metrics.collapse_day = day
            metrics.collapse_rate = 1.0
        if metrics.primary_failure_day is None and primary["triggered"]:
            metrics.primary_failure_day = day
            metrics.primary_failure_rate = 1.0

        entry = {
            "day": day,
            "prices": {a: float(s.price) for a, s in self.market.assets.items()},
            "fundamentals": {a: float(s.fundamental) for a, s in self.market.assets.items()},
            "components": comp,
            "mechanism_components": mechanism,
            "primary_failure": primary,
            "collapse_score": score,
            "wolfguard_actions": actions,
        }
        close_prices = entry["prices"]
        retail_initial = sum(agent.portfolio.initial_wealth for agent in self.society.retail)
        retail_wealth = sum(agent.portfolio.mark_to_market(close_prices) for agent in self.society.retail)
        individual_returns = [
            agent.portfolio.mark_to_market(close_prices) / agent.portfolio.initial_wealth - 1
            for agent in self.society.retail
        ]
        role_wealth: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for agent in self.society.retail:
            role_wealth[agent.sub_role][0] += agent.portfolio.mark_to_market(close_prices)
            role_wealth[agent.sub_role][1] += agent.portfolio.initial_wealth
        # Count retail sides at the pre-trade mark, matching the order denominator.
        executed_retail_notional = sum(
            trade.quantity * prices[trade.asset]
            * (int(trade.buyer_id in retail_ids) + int(trade.seller_id in retail_ids))
            for trade in self.market.trades[first_trade:]
        )
        entry["retail_outcomes"] = {
            "wealth": retail_wealth,
            "initial_wealth": retail_initial,
            "return": retail_wealth / retail_initial - 1 if retail_initial else 0.0,
            "individual_return_q10": float(np.quantile(individual_returns, .1)) if individual_returns else 0.0,
            "role_returns": {role: wealth / initial - 1 for role, (wealth, initial) in role_wealth.items()},
            "requested_notional": requested_retail_notional,
            "blocked_notional": max(0.0, requested_retail_notional - submitted_retail_notional),
            "submitted_notional": submitted_retail_notional,
            "executed_notional": executed_retail_notional,
            "unexecuted_notional": max(0.0, submitted_retail_notional - executed_retail_notional),
        }
        if self.record_trajectory:
            entry["observation"] = public_summary or self._system_summary(day, recent_ret, include_oracle=False)
            entry["oracle_actions"] = oracle_actions
        self._daily_log.append(entry)
        self._next_day += 1
        return entry

    def _record_observable_entity_activity(
        self,
        day: int,
        orders: list[Order],
        messages: list[Message],
    ) -> None:
        """Record public order/message behavior without ground-truth labels."""
        per_entity: dict[str, dict[str, float]] = defaultdict(lambda: {
            "message_intensity": 0.0,
            "potential_message_reach": 0.0,
            "order_quantity": 0.0,
            "order_count": 0.0,
            "cancellation_count": 0.0,
        })
        degrees = self.graph.degrees()
        max_degree = max(degrees.values(), default=1)
        for message in messages:
            stats = per_entity[message.sender_id]
            intensity = max(0.0, float(message.intensity))
            stats["message_intensity"] += intensity
            stats["potential_message_reach"] += (
                intensity * degrees.get(message.sender_id, 0) / max(max_degree, 1)
            )
        for order in orders:
            stats = per_entity[order.agent_id]
            stats["order_quantity"] += max(0.0, float(order.quantity))
            stats["order_count"] += 1.0
        for entity_id, activity in self.market.normal_quote_activity.items():
            for key, value in activity.items():
                per_entity[entity_id][key] += value
        for agent_id, count in self.market.cancellations_by_agent_today.items():
            # This event log is populated by market clearing after cancellation,
            # so telemetry never reads an order's hidden mechanism flags.
            per_entity[agent_id]["cancellation_count"] += float(count)
        self._observable_entity_history.append({
            "day": int(day),
            "entities": {entity_id: dict(stats) for entity_id, stats in per_entity.items()},
        })

    def _episode_result(self) -> EpisodeResult:
        """Finalize the legacy episode metrics after the configured horizon."""
        metrics = self._metrics
        prices = {asset_id: asset.price for asset_id, asset in self.market.assets.items()}
        retail_wealth_now = sum(a.portfolio.mark_to_market(prices) for a in self.society.retail)
        metrics.retail_loss_30d = self._initial_retail_wealth - retail_wealth_now
        metrics.retail_loss_pct_30d = metrics.retail_loss_30d / max(self._initial_retail_wealth, 1e-6)
        harmful_now = sum(a.portfolio.mark_to_market(prices) for a in self.society.attackers)
        metrics.harmful_profit = harmful_now - self._initial_harmful_wealth
        if abs(metrics.retail_loss_30d) > 1e-6:
            metrics.wealth_transfer = metrics.harmful_profit / abs(metrics.retail_loss_30d)
        metrics.intervention_cost = self._intervention_cost
        metrics.utility_loss = self._utility_loss
        denom = max(self._false_positive_opportunities, 1)
        metrics.false_positive_rate = self._false_positive_count / denom

        return EpisodeResult(
            scenario_id=self.scenario.id,
            n_society=self.n_society,
            alpha=self.alpha,
            seed=self.seed,
            metrics=metrics,
            target_asset=self.target_asset,
            daily_log=self._daily_log,
            config_snapshot={
                "simulator_version": SIMULATOR_VERSION,
                "target_asset": self.target_asset,
                "n_harmful": self.society.n_harmful,
                "n_market_makers": len(self.society.market_makers),
                "n_all_agents": len(self.society.all_agents),
                "harmful_wealth_multiplier": self.scenario.harmful_wealth_multiplier,
                "initial_harmful_capital_share": self._initial_harmful_wealth / max(
                    self._initial_retail_wealth + self._initial_harmful_wealth, 1e-12,
                ),
                "placement": self.society.placement,
                "intraday_steps": self.intraday_steps,
            },
        )

    # ---------------------------------------------------------------- helpers

    def _primary_failure_eval_start_day(self) -> int:
        """First day counted for scenario primary-failure triggers.

        S3 spoofing leaves an observable footprint only after the first close.
        Counting day-0/day-1 spoof prints as official failure makes every
        public defense lose before it can observe any public signal.
        """
        scenario = str(self.scenario.id).lower()
        if scenario.startswith("s3") or "spoof" in scenario:
            return 2
        if scenario.startswith("s4") or "wash" in scenario:
            return 6
        return 0

    def _is_s4_scenario(self) -> bool:
        scenario = str(self.scenario.id).lower()
        return scenario.startswith("s4") or "wash" in scenario

    def _ordered_s4_primary_signal(
        self,
        raw_primary: dict[str, Any],
        mechanism: dict[str, float],
        fake_liquidity_score_max: float,
    ) -> dict[str, Any]:
        """Evaluate S4 as an ordered fake-liquidity then withdrawal mechanism."""
        thresholds = raw_primary.get("thresholds", {}) or {}
        wash_threshold = max(float(thresholds.get("wash_share", 0.45)), 1e-12)
        distortion_threshold = max(float(thresholds.get("volume_distortion", 0.8)), 1e-12)
        withdrawal_threshold = max(float(thresholds.get("withdrawal_loss", 0.06)), 1e-12)
        fake_score_today = min(
            float(mechanism.get("wash_share", 0.0)) / wash_threshold,
            float(mechanism.get("volume_distortion", 0.0)) / distortion_threshold,
        )
        fake_score_max = max(float(fake_liquidity_score_max), max(0.0, fake_score_today))
        withdrawal_score = max(
            0.0,
            float(mechanism.get("withdrawal_loss", 0.0)) / withdrawal_threshold,
        )
        ordered_score = float(min(fake_score_max, withdrawal_score))
        out = dict(raw_primary)
        components = dict(out.get("components", {}) or {})
        components["ordered_fake_liquidity_score_max"] = fake_score_max
        components["withdrawal_score"] = withdrawal_score
        out["components"] = components
        out["ordered_fake_liquidity_score_max"] = fake_score_max
        out["primary_metric_value"] = ordered_score
        out["primary_failure_score"] = ordered_score
        out["triggered"] = ordered_score >= 1.0
        return out

    def _decay_asset_controls(self) -> None:
        if not self._asset_controls:
            return
        kept: dict[str, dict[str, float]] = {}
        for asset, controls in self._asset_controls.items():
            decayed = {
                key: float(np.clip(value * self._control_decay.get(key, 1.0), 0.0, 1.0))
                for key, value in controls.items()
            }
            if max(decayed.values(), default=0.0) > 1e-4:
                kept[asset] = decayed
        self._asset_controls = kept

    def _build_observation(self, day, prices, recent_ret):
        market_view = self.market.snapshot()
        # volume_z relative to running mean
        volume_z: dict[str, float] = {}
        for aid, s in self.market.assets.items():
            # Retail sees the public tape, not oracle-cleaned wash-free volume.
            # Compare the last close with earlier closes, excluding itself.
            hist = s.history["volume"][:-1]
            if len(hist) > 3:
                arr = np.array(hist, dtype=float)
                mu, sd = arr.mean(), arr.std() + 1e-6
                volume_z[aid] = float((s.volume_today - mu) / sd)
            else:
                volume_z[aid] = 0.0
        # social per-agent exposure is fetched lazily; precompute per asset for retail
        observation = {
            "day": day,
            "prices": prices,
            "market": market_view,
            "recent_return": recent_ret,
            # Completed closes only; immutable sequences prevent a policy
            # from changing another participant's historical observations.
            "return_history": {asset: tuple(values) for asset, values in self._recent_returns.items()},
            "volume_z": volume_z,
            "social_env": self.social,
        }
        return observation

    def _system_summary(self, day, recent_ret,
                        include_oracle: bool | None = None) -> dict[str, Any]:
        market_view = self.market.snapshot()
        social_view = {a: self.social.asset_signal(a) for a in self.market.assets}
        # Public WolfGuard observations must not expose simulator ground-truth
        # harmful-source labels. Hidden oracle pressure is provided separately
        # through oracle_view for the non-ranked oracle policy.
        if not (self.expose_oracle if include_oracle is None else include_oracle):
            for signals in social_view.values():
                signals.pop("harmful_msg_share", None)
        summary = {
            "day": day,
            "market": market_view,
            "social": social_view,
            "recent_return": recent_ret,
        }
        should_include_oracle = self.expose_oracle if include_oracle is None else include_oracle
        if should_include_oracle:
            summary["oracle_view"] = self._oracle_view(day)
        return summary

    def _oracle_label_actions(self, day: int,
                              public_summary: dict[str, Any]) -> dict[str, dict]:
        """Return internal oracle labels without exposing them to a policy."""
        from wolfgym.core.defense.baselines import OracleWolfGuardPolicy

        summary = dict(public_summary)
        summary["oracle_view"] = self._oracle_view(day)
        return validate_interventions(
            OracleWolfGuardPolicy().decide(day, summary),
            set(self.market.assets),
        )

    def _oracle_view(self, day: int) -> dict[str, dict[str, float]]:
        """Ground-truth per-asset harmful pressure. Used only by the Oracle
        defense baseline (upper bound); regular submissions ignore this key.
        """
        per_asset_harmful_eq = {a: 0.0 for a in self.market.assets}
        per_asset_total_eq = {a: 1e-9 for a in self.market.assets}
        per_asset_harmful_count = {a: 0 for a in self.market.assets}
        per_asset_active_count = {a: 0 for a in self.market.assets}
        prices = {aid: s.price for aid, s in self.market.assets.items()}
        for ag in self.society.all_agents:
            equity = ag.portfolio.mark_to_market(prices)
            asset = getattr(ag, "target_asset", self.target_asset)
            per_asset_total_eq[asset] += max(equity, 0.0)
            if getattr(ag, "is_harmful", False):
                per_asset_harmful_eq[asset] += max(equity, 0.0)
                per_asset_harmful_count[asset] += 1
                if self._harmful_agent_active(ag, day):
                    per_asset_active_count[asset] += 1
        return {
            a: {
                "harmful_pressure": float(
                    per_asset_harmful_eq[a] / per_asset_total_eq[a]
                ),
                "harmful_agent_count": float(per_asset_harmful_count[a]),
                "active_harmful_count": float(per_asset_active_count[a]),
                "harmful_agent_share": float(
                    per_asset_harmful_count[a] / max(self.society.n_total, 1)
                ),
                "active_harmful_share": float(
                    per_asset_active_count[a] / max(self.society.n_total, 1)
                ),
                "is_attack_target": float(
                    a == self.target_asset and self.society.n_harmful > 0
                ),
            }
            for a in self.market.assets
        }

    def _harmful_agent_active(self, agent, day: int) -> bool:
        if not getattr(agent, "is_harmful", False):
            return False
        windows = (
            "accumulate_days", "promote_days", "dump_days", "sell_days",
            "wash_days", "withdraw_days",
        )
        found_window = False
        for attr in windows:
            window = getattr(agent, attr, None)
            if window is None:
                continue
            found_window = True
            start, end = int(window[0]), int(window[1])
            if start <= day <= end:
                return True
        return not found_window

    def _apply_defense(self, day: int, actions: dict[str, dict]) -> None:
        cfg = self.evaluator_config
        for asset, act in actions.items():
            action = act["action"]
            if action == "none":
                continue
            self._intervention_count += 1
            cost = 0.0
            reason = act.get("reason", "")
            if action == "warning":
                cost = cfg.intervention_cost_warning
                self._set_asset_controls(asset, action, act["risk"])
                for r in self.society.retail:
                    r.warning_level[asset] = max(r.warning_level.get(asset, 0.0), act["risk"])
            elif action == "cooldown":
                cost = cfg.intervention_cost_cooldown
                self._set_asset_controls(asset, action, act["risk"])
                for r in self.society.retail:
                    r.cooldown_until[asset] = day  # skip just today
                    r.warning_level[asset] = max(r.warning_level.get(asset, 0.0), act["risk"])
            elif action == "block":
                cost = cfg.intervention_cost_block
                if reason == "oracle":
                    self._oracle_suppressed_assets.add(asset)
                    for r in self.society.retail:
                        r.blocked_today[asset] = True
                        r.warning_level[asset] = max(r.warning_level.get(asset, 0.0), act["risk"])
                    self._intervention_cost += cost
                    self._utility_loss += cost
                    continue
                self._set_asset_controls(asset, action, act["risk"])
                # Block only suspected erroneous trades: high harmful exposure
                for r in self.society.retail:
                    self._false_positive_opportunities += 1
                    bb = r.last_belief_breakdown.get(asset, {})
                    if bb.get("harmful_exposure", 0.0) > cfg.err_trade_exposure_threshold:
                        r.blocked_today[asset] = True
                    else:
                        # over-blocked: false positive on this retail agent
                        if act["risk"] > 0.9:
                            self._false_positive_count += 1
            self._intervention_cost += cost
            self._utility_loss += cost

    def _set_asset_controls(self, asset: str, action: str, risk: float) -> None:
        """Translate a public intervention into market/platform controls.

        These controls do not expose oracle labels to a policy. They model
        public action semantics: warnings reduce message credibility, cooldowns
        throttle asset activity, and blocks/filtering suppress suspicious order
        patterns such as spoof layers and wash trades.
        """
        risk = float(np.clip(risk, 0.0, 1.0))
        risk_factor = 0.5 + 0.5 * risk
        base = {
            "warning": {
                "message_filter": 0.60,
                "trade_throttle": 0.00,
                "spoof_filter": 0.70,
                "wash_filter": 0.70,
            },
            "cooldown": {
                "message_filter": 0.97,
                "trade_throttle": 0.00,
                "spoof_filter": 0.98,
                "wash_filter": 0.98,
            },
            "block": {
                "message_filter": 1.00,
                "trade_throttle": 0.20,
                "spoof_filter": 1.00,
                "wash_filter": 1.00,
            },
        }.get(action)
        if base is None:
            return
        current = self._asset_controls.setdefault(asset, {
            "message_filter": 0.0,
            "trade_throttle": 0.0,
            "spoof_filter": 0.0,
            "wash_filter": 0.0,
        })
        for key, value in base.items():
            if key in {"message_filter", "spoof_filter", "wash_filter"}:
                scaled = value
            else:
                scaled = value * risk_factor
            current[key] = max(current[key], float(np.clip(scaled, 0.0, 1.0)))

    def set_persistent_regulatory_controls(
        self,
        scope: Literal["asset", "agent"],
        target_id: str,
        controls: dict[str, float],
    ) -> None:
        """Install explicit controls that persist to the episode horizon.

        This state is separate from decaying legacy defense controls. It is
        snapshotted with the simulator and therefore has unambiguous duration
        in counterfactual branches.
        """
        valid_keys = {
            "message_filter", "trade_throttle", "spoof_filter", "wash_filter",
        }
        if set(controls) != valid_keys:
            raise ValueError(f"regulatory controls must define {sorted(valid_keys)}")
        bounded = {
            key: float(np.clip(float(value), 0.0, 1.0))
            for key, value in controls.items()
        }
        if scope == "asset":
            if target_id not in self.market.assets:
                raise ValueError(f"unknown asset target {target_id!r}")
            registry = self._regulatory_asset_controls
        elif scope == "agent":
            if target_id not in {agent.agent_id for agent in self.society.all_agents}:
                raise ValueError(f"unknown agent target {target_id!r}")
            registry = self._regulatory_agent_controls
        else:
            raise ValueError(f"unknown regulatory scope {scope!r}")
        current = registry.setdefault(target_id, {key: 0.0 for key in valid_keys})
        for key, value in bounded.items():
            current[key] = max(current[key], value)

    def clear_persistent_regulatory_controls(
        self,
        scope: Literal["asset", "agent"],
        target_id: str,
    ) -> None:
        """Remove a WolfGym persistent control without touching legacy defense state."""
        if scope == "asset":
            registry = self._regulatory_asset_controls
        elif scope == "agent":
            registry = self._regulatory_agent_controls
        else:
            raise ValueError(f"unknown regulatory scope {scope!r}")
        if target_id not in registry:
            raise ValueError(f"no active regulatory control for {scope} {target_id!r}")
        del registry[target_id]

    def persistent_regulatory_controls(
        self,
    ) -> dict[Literal["asset", "agent"], dict[str, dict[str, float]]]:
        """Return a defensive view for the WolfGym wrapper and observation builder."""
        return {
            "asset": deepcopy(self._regulatory_asset_controls),
            "agent": deepcopy(self._regulatory_agent_controls),
        }

    def _combined_controls(self, asset: str, agent_id: str) -> dict[str, float]:
        sources = (
            self._asset_controls.get(asset, {}),
            self._regulatory_asset_controls.get(asset, {}),
            self._regulatory_agent_controls.get(agent_id, {}),
        )
        keys = ("message_filter", "trade_throttle", "spoof_filter", "wash_filter")
        return {key: max(float(source.get(key, 0.0)) for source in sources) for key in keys}

    def _apply_public_market_controls(
        self,
        orders: list[Order],
        messages: list[Message],
    ) -> tuple[list[Order], list[Message]]:
        if not (
            self._asset_controls
            or self._regulatory_asset_controls
            or self._regulatory_agent_controls
        ):
            return orders, messages

        filtered_messages: list[Message] = []
        for message in messages:
            controls = self._combined_controls(message.asset, message.sender_id)
            if not any(controls.values()):
                filtered_messages.append(message)
                continue
            scale = 1.0 - float(controls.get("message_filter", 0.0))
            if scale <= 1e-6:
                continue
            filtered_messages.append(Message(
                sender_id=message.sender_id,
                asset=message.asset,
                sentiment=message.sentiment,
                intensity=message.intensity * scale,
                is_harmful=message.is_harmful,
                is_bot=message.is_bot,
                day=message.day,
                message_id=message.message_id,
                kind=message.kind,
                root_sender_id=message.root_sender_id,
                confidence=message.confidence,
                social_proof=message.social_proof,
                text=message.text,
            ))

        filtered_orders: list[Order] = []
        for order in orders:
            controls = self._combined_controls(order.asset, order.agent_id)
            if not any(controls.values()):
                filtered_orders.append(order)
                continue
            if order.is_spoof:
                scale = 1.0 - float(controls.get("spoof_filter", 0.0))
            elif order.is_wash:
                scale = 1.0 - float(controls.get("wash_filter", 0.0))
            else:
                scale = 1.0 - float(controls.get("trade_throttle", 0.0))
            if scale <= 1e-6:
                continue
            filtered_orders.append(Order(
                agent_id=order.agent_id,
                asset=order.asset,
                side=order.side,
                quantity=order.quantity * scale,
                is_spoof=order.is_spoof,
                cancel_after_steps=order.cancel_after_steps,
                is_wash=order.is_wash,
                counterparty_id=order.counterparty_id,
                is_harmful=order.is_harmful,
            ))
        return filtered_orders, filtered_messages

    def _collapse_components(self, day: int) -> dict[str, float]:
        s = self.market.assets[self.target_asset]
        price_disloc = abs(s.price - s.fundamental) / max(s.fundamental, 1e-6)
        spread_z = (s.spread_bps / max(self.market.base_spread_bps, 1e-6)) - 1.0
        spoof_depth = s.spoof_buy_size + s.spoof_sell_size
        cancel_rate = s.cancel_count / max(s.order_count, 1)
        spoof_stress = cancel_rate * np.log1p(spoof_depth / max(s.base_liquidity, 1e-6))

        prices = {aid: a.price for aid, a in self.market.assets.items()}
        retail_now = sum(a.portfolio.mark_to_market(prices) for a in self.society.retail)
        retail_loss_pct = max(0.0, (self._initial_retail_wealth - retail_now)
                              / max(self._initial_retail_wealth, 1e-6))

        n_retail = max(len(self.society.retail), 1)
        cascade = float(len(self.social.state.cascade_size.get(self.target_asset, set()))) / n_retail

        harmful_now = sum(a.portfolio.mark_to_market(prices) for a in self.society.attackers)
        harmful_profit = harmful_now - self._initial_harmful_wealth
        # Wealth transferred from retail population to harmful actors,
        # expressed as a fraction of total initial retail wealth.
        wt = harmful_profit / max(self._initial_retail_wealth, 1e-6)
        wt = float(np.clip(wt, -1.0, 1.0))

        return {
            "price_dislocation": float(price_disloc),
            # New unified market separates market quality from attack activity.
            # Spread is still a model-derived price-gap proxy, not an independent
            # empirically validated liquidity-loss endpoint.
            "liquidity_stress": float(max(spread_z, 0.0) if self.scenario.id == "stock_market_v2"
                                      else max(spread_z, spoof_stress, 0.0)),
            "retail_loss": float(retail_loss_pct),
            "social_cascade": float(min(cascade, 1.0)),
            "wealth_transfer": float(wt),
        }

    def _mechanism_components(self, day: int) -> dict[str, float]:
        s = self.market.assets[self.target_asset]
        wash_volume = max(0.0, s.volume_today - s.real_volume_today)
        expected_clean_volume = self._expected_clean_volume(s)
        raw_clean_volume_floor = 0.05 * max(s.base_liquidity, 1.0)
        raw_volume_distortion = wash_volume / max(s.real_volume_today, raw_clean_volume_floor, 1.0)
        volume_distortion = wash_volume / max(expected_clean_volume, 1.0)
        cancel_rate = s.cancel_count / max(s.order_count, 1)
        spoof_depth = s.spoof_buy_size + s.spoof_sell_size
        hist = np.array(s.history["volume"], dtype=float)
        if hist.size > 3:
            volume_signal_z = (s.volume_today - float(hist.mean())) / (float(hist.std()) + 1e-6)
        else:
            volume_signal_z = 0.0

        withdrawal_loss = 0.0
        wash_cfg = self.scenario.attackers.get("wash_trading", {}) or {}
        window = wash_cfg.get("withdraw_days")
        if window is not None:
            start, end = int(window[0]), int(window[1])
            if start <= day <= end:
                withdrawal_loss = max(0.0, (s.last_price - s.price) / max(s.last_price, 1e-6))

        return {
            "wash_share": float(
                wash_volume / max(wash_volume + expected_clean_volume, 1.0)
            ),
            "volume_distortion": float(volume_distortion),
            "raw_wash_share": float(
                1.0 - s.real_volume_today / s.volume_today
                if s.volume_today > 0 else 0.0
            ),
            "raw_volume_distortion": float(raw_volume_distortion),
            "expected_clean_volume": float(expected_clean_volume),
            "volume_signal_z": float(max(0.0, volume_signal_z)),
            "cancel_rate": float(cancel_rate),
            "spoof_depth_to_liquidity": float(spoof_depth / max(s.base_liquidity, 1e-6)),
            "withdrawal_loss": float(withdrawal_loss),
        }

    def _expected_clean_volume(self, s) -> float:
        """Stable clean-volume baseline for fake-liquidity ratios.

        S4 interventions can legitimately reduce retail participation. Using
        same-day real volume as the denominator then makes successful warnings
        look like worse wash trading. Anchor the denominator to recent clean
        real volume and a small liquidity floor instead.
        """
        hist = [float(v) for v in s.history["real_volume"] if float(v) > 1e-9]
        hist_baseline = float(np.median(hist[-5:])) if hist else 0.0
        liquidity_floor = 0.10 * max(float(s.base_liquidity), 1.0)
        return float(max(hist_baseline, liquidity_floor, 1.0))
