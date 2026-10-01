"""Order types and stylized market with depth, spread, and short-horizon dynamics.

This is intentionally not a full order-book simulator. Each day runs a fixed
number of clearing rounds. Displayed attack quotes survive one reaction round:
opposite real orders may fill them before the remainder is cancelled. This is
a discrete quote-and-reaction model, not a continuous limit-order exchange.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from wolfgym.core.agents.base import Portfolio

Side = Literal["buy", "sell"]


@dataclass
class Order:
    agent_id: str
    asset: str
    side: Side
    quantity: float
    is_spoof: bool = False
    cancel_after_steps: int = 0     # 0 = real order
    is_wash: bool = False
    counterparty_id: str | None = None  # for wash trades, the colluder
    is_harmful: bool = False        # comes from a harmful agent


@dataclass
class TradeRecord:
    day: int
    step: int
    asset: str
    buyer_id: str
    seller_id: str
    quantity: float
    price: float
    is_wash: bool = False
    buyer_is_harmful: bool = False
    seller_is_harmful: bool = False


@dataclass
class AssetState:
    asset_id: str
    fundamental: float
    fundamental_vol: float
    base_liquidity: float
    daily_execution_liquidity: float
    price: float
    last_price: float = 0.0
    volume_today: float = 0.0
    real_volume_today: float = 0.0   # excludes wash
    spoof_buy_size: float = 0.0
    spoof_sell_size: float = 0.0
    cancel_count: int = 0
    order_count: int = 0
    spread_bps: float = 20.0
    displayed_buy_size: float = 0.0
    displayed_sell_size: float = 0.0
    price_noise_multiplier: float = 0.5
    volatility_alpha: float = 0.0
    volatility_beta: float = 0.0
    volume_variance_elasticity: float = 0.0
    volume_lookback: int = 10
    volume_log_ratio_clip: float = 2.0
    base_conditional_variance: float = 0.0
    previous_base_price_noise: float = 0.0

    history: dict[str, list[float]] = field(default_factory=lambda: {
        "price": [],
        "fundamental": [],
        "volume": [],
        "real_volume": [],
        "spread_bps": [],
        "depth_imbalance": [],
        "cancel_rate": [],
        "wash_share": [],
    })


class MarketEnv:
    """Multi-asset market with daily clearing.

    Each clearing round applies bounded impact from executed net flow, noise,
    and mean reversion. Noise and drift are scaled by rounds per day.
    Spoof and wash do NOT enter ``net_pressure`` directly (they were not real
    intent), but they leave observable footprints (depth imbalance, cancel
    rate, wash share) that other agents can read.
    """

    def __init__(self, scenario, rng: np.random.Generator,
                 accounts: dict[str, "Portfolio"], market_maker_ids: list[str],
                 liquidity_scale: float = 1.0):
        self.rng = rng
        self.accounts = accounts
        self.market_maker_ids = tuple(market_maker_ids)
        self.rounds_per_day = max(1, int(scenario.market_makers.get("intraday_steps", 1)))
        if any(account_id not in accounts for account_id in market_maker_ids):
            raise ValueError("every market maker must have a settlement account")
        for account in accounts.values():
            balances = (account.cash, *account.holdings.values())
            if not all(np.isfinite(value) and value >= 0 for value in balances):
                raise ValueError("initial account balances must be finite and nonnegative")
        self.order_adjustments: list[dict] = []
        self.assets: dict[str, AssetState] = {}
        for a in scenario.assets:
            if not all(np.isfinite(value) and value > 0
                       for value in (a.fundamental, a.initial_liquidity, liquidity_scale)):
                raise ValueError("initial price and liquidity must be positive and finite")
            if not np.isfinite(a.fundamental_vol) or a.fundamental_vol < 0:
                raise ValueError("fundamental volatility must be finite and nonnegative")
            execution_liquidity = (
                a.initial_liquidity
                if a.daily_execution_liquidity is None
                else a.daily_execution_liquidity
            )
            if not np.isfinite(execution_liquidity) or execution_liquidity <= 0:
                raise ValueError("daily execution liquidity must be positive and finite")
            self._validate_volatility_config(a)
            unconditional_variance = (a.fundamental_vol * a.price_noise_multiplier) ** 2
            self.assets[a.id] = AssetState(
                asset_id=a.id,
                fundamental=a.fundamental,
                fundamental_vol=a.fundamental_vol,
                base_liquidity=a.initial_liquidity * liquidity_scale,
                daily_execution_liquidity=execution_liquidity * liquidity_scale,
                price=a.fundamental,
                last_price=a.fundamental,
                spread_bps=scenario.market_makers["base_spread_bps"],
                price_noise_multiplier=a.price_noise_multiplier,
                volatility_alpha=a.volatility_alpha,
                volatility_beta=a.volatility_beta,
                volume_variance_elasticity=a.volume_variance_elasticity,
                volume_lookback=int(a.volume_lookback),
                volume_log_ratio_clip=a.volume_log_ratio_clip,
                base_conditional_variance=unconditional_variance,
            )
        self.base_spread_bps = scenario.market_makers["base_spread_bps"]
        if not np.isfinite(self.base_spread_bps) or self.base_spread_bps < 0:
            raise ValueError("base spread must be finite and nonnegative")
        self.inventory_aversion = scenario.market_makers["inventory_aversion"]
        self.trades: list[TradeRecord] = []
        self.cancellations_by_agent_today: dict[str, int] = {}
        self.normal_quotes_enabled = bool(scenario.market_makers.get("normal_quotes_enabled", False))
        self.quote_jitter = float(scenario.market_makers.get("quote_size_jitter", 0.25))
        if not np.isfinite(self.quote_jitter) or not 0 <= self.quote_jitter < 1:
            raise ValueError("quote_size_jitter must be in [0, 1)")
        self.live_maker_quotes: dict[tuple[str, str, str], float] = {}
        self.maker_execution_capacity: dict[tuple[str, str, str], float] = {}
        self.normal_quote_activity: dict[str, dict[str, float]] = {}
        # Trusted, opt-in diagnostics: never exposed as regulator observations.
        # Logging consumes no randomness and does not change price dynamics.
        self.record_price_formation = False
        self.price_formation_log: list[dict] = []
        self.record_clearing_flow = False
        self.clearing_flow_log: list[dict] = []

    # ------------------------------------------------------------------ daily

    def begin_day(self) -> None:
        self.cancellations_by_agent_today = {}
        self.order_adjustments = []
        self.live_maker_quotes = {}
        self.maker_execution_capacity = {}
        self.normal_quote_activity = {}
        for s in self.assets.values():
            s.last_price = s.price
            s.volume_today = 0.0
            s.real_volume_today = 0.0
            s.spoof_buy_size = 0.0
            s.spoof_sell_size = 0.0
            s.cancel_count = 0
            s.order_count = 0
            s.displayed_buy_size = s.displayed_sell_size = 0.0

    def prepare_normal_quotes(self, controls=None) -> None:
        """Funded one-day bid/ask supply, not synthetic depth-only decorations.

        Quote sizes are drawn for every maker/asset/side even under a freeze.
        Cash is reserved across assets; today's sells cannot finance these bids.
        This is a daily quote model, not an empirically calibrated order book.
        """
        if not self.normal_quotes_enabled:
            return
        if self.live_maker_quotes:
            raise RuntimeError("normal quotes already prepared for this day")
        for maker in self.market_maker_ids:
            cash = self.accounts[maker].cash
            for asset, state in self.assets.items():
                bid, _ = self._quotes(state)
                scale = 1.0 - float((controls(asset, maker) if controls else {}).get("trade_throttle", 0))
                sizes = self.rng.uniform(1 - self.quote_jitter, 1 + self.quote_jitter, 2)
                maker_count = max(len(self.market_maker_ids), 1)
                displayed_size = state.base_liquidity / maker_count
                execution_size = state.daily_execution_liquidity / maker_count
                buy_capacity = min(execution_size * sizes[0] * scale, cash / bid)
                sell_capacity = min(
                    execution_size * sizes[1] * scale,
                    self.accounts[maker].position(asset),
                )
                buy = min(displayed_size * sizes[0] * scale, buy_capacity)
                sell = min(displayed_size * sizes[1] * scale, sell_capacity)
                cash -= buy_capacity * bid
                self.live_maker_quotes[maker, asset, "buy"] = buy
                self.live_maker_quotes[maker, asset, "sell"] = sell
                self.maker_execution_capacity[maker, asset, "buy"] = buy_capacity
                self.maker_execution_capacity[maker, asset, "sell"] = sell_capacity
                state.displayed_buy_size += buy
                state.displayed_sell_size += sell
                state.order_count += int(buy > 0) + int(sell > 0)
                stats = self.normal_quote_activity.setdefault(maker, {"order_quantity": 0., "order_count": 0.})
                stats["order_quantity"] += buy + sell
                stats["order_count"] += int(buy > 0) + int(sell > 0)

    def end_day(self, day: int) -> None:
        # Non-executed normal quotes expire too; cancellation is not an attack label.
        for (maker, asset, _), quantity in self.live_maker_quotes.items():
            if quantity > 1e-12:
                self.assets[asset].cancel_count += 1
                self.cancellations_by_agent_today[maker] = self.cancellations_by_agent_today.get(maker, 0) + 1
        self.live_maker_quotes = {}
        self.maker_execution_capacity = {}
        for s in self.assets.values():
            # fundamental random walk
            drift = self.rng.normal(0.0, s.fundamental_vol)
            s.fundamental *= float(np.exp(drift))

            # spread reacts to inventory aversion and noise
            stress = abs(s.price - s.fundamental) / max(s.fundamental, 1e-6)
            s.spread_bps = self.base_spread_bps * (1.0 + 4.0 * stress)

            depth_imb = self._depth_imbalance(s)
            cancel_rate = (s.cancel_count / max(s.order_count, 1))
            wash_share = 0.0
            if s.volume_today > 0:
                wash_share = 1.0 - (s.real_volume_today / s.volume_today)

            s.history["price"].append(s.price)
            s.history["fundamental"].append(s.fundamental)
            s.history["volume"].append(s.volume_today)
            s.history["real_volume"].append(s.real_volume_today)
            s.history["spread_bps"].append(s.spread_bps)
            s.history["depth_imbalance"].append(depth_imb)
            s.history["cancel_rate"].append(cancel_rate)
            s.history["wash_share"].append(wash_share)

    # ------------------------------------------------------------------ orders

    def displayed_depth(self, orders: list[Order]) -> dict[str, float]:
        """Funded, live quote imbalance before traders react; no settlement.

        Include background two-sided depth so a tiny quote cannot imply 100%
        imbalance. Reservation is repeated atomically with all clearing orders.
        No participant can spend or change inventory between these two phases.
        """
        adjustment_start = len(self.order_adjustments)
        funded = self._funded_orders(orders)
        del self.order_adjustments[adjustment_start:]
        result = {}
        for asset, state in self.assets.items():
            quotes = [order for order in funded if order.asset == asset and order.is_spoof]
            buy = state.displayed_buy_size + sum(order.quantity for order in quotes if order.side == "buy")
            sell = state.displayed_sell_size + sum(order.quantity for order in quotes if order.side == "sell")
            background = 0 if self.normal_quotes_enabled else 2 * state.base_liquidity
            result[asset] = (buy - sell) / max(background + buy + sell, 1e-12)
        return result

    def submit_orders(self, day: int, step: int, orders: list[Order]) -> list[TradeRecord]:
        """Validate, reserve, clear and settle one batch (cash-only, no shorting).

        Buy orders reserve the ask, even when eventually matched at mid. Sell
        proceeds are usable in the next batch, not for funding this batch.
        Displayed spoof quotes may execute before their one-round cancellation.
        """
        orders = self._funded_orders(orders)
        new_trades: list[TradeRecord] = []
        # group by asset
        by_asset: dict[str, list[Order]] = {a: [] for a in self.assets}
        for o in orders:
            if o.asset in by_asset:
                by_asset[o.asset].append(o)

        for asset_id, asset_orders in by_asset.items():
            trade_start = len(new_trades)
            s = self.assets[asset_id]
            real_buy = 0.0
            real_sell = 0.0

            # wash trades cross internally and add only to volume
            washes: list[Order] = [o for o in asset_orders if o.is_wash]
            spoofs = [o for o in asset_orders if o.is_spoof]
            reals = [o for o in asset_orders if not o.is_spoof and not o.is_wash]

            for o in reals:
                s.order_count += 1
                if o.side == "buy":
                    real_buy += o.quantity
                else:
                    real_sell += o.quantity

            # A manipulator cannot guarantee free cancellation: a displayed
            # quote is executable against opposite real interest at this mid.
            # Self fills are excluded. The unfilled remainder expires below.
            for o in spoofs:
                s.order_count += 1
                if o.side == "buy":
                    s.spoof_buy_size += o.quantity
                    s.displayed_buy_size += o.quantity
                else:
                    s.spoof_sell_size += o.quantity
                    s.displayed_sell_size += o.quantity
                remaining = o.quantity
                for counter in reals:
                    if counter.side == o.side or counter.agent_id == o.agent_id:
                        continue
                    quantity = min(remaining, counter.quantity)
                    if quantity <= 1e-12:
                        continue
                    buyer, seller = (o, counter) if o.side == "buy" else (counter, o)
                    new_trades.append(TradeRecord(
                        day, step, asset_id, buyer.agent_id, seller.agent_id,
                        quantity, s.price, buyer_is_harmful=buyer.is_harmful,
                        seller_is_harmful=seller.is_harmful,
                    ))
                    counter.quantity -= quantity
                    remaining -= quantity
                    s.volume_today += quantity
                    s.real_volume_today += quantity
                if remaining > 1e-12:
                    s.cancel_count += 1
                    self.cancellations_by_agent_today[o.agent_id] = (
                        self.cancellations_by_agent_today.get(o.agent_id, 0) + 1
                    )

            real_buy = sum(o.quantity for o in reals if o.side == "buy")
            real_sell = sum(o.quantity for o in reals if o.side == "sell")

            # match real orders against pooled liquidity at mid; price impact applied
            net = real_buy - real_sell
            matched = min(real_buy, real_sell)
            # pseudo-counterparty matching -- record for wealth bookkeeping
            if matched > 0 and reals:
                buyers = [o for o in reals if o.side == "buy"]
                sellers = [o for o in reals if o.side == "sell"]
                self._match_pool(day, step, asset_id, buyers, sellers, matched, s.price, new_trades)
            # remaining net order trades against market makers at mid +/- spread
            bid, ask = self._quotes(s)
            if net > 0:
                # buyers lift offer
                self._match_with_mm(day, step, asset_id, [o for o in reals if o.side == "buy"],
                                    abs(net), ask, "buy", new_trades)
            elif net < 0:
                self._match_with_mm(day, step, asset_id, [o for o in reals if o.side == "sell"],
                                    abs(net), bid, "sell", new_trades)

            # wash volume
            for o in washes:
                # wash matched at current mid, both sides held by colluders
                qty = o.quantity
                price = s.price
                s.volume_today += qty
                # do not affect real_volume_today
                tr = TradeRecord(
                    day=day, step=step, asset=asset_id,
                    buyer_id=o.agent_id if o.side == "buy" else o.counterparty_id,
                    seller_id=o.counterparty_id if o.side == "buy" else o.agent_id,
                    quantity=qty, price=price, is_wash=True,
                    buyer_is_harmful=True, seller_is_harmful=True,
                )
                new_trades.append(tr)

            asset_trades = new_trades[trade_start:]
            self._settle(asset_trades)
            # Only executed net flow moves the price; rejected/unfilled demand
            # cannot create impact. Internal matches and wash transfers net out.
            net = sum(
                trade.quantity * (1 if trade.seller_id in self.market_maker_ids else -1)
                for trade in asset_trades
                if not trade.is_wash and (
                    trade.seller_id in self.market_maker_ids
                    or trade.buyer_id in self.market_maker_ids
                )
            )
            maker_buy = sum(
                trade.quantity for trade in asset_trades
                if not trade.is_wash and trade.buyer_id in self.market_maker_ids
            )
            maker_sell = sum(
                trade.quantity for trade in asset_trades
                if not trade.is_wash and trade.seller_id in self.market_maker_ids
            )
            internal_match = sum(
                trade.quantity for trade in asset_trades
                if not trade.is_wash
                and trade.buyer_id not in self.market_maker_ids
                and trade.seller_id not in self.market_maker_ids
            )
            if self.record_clearing_flow:
                self.clearing_flow_log.append({
                    "day": day,
                    "step": step,
                    "asset": asset_id,
                    "accepted_buy": float(real_buy),
                    "accepted_sell": float(real_sell),
                    "harmful_buy": float(sum(
                        order.quantity for order in reals
                        if order.side == "buy" and order.is_harmful
                    )),
                    "harmful_sell": float(sum(
                        order.quantity for order in reals
                        if order.side == "sell" and order.is_harmful
                    )),
                    "internal_match": float(internal_match),
                    "maker_buy": float(maker_buy),
                    "maker_sell": float(maker_sell),
                    "unfilled_buy": float(max(real_buy - internal_match - maker_sell, 0.0)),
                    "unfilled_sell": float(max(real_sell - internal_match - maker_buy, 0.0)),
                })

            liquidity = s.daily_execution_liquidity
            gap = (s.price - s.fundamental) / max(s.fundamental, 1e-6)
            # MMs widen effective depth as price diverges from fundamental
            eff_liquidity = liquidity * (1.0 + 5.0 * abs(gap))
            raw = (net / max(eff_liquidity, 1e-6))
            impact = 0.06 * float(np.tanh(raw)) / self.rounds_per_day
            noise, base_variance, variance, volume_ratio, volume_multiplier = self._sample_price_noise(s)
            # mean-reversion toward fundamental, log-shaped
            mean_rev = 0.05 * float(np.tanh(2.0 * (s.fundamental - s.price) / max(s.price, 1e-6))) / self.rounds_per_day
            prior_price = s.price
            s.price = max(0.01, s.price * (1.0 + impact + noise + mean_rev))
            if self.record_price_formation:
                realized = s.price / prior_price - 1.0
                self.price_formation_log.append({
                    "day": day, "step": step, "asset": asset_id,
                    "impact": float(impact), "noise": float(noise), "reversion": float(mean_rev),
                    "floor_residual": float(realized - impact - noise - mean_rev),
                    "return": float(realized), "net_flow": float(net),
                    "executed_volume": float(sum(t.quantity for t in asset_trades)),
                    "pressure": float(raw), "price_gap": float(gap),
                    "fundamental_vol": float(s.fundamental_vol),
                    "base_conditional_variance": float(base_variance),
                    "conditional_variance": float(variance),
                    "volume_ratio": float(volume_ratio),
                    "volume_variance_multiplier": float(volume_multiplier),
                })

        self.trades.extend(new_trades)
        return new_trades

    # -------------------------------------------------------------- internals

    @staticmethod
    def _quotes(asset: AssetState) -> tuple[float, float]:
        # The simplified quote must remain positive even in extreme stress.
        half_spread = min(asset.price * asset.spread_bps / 20000.0, asset.price * 0.99)
        return asset.price - half_spread, asset.price + half_spread

    @staticmethod
    def _validate_volatility_config(asset) -> None:
        values = (
            asset.price_noise_multiplier,
            asset.volatility_alpha,
            asset.volatility_beta,
            asset.volume_variance_elasticity,
            asset.volume_log_ratio_clip,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("price volatility parameters must be finite")
        if asset.price_noise_multiplier < 0:
            raise ValueError("price_noise_multiplier must be nonnegative")
        if asset.volatility_alpha < 0 or asset.volatility_beta < 0:
            raise ValueError("volatility_alpha and volatility_beta must be nonnegative")
        if asset.volatility_alpha + asset.volatility_beta >= 1:
            raise ValueError("volatility_alpha + volatility_beta must be below one")
        if asset.volume_variance_elasticity < 0:
            raise ValueError("volume_variance_elasticity must be nonnegative")
        if isinstance(asset.volume_lookback, bool) or asset.volume_lookback < 1:
            raise ValueError("volume_lookback must be a positive integer")
        if int(asset.volume_lookback) != asset.volume_lookback:
            raise ValueError("volume_lookback must be a positive integer")
        if asset.volume_log_ratio_clip <= 0:
            raise ValueError("volume_log_ratio_clip must be positive")

    def _sample_price_noise(
        self, asset: AssetState,
    ) -> tuple[float, float, float, float, float]:
        """Draw one causal close-to-close innovation for this clearing round."""
        target_variance = (asset.fundamental_vol * asset.price_noise_multiplier) ** 2
        alpha = asset.volatility_alpha
        beta = asset.volatility_beta
        if alpha == 0 and beta == 0:
            base_variance = target_variance
        else:
            omega = (1.0 - alpha - beta) * target_variance
            base_variance = (
                omega
                + alpha * asset.previous_base_price_noise ** 2
                + beta * asset.base_conditional_variance
            )

        volume_ratio = self._current_volume_ratio(asset)
        log_ratio = np.clip(
            np.log(volume_ratio),
            -asset.volume_log_ratio_clip,
            asset.volume_log_ratio_clip,
        )
        volume_multiplier = float(np.exp(asset.volume_variance_elasticity * log_ratio))
        variance = max(0.0, float(base_variance * volume_multiplier))
        standardized_innovation = float(self.rng.normal())
        base_noise = standardized_innovation * np.sqrt(base_variance / self.rounds_per_day)
        noise = float(base_noise * np.sqrt(volume_multiplier))
        # Persist only the baseline GARCH state. Feeding the volume-amplified
        # innovation back into the same recurrence would count a volume shock
        # twice and can create an artificial explosive variance loop.
        asset.base_conditional_variance = float(base_variance)
        asset.previous_base_price_noise = float(base_noise)
        return noise, float(base_variance), variance, volume_ratio, volume_multiplier

    @staticmethod
    def _current_volume_ratio(asset: AssetState) -> float:
        history = asset.history["real_volume"][-asset.volume_lookback:]
        positive_history = [volume for volume in history if volume > 0]
        if asset.real_volume_today <= 0 or not positive_history:
            return 1.0
        baseline = float(np.mean(positive_history))
        return asset.real_volume_today / baseline if baseline > 0 else 1.0

    def _funded_orders(self, orders: list[Order]) -> list[Order]:
        """Reserve across assets and orders without mutating account balances."""
        for order in orders:
            if order.agent_id not in self.accounts or order.asset not in self.assets:
                raise ValueError("order references an unknown account or asset")
            if order.agent_id in self.market_maker_ids:
                raise ValueError("pooled market makers cannot submit taker orders")
            if order.side not in ("buy", "sell"):
                raise ValueError("order side must be buy or sell")
            if not np.isfinite(order.quantity) or order.quantity <= 0:
                raise ValueError("order quantity must be positive and finite")
            if order.is_spoof and order.cancel_after_steps != 1:
                raise ValueError("displayed quotes currently support exactly one reaction round")
            if order.is_wash and (order.is_spoof or order.counterparty_id not in self.accounts
                                  or order.counterparty_id == order.agent_id
                                  or order.counterparty_id in self.market_maker_ids):
                raise ValueError("wash orders require a distinct participant counterparty")

        cash = {key: account.cash for key, account in self.accounts.items()}
        holdings = {key: dict(account.holdings) for key, account in self.accounts.items()}
        accepted = []
        for order in orders:
            asset = self.assets[order.asset]
            _, ask = self._quotes(asset)
            buyer = order.agent_id if order.side == "buy" else None
            seller = order.agent_id if order.side == "sell" else None
            price = ask
            if order.is_wash:
                buyer = order.agent_id if order.side == "buy" else order.counterparty_id
                seller = order.counterparty_id if order.side == "buy" else order.agent_id
                price = asset.price
            quantity = order.quantity
            if buyer is not None:
                quantity = min(quantity, max(0.0, cash[buyer]) / price)
            if seller is not None:
                quantity = min(quantity, max(0.0, holdings[seller].get(order.asset, 0.0)))
            if quantity < order.quantity:
                self.order_adjustments.append({
                    "agent_id": order.agent_id, "asset": order.asset,
                    "requested_quantity": order.quantity, "accepted_quantity": quantity,
                    "reason": "insufficient_cash_or_inventory",
                })
            if quantity <= 1e-12:
                continue
            if buyer is not None:
                cash[buyer] -= quantity * price
            if seller is not None:
                holdings[seller][order.asset] -= quantity
            accepted.append(replace(order, quantity=quantity))
        return accepted

    def _settle(self, trades: list[TradeRecord]) -> None:
        """Every fill, including a wash transfer, has two real ledger entries."""
        for trade in trades:
            buyer = self.accounts[trade.buyer_id]
            seller = self.accounts[trade.seller_id]
            value = trade.quantity * trade.price
            buyer.cash -= value
            seller.cash += value
            buyer.holdings[trade.asset] = buyer.position(trade.asset) + trade.quantity
            seller.holdings[trade.asset] = seller.position(trade.asset) - trade.quantity

    def _match_pool(self, day, step, asset_id, buyers, sellers, total_qty, price, out):
        if total_qty <= 0 or not buyers or not sellers:
            return
        s = self.assets[asset_id]
        # proportional fill
        b_total = sum(o.quantity for o in buyers)
        s_total = sum(o.quantity for o in sellers)
        scale_b = total_qty / b_total if b_total > 0 else 0.0
        scale_s = total_qty / s_total if s_total > 0 else 0.0
        bi, si = 0, 0
        b_left = buyers[0].quantity * scale_b if buyers else 0.0
        s_left = sellers[0].quantity * scale_s if sellers else 0.0
        while bi < len(buyers) and si < len(sellers):
            q = min(b_left, s_left)
            if q > 1e-9:
                out.append(TradeRecord(
                    day=day, step=step, asset=asset_id,
                    buyer_id=buyers[bi].agent_id, seller_id=sellers[si].agent_id,
                    quantity=q, price=price,
                    buyer_is_harmful=buyers[bi].is_harmful,
                    seller_is_harmful=sellers[si].is_harmful,
                ))
                s.volume_today += q
                s.real_volume_today += q
            b_left -= q
            s_left -= q
            if b_left <= 1e-9:
                bi += 1
                if bi < len(buyers):
                    b_left = buyers[bi].quantity * scale_b
            if s_left <= 1e-9:
                si += 1
                if si < len(sellers):
                    s_left = sellers[si].quantity * scale_s

    def _match_with_mm(self, day, step, asset_id, takers, total_qty, price, side, out):
        if total_qty <= 0 or not takers:
            return
        s = self.assets[asset_id]
        t_total = sum(o.quantity for o in takers)
        if t_total <= 0:
            return
        remaining = total_qty
        for maker_id in self.market_maker_ids:
            account = self.accounts[maker_id]
            available = account.position(asset_id) if side == "buy" else account.cash / price
            quote_key = (maker_id, asset_id, "sell" if side == "buy" else "buy")
            if self.normal_quotes_enabled:
                available = min(
                    available,
                    self.maker_execution_capacity.get(quote_key, 0.0),
                )
            fill = min(remaining, max(0.0, available))
            if fill <= 0:
                continue
            for order in takers:
                quantity = fill * order.quantity / t_total
                out.append(TradeRecord(
                    day=day, step=step, asset=asset_id,
                    buyer_id=order.agent_id if side == "buy" else maker_id,
                    seller_id=maker_id if side == "buy" else order.agent_id,
                    quantity=quantity, price=price,
                    buyer_is_harmful=order.is_harmful if side == "buy" else False,
                    seller_is_harmful=order.is_harmful if side == "sell" else False,
                ))
                s.volume_today += quantity
                s.real_volume_today += quantity
            remaining -= fill
            if self.normal_quotes_enabled:
                self.maker_execution_capacity[quote_key] -= fill
                self.live_maker_quotes[quote_key] = max(
                    0.0,
                    self.live_maker_quotes.get(quote_key, 0.0) - fill,
                )
            if remaining <= 1e-12:
                break

    def _depth_imbalance(self, s: AssetState) -> float:
        total = s.displayed_buy_size + s.displayed_sell_size
        if total <= 1e-9:
            return 0.0
        return (s.displayed_buy_size - s.displayed_sell_size) / total

    # ------------------------------------------------------------------ views

    def snapshot(self) -> dict[str, dict[str, float]]:
        out = {}
        for aid, s in self.assets.items():
            out[aid] = {
                "price": s.price,
                "fundamental": s.fundamental,
                "spread_bps": s.spread_bps,
                "volume": s.volume_today,
                "real_volume": s.real_volume_today,
                "depth_imbalance": self._depth_imbalance(s),
                "displayed_bid_size": s.displayed_buy_size,
                "displayed_ask_size": s.displayed_sell_size,
                "cancel_rate": s.cancel_count / max(s.order_count, 1),
                "wash_share": (1.0 - s.real_volume_today / s.volume_today) if s.volume_today > 0 else 0.0,
            }
        return out
