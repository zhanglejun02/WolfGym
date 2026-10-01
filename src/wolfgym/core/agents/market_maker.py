"""Stylized market makers: stabilise spread, mean-revert, no LLM."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from wolfgym.core.agents.base import AgentBase, Portfolio
from wolfgym.core.env.market import Order
from wolfgym.core.env.social import Message


@dataclass
class MarketMaker(AgentBase):
    inventory_aversion: float = 0.05

    def decide(self, day: int, observation: dict) -> tuple[list[Order], list[Message]]:
        # Market maker fills are handled inside MarketEnv via pooled liquidity.
        # We expose this stub so the loop is uniform.
        return [], []


def build_market_makers(scenario) -> list[MarketMaker]:
    n = int(scenario.market_makers.get("count", 3))
    out: list[MarketMaker] = []
    inv_av = float(scenario.market_makers.get("inventory_aversion", 0.05))
    # Explicit finite external liquidity endowment, outside the N participants.
    cash = float(scenario.market_makers.get("initial_cash", 1e9))
    inventory_value = float(scenario.market_makers.get("inventory_value_per_asset", 1e9))
    if not all(isfinite(value) and value >= 0 for value in (cash, inventory_value)):
        raise ValueError("market-maker endowments must be finite and nonnegative")
    if any(not isfinite(asset.fundamental) or asset.fundamental <= 0 for asset in scenario.assets):
        raise ValueError("initial asset prices must be positive and finite")
    for i in range(n):
        out.append(MarketMaker(
            agent_id=f"mm_{i}",
            role="market_maker",
            is_harmful=False,
            portfolio=Portfolio(
                cash=cash,
                holdings={asset.id: inventory_value / asset.fundamental for asset in scenario.assets},
                initial_wealth=cash + inventory_value * len(scenario.assets),
            ),
            inventory_aversion=inv_av,
        ))
    return out
