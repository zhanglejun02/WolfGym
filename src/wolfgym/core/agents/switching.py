"""Experimental v3 tactic switching over one funded account per participant.

The v8 market and existing S1--S4 policies remain unchanged. This adapter is
installed only by the explicit v3 research harness, never by legacy scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math

import numpy as np

from wolfgym.core.agents.base import AgentBase
from wolfgym.core.scenarios.society import _build_pump_and_dump, _build_spoofers, _build_wash_traders
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.core.scenarios.stock_market import _PROFILE_PATH


TACTICS = ("S1", "S2", "S3", "S4")


@dataclass
class SwitchingManipulator(AgentBase):
    target_asset: str = "asset_2"
    modules: dict = field(default_factory=dict)
    selected_tactic: str = "WAIT"
    daily_notional_multiple: float = 1.0
    daily_message_cap: float = 1.8
    last_activity: dict = field(default_factory=dict)

    def decide(self, day, observation):
        tactic = self.selected_tactic
        if tactic == "WAIT":
            self.last_activity = {"tactic": tactic, "orders": 0, "messages": 0,
                                  "submitted_notional": 0., "message_intensity": 0.}
            return [], []
        module = self.modules[tactic]
        # Continuation seeding can replace the wrapper RNG. Rebind here so
        # modules never retain a stale RNG or a separate settlement wallet.
        module.portfolio, module.rng = self.portfolio, self.rng
        # Existing tactics need only public prices/tape plus their own account.
        public_view = {"prices": dict(observation["prices"]), "market": {
            key: {"price": value["price"], "volume": value.get("volume", 0.)}
            for key, value in observation["market"].items()}}
        orders, messages = module.decide(day, public_view)
        notional = math.fsum(o.quantity * public_view["prices"][o.asset] for o in orders)
        scale = min(1., self.daily_notional_multiple * self.portfolio.initial_wealth / notional) if notional else 1.
        orders = [replace(o, quantity=o.quantity * scale) for o in orders]
        intensity = math.fsum(m.intensity for m in messages)
        scale = min(1., self.daily_message_cap / intensity) if intensity else 1.
        messages = [replace(m, intensity=m.intensity * scale) for m in messages]
        self.last_activity = {"tactic": tactic, "orders": len(orders), "messages": len(messages),
                              "submitted_notional": math.fsum(o.quantity * public_view["prices"][o.asset] for o in orders),
                              "message_intensity": math.fsum(m.intensity for m in messages)}
        return orders, messages


def install_switching_cohort(simulator, *, initial_inventory_share=.20):
    """Install at day zero, preserving IDs, graph placement and initial wealth.

    A common 20% stock endowment funds all tactics; it is an explicit v3
    endowment change, not a free trade or extra capital at a tactic switch.
    """
    if simulator._next_day != 0 or not 0 <= initial_inventory_share <= 1:
        raise ValueError("v3 installation requires day zero and a valid inventory share")
    originals = simulator.society.attackers
    if len(originals) < 2 or any(isinstance(a, SwitchingManipulator) for a in originals):
        raise ValueError("install once on a finfluencer cohort of at least two accounts")
    if tuple(simulator.scenario.attackers) != ("finfluencer",):
        raise ValueError("v3 fixes the same finfluencer role allocation and hub placement for all branches")
    catalog = load_scenario(str(_PROFILE_PATH)).attackers
    delay = originals[0].accumulate_days[0]
    shifted = {}
    for family, raw in catalog.items():
        cfg = dict(raw)
        for name in ("accumulate_days", "promote_days", "dump_days", "sell_days", "wash_days", "withdraw_days", "active_days"):
            if name in cfg:
                cfg[name] = tuple(value + delay for value in cfg[name])
        shifted[family] = cfg
    module_rng = np.random.default_rng(simulator.seed + 3000000)
    n = len(originals)
    banks = {
        "S1": _build_pump_and_dump(shifted["pump_and_dump"], n, module_rng,
                                   n_society=simulator.n_society, alpha=simulator.alpha),
        "S2": originals,
        "S3": _build_spoofers(shifted["spoofing"], n, module_rng),
        "S4": _build_wash_traders(shifted["wash_trading"], n, module_rng),
    }
    wash_ids = {module.agent_id: original.agent_id for module, original in zip(banks["S4"], originals)}
    for module in banks["S4"]:
        module.counterparty_id = wash_ids[module.counterparty_id]
    replacements = {}
    for index, original in enumerate(originals):
        portfolio = original.portfolio
        stock_value = portfolio.initial_wealth * initial_inventory_share
        portfolio.cash = portfolio.initial_wealth - stock_value
        portfolio.holdings = {simulator.target_asset: stock_value / simulator.market.assets[simulator.target_asset].price}
        modules = {tactic: bank[index] for tactic, bank in banks.items()}
        for module in modules.values():
            module.agent_id, module.portfolio, module.rng = original.agent_id, portfolio, original.rng
        replacements[original.agent_id] = SwitchingManipulator(
            agent_id=original.agent_id, role=original.role, is_harmful=True,
            portfolio=portfolio, rng=original.rng, target_asset=simulator.target_asset, modules=modules)
    simulator.society.attackers = [replacements[a.agent_id] for a in originals]
    simulator.society.all_agents = [replacements.get(a.agent_id, a) for a in simulator.society.all_agents]
    simulator.switching_start_delay = delay
    simulator.switching_initial_price = simulator.market.assets[simulator.target_asset].price
    return simulator


def select_tactic(mode, *, day, start_delay, public_signals, inventory_share, selection_seed, disabled=None):
    """Pure group selector. No fundamentals, victim accounts, or future state."""
    if mode not in (*TACTICS, "off", "random", "adaptive", "campaign"):
        raise ValueError(f"unknown switching mode {mode}")
    age = day - start_delay
    if age < 0 or age > 25 or mode == "off":
        planned = "WAIT"
    elif mode in TACTICS:
        planned = mode
    elif mode == "random":
        number = int.from_bytes(hashlib.blake2s(f"v3:{selection_seed}:{day}".encode(), digest_size=8).digest(), "big")
        planned = (*TACTICS, "WAIT")[number % 5]
    elif mode == "campaign":
        # Development candidate: retain accumulation/promotion continuity before
        # switching to distribution. Decisions use only public tape and own stock.
        if age <= 4:
            planned = "S1"
        elif age <= 8:
            planned = "S4" if public_signals["volume_ratio"] < 1. else "S1"
        elif age <= 18:
            planned = "S3" if age % 5 == 0 and abs(public_signals["depth_imbalance"]) < .25 else "S2"
        else:
            planned = "S1" if public_signals["return_3d"] > .03 and inventory_share > .10 else ("S2" if inventory_share > 0 else "WAIT")
    elif mode == "adaptive":
        if age <= 4:
            planned = "S1"
        elif age >= 19:
            planned = "S2" if inventory_share > 0 else "WAIT"
        elif public_signals["return_3d"] > .03 and inventory_share > .10:
            planned = "S2"
        elif public_signals["volume_ratio"] < 1.:
            planned = "S4"
        elif abs(public_signals["depth_imbalance"]) < .25:
            planned = "S3"
        else:
            planned = "S1"
    else:
        raise ValueError(f"unknown switching mode {mode}")
    if disabled is not None and disabled not in TACTICS:
        raise ValueError("unknown ablated tactic")
    return planned, "WAIT" if planned == disabled else planned


def prepare_tactic(simulator, *, mode, selection_seed, disabled=None):
    """Extract an explicit public/own-account whitelist before the next day."""
    target = simulator.market.assets[simulator.target_asset]
    prices = [simulator.switching_initial_price, *target.history["price"]]
    volumes = target.history["volume"]
    prior_volume = float(np.mean(volumes[-4:-1])) if len(volumes) >= 2 else 0.
    signals = {"return_3d": prices[-1] / prices[max(0, len(prices) - 4)] - 1,
               "volume_ratio": volumes[-1] / prior_volume if prior_volume > 0 else 1.,
               "depth_imbalance": target.history["depth_imbalance"][-1] if target.history["depth_imbalance"] else 0.}
    marks = {key: asset.price for key, asset in simulator.market.assets.items()}
    equity = math.fsum(a.portfolio.mark_to_market(marks) for a in simulator.society.attackers)
    inventory = math.fsum(a.portfolio.position(simulator.target_asset) * target.price for a in simulator.society.attackers)
    share = inventory / equity if equity > 0 else 0.
    planned, actual = select_tactic(mode, day=simulator._next_day, start_delay=simulator.switching_start_delay,
                                    public_signals=signals, inventory_share=share,
                                    selection_seed=selection_seed, disabled=disabled)
    for agent in simulator.society.attackers:
        agent.selected_tactic = actual
    return {"decision_day": simulator._next_day, "planned": planned, "executed": actual,
            "public_signals": signals, "own_inventory_share": share}
