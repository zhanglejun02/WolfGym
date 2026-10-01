"""Behavioral contracts under controlled information, not fitted profit targets."""
from dataclasses import replace

import numpy as np
import pytest

from wolfgym.core.agents.retail import RetailAgent
from wolfgym.core.agents.base import Portfolio
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.stock_market import ATTACK_FAMILIES, load_stock_market


def observation(history, fundamental=100):
    return {"prices": {"a": 100}, "market": {"a": {"price": 100, "fundamental": fundamental}},
            "recent_return": {"a": 0}, "return_history": {"a": tuple(history)}, "volume_z": {}}


def trader(**kwargs):
    return RetailAgent("r", "retail", portfolio=Portfolio(8000, {"a": 20}, initial_wealth=10000),
                       rng=np.random.default_rng(17), **kwargs)


def test_risk_averse_reduces_exposure_when_volatility_rises_without_directional_signal():
    calm = trader(sub_role="risk_averse", max_asset_share=.25, volatility_sensitivity=10)
    volatile = trader(sub_role="risk_averse", max_asset_share=.25, volatility_sensitivity=10)
    assert calm.decide(0, observation([0, 0]))[0] == []
    orders, _ = volatile.decide(0, observation([-.10, .10]))
    assert len(orders) == 1 and orders[0].side == "sell"
    assert orders[0].quantity * 100 <= volatile.risk_appetite * 10000
    assert volatile.last_belief_breakdown["a"]["position_limit"] < .25


def test_same_history_different_windows_produce_different_momentum():
    short = trader(momentum_window=1)
    long = trader(momentum_window=3)
    obs = observation([-.1, -.1, .1])
    short.decide(0, obs)
    long.decide(0, obs)
    assert short.last_belief_breakdown["a"]["momentum"] > 0
    assert long.last_belief_breakdown["a"]["momentum"] < 0


def test_inventory_limit_blocks_further_buys_even_when_asset_is_undervalued():
    for seed in range(32):
        agent = trader(max_asset_share=.2)
        agent.rng = np.random.default_rng(seed)
        assert agent.decide(0, observation([], fundamental=200))[0] == []


@pytest.mark.parametrize("family", ATTACK_FAMILIES)
def test_attack_delay_is_bounded_shared_and_snapshot_stable(family):
    starts = set()
    for seed in range(6):
        env = WolfSocietySimulator(load_stock_market(family), 40, .1, seed)
        field = "active_days" if family == "spoofing" else "accumulate_days"
        agents = [a for a in env.society.attackers if hasattr(a, field)]
        starts.add(getattr(agents[0], field)[0])
        assert len({getattr(a, field) for a in agents}) == 1
        branch = env.clone()
        assert [getattr(a, field, None) for a in branch.society.attackers] == [
            getattr(a, field, None) for a in env.society.attackers]
    assert len(starts) > 1 and starts <= {0, 1, 2, 3}


def test_history_is_only_completed_closes_and_snapshot_replays():
    env = WolfSocietySimulator(load_stock_market(), 40, 0, 17)
    env.step_day()
    prices = {a: s.price for a, s in env.market.assets.items()}
    obs = env._build_observation(1, prices, {})
    assert all(isinstance(v, tuple) and len(v) == 1 for v in obs["return_history"].values())
    branch = env.clone()
    assert env.step_day() == branch.step_day()
    with pytest.raises(ValueError, match="snapshot schema"):
        env.set_state(replace(env.get_state(), schema_version=3))
