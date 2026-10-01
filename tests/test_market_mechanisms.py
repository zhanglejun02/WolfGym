"""Controlled mechanism checks; none assumes attacks must cause collapse."""
from dataclasses import replace

import numpy as np
import pytest

from wolfgym.core.agents.attackers import CoordinatedTrader
from wolfgym.core.agents.base import Portfolio
from wolfgym.core.agents.retail import RetailAgent
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.env.market import Order
from wolfgym.core.scenarios.stock_market import ATTACK_FAMILIES, load_stock_market


def retail_observation(fundamental=100, volume=0, signal=0):
    class SocialSignal:
        def agent_signal(self, agent_id, asset):
            return signal, 0
    return {"prices": {"asset_0": 100},
            "market": {"asset_0": {"price": 100, "fundamental": fundamental}},
            "recent_return": {}, "volume_z": {"asset_0": volume}, "social_env": SocialSignal()}


def agent(role="value_investor", seed=0, **kwargs):
    return RetailAgent("test", f"retail_{role}", sub_role=role,
                       portfolio=Portfolio(cash=5000, holdings={"asset_0": 50}, initial_wealth=10000),
                       rng=np.random.default_rng(seed), **kwargs)


@pytest.mark.parametrize("role", ["risk_averse", "value_investor"])
@pytest.mark.parametrize("fundamental,side", [(95, "sell"), (105, "buy")])
def test_small_fundamental_opportunities_are_tradeable_in_both_directions(role, fundamental, side):
    orders = [order for seed in range(128)
              for order in agent(role, seed).decide(0, retail_observation(fundamental))[0]]
    assert orders and {order.side for order in orders} == {side}
    # Neutral roles remain inactive: this is not a minimum-turnover quota.
    assert agent(role).decide(0, retail_observation())[0] == []


def test_volume_amplifies_evidence_without_inventing_a_bullish_direction():
    neutral, cheap, expensive = agent(), agent(), agent()
    for trader, fundamental in [(neutral, 100), (cheap, 95), (expensive, 105)]:
        trader.decide(0, retail_observation(fundamental, volume=10))
    assert neutral.last_belief_breakdown["asset_0"]["volume"] == 0
    assert cheap.last_belief_breakdown["asset_0"]["volume"] < 0
    assert expensive.last_belief_breakdown["asset_0"]["volume"] > 0


def test_skepticism_discounts_both_social_directions_symmetrically():
    values = []
    for signal in [-1, 1]:
        trader = agent("risk_averse", skepticism=.75)
        trader.decide(0, retail_observation(signal=signal))
        values.append(trader.last_belief_breakdown["asset_0"]["social"])
    assert values[0] == pytest.approx(-values[1])


def test_initial_basket_is_funded_without_changing_initial_wealth():
    env = WolfSocietySimulator(load_stock_market(), 20, 0, seed=1)
    prices = {a.id: a.fundamental for a in env.scenario.assets}
    for trader in env.society.retail:
        assert trader.portfolio.cash == 5000
        assert trader.portfolio.mark_to_market(prices) == pytest.approx(10000)
        assert all(trader.portfolio.position(asset) > 0 for asset in prices)


def test_normal_posts_have_useful_direction_but_no_intent_encoded_in_text():
    trader = agent(post_probability=1, research_noise=0)
    _, messages = trader.decide(0, retail_observation(fundamental=105))
    assert len(messages) == 1 and not messages[0].is_harmful
    assert messages[0].sentiment > 0
    clone = replace(messages[0], is_harmful=True, text="")
    assert clone.text == messages[0].text


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_fully_filled_displayed_quote_is_not_recorded_as_cancelled(side):
    env = WolfSocietySimulator(load_stock_market("spoofing"), 20, .1, seed=1)
    spoofer, retail = env.society.attackers[0], env.society.retail[0]
    asset = env.target_asset
    quote = Order(spoofer.agent_id, asset, side, 5, is_spoof=True, cancel_after_steps=1)
    other = Order(retail.agent_id, asset, "sell" if side == "buy" else "buy", 5)
    trades = env.market.submit_orders(0, 0, [quote, other])
    assert sum(trade.quantity for trade in trades) == 5
    assert env.market.assets[asset].cancel_count == 0
    assert all(trade.buyer_id != trade.seller_id for trade in trades)


def test_quote_reservation_is_shared_with_real_orders_and_forks_replay():
    env = WolfSocietySimulator(load_stock_market("spoofing"), 20, .1, seed=1)
    spoofer = env.society.attackers[0]
    spoofer.portfolio.cash = 10
    asset = env.target_asset
    orders = [Order(spoofer.agent_id, asset, "buy", 100, is_spoof=True, cancel_after_steps=1),
              Order(spoofer.agent_id, asset, "buy", 100),
              Order(env.society.retail[0].agent_id, asset, "sell", 100)]
    left, right = env.clone(), env.clone()
    assert left.market.submit_orders(0, 0, orders) == right.market.submit_orders(0, 0, orders)
    assert left.market.accounts[spoofer.agent_id].cash >= 0
    assert sum(t.price * t.quantity for t in left.market.trades if t.buyer_id == spoofer.agent_id) <= 10


def test_retail_volume_signal_uses_reported_volume_not_hidden_wash_labels():
    env = WolfSocietySimulator(load_stock_market("wash_trading"), 20, .1, seed=1)
    state = env.market.assets[env.target_asset]
    state.history["volume"] = [100, 90, 110, 100, 1000]
    state.history["real_volume"] = [100, 90, 110, 100, 100]
    state.volume_today, state.real_volume_today = 1000, 100
    prices = {aid: s.price for aid, s in env.market.assets.items()}
    assert env._build_observation(5, prices, {})["volume_z"][env.target_asset] > 10


def test_overlapping_promotion_does_not_delay_coordinated_sales():
    trader = CoordinatedTrader("seller", "harmful", portfolio=Portfolio(5000, {"asset_0": 50}),
                                target_asset="asset_0", promote_days=(4, 18), dump_days=(12, 25))
    orders, _ = trader.decide(12, retail_observation())
    assert len(orders) == 1 and orders[0].side == "sell"


def test_displayed_quote_is_funded_can_fill_and_cancels_only_the_remainder():
    env = WolfSocietySimulator(load_stock_market("spoofing"), 20, .1, seed=1)
    spoofer, seller = env.society.attackers[0], env.society.retail[0]
    asset = env.target_asset
    quote = Order(spoofer.agent_id, asset, "buy", 20, is_spoof=True, cancel_after_steps=1)
    env.market.prepare_normal_quotes()
    balances = {key: (p.cash, dict(p.holdings)) for key, p in env.market.accounts.items()}
    imbalance = env.market.displayed_depth([quote])[asset]
    assert 0 < imbalance < 1
    assert balances == {key: (p.cash, dict(p.holdings)) for key, p in env.market.accounts.items()}
    trades = env.market.submit_orders(0, 0, [quote, Order(seller.agent_id, asset, "sell", 5)])
    assert len(trades) == 1
    assert trades[0].buyer_id == spoofer.agent_id and trades[0].quantity == 5
    assert env.market.assets[asset].cancel_count == 1
    assert sum(p.cash for p in env.market.accounts.values()) == pytest.approx(sum(v[0] for v in balances.values()))
    assert sum(p.position(asset) for p in env.market.accounts.values()) == pytest.approx(sum(v[1].get(asset, 0) for v in balances.values()))
    assert all(p.cash >= -1e-8 and all(q >= -1e-8 for q in p.holdings.values()) for p in env.market.accounts.values())
    with pytest.raises(ValueError, match="one reaction round"):
        env.market.submit_orders(0, 1, [replace(quote, cancel_after_steps=3)])


def test_retail_observes_current_displayed_depth_not_cancelled_yesterday_quotes():
    env = WolfSocietySimulator(load_stock_market("spoofing"), 20, .1, seed=1)
    for attacker in env.society.attackers:
        attacker.side_buy_prob = 1
    env.step_day()
    assert any(trader.last_belief_breakdown[env.target_asset]["imbalance"] > 0
               for trader in env.society.retail)
    branch = env.clone()
    branch.set_persistent_regulatory_controls(scope="asset", target_id=branch.target_asset,
        controls={"message_filter": 0, "trade_throttle": 0, "spoof_filter": 1, "wash_filter": 0})
    branch.step_day()
    assert branch.market.assets[branch.target_asset].spoof_buy_size == 0
    assert branch.market.assets[branch.target_asset].spoof_sell_size == 0
    # Filtering attack quotes must leave a genuine normal depth background.
    assert any(trader.last_belief_breakdown[env.target_asset]["imbalance"] != 0
               for trader in branch.society.retail)


def test_pre_v3_snapshots_are_rejected():
    env = WolfSocietySimulator(load_stock_market(), 20, 0, seed=1)
    with pytest.raises(ValueError, match="snapshot schema"):
        env.set_state(replace(env.get_state(), schema_version=2))


@pytest.mark.parametrize("family", ATTACK_FAMILIES)
def test_all_four_mechanisms_preserve_the_ledger_through_thirty_days(family):
    env = WolfSocietySimulator(load_stock_market(family), 40, .1, seed=17)
    accounts = env.market.accounts
    cash = sum(p.cash for p in accounts.values())
    shares = {asset: sum(p.position(asset) for p in accounts.values()) for asset in env.market.assets}
    for _ in range(30):
        env.step_day()
        assert sum(p.cash for p in accounts.values()) == pytest.approx(cash)
        for asset, total in shares.items():
            assert sum(p.position(asset) for p in accounts.values()) == pytest.approx(total)
        assert all(p.cash >= -1e-8 and all(q >= -1e-8 for q in p.holdings.values()) for p in accounts.values())
