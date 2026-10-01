"""Small ledger invariants; no model calls or benchmark generation."""
from dataclasses import replace

import numpy as np
import pytest

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.env.market import Order
from wolfgym.core.scenarios.base import load_scenario


def make_env():
    return WolfSocietySimulator(load_scenario("s1"), 5, 0.0, seed=17)


def ledger_totals(env):
    accounts = env.market.accounts.values()
    return [sum(account.cash for account in accounts)] + [
        sum(account.position(asset) for account in env.market.accounts.values())
        for asset in env.market.assets
    ]


def assert_ledger(env, before):
    assert ledger_totals(env) == pytest.approx(before)
    for account in env.market.accounts.values():
        assert account.cash >= -1e-8
        assert all(quantity >= -1e-8 for quantity in account.holdings.values())


def test_unaffordable_buy_is_clipped_and_market_maker_is_settled():
    env = make_env()
    buyer = env.society.retail[0]
    buyer.portfolio.cash = 1.0
    before = ledger_totals(env)
    trades = env.market.submit_orders(0, 0, [Order(buyer.agent_id, "asset_0", "buy", 10)])
    assert len(trades) == 1
    assert trades[0].seller_id in env.market.market_maker_ids
    assert buyer.portfolio.cash == pytest.approx(0.0)
    assert trades[0].quantity * trades[0].price == pytest.approx(1.0)
    assert env.market.order_adjustments[0]["reason"] == "insufficient_cash_or_inventory"
    assert_ledger(env, before)


def test_no_inventory_means_no_sale_and_no_flow_impact():
    env = make_env()
    control = env.clone()
    seller = env.society.retail[0]
    before = ledger_totals(env)
    assert env.market.submit_orders(0, 0, [Order(seller.agent_id, "asset_0", "sell", 10)]) == []
    control.market.submit_orders(0, 0, [])
    assert env.market.snapshot() == control.market.snapshot()
    assert_ledger(env, before)


def test_reservations_span_orders_and_assets():
    env = make_env()
    buyer = env.society.retail[0]
    buyer.portfolio.cash = 100.0
    before = ledger_totals(env)
    trades = env.market.submit_orders(0, 0, [
        Order(buyer.agent_id, "asset_0", "buy", 0.5),
        Order(buyer.agent_id, "asset_1", "buy", 10),
        Order(buyer.agent_id, "asset_0", "buy", 10),
    ])
    assert sum(trade.price * trade.quantity for trade in trades) == pytest.approx(100)
    assert_ledger(env, before)


def test_partial_pool_and_maker_fills_do_not_overfill_orders():
    env = make_env()
    buyer, seller = env.society.retail[:2]
    seller.portfolio.holdings["asset_0"] = 3
    before = ledger_totals(env)
    trades = env.market.submit_orders(0, 0, [
        Order(buyer.agent_id, "asset_0", "buy", 5),
        Order(seller.agent_id, "asset_0", "sell", 3),
    ])
    assert sum(t.quantity for t in trades if t.buyer_id == buyer.agent_id) == pytest.approx(5)
    assert sum(t.quantity for t in trades if t.seller_id == seller.agent_id) == pytest.approx(3)
    assert_ledger(env, before)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_finite_maker_endowment_limits_fills(side):
    env = make_env()
    agent = env.society.retail[0]
    agent.portfolio.holdings["asset_0"] = 10
    for maker in env.society.market_makers:
        maker.portfolio.cash = 0
        maker.portfolio.holdings["asset_0"] = 0
    maker = env.society.market_makers[0]
    if side == "buy":
        maker.portfolio.holdings["asset_0"] = 2
    else:
        bid, _ = env.market._quotes(env.market.assets["asset_0"])
        maker.portfolio.cash = 2 * bid
    before = ledger_totals(env)
    trades = env.market.submit_orders(0, 0, [Order(agent.agent_id, "asset_0", side, 10)])
    assert sum(trade.quantity for trade in trades) == pytest.approx(2)
    assert_ledger(env, before)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_wash_is_a_funded_two_sided_transfer(side):
    env = make_env()
    buyer, seller = env.society.retail[:2]
    buyer.portfolio.cash = 50
    seller.portfolio.holdings["asset_0"] = 2
    before = ledger_totals(env)
    origin, counterparty = (buyer, seller) if side == "buy" else (seller, buyer)
    trades = env.market.submit_orders(0, 0, [Order(
        origin.agent_id, "asset_0", side, 10,
        is_wash=True, counterparty_id=counterparty.agent_id,
    )])
    assert trades[0].quantity == pytest.approx(0.5)
    assert buyer.portfolio.position("asset_0") == pytest.approx(0.5)
    assert seller.portfolio.position("asset_0") == pytest.approx(1.5)
    assert env.market.assets["asset_0"].real_volume_today == 0
    assert_ledger(env, before)


@pytest.mark.parametrize("changes", [
    {"quantity": -1}, {"quantity": 0}, {"quantity": float("nan")},
    {"quantity": float("inf")}, {"side": "hold"}, {"asset": "missing"},
    {"agent_id": "missing"}, {"is_wash": True, "counterparty_id": "missing"},
])
def test_invalid_batch_is_rejected_before_any_mutation(changes):
    env = make_env()
    order = Order(env.society.retail[0].agent_id, "asset_0", "buy", 1)
    before = ledger_totals(env)
    prices = env.market.snapshot()
    with pytest.raises(ValueError):
        env.market.submit_orders(0, 0, [order, replace(order, **changes)])
    assert env.market.snapshot() == prices
    assert env.market.trades == []
    assert_ledger(env, before)


def test_final_valuation_uses_current_market_prices():
    env = make_env()
    env.society.retail[0].portfolio.holdings["asset_0"] = 1
    env._last_pre_step_prices["asset_0"] = 100
    env.market.assets["asset_0"].price = 120
    assert env._episode_result().metrics.retail_loss_30d == pytest.approx(-120)


def test_snapshot_preserves_ledger_aliases_and_replay():
    env = make_env()
    snapshot = env.get_state()
    left, right = env.fork_from_state(snapshot), env.fork_from_state(snapshot)
    account_id = left.society.retail[0].agent_id
    assert left.market.accounts[account_id] is left.society.retail[0].portfolio
    assert left.market.accounts[account_id] is not right.market.accounts[account_id]
    order = Order(account_id, "asset_0", "buy", 1)
    assert left.market.submit_orders(0, 0, [order]) == right.market.submit_orders(0, 0, [order])
    assert ledger_totals(left) == ledger_totals(right)
    with pytest.raises(ValueError, match="snapshot schema"):
        env.set_state(replace(snapshot, schema_version=1))


def test_small_episode_conserves_cash_and_shares_each_day():
    env = WolfSocietySimulator(load_scenario("s1"), 20, 0.1, seed=7)
    before = ledger_totals(env)
    for _ in range(30):
        env.step_day()
        assert_ledger(env, before)
        assert all(np.isfinite(asset.price) and asset.price > 0 for asset in env.market.assets.values())


def test_trader_observes_previous_close_before_daily_reset(monkeypatch):
    env = make_env()
    env.market.assets["asset_0"].volume_today = 123
    observed = []
    for agent in env.society.all_agents:
        def decide(day, observation):
            observed.append(observation["market"]["asset_0"]["volume"])
            return [], []
        monkeypatch.setattr(agent, "decide", decide)
    env.step_day()
    assert observed and set(observed) == {123}
    assert env.market.assets["asset_0"].volume_today == 0
    assert env.market.assets["asset_0"].price != 100  # no orders does not stop the clock


@pytest.mark.parametrize("harmful_count", [2, 3, 4, 5, 7, 9])
def test_wash_groups_preserve_exact_participant_count(harmful_count):
    env = WolfSocietySimulator(load_scenario("s4"), 20, harmful_count / 20, seed=7)
    assert len(env.society.attackers) == harmful_count
    assert len(env.society.retail) + len(env.society.attackers) == 20
    ids = {agent.agent_id for agent in env.society.attackers}
    assert all(agent.counterparty_id in ids and agent.counterparty_id != agent.agent_id
               for agent in env.society.attackers)
