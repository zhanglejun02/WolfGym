from dataclasses import replace
from datetime import date, timedelta
import json

import numpy as np
import pytest

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.env.market import Order
from wolfgym.core.metrics.collapse import primary_failure_signal
from wolfgym.core.scenarios.stock_market import load_stock_market
from wolfgym.evaluation.real_market import (
    MarketWindow, baseline_windows, compare_distributions, correlation,
    load_reference, summarize, wasserstein_1,
)


def test_normal_quotes_are_funded_executable_and_expire():
    env = WolfSocietySimulator(load_stock_market(), 20, 0, seed=11)
    market = env.market
    for maker in market.market_maker_ids:
        market.accounts[maker].cash = 100
        market.accounts[maker].holdings = {asset: 2. for asset in market.assets}
    before = sum(p.cash for p in market.accounts.values())
    shares = {a: sum(p.position(a) for p in market.accounts.values()) for a in market.assets}
    market.prepare_normal_quotes()
    for maker in market.market_maker_ids:
        committed = sum(q * market._quotes(market.assets[a])[0] for (m, a, side), q in market.live_maker_quotes.items() if m == maker and side == "buy")
        assert committed <= 100 + 1e-10
    assert any(abs(x) > 0 for x in market.displayed_depth([]).values())
    buyer = env.society.retail[0]
    fills = market.submit_orders(0, 0, [Order(buyer.agent_id, "asset_0", "buy", 10)])
    assert sum(t.quantity for t in fills) == pytest.approx(6)
    assert sum(p.cash for p in market.accounts.values()) == pytest.approx(before)
    for a, total in shares.items():
        assert sum(p.position(a) for p in market.accounts.values()) == pytest.approx(total)
    market.end_day(0)
    assert market.assets["asset_0"].cancel_count > 0
    assert not market.live_maker_quotes


def test_asset2_top_depth_replenishes_with_funded_daily_capacity():
    env = WolfSocietySimulator(load_stock_market(), 20, 0, seed=11)
    market = env.market
    buyer = env.society.retail[0]
    buyer.portfolio.cash = 1_000_000
    market.prepare_normal_quotes()
    displayed = sum(
        quantity
        for (maker, asset, side), quantity in market.live_maker_quotes.items()
        if asset == "asset_2" and side == "sell"
    )
    capacity = sum(
        quantity
        for (maker, asset, side), quantity in market.maker_execution_capacity.items()
        if asset == "asset_2" and side == "sell"
    )
    assert displayed < 500
    assert capacity > 10_000

    market.record_clearing_flow = True
    fills = market.submit_orders(0, 0, [Order(buyer.agent_id, "asset_2", "buy", 1_000)])
    assert sum(trade.quantity for trade in fills) == pytest.approx(1_000)
    assert sum(trade.quantity for trade in fills) > displayed
    flow = market.clearing_flow_log[-1]
    assert flow["accepted_buy"] == pytest.approx(1_000)
    assert flow["maker_sell"] == pytest.approx(1_000)
    assert flow["unfilled_buy"] == pytest.approx(0)


def test_execution_capacity_remains_finite_and_records_unfilled_residual():
    env = WolfSocietySimulator(load_stock_market(), 20, 0, seed=11)
    market = env.market
    buyer = env.society.retail[0]
    buyer.portfolio.cash = 1_000_000
    market.prepare_normal_quotes()
    capacity = sum(
        quantity
        for (maker, asset, side), quantity in market.maker_execution_capacity.items()
        if asset == "asset_2" and side == "sell"
    )
    market.record_clearing_flow = True
    fills = market.submit_orders(0, 0, [
        Order(buyer.agent_id, "asset_2", "buy", capacity + 1_000),
    ])
    assert sum(trade.quantity for trade in fills) == pytest.approx(capacity)
    assert market.clearing_flow_log[-1]["unfilled_buy"] == pytest.approx(1_000)


def test_normal_quotes_fork_exactly_and_freeze_suppresses_supply():
    env = WolfSocietySimulator(load_stock_market(), 20, 0, seed=11)
    env.step_day()
    activity = env.observable_entity_history()[-1]["entities"]
    assert all(activity[m.agent_id]["order_count"] > 0 for m in env.society.market_makers)
    assert sum(row["cancellation_count"] for row in activity.values()) == sum(s.cancel_count for s in env.market.assets.values())
    branch = env.clone()
    assert env.step_day() == branch.step_day()
    assert env.market.snapshot() == branch.market.snapshot()
    frozen = branch.clone()
    for asset in frozen.market.assets:
        frozen.set_persistent_regulatory_controls(scope="asset", target_id=asset,
            controls={"trade_throttle": 1., "message_filter": 0., "spoof_filter": 0., "wash_filter": 0.})
    frozen.step_day()
    assert all(s.displayed_buy_size == s.displayed_sell_size == 0 for s in frozen.market.assets.values())
    assert all(capacity == 0 for capacity in frozen.market.maker_execution_capacity.values())
    with pytest.raises(ValueError, match="snapshot schema"):
        env.set_state(replace(env.get_state(), schema_version=4))


def test_unified_harm_does_not_change_when_activity_counters_change():
    env = WolfSocietySimulator(load_stock_market("spoofing"), 20, .1, seed=11)
    before = env._collapse_components(0)
    state = env.market.assets[env.target_asset]
    state.cancel_count, state.order_count, state.spoof_buy_size = 1000, 1000, 1e12
    after = env._collapse_components(0)
    assert before == after
    signal = primary_failure_signal(env.scenario.id, after, {"wash_share": 1., "volume_distortion": 1e12})
    assert set(signal["components"]) == {"retail_loss", "price_dislocation", "liquidity_stress"}
    assert signal["status"] == "not_calibrated_not_a_failure_label"
    assert not signal["triggered"]


def windows(n=10):
    rng = np.random.default_rng(19)
    return [MarketWindow(str(i), tuple(rng.normal(0, .01, 30)), tuple(rng.uniform(1, 3, 30))) for i in range(n)]


def test_moments_never_pair_across_episode_boundaries():
    rows = windows()
    r = np.asarray([w.returns for w in rows])
    result = summarize(rows)
    assert result["metrics"]["abs_acf5"] == pytest.approx(correlation(abs(r[:, :-5]).ravel(), abs(r[:, 5:]).ravel()))
    zero = MarketWindow("constant", (0.,) * 30, (0.,) * 30)
    assert summarize([zero])["metrics"]["volume_cv"] is None
    assert summarize([zero])["metric_coverage"]["volume_cv"] == 0
    with pytest.raises(ValueError):
        summarize([zero, zero])


def test_wasserstein_and_matched_sample_sizes():
    assert wasserstein_1([0, 2], [1]) == 1
    assert wasserstein_1([0, 2], [0, 2]) == 0
    result = compare_distributions(windows(3), windows(10), repetitions=10)
    assert all(row["matched_n"] == 3 for row in result.values())
    assert result == compare_distributions(windows(3), windows(10), repetitions=10)
    for kind in ("gaussian", "block_bootstrap"):
        assert baseline_windows(windows(), kind=kind, count=3, seed=4) == baseline_windows(windows(), kind=kind, count=3, seed=4)


def test_reference_split_adjustment_nonoverlap_and_closed_test(tmp_path):
    data = {}
    days = [(date(2010, 1, 1) + timedelta(days=i)).isoformat() for i in range(62)]
    for i, day in enumerate(days):
        price = 100 if i < 10 else 50
        data[day] = {f"{j}. {key}": str(price) for j, key in enumerate(("open", "high", "low", "close"), 1)}
        data[day].update({"6. volume": 100 if i < 10 else 200, "8. split coefficient": 2 if i == 10 else 1})
    # A malformed final-test row must not be inspected by calibration.
    data["2025-01-01"] = {"not": "valid market data"}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"Meta Data": {"2. Symbol": "IBM"}, "Time Series (Daily)": data}))
    rows = load_reference(path, days[0], days[-1], symbol="IBM")
    assert len(rows) == 2
    assert all(r == 0 for w in rows for r in w.returns)
    assert rows[0].volumes == (100.,) * 30
    assert days[30] in rows[0].identity and days[31] in rows[1].identity
    with pytest.raises(ValueError, match="symbol"):
        load_reference(path, days[0], days[-1], symbol="WRONG")
