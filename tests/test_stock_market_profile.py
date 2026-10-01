"""Unified-market configuration contracts, not population validity claims."""
from dataclasses import asdict

import pytest

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.stock_market import ATTACK_FAMILIES, STOCK_SOCIETY_SIZE, load_stock_market
from wolfgym.counterfactual.failure import failure_criterion_for_scenario


def test_attack_selection_never_changes_market_contract():
    configs = [load_stock_market(family) for family in (None, *ATTACK_FAMILIES)]
    markets = []
    for scenario in configs:
        market = asdict(scenario)
        market.pop("attackers")
        markets.append(market)
    assert all(market == markets[0] for market in markets)
    assert STOCK_SOCIETY_SIZE == 500
    assert configs[0].horizon_days == 30
    assert len(configs[0].assets) == 3


@pytest.mark.parametrize("family", [None, *ATTACK_FAMILIES])
def test_population_and_capital_are_explicit(family):
    alpha = 0 if family is None else 0.02  # structural example, not a chosen operating point
    env = WolfSocietySimulator(load_stock_market(family), STOCK_SOCIETY_SIZE, alpha, seed=2)
    assert len(env.society.retail) + len(env.society.attackers) == 500
    assert len(env.society.attackers) == round(500 * alpha)
    assert len(env.society.market_makers) == 3
    assert env.market.assets["asset_2"].base_liquidity == 300
    assert env.market.assets["asset_2"].daily_execution_liquidity == 12000
    if family:
        assert env.society.attackers[0].portfolio.initial_wealth == 50000


def test_configs_are_independent_and_unapproved_modes_fail_closed():
    first, second = load_stock_market("pump_and_dump"), load_stock_market("pump_and_dump")
    first.retail["initial_wealth"] = 1
    assert second.retail["initial_wealth"] == 10000
    with pytest.raises(ValueError, match="unknown attack family"):
        load_stock_market("unknown")
    with pytest.raises(KeyError, match="no audited"):
        failure_criterion_for_scenario(second.id)
    with pytest.raises(ValueError, match="requires an attack"):
        WolfSocietySimulator(load_stock_market(), 500, 0.02)
