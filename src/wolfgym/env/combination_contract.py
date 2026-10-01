"""Fail-closed primary entry contract, distinct from the parameterized base."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from wolfgym.core.scenarios.stock_market import load_stock_market
from wolfgym.evaluation.switching_probe import check_wallets


def primary_contract():
    return json.loads(Path(__file__).with_name('combination_v51.json').read_text())


@dataclass(frozen=True)
class CombinationSettings:
    society_size: int = 500
    alpha: float = .02
    initial_budget: float = 12.
    horizon_days: int = 30
    initial_stock_share: float = .20
    daily_notional_multiple: float = 1.
    daily_message_cap: float = 1.8

    def __post_init__(self):
        contract = primary_contract()
        p = contract['population']
        expected = {'society_size': p['society_size'], 'alpha': p['harmful_fraction'],
                    'initial_budget': contract['initial_budget'], 'horizon_days': contract['horizon_days'],
                    'initial_stock_share': p['harmful_initial_stock_share'],
                    'daily_notional_multiple': p['daily_submitted_notional_per_initial_wealth_max'],
                    'daily_message_cap': p['daily_message_intensity_sum_max']}
        for key, value in asdict(self).items():
            if isinstance(value, bool) or value != expected[key]:
                raise ValueError(f'primary task locks {key}={expected[key]}; use a separate extension')
        if type(self.society_size) is not int or type(self.horizon_days) is not int:
            raise ValueError('society_size and horizon_days must be integers')


def validate_primary_scenario(scenario=None):
    # Check deployed package data/source before allocating or advancing a market.
    root = Path(__file__).resolve().parents[1]
    for path, expected in primary_contract()['source_files'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'primary task source/resource contract changed: {path}')
    canonical = load_stock_market('finfluencer')
    if scenario is not None and asdict(scenario) != asdict(canonical):
        raise ValueError('primary task scenario/resources cannot be overridden')
    return canonical


def validate_primary_population(simulator):
    p = primary_contract()['population']
    if (simulator.n_society != p['society_size'] or simulator.alpha != p['harmful_fraction']
            or len(simulator.society.attackers) != p['harmful_accounts']
            or len(simulator.society.retail) != p['ordinary_accounts']
            or len(simulator.society.market_makers) != p['market_maker_accounts']
            or simulator.society.placement != p['harmful_placement']):
        raise ValueError('constructed primary population violates the contract')
    check_wallets(simulator)
    prices = {key: a.price for key, a in simulator.market.assets.items()}
    for agents, wealth, share in (
        (simulator.society.retail, p['ordinary_initial_wealth'], p['ordinary_initial_stock_share']),
        (simulator.society.attackers, p['harmful_initial_wealth'], p['harmful_initial_stock_share'])):
        for agent in agents:
            account = agent.portfolio
            stock = math.fsum(q * prices[k] for k, q in account.holdings.items())
            if (account.initial_wealth != wealth or not math.isclose(stock / wealth, share)
                    or not math.isclose(account.cash + stock, wealth)):
                raise ValueError('constructed primary endowment violates the contract')
    for agent in simulator.society.attackers:
        if (agent.daily_notional_multiple != p['daily_submitted_notional_per_initial_wealth_max']
                or agent.daily_message_cap != p['daily_message_intensity_sum_max']):
            raise ValueError('constructed primary attack resource cap differs')
