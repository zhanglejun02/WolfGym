"""One market contract, with attack selection independent of market parameters.

The profile deliberately has no registered benchmark failure criterion yet.
Mechanism validity and a calibrated alpha must precede benchmark generation.
"""
from dataclasses import replace
from pathlib import Path

from wolfgym.core.scenarios.base import ScenarioConfig, load_scenario


STOCK_SOCIETY_SIZE = 500
ATTACK_FAMILIES = ("pump_and_dump", "finfluencer", "spoofing", "wash_trading")
_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config/scenarios/stock_market_v2.yaml"


def load_stock_market(attack_family: str | None = None) -> ScenarioConfig:
    """Select a single mechanism from the four-family development catalog.

    None selects the normal-market control (use alpha=0). Attacker wealth and
    behavior may differ; assets, retail composition and market rules do not.
    No default alpha is selected until the revised dynamics are validated.
    """
    scenario = load_scenario(str(_PROFILE_PATH))
    if attack_family is None:
        return replace(scenario, attackers={})
    if attack_family not in scenario.attackers:
        raise ValueError(f"unknown attack family {attack_family!r}; expected {ATTACK_FAMILIES}")
    return replace(scenario, attackers={attack_family: scenario.attackers[attack_family]})
