"""Versioned regression values, separate from accounting correctness tests.

V4 adds configured inventory/volatility limits and attack scheduling variation.
These legacy presets keep their prior parameters and numerical values. These
values guard reproducibility, not economic validity or a collapse threshold.
"""

import pytest

from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.core.version import SIMULATOR_VERSION


GOLDEN = {
    "s0": (0.0, None, 0.9091468297134168, -1806.1074597931583),
    "s1": (1.0, 4, 2.1236077790709795, 856.251346989593),
    "s2": (1.0, 4, 2.316308761107374, 18789.29073055915),
    "s3": (0.0, None, 0.8631765401112181, 668.4270951676881),
    "s4": (1.0, 4, 1.13605689354736, -2147.5432144528604),
}


@pytest.mark.parametrize("scenario_id", sorted(GOLDEN))
def test_no_regulator_matches_current_golden_episode(scenario_id: str) -> None:
    assert SIMULATOR_VERSION == "wolfgym-market-ledger-v8"
    alpha = 0.0 if scenario_id == "s0" else 0.10
    result = WolfSocietySimulator(
        load_scenario(scenario_id),
        n_society=40,
        alpha=alpha,
        seed=7,
    ).run()
    expected = GOLDEN[scenario_id]
    actual = (
        result.metrics.collapse_rate,
        result.metrics.collapse_day,
        result.metrics.max_collapse_score,
        result.metrics.retail_loss_30d,
    )
    assert actual == expected
