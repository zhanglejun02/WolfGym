"""Trusted terminal scoring for the predeclared investor-protection protocol.

Pure accounting: no simulation, threshold fitting, policy observations, or
automatic benchmark release. Reference/WAIT/policy must share one initial
population and exogenous continuation. Common terminal fundamentals are a
declared simulator valuation convention, not observable real-world fair value.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


WELFARE_PROTOCOL_VERSION = "wolfgym-investor-protection-v2"
COLLAPSE_DEFINITION_VERSION = "wolfgym-investor-harm-collapse-v1"
COLLAPSE_INDIVIDUAL_SHORTFALL = .05
COLLAPSE_COHORT_FRACTION = .10
COLLAPSE_PERSISTENCE_DAYS = 2


@dataclass(frozen=True)
class InvestorAccount:
    initial_wealth: float
    cash: float
    holdings: Mapping[str, float]
    initial_group: str


@dataclass(frozen=True)
class WelfareSnapshot:
    # Harness-owned identity binding initial snapshot, continuation seed,
    # simulator/source/config fingerprint and the protected cohort.
    pair_id: str
    day: int
    fundamentals: Mapping[str, float]
    accounts: Mapping[str, InvestorAccount]


def capture_welfare_snapshot(simulator, *, pair_id: str) -> WelfareSnapshot:
    """Capture trusted daily accounts without advancing or mutating state."""
    return WelfareSnapshot(
        pair_id=pair_id, day=simulator._next_day,
        fundamentals={key: float(asset.fundamental) for key, asset in simulator.market.assets.items()},
        accounts={agent.agent_id: InvestorAccount(
            float(agent.portfolio.initial_wealth), float(agent.portfolio.cash),
            dict(agent.portfolio.holdings), str(agent.sub_role),
        ) for agent in simulator.society.retail},
    )


def _validate(snapshot: WelfareSnapshot, *, terminal: bool = True) -> None:
    if terminal and snapshot.day != 30:
        raise ValueError("v2 requires a paired, nonempty cohort at the fixed day-30 endpoint")
    if (not snapshot.pair_id or not snapshot.accounts or not snapshot.fundamentals
            or type(snapshot.day) is not int or not 1 <= snapshot.day <= 30):
        raise ValueError("v2 requires a paired, nonempty cohort at a day-1 to day-30 close")
    if any(not math.isfinite(value) or value <= 0 for value in snapshot.fundamentals.values()):
        raise ValueError("terminal fundamental marks must be finite and positive")
    for account in snapshot.accounts.values():
        if not math.isfinite(account.initial_wealth) or account.initial_wealth <= 0:
            raise ValueError("initial investor wealth must be finite and positive")
        if not math.isfinite(account.cash) or account.cash < -1e-8:
            raise ValueError("invalid cash balance")
        if not account.initial_group:
            raise ValueError("initial group is required for distribution diagnostics")
        if set(account.holdings) - set(snapshot.fundamentals):
            raise ValueError("holding has no common terminal mark")
        if any(not math.isfinite(value) or value < -1e-8 for value in account.holdings.values()):
            raise ValueError("invalid share balance")


def _wealth(account: InvestorAccount, marks: Mapping[str, float]) -> float:
    return account.cash + math.fsum(quantity * marks[asset] for asset, quantity in account.holdings.items())


def _validate_cohort(reference: WelfareSnapshot, branch: WelfareSnapshot) -> None:
    if branch.pair_id != reference.pair_id:
        raise ValueError("counterfactual branches have different pairing identities")
    if set(branch.accounts) != set(reference.accounts):
        raise ValueError("counterfactual branches must preserve the protected cohort")
    for key, account in branch.accounts.items():
        initial = reference.accounts[key]
        if account.initial_wealth != initial.initial_wealth or account.initial_group != initial.initial_group:
            raise ValueError("initial endowment/group changed between branches")


def _validate_pair(reference: WelfareSnapshot, branch: WelfareSnapshot) -> None:
    _validate_cohort(reference, branch)
    if branch.day != reference.day:
        raise ValueError("counterfactual closes must have the same day")
    if set(branch.fundamentals) != set(reference.fundamentals) or any(
        not math.isclose(branch.fundamentals[key], reference.fundamentals[key], rel_tol=1e-12, abs_tol=1e-12)
        for key in reference.fundamentals
    ):
        raise ValueError("exogenous fundamentals diverged; paired scoring is invalid")


def score_collapse_history(reference: Sequence[WelfareSnapshot], branch: Sequence[WelfareSnapshot]) -> dict:
    """Score a complete day-1 prefix, using only each close's common marks.

    Collapse requires the SAME >=10% of initial retail accounts to each have
    >=5% initial-wealth shortfall at two consecutive closes. Once observed,
    recovery cannot erase the episode event. This trusted evaluator neither
    exposes reference accounts to a policy nor terminates the simulator.
    """
    if not reference or len(reference) != len(branch):
        raise ValueError("collapse requires nonempty equal-length paired histories")
    # Exact integer ceiling for the fixed 10% fraction, including small cohorts.
    required = (len(reference[0].accounts) + 9) // 10
    previous_harmed: set[str] = set()
    first_day = None
    daily = []
    for day, (ref, actual) in enumerate(zip(reference, branch), start=1):
        _validate(ref, terminal=False)
        _validate(actual, terminal=False)
        if ref.day != day or actual.day != day:
            raise ValueError("collapse histories must contain every close starting at day 1")
        _validate_cohort(reference[0], ref)
        if set(ref.fundamentals) != set(reference[0].fundamentals):
            raise ValueError("asset universe changed during the episode")
        _validate_pair(ref, actual)
        harmed = {
            key for key, account in ref.accounts.items()
            if _wealth(account, ref.fundamentals) - _wealth(actual.accounts[key], ref.fundamentals)
            >= COLLAPSE_INDIVIDUAL_SHORTFALL * account.initial_wealth
        }
        sustained_count = len(harmed & previous_harmed)
        event = day >= COLLAPSE_PERSISTENCE_DAYS and sustained_count >= required
        if event and first_day is None:
            first_day = day
        daily.append({"day": day, "harmed_accounts": len(harmed),
                      "sustained_harmed_accounts": sustained_count, "collapse_at_close": event})
        previous_harmed = harmed
    return {
        "definition": COLLAPSE_DEFINITION_VERSION,
        "individual_shortfall_threshold": COLLAPSE_INDIVIDUAL_SHORTFALL,
        "cohort_fraction_threshold": COLLAPSE_COHORT_FRACTION,
        "persistence_days": COLLAPSE_PERSISTENCE_DAYS,
        "protected_investors": len(reference[0].accounts),
        "required_harmed_accounts": required,
        "collapse": first_day is not None,
        "first_collapse_day": first_day,
        "observed_through_day": reference[-1].day,
        "horizon_complete": reference[-1].day == 30,
        "daily": daily,
        "release_eligible": False,
    }


def _shortfall(reference: WelfareSnapshot, branch: WelfareSnapshot) -> dict:
    marks = reference.fundamentals
    ids = sorted(reference.accounts)
    gaps = {key: _wealth(reference.accounts[key], marks) - _wealth(branch.accounts[key], marks) for key in ids}
    initial_total = math.fsum(reference.accounts[key].initial_wealth for key in ids)
    positive = {key: max(0., gaps[key]) for key in ids}
    individual_rates = sorted((positive[key] / reference.accounts[key].initial_wealth for key in ids), reverse=True)
    tail_count = max(1, math.ceil(.1 * len(ids)))
    groups = sorted({account.initial_group for account in reference.accounts.values()})
    return {
        "positive_shortfall_rate": math.fsum(positive.values()) / initial_total,
        "signed_net_gap_rate": math.fsum(gaps.values()) / initial_total,
        "worst_decile_mean_shortfall_rate": math.fsum(individual_rates[:tail_count]) / tail_count,
        "initial_group_shortfall_rates": {
            group: math.fsum(positive[key] for key in ids if reference.accounts[key].initial_group == group)
            / math.fsum(reference.accounts[key].initial_wealth for key in ids if reference.accounts[key].initial_group == group)
            for group in groups},
    }


def score_welfare_triplet(reference: WelfareSnapshot, wait: WelfareSnapshot, policy: WelfareSnapshot) -> dict:
    """Compare policy and WAIT to the SAME attacker-off/WAIT reference.

    In clean episodes reference is exactly the clean WAIT snapshot. Reference
    is never recomputed under the evaluated policy. The signed reduction is
    retained even when intervention makes protected investors worse off.
    """
    for branch in (reference, wait, policy):
        _validate(branch)
        _validate_pair(reference, branch)
    baseline, controlled = _shortfall(reference, wait), _shortfall(reference, policy)
    return {
        "protocol": WELFARE_PROTOCOL_VERSION,
        "valuation": "common_exogenous_terminal_fundamentals",
        "protected_investors": len(reference.accounts),
        "wait": baseline, "policy": controlled,
        "shortfall_reduction": baseline["positive_shortfall_rate"] - controlled["positive_shortfall_rate"],
        "net_wealth_gain": baseline["signed_net_gap_rate"] - controlled["signed_net_gap_rate"],
        "worst_decile_shortfall_reduction": baseline["worst_decile_mean_shortfall_rate"] - controlled["worst_decile_mean_shortfall_rate"],
        "release_eligible": False,
    }
