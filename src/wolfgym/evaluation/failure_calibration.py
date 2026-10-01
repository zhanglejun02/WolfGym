"""Candidate harm-event calibration. Never registers a benchmark failure event.

Calibrate whole-episode sustained loss, not correlated day rows. A low clean
false-alarm rate is a statistical property, not certification of market realism.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

from wolfgym.core.version import SIMULATOR_VERSION


@dataclass(frozen=True)
class LossEpisode:
    trajectory_id: str
    seed: int
    losses: tuple[float, ...]  # max(0, 1 - retail wealth / initial retail wealth)
    clean: bool

    def __post_init__(self):
        if not self.trajectory_id or not self.losses:
            raise ValueError("episode identity and complete loss series are required")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in self.losses):
            raise ValueError("retail loss must be finite and in [0, 1]")


def sustained_loss_score(losses: Sequence[float], persistence: int) -> float:
    """max(window minima): score > threshold iff a complete run exceeds it.

    Strict comparison handles tied/zero calibration scores conservatively.
    This event contains no cancellation, wash, cascade or attacker-profit data.
    """
    if isinstance(persistence, bool) or not isinstance(persistence, int) or persistence < 1:
        raise ValueError("persistence must be a positive integer")
    if len(losses) < persistence:
        raise ValueError("episode is shorter than the persistence window")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in losses):
        raise ValueError("invalid retail loss series")
    return max(min(losses[start:start + persistence])
               for start in range(len(losses) - persistence + 1))


def binomial_upper_bound(failures: int, n: int, confidence: float) -> float:
    """Invert the binomial CDF for a one-sided exact bound, without SciPy ABI dependencies."""
    if (any(isinstance(x, bool) or not isinstance(x, int) for x in (failures, n))
            or not 0 <= failures <= n or not 0 < confidence < 1):
        raise ValueError("invalid binomial counts/confidence")
    if n == 0 or failures == n:
        return 1.0
    if failures == 0:
        return -math.expm1(math.log1p(-confidence) / n)
    log_choose = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                  for j in range(failures + 1)]
    low, high = 0., 1.
    for _ in range(60):
        p = (low + high) / 2
        if p == low or p == high:
            break
        logs = [c + j * math.log(p) + (n - j) * math.log1p(-p)
                for j, c in enumerate(log_choose)]
        peak = max(logs)
        cdf = math.exp(peak) * math.fsum(math.exp(x - peak) for x in logs)
        if cdf > 1 - confidence:
            low = p
        else:
            high = p
    return high


def false_alarm_summary(episodes, threshold, persistence, confidence):
    n = len(episodes)
    failures = sum(sustained_loss_score(e.losses, persistence) > threshold for e in episodes)
    # Exact one-sided Clopper-Pearson upper bound; zero observations != zero risk.
    upper = binomial_upper_bound(failures, n, confidence)
    return {"episodes": n, "failures": failures, "rate": failures / n if n else None,
            "confidence": confidence, "one_sided_upper": upper,
            "interval_method": "Clopper-Pearson"}


def calibrate_loss_event(calibration: Sequence[LossEpisode], validation: Sequence[LossEpisode], *,
                         source_version: str, context_id: str, alpha: float = .01,
                         persistence: int = 2, confidence: float = .95) -> dict:
    """One preselected score, one threshold, disjoint held-out clean validation.

    The rank correction is marginal under exchangeable episodes; it is not a
    guarantee for a chosen calibration sample or a changed market population.
    Validation is not used to tune threshold/persistence or choose a sensitivity.
    """
    if source_version != SIMULATOR_VERSION:
        raise ValueError("calibration data simulator version does not match current simulator")
    if not context_id or not 0 < alpha < 1 or not 0 < confidence < 1:
        raise ValueError("context and probabilities must be valid")
    all_episodes = [*calibration, *validation]
    if not all_episodes or any(not e.clean for e in all_episodes):
        raise ValueError("calibration and validation must contain only clean episodes")
    if len({e.trajectory_id for e in all_episodes}) != len(all_episodes):
        raise ValueError("duplicate or overlapping trajectories")
    if len({e.seed for e in all_episodes}) != len(all_episodes):
        raise ValueError("seed reuse violates independent trajectory splits")
    horizons = {len(e.losses) for e in all_episodes}
    if len(horizons) != 1:
        raise ValueError("all episodes must use the same full horizon")
    scores = sorted(sustained_loss_score(e.losses, persistence) for e in calibration)
    for e in validation:
        sustained_loss_score(e.losses, persistence)
    # Never clamp an unsupported extreme quantile to the observed maximum.
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    threshold = scores[rank - 1] if 1 <= rank <= len(scores) else None
    identity_payload = {
        "event": "sustained_retail_initial_wealth_loss", "event_schema": "candidate-v1",
        "simulator_version": source_version, "context_id": context_id,
        "horizon": next(iter(horizons)), "persistence": persistence,
        "comparison": ">", "threshold": threshold, "target_false_alarm_rate": alpha,
        "calibration": [asdict(e) for e in calibration],
    }
    digest = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    report = {
        "schema_version": "wolfgym-clean-loss-calibration-v1", "status": "candidate_only",
        "criterion_identity": {"id": "sustained_retail_loss", "version": f"candidate-v1:{digest}"},
        "definition": {k: v for k, v in identity_payload.items() if k != "calibration"},
        "calibration_trajectory_ids": [e.trajectory_id for e in calibration],
        "validation_trajectory_ids": [e.trajectory_id for e in validation],
        "calibration_digest": digest, "rank": rank, "calibration_n": len(scores),
        "finite_threshold_available": threshold is not None,
        "clean_validation_passed": False, "release_eligible": False,
        "reason": "insufficient calibration episodes for the requested tail probability",
        "limitations": ["Null calibration does not validate financial realism or material harm.",
                        "No attack power, independent harm axis or normative severity floor certified.",
                        "Do not select parameters using held-out or sensitivity results."],
    }
    if threshold is None:
        return report
    report["calibration_false_alarm"] = false_alarm_summary(calibration, threshold, persistence, confidence)
    report["heldout_false_alarm"] = false_alarm_summary(validation, threshold, persistence, confidence)
    report["clean_validation_passed"] = report["heldout_false_alarm"]["one_sided_upper"] <= alpha
    report["reason"] = ("clean statistical check passed; financial/power/release review still required"
                        if report["clean_validation_passed"] else "held-out clean evidence insufficient or false-alarm rate too high")
    # Sensitivities change only one parameter, never silently replace the candidate.
    report["sensitivity"] = [
        {"threshold_factor": factor, "persistence": window,
         **false_alarm_summary(validation, threshold * factor, window, confidence)}
        for factor, window in sorted({(.8, persistence), (1., persistence), (1.2, persistence), (1., 1)})
    ]
    return report
