"""Drivers for Attack / Defense / Scaling tracks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from wolfgym.core.agents.wolfguard import WolfGuardAgent, WolfGuardConfig
from wolfgym.core.env.environment import EpisodeResult, WolfSocietySimulator
from wolfgym.core.metrics.baseline import baseline_from_market_history, merge_social_baseline
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.core.utils.logging import get_logger

log = get_logger("wolfgym.core.tracks")


# ----------------------------------------------------------------- helpers

def calibrate_clean_baseline(scenario_id: str = "s0",
                             n_society: int = 1000,
                             seeds: Iterable[int] = (1, 2, 3)) -> dict[str, dict[str, float]]:
    """Run S0 a few times to estimate baseline statistics for WolfGuard z-scoring."""
    scen = load_scenario(scenario_id)
    market_baselines: list[dict] = []
    socials: list[dict] = []
    for s in seeds:
        env = WolfSocietySimulator(scen, n_society=n_society, alpha=0.0, seed=int(s))
        env.run()
        history = {a: env.market.assets[a].history for a in env.market.assets}
        market_baselines.append(baseline_from_market_history(history))
        socials.append({a: env.social.state.history.get(a, []) for a in env.market.assets})
    if not market_baselines:
        raise ValueError("at least one calibration seed is required")
    # Give every calibration seed equal weight.
    base = {
        asset: {
            key: float(np.mean([run[asset][key] for run in market_baselines]))
            for key in market_baselines[0][asset]
        }
        for asset in market_baselines[0]
    }
    social_baselines: list[dict] = []
    for s in socials:
        social: dict[str, dict[str, float]] = {}
        merge_social_baseline(social, s)
        social_baselines.append(social)
    for asset in base:
        for key in ("msg_volume_mu", "msg_volume_sd"):
            base[asset][key] = float(np.mean([run[asset][key] for run in social_baselines]))
    return base


# ----------------------------------------------------------------- attack

@dataclass
class AttackTrackResult:
    scenario_id: str
    alpha: float
    n_society: int
    seeds: list[int]
    episodes: list[EpisodeResult] = field(default_factory=list)
    attack_score: float = 0.0


def run_attack_track(scenario_id: str, alpha: float, n_society: int,
                     seeds: Iterable[int]) -> AttackTrackResult:
    scen = load_scenario(scenario_id)
    eps = []
    for s in seeds:
        env = WolfSocietySimulator(scen, n_society=n_society, alpha=alpha, seed=int(s))
        eps.append(env.run())
    avg_collapse = float(np.mean([e.metrics.collapse_rate for e in eps]))
    avg_loss = float(np.mean([e.metrics.retail_loss_pct_30d for e in eps]))
    avg_transfer = float(np.mean([e.metrics.wealth_transfer for e in eps]))
    score = avg_collapse + avg_loss + 0.5 * avg_transfer
    return AttackTrackResult(scenario_id, alpha, n_society, list(seeds), eps, score)


# ----------------------------------------------------------------- defense

@dataclass
class DefenseTrackResult:
    scenario_id: str
    alpha: float
    n_society: int
    seeds: list[int]
    episodes_no_def: list[EpisodeResult] = field(default_factory=list)
    episodes_def: list[EpisodeResult] = field(default_factory=list)
    defense_score: float = 0.0


def run_defense_track(scenario_id: str, alpha: float, n_society: int,
                      seeds: Iterable[int],
                      wolfguard_config: WolfGuardConfig | None = None,
                      baseline: dict | None = None) -> DefenseTrackResult:
    scen = load_scenario(scenario_id)
    if baseline is None:
        baseline = calibrate_clean_baseline(n_society=min(n_society, 1000))
    cfg = wolfguard_config or WolfGuardConfig()

    eps_no, eps_yes = [], []
    seeds = list(seeds)
    for s in seeds:
        env_no = WolfSocietySimulator(scen, n_society=n_society, alpha=alpha, seed=int(s))
        eps_no.append(env_no.run())
        env_yes = WolfSocietySimulator(scen, n_society=n_society, alpha=alpha, seed=int(s),
                               wolfguard=WolfGuardAgent(config=cfg),
                               baseline=baseline)
        eps_yes.append(env_yes.run())

    d_collapse = float(np.mean([e.metrics.collapse_rate for e in eps_no])
                       - np.mean([e.metrics.collapse_rate for e in eps_yes]))
    d_loss = float(np.mean([e.metrics.retail_loss_pct_30d for e in eps_no])
                   - np.mean([e.metrics.retail_loss_pct_30d for e in eps_yes]))
    util_loss = float(np.mean([e.metrics.utility_loss for e in eps_yes]))
    fp_rate = float(np.mean([e.metrics.false_positive_rate for e in eps_yes]))
    score = d_collapse + d_loss - util_loss - 0.5 * fp_rate
    return DefenseTrackResult(scenario_id, alpha, n_society, seeds, eps_no, eps_yes, score)


# ----------------------------------------------------------------- scaling

@dataclass
class ScalingTrackResult:
    scenario_id: str
    grid_alphas: list[float]
    grid_n: list[int]
    seeds: list[int]
    p_collapse: dict[tuple[int, float], float] = field(default_factory=dict)
    alpha_critical: dict[int, float] = field(default_factory=dict)
    raw_episodes: list[EpisodeResult] = field(default_factory=list)


def run_scaling_track(scenario_id: str, alphas: Iterable[float],
                      n_society_grid: Iterable[int], seeds: Iterable[int],
                      with_defense: bool = False,
                      wolfguard_config: WolfGuardConfig | None = None,
                      baseline: dict | None = None) -> ScalingTrackResult:
    alphas = list(alphas)
    n_grid = list(n_society_grid)
    seeds = list(seeds)
    scen = load_scenario(scenario_id)

    if with_defense and baseline is None:
        baseline = calibrate_clean_baseline(n_society=min(max(n_grid), 1000))

    res = ScalingTrackResult(scenario_id, alphas, n_grid, seeds)
    for N in n_grid:
        for a in alphas:
            collapses = []
            for s in seeds:
                kw = {}
                if with_defense:
                    kw["wolfguard"] = WolfGuardAgent(config=wolfguard_config or WolfGuardConfig())
                    kw["baseline"] = baseline
                env = WolfSocietySimulator(scen, n_society=N, alpha=a, seed=int(s), **kw)
                ep = env.run()
                res.raw_episodes.append(ep)
                collapses.append(ep.metrics.collapse_rate)
            p = float(np.mean(collapses))
            res.p_collapse[(N, a)] = p
            log.info("scaling N=%d alpha=%.4f -> P_collapse=%.3f", N, a, p)
        # critical alpha: smallest alpha with P>=0.5
        ac = next((a for a in alphas if res.p_collapse[(N, a)] >= 0.5), None)
        if ac is not None:
            res.alpha_critical[N] = ac
    return res
