"""Matched-episode CR evaluation with public observations and raw cost axes."""
from __future__ import annotations

import hashlib
import random
from statistics import mean
from collections import defaultdict
from typing import Callable, Mapping, Sequence

from wolfgym.benchmark.management import public_candidates
from wolfgym.benchmark.sampling import StateSample
from wolfgym.env import WolfGymEnv, WolfGymEnvConfig
from wolfgym.evaluation.interactive import InteractiveEvaluator
from wolfgym.observations import ObservationConfig
from wolfgym.counterfactual.failure import SystemFailureCriterion
from wolfgym.regulation import DEFAULT_ACTION_COSTS, RegulatoryAction, RegulatoryActionType


Controller = Callable[[Mapping, Sequence[Mapping]], RegulatoryAction]


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, resamples: int = 2000,
    groups: Sequence[str] | None = None,
) -> list[float] | None:
    if groups is not None:
        values = _group_means(values, groups)
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    return [
        float(estimates[int(0.025 * (resamples - 1))]),
        float(estimates[int(0.975 * (resamples - 1))]),
    ]


def _group_means(values: Sequence[float], groups: Sequence[str]) -> list[float]:
    if len(values) != len(groups):
        raise ValueError("one trajectory group is required for each outcome")
    grouped = defaultdict(list)
    for group, value in zip(groups, values):
        grouped[group].append(float(value))
    return [mean(grouped[group]) for group in sorted(grouped)]


def grouped_summary(values: Sequence[float], groups: Sequence[str], *, seed: int = 42) -> dict:
    means = _group_means(values, groups)
    return {"mean": mean(means) if means else None,
            "ci_95": _bootstrap_mean_ci(means, seed=seed),
            "independent_groups": len(means), "weighting": "equal_trajectory"}


def _bootstrap_seed(label: str) -> int:
    return int.from_bytes(hashlib.blake2s(label.encode(), digest_size=4).digest(), "big")


def baseline_controller(kind: str) -> Controller:
    """Simple public-observation anchors, not oracle-guided policies."""
    if kind not in {"wait", "throttle_once", "freeze_once"}:
        raise ValueError(f"unknown baseline {kind!r}")

    issued = False

    def decide(observation, menu):
        nonlocal issued
        if not issued and kind != "wait":
            wanted = "THROTTLE" if kind == "throttle_once" else "FREEZE"
            for action in menu:
                if action["type"] == wanted:
                    issued = True
                    return RegulatoryAction(RegulatoryActionType[wanted], action["target_id"])
        return RegulatoryAction(RegulatoryActionType.WAIT)

    return decide


def evaluate_controllers(
    samples: Sequence[StateSample],
    controller_factories: Mapping[str, Callable[[], Controller]],
    seeds: Sequence[int], *, horizon: int = 8, budget: float = 12.0,
    failure_criterion: SystemFailureCriterion | None = None,
    allow_candidate_criterion: bool = False,
) -> dict:
    """Each controller gets a fresh instance and the same snapshot/seed pairs.

    The controller sees neither simulator objects nor trace metadata. Only
    direct public candidates within budget are allowed in this main CR track.
    """
    if not samples or not controller_factories:
        raise ValueError("nonempty samples and controllers are required")
    if horizon <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("positive horizon and unique nonempty seeds are required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    if len({sample.instance_id for sample in samples}) != len(samples):
        raise ValueError("CR initial states must be unique")
    results = {}
    for name, factory in controller_factories.items():
        episodes = []
        for sample in samples:
            if sample.state.horizon_days - sample.state.next_day < horizon:
                raise ValueError("CR snapshot has insufficient remaining horizon")
            for seed in seeds:
                controller = factory()
                env = WolfGymEnv(
                    sample.simulator, sample.state,
                    config=WolfGymEnvConfig(initial_budget=budget, max_interactions=horizon,
                                            observation=ObservationConfig(),
                                            intervention_duration_steps=1,
                                            allow_candidate_criterion=allow_candidate_criterion),
                    episode_id=f"{sample.instance_id}:{seed}:{name}",
                    trajectory_id=sample.trajectory_id,
                    failure_criterion=failure_criterion,
                )
                observation, _ = env.reset(seed=seed)
                if observation.active_controls:
                    raise ValueError("main CR requires an uncontrolled initial snapshot")
                for _ in range(horizon):
                    candidates = tuple(
                        action for action in public_candidates(observation.to_dict())
                        if DEFAULT_ACTION_COSTS[action.action_type] <= observation.remaining_budget
                    )
                    menu = tuple({**action.to_dict(), "cost": DEFAULT_ACTION_COSTS[action.action_type]}
                                 for action in candidates)
                    action = controller(observation.to_dict(), menu)
                    if action not in candidates:
                        raise ValueError(f"controller {name!r} selected an unavailable action")
                    observation, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                metrics = InteractiveEvaluator().evaluate(env.trace).to_dict()
                episodes.append({
                    "instance_id": sample.instance_id, "trajectory_id": sample.trajectory_id,
                    "bootstrap_group": repr((sample.scenario_id, sample.society_size,
                                             sample.alpha, sample.trajectory_seed)),
                    "seed": seed, "metrics": metrics, "trace": env.trace.to_dict(),
                })
        groups = [row["bootstrap_group"] for row in episodes]
        results[name] = {
            "episode_count": len(episodes), "episodes": episodes,
            "system_failure_rate": mean(_group_means([row["metrics"]["system_failure"] for row in episodes], groups)) if episodes else None,
            "mean_intervention_cost": mean(_group_means([row["metrics"]["intervention_cost"] for row in episodes], groups)) if episodes else None,
            "system_failure_rate_bootstrap_ci_95": _bootstrap_mean_ci(
                [row["metrics"]["system_failure"] for row in episodes],
                seed=_bootstrap_seed(f"{name}:failure"), groups=groups,
            ),
            "mean_intervention_cost_bootstrap_ci_95": _bootstrap_mean_ci(
                [row["metrics"]["intervention_cost"] for row in episodes],
                seed=_bootstrap_seed(f"{name}:cost"), groups=groups,
            ),
        }
    frontier = []
    for name, result in results.items():
        if result["system_failure_rate"] is None:
            continue
        point = (result["system_failure_rate"], result["mean_intervention_cost"])
        dominated = any(
            other["system_failure_rate"] is not None
            and other["system_failure_rate"] <= point[0]
            and other["mean_intervention_cost"] <= point[1]
            and (other["system_failure_rate"], other["mean_intervention_cost"]) != point
            for other in results.values()
        )
        if not dominated:
            frontier.append(name)
    paired = {}
    if "wait" in results:
        reference = {
            (row["instance_id"], row["seed"]): row for row in results["wait"]["episodes"]
        }
        for name, result in results.items():
            if name == "wait":
                continue
            rows = {
                (row["instance_id"], row["seed"]): row for row in result["episodes"]
            }
            if set(rows) != set(reference):
                raise RuntimeError("controllers do not share identical snapshot/seed pairs")
            failure_differences = [
                rows[key]["metrics"]["system_failure"]
                - reference[key]["metrics"]["system_failure"]
                for key in sorted(rows)
            ]
            cost_differences = [
                rows[key]["metrics"]["intervention_cost"]
                - reference[key]["metrics"]["intervention_cost"]
                for key in sorted(rows)
            ]
            paired_groups = [rows[key]["bootstrap_group"] for key in sorted(rows)]
            paired[name] = {
                "reference": "wait",
                "system_failure_rate_difference": mean(_group_means(failure_differences, paired_groups)),
                "system_failure_rate_difference_bootstrap_ci_95": _bootstrap_mean_ci(
                    failure_differences, seed=_bootstrap_seed(f"{name}:paired-failure"), groups=paired_groups,
                ),
                "mean_intervention_cost_difference": mean(_group_means(cost_differences, paired_groups)),
                "mean_intervention_cost_difference_bootstrap_ci_95": _bootstrap_mean_ci(
                    cost_differences, seed=_bootstrap_seed(f"{name}:paired-cost"), groups=paired_groups,
                ),
            }
    return {
        "protocol": "wolfgym-cr-v4",
        "bootstrap_unit": "physical_trajectory", "weighting": "equal_trajectory", "horizon": horizon, "initial_budget": budget,
        "intervention_duration_steps": 1,
        "intervention_cost_basis": "sum_of_per_step_issued_action_costs",
        "unique_initial_states": len(samples), "matched_seeds": list(seeds),
        "controllers": results, "paired_comparisons_vs_wait": paired,
        "pareto_frontier": sorted(frontier),
    }
