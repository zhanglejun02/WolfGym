"""Predeclared, split-safe validation of harm, public discrimination and control."""
from dataclasses import asdict, replace
import math

import numpy as np

from wolfgym.core.agents.switching import prepare_tactic
from wolfgym.evaluation.counterfactual_welfare import (
    capture_welfare_snapshot, score_collapse_history, _shortfall,
)
from wolfgym.evaluation.failure_calibration import binomial_upper_bound
from wolfgym.evaluation.switching_probe import account_totals, check_wallets
from wolfgym.observations import ObservationBuilder, ObservationConfig
from wolfgym.observations.schema import InformationLevel, RegulatoryObservation
from wolfgym.regulation import DEFAULT_ACTION_COSTS, RegulatoryAction, RegulatoryActionType
from wolfgym.regulation.executor import RegulatoryActionExecutor


def public_observation(simulator, remaining_budget):
    """Read existing public-schema builders without making two snapshot copies.

    Only the synchronous trusted runner calls this. Equality to the snapshot
    builder and absence of state changes are tested; policies get the value only.
    """
    builder = ObservationBuilder(ObservationConfig(top_k_entities=5))
    market, assets = builder._build_market(simulator)
    return RegulatoryObservation(day=simulator._next_day, society_size=simulator.n_society,
                                 information_level=InformationLevel.TOP_ENTITIES,
                                 market=market, assets=assets, social=builder._build_social(simulator),
                                 entities=builder._build_entities(simulator), remaining_budget=remaining_budget)


def public_features(observation):
    """No IDs, dates, budgets, hidden target, fundamentals or evaluator metrics."""
    result = {f"market_{key}": float(value) for key, value in asdict(observation.market).items()}
    result.update({f"social_{key}": float(value) for key, value in asdict(observation.social).items()
                   if key != "network_degree_concentration"})
    for field in ("return_3d", "volatility_3d", "volume_zscore", "order_imbalance", "cancellation_rate"):
        values = [getattr(asset, field) for asset in observation.assets]
        result[f"asset_max_{field}"] = float(max(values, default=0.))
        result[f"asset_min_{field}"] = float(min(values, default=0.))
    for field in ("message_reach", "trading_concentration", "cancellation_rate", "social_influence", "activity_change"):
        result[f"entity_max_{field}"] = float(max((getattr(entity, field) for entity in observation.entities), default=0.))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite public feature")
    return result


def fit_detector(features, labels, config):
    names = sorted(features[0])
    if any(sorted(row) != names for row in features):
        raise ValueError("feature schema mismatch")
    x = np.array([[row[key] for key in names] for row in features], dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(y) != len(x) or set(y) != {0., 1.}:
        raise ValueError("both development classes are required")
    weights = np.where(y == 1., .5 / sum(y == 1.), .5 / sum(y == 0.))
    center = np.sum(x * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((x - center) ** 2 * weights[:, None], axis=0))
    scale[scale < 1e-10] = 1.
    x = np.clip((x - center) / scale, -8., 8.)
    coefficient = np.zeros(x.shape[1])
    intercept = 0.
    for _ in range(config["iterations"]):
        probabilities = 1. / (1. + np.exp(-np.clip(x @ coefficient + intercept, -30., 30.)))
        error = (probabilities - y) * weights
        coefficient -= config["step_size"] * (x.T @ error + config["l2"] * coefficient)
        intercept -= config["step_size"] * float(error.sum())
    return {"features": names, "center": center.tolist(), "scale": scale.tolist(),
            "coefficient": coefficient.tolist(), "intercept": intercept,
            "training_rows": len(y), "positive_rows": int(sum(y)), "config": config}


def detector_score(features, model):
    if set(features) != set(model["features"]):
        raise ValueError("detector feature contract mismatch")
    x = np.asarray([features[key] for key in model["features"]], dtype=float)
    z = np.clip((x - model["center"]) / model["scale"], -8., 8.)
    logit = float(np.dot(z, model["coefficient"]) + model["intercept"])
    return 1. / (1. + math.exp(-max(-30., min(30., logit))))


def damage_series(reference, actual):
    old = score_collapse_history(reference, actual)  # Complete paired-prefix validation.
    daily = [_shortfall(r, a) for r, a in zip(reference, actual)]
    return {"H": [row["positive_shortfall_rate"] for row in daily],
            "G": [row["signed_net_gap_rate"] for row in daily],
            "tail": [row["worst_decile_mean_shortfall_rate"] for row in daily],
            "old_collapse": old}


def sustained_damage_score(harm, net, *, net_floor=.001):
    if not harm or len(harm) != len(net):
        raise ValueError("aligned nonempty daily harm and net series required")
    if any(not math.isfinite(x) or x < 0 for x in harm) or any(not math.isfinite(x) for x in net):
        raise ValueError("invalid daily damage")
    return max((min(harm[i - 1], harm[i]) for i in range(1, len(harm))
                if net[i - 1] >= net_floor and net[i] >= net_floor), default=0.)


def damage_event(harm, net, threshold, *, net_floor=.001):
    if not math.isfinite(threshold) or threshold < .0025:
        raise ValueError("v4 cannot go below its predeclared materiality floor")
    sustained_damage_score(harm, net, net_floor=net_floor)
    first = next((i + 1 for i in range(1, len(harm))
                  if min(harm[i - 1], harm[i]) > threshold and min(net[i - 1], net[i]) >= net_floor), None)
    return {"event": first is not None, "first_day": first, "threshold": threshold,
            "net_floor": net_floor, "horizon_complete": len(harm) == 30}


def exact_rate(successes, n):
    return {"successes": successes, "n": n, "rate": successes / n if n else None,
            "lower_95_one_sided": 1 - binomial_upper_bound(n - successes, n, .95),
            "upper_95_one_sided": binomial_upper_bound(successes, n, .95)}


def reseed_normal_decisions(simulator, seed):
    """Normal decision/propagation replica; do not alter the market RNG."""
    streams = np.random.SeedSequence(seed).spawn(1 + len(simulator.society.retail))
    simulator.social.set_rng(np.random.default_rng(streams[0]))
    for agent, stream in zip(simulator.society.retail, streams[1:]):
        agent.rng = np.random.default_rng(stream)
        if hasattr(agent, "base") and hasattr(agent.base, "rng"):
            agent.base.rng = agent.rng


def run_validation_branch(template, initial, *, seed, name, config, pair_id, model=None, frozen=None, policy_factory=None,
                          tactic_mode=None, normal_decision_replica=False):
    if name not in {"reference", "normal_replica", "attack", "attack_public", "normal_public", "attack_privileged"}:
        raise ValueError("unknown v4 branch")
    if name.endswith("public") and (model is None or frozen is None):
        raise ValueError("public control requires frozen model and calibration")
    sim = template.fork_from_state(initial)
    sim.set_continuation_seed(seed + config["continuation_seed_offset"])
    if name == "normal_replica" or normal_decision_replica:
        reseed_normal_decisions(sim, seed + config["normal_replica_seed_offset"])
    check_wallets(sim)
    initial_totals = account_totals(sim)
    executor = RegulatoryActionExecutor()
    remaining = config["initial_budget"]
    snapshots, rows = [], []
    policy = policy_factory() if policy_factory is not None else None
    for _ in range(config["horizon_days"]):
        choice = prepare_tactic(sim, mode=(tactic_mode or "S2") if name.startswith("attack") else "off",
                                selection_seed=seed + config["continuation_seed_offset"])
        observation = public_observation(sim, remaining)
        if policy is not None:
            messages = ObservationBuilder(ObservationConfig(include_messages=True))._build_messages(sim)
            observation = replace(observation, messages=messages)
        features = public_features(observation)
        score = detector_score(features, model) if model is not None else None
        action = RegulatoryAction(RegulatoryActionType.WAIT)
        if name.endswith("public") and policy is not None:
            action = policy(observation, score)
        elif name.endswith("public") and score > frozen["alarm_threshold"] and remaining >= 2:
            target = max(observation.assets, key=lambda a: (a.return_3d, a.volume_zscore, a.asset_id))
            action = RegulatoryAction(RegulatoryActionType.THROTTLE, target.asset_id)
        elif name == "attack_privileged" and observation.day in config["privileged_diagnostic"]["decision_days"]:
            action = RegulatoryAction(RegulatoryActionType.FREEZE, sim.public_asset_id(sim.target_asset))
        cost = float(DEFAULT_ACTION_COSTS[action.action_type])
        if cost > remaining:
            raise ValueError("v4 budget exceeded")
        entry = sim.step_day(pre_step=lambda state, day: executor.apply(state, action, day))
        if cost:
            executor.release(sim, action.target_id)
        remaining -= cost
        check_wallets(sim)
        totals = account_totals(sim)
        if any(not math.isclose(totals[key], value, rel_tol=1e-12, abs_tol=1e-5) for key, value in initial_totals.items()):
            raise ValueError("v4 cash/share conservation failed")
        maximum_order_multiple = max(a.last_activity["submitted_notional"] / a.portfolio.initial_wealth for a in sim.society.attackers)
        maximum_message = max(a.last_activity["message_intensity"] for a in sim.society.attackers)
        if maximum_order_multiple > 1 + 1e-10 or maximum_message > 1.8 + 1e-10:
            raise ValueError("v4 attacker resource cap exceeded")
        snapshots.append(capture_welfare_snapshot(sim, pair_id=pair_id))
        rows.append({"day": sim._next_day, "observation": observation.to_dict(), "features": features,
                     "detector_score": score, "action": action.to_dict(), "cost": cost,
                     "financial_outcomes": entry["retail_outcomes"],
                     "resource_maxima": {"order_multiple": maximum_order_multiple, "message_intensity": maximum_message},
                     "conservation_residual": {key: totals[key] - value for key, value in initial_totals.items()}})
        if tactic_mode is not None:
            rows[-1]["choice"] = choice
            rows[-1]["tactic_activity"] = {key: sum(a.last_activity[key] for a in sim.society.attackers)
                                           for key in ("orders", "messages", "submitted_notional", "message_intensity")}
    assert template._next_day == 0
    return snapshots, rows
