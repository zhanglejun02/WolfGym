"""Bounded development test of v2 harm metrics, without changing market dynamics."""
from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

import yaml

from wolfgym.benchmark.management import public_candidates
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.stock_market import load_stock_market
from wolfgym.evaluation.closed_loop import grouped_summary
from wolfgym.evaluation.counterfactual_welfare import (
    capture_welfare_snapshot, score_collapse_history, score_welfare_triplet, _shortfall,
)
from wolfgym.evaluation.real_market import source_fingerprint
from wolfgym.observations import ObservationBuilder, ObservationConfig
from wolfgym.regulation import DEFAULT_ACTION_COSTS, RegulatoryAction, RegulatoryActionType
from wolfgym.regulation.executor import RegulatoryActionExecutor


def choose_action(kind, observation, config, *, privileged_target=None):
    """Public policies receive only an observation; privileged target is explicit."""
    wait = RegulatoryAction(RegulatoryActionType.WAIT)
    if kind in {"reference", "wait"}:
        return wait
    target = max(observation.assets, key=lambda a: (a.return_3d, a.volume_zscore, a.asset_id))
    if kind == "public_rule":
        rule = config["public_rule"]
        if not (target.return_3d > rule["return_3d_threshold"]
                and target.volume_zscore > rule["volume_zscore_threshold"]):
            return wait
        action = RegulatoryAction(RegulatoryActionType.THROTTLE, target.asset_id)
    elif kind in {"scheduled_public_freeze", "privileged_target_freeze"}:
        if observation.day not in config["decision_days"]:
            return wait
        if kind == "privileged_target_freeze" and privileged_target is None:
            raise ValueError("privileged diagnostic requires an explicit target")
        action = RegulatoryAction(RegulatoryActionType.FREEZE,
                                  privileged_target if kind == "privileged_target_freeze" else target.asset_id)
    else:
        raise ValueError(f"unknown controller {kind}")
    menu = public_candidates(observation.to_dict())
    return action if action in menu and DEFAULT_ACTION_COSTS[action.action_type] <= observation.remaining_budget else wait


def run_branch(template, initial, *, pair_id, continuation_seed, kind, config):
    simulator = template.fork_from_state(initial)
    simulator.set_continuation_seed(continuation_seed)
    executor = RegulatoryActionExecutor()
    builder = ObservationBuilder(ObservationConfig(top_k_entities=5))
    if kind == "reference":
        for agent in simulator.society.attackers:
            executor.apply(simulator, RegulatoryAction(
                RegulatoryActionType.FREEZE, simulator.public_entity_id(agent.agent_id)), 0)
    remaining = float(config["initial_budget"])
    snapshots, rows = [], []
    for _ in range(config["horizon_days"]):
        observation = builder.build(simulator, simulator.get_state(), remaining_budget=remaining)
        action = choose_action(kind, observation, config, privileged_target=(
            simulator.public_asset_id(simulator.target_asset) if kind == "privileged_target_freeze" else None))
        cost = float(DEFAULT_ACTION_COSTS[action.action_type])
        if cost > remaining:
            raise ValueError("budget exceeded before stepping")
        entry = simulator.step_day(pre_step=lambda sim, day: executor.apply(sim, action, day))
        if action.action_type is not RegulatoryActionType.WAIT:
            executor.release(simulator, action.target_id)
        remaining -= cost
        snapshots.append(capture_welfare_snapshot(simulator, pair_id=pair_id))
        rows.append({"day": simulator._next_day, "action": action.to_dict(), "cost": cost,
                     "financial_outcomes": entry["retail_outcomes"]})
    assert template._next_day == 0
    return snapshots, rows


def summarize_branch(reference, wait, actual, rows):
    collapse = score_collapse_history(reference, actual)
    welfare = score_welfare_triplet(reference[-1], wait[-1], actual[-1])
    daily = [_shortfall(ref, branch) for ref, branch in zip(reference, actual)]
    totals = {key: sum(row["financial_outcomes"][key] for row in rows)
              for key in ("requested_notional", "blocked_notional", "unexecuted_notional", "executed_notional")}
    positive = welfare["policy"]["positive_shortfall_rate"]
    net = welfare["policy"]["signed_net_gap_rate"]
    return {
        "welfare": welfare, "collapse": collapse,
        "peak_positive_shortfall_rate": max(row["positive_shortfall_rate"] for row in daily),
        "peak_worst_decile_shortfall_rate": max(row["worst_decile_mean_shortfall_rate"] for row in daily),
        "peak_sustained_harmed_fraction": max(row["sustained_harmed_accounts"] for row in collapse["daily"]) / collapse["protected_investors"],
        "terminal_gain_rate": positive - net,
        "intervention_cost": sum(row["cost"] for row in rows),
        "intervention_count": sum(row["cost"] > 0 for row in rows),
        "terminal_market_return": rows[-1]["financial_outcomes"]["return"],
        "notional": totals,
    }


def aggregate(records):
    results = {}
    for family in sorted({row["family"] for row in records}):
        results[family] = {}
        for kind in sorted({row["controller"] for row in records if row["family"] == family}):
            group = [row for row in records if row["family"] == family and row["controller"] == kind]
            summaries = [row["summary"] for row in group]
            seeds = [str(row["seed"]) for row in group]
            events = sum(row["collapse"]["collapse"] for row in summaries)
            positive_effects = [row["welfare"]["shortfall_reduction"] for row in summaries]
            results[family][kind] = {
                "n": len(group), "collapse_count": events,
                "collapse_rate": events / len(group),
                "zero_event_one_sided_95_upper": 1 - .05 ** (1 / len(group)) if events == 0 else None,
                "positive_shortfall": grouped_summary([r["welfare"]["policy"]["positive_shortfall_rate"] for r in summaries], seeds),
                "signed_net_gap": grouped_summary([r["welfare"]["policy"]["signed_net_gap_rate"] for r in summaries], seeds),
                "worst_decile_shortfall": grouped_summary([r["welfare"]["policy"]["worst_decile_mean_shortfall_rate"] for r in summaries], seeds),
                "shortfall_reduction": grouped_summary(positive_effects, seeds),
                "improved_trajectories": sum(value > 1e-12 for value in positive_effects),
                "worsened_trajectories": sum(value < -1e-12 for value in positive_effects),
                "net_gain_despite_individual_harm": sum(r["welfare"]["policy"]["signed_net_gap_rate"] < 0
                                                         and r["welfare"]["policy"]["positive_shortfall_rate"] > 0 for r in summaries),
                "maximum_sustained_harmed_fraction": max(r["peak_sustained_harmed_fraction"] for r in summaries),
                "mean_intervention_cost": mean(r["intervention_cost"] for r in summaries),
                "total_interventions": sum(r["intervention_count"] for r in summaries),
                "mean_blocked_notional": mean(r["notional"]["blocked_notional"] for r in summaries),
                "executed_notional_change_vs_wait": grouped_summary([r["executed_notional_change_vs_wait"] for r in summaries], seeds),
            }
    return results


def run_probe(config_path, output, *, progress=print):
    config_path, output = Path(config_path), Path(output)
    config = yaml.safe_load(config_path.read_text())
    if (config["schema"] != "wolfgym-harm-metric-probe-v1" or config["society_size"] != 500
            or config["horizon_days"] != 30 or config["alpha"] != .02
            or len(config["seeds"]) != 8 or len(set(config["seeds"])) != 8
            or config["families"] != ["clean", "pump_and_dump", "finfluencer"]
            or config["controllers"] != ["wait", "public_rule", "scheduled_public_freeze", "privileged_target_freeze"]
            or config["decision_days"] != [5, 10, 15] or config["initial_budget"] != 12
            or config["maximum_simulated_branches"] != 107
            or config["release_eligible"] is not False):
        raise ValueError("requires the fixed bounded development probe")
    output.mkdir(parents=True, exist_ok=False)
    fingerprint = source_fingerprint()
    protocol_path = Path(config["protocol"])
    manifest = {"config": config, "source_sha256": fingerprint,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "started_unix": time.time(), "release_eligible": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records, replay_checks, simulated = [], [], 0
    for seed in config["seeds"]:
        for family in config["families"]:
            template = WolfSocietySimulator(load_stock_market(None if family == "clean" else family),
                                           config["society_size"], 0. if family == "clean" else config["alpha"], seed=seed)
            initial = template.get_state()
            pair_id = f"{fingerprint}:{manifest['protocol_sha256']}:{family}:{seed}"
            branch_args = dict(pair_id=pair_id, continuation_seed=seed + config["continuation_seed_offset"], config=config)
            reference, reference_rows = run_branch(template, initial, kind="reference", **branch_args)
            simulated += 1
            with gzip.open(output / f"{family}_{seed}_reference.json.gz", "wt") as stream:
                json.dump({"snapshots": [asdict(s) for s in reference], "days": reference_rows}, stream)
            if seed == config["seeds"][0]:
                replay, replay_rows = run_branch(template, initial, kind="reference", **branch_args)
                simulated += 1
                if replay != reference or replay_rows != reference_rows:
                    raise ValueError("identical replay diverged")
                replay_checks.append({"family": family, "exact_snapshots_and_rows": True})
            wait = None
            wait_notional = None
            for kind in config["controllers"]:
                if family == "clean" and kind == "privileged_target_freeze":
                    continue
                if family == "clean" and kind == "wait":
                    actual, rows = reference, reference_rows
                else:
                    actual, rows = run_branch(template, initial, kind=kind, **branch_args)
                    simulated += 1
                if kind == "wait":
                    wait = actual
                    wait_notional = sum(r["financial_outcomes"]["executed_notional"] for r in rows)
                summary = summarize_branch(reference, wait, actual, rows)
                summary["executed_notional_change_vs_wait"] = summary["notional"]["executed_notional"] - wait_notional
                record = {"family": family, "seed": seed, "controller": kind, "summary": summary}
                records.append(record)
                with gzip.open(output / f"{family}_{seed}_{kind}.json.gz", "wt") as stream:
                    json.dump({"snapshots": [asdict(s) for s in actual], "days": rows}, stream)
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                progress(f"{family} seed={seed} {kind}: H={summary['welfare']['policy']['positive_shortfall_rate']:.6%}, "
                         f"benefit={summary['welfare']['shortfall_reduction']:.6%}, "
                         f"collapse={summary['collapse']['collapse']}, simulations={simulated}")
    assert simulated == config["maximum_simulated_branches"]
    assert source_fingerprint() == fingerprint
    report = {"aggregates": aggregate(records), "replay_checks": replay_checks,
              "simulated_branches": simulated, "scored_branches": len(records),
              "elapsed_seconds": time.time() - manifest["started_unix"],
              "source_verified_at_completion": True,
              "interpretation": "Exploratory eight-seed paired diagnostic; no model fitting or threshold selection.",
              "limitations": ["clean WAIT zero is true by construction", "attacker-off reference is not a legal controller",
                              "positive individual shortfalls need not imply negative aggregate welfare",
                              "policy failures do not establish impossibility of intervention",
                              "public observability not established by this experiment"],
              "release_eligible": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
