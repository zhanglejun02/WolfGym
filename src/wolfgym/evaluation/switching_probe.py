"""Bounded, separate v3 development harness; no change to the Gym registry."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import yaml

from wolfgym.core.agents.switching import TACTICS, install_switching_cohort, prepare_tactic
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.stock_market import load_stock_market
from wolfgym.evaluation.composite_harm import score_composite_harm
from wolfgym.evaluation.counterfactual_welfare import capture_welfare_snapshot
from wolfgym.evaluation.harm_metric_probe import choose_action, summarize_branch
from wolfgym.evaluation.closed_loop import grouped_summary
from wolfgym.evaluation.real_market import source_fingerprint
from wolfgym.observations import ObservationBuilder, ObservationConfig
from wolfgym.regulation import DEFAULT_ACTION_COSTS, RegulatoryAction, RegulatoryActionType
from wolfgym.regulation.executor import RegulatoryActionExecutor


def branch_specs():
    return {
        "reference": ("off", "wait", None),
        **{f"single_{t}": (t, "wait", None) for t in TACTICS},
        "random": ("random", "wait", None), "adaptive": ("adaptive", "wait", None),
        **{f"minus_{t}": ("adaptive", "wait", t) for t in TACTICS},
        **{f"{mode}_{policy}": (mode, policy, None) for mode in ("adaptive", "off")
           for policy in ("public_rule", "scheduled_public_freeze")},
        "off_blanket_freeze": ("off", "blanket_freeze", None),
    }


def account_totals(simulator):
    accounts = simulator.market.accounts.values()
    return {"cash": math.fsum(a.cash for a in accounts),
            **{key: math.fsum(a.position(key) for a in accounts) for key in simulator.market.assets}}


def check_wallets(simulator):
    for agent in simulator.society.attackers:
        if simulator.market.accounts[agent.agent_id] is not agent.portfolio:
            raise ValueError("settlement wallet alias lost")
        if any(module.portfolio is not agent.portfolio for module in agent.modules.values()):
            raise ValueError("tactic wallet alias lost")


def run_switching_branch(template, initial, *, pair_id, continuation_seed, spec, config):
    simulator = template.fork_from_state(initial)
    simulator.set_continuation_seed(continuation_seed)
    check_wallets(simulator)
    totals = account_totals(simulator)
    mode, controller, disabled = spec
    executor = RegulatoryActionExecutor()
    builder = ObservationBuilder(ObservationConfig(top_k_entities=5)) if controller not in {"wait", "blanket_freeze"} else None
    remaining = float(config["initial_budget"])
    snapshots, rows = [], []
    for _ in range(config["horizon_days"]):
        choice = prepare_tactic(simulator, mode=mode, selection_seed=continuation_seed, disabled=disabled)
        if controller == "blanket_freeze":
            actions = [RegulatoryAction(RegulatoryActionType.FREEZE, simulator.public_asset_id(key))
                       for key in simulator.market.assets]
        elif builder is not None:
            observation = builder.build(simulator, simulator.get_state(), remaining_budget=remaining)
            actions = [choose_action(controller, observation, config)]
        else:
            actions = [RegulatoryAction(RegulatoryActionType.WAIT)]
        cost = sum(float(DEFAULT_ACTION_COSTS[a.action_type]) for a in actions)
        if controller != "blanket_freeze" and cost > remaining:
            raise ValueError("legal controller exceeded budget")

        def apply(sim, day):
            for action in actions:
                executor.apply(sim, action, day)

        entry = simulator.step_day(pre_step=apply)
        for action in actions:
            if action.action_type is not RegulatoryActionType.WAIT:
                executor.release(simulator, action.target_id)
        remaining -= cost
        check_wallets(simulator)
        current = account_totals(simulator)
        residual = {key: current[key] - value for key, value in totals.items()}
        if any(not math.isclose(current[key], value, rel_tol=1e-12, abs_tol=1e-5)
               for key, value in totals.items()):
            raise ValueError(f"cash/share conservation failed: {residual}")
        activity = [a.last_activity for a in simulator.society.attackers]
        for agent, row in zip(simulator.society.attackers, activity):
            if row["submitted_notional"] > agent.portfolio.initial_wealth + 1e-7 or row["message_intensity"] > 1.8 + 1e-12:
                raise ValueError("shared daily resource cap exceeded")
        snapshots.append(capture_welfare_snapshot(simulator, pair_id=pair_id))
        rows.append({"day": simulator._next_day, "actions": [a.to_dict() for a in actions],
                     "cost": cost, "financial_outcomes": entry["retail_outcomes"], "choice": choice,
                     "activity": {key: sum(row[key] for row in activity) for key in
                                  ("orders", "messages", "submitted_notional", "message_intensity")},
                     "conservation_residual": residual})
    assert template._next_day == 0
    return snapshots, rows


def run_seed(seed, config, output, fingerprint, protocol_hash):
    output = Path(output)
    template = install_switching_cohort(WolfSocietySimulator(
        load_stock_market("finfluencer"), config["society_size"], config["alpha"], seed=seed))
    initial = template.get_state()
    args = dict(pair_id=f"{fingerprint}:{protocol_hash}:v3:{seed}",
                continuation_seed=seed + config["continuation_seed_offset"], config=config)
    branches, replays = {}, []
    for name, spec in branch_specs().items():
        snapshots, rows = run_switching_branch(template, initial, spec=spec, **args)
        branches[name] = (snapshots, rows)
        with gzip.open(output / f"{seed}_{name}.json.gz", "wt") as stream:
            json.dump({"snapshots": [asdict(s) for s in snapshots], "days": rows}, stream)
        print(f"seed={seed} completed {name}", flush=True)
        if seed == config["seeds"][0] and name in {"reference", "adaptive"}:
            replay = run_switching_branch(template, initial, spec=spec, **args)
            if replay != (snapshots, rows):
                raise ValueError("exact replay diverged")
            replays.append({"seed": seed, "branch": name, "exact_snapshots_and_rows": True})
    reference, ref_rows = branches["reference"]
    adaptive, _ = branches["adaptive"]
    records = []
    for name, (snapshots, rows) in branches.items():
        baseline = reference if name.startswith("off_") or name == "reference" else adaptive
        summary = summarize_branch(reference, baseline, snapshots, rows)
        summary["composite"] = score_composite_harm(reference, snapshots, ref_rows, rows)
        summary["tactic_days"] = dict(Counter(row["choice"]["executed"] for row in rows))
        summary["active_tactic_days"] = dict(Counter(row["choice"]["executed"] for row in rows
                                                     if row["activity"]["orders"] or row["activity"]["messages"]))
        summary["activity_by_tactic"] = {t: {key: sum(row["activity"][key] for row in rows if row["choice"]["executed"] == t)
                                                   for key in ("orders", "messages", "submitted_notional", "message_intensity")}
                                         for t in TACTICS}
        records.append({"seed": seed, "branch": name, "summary": summary})
    result = {"records": records, "replays": replays}
    (output / f"{seed}_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def aggregate_switching(records):
    result = {}
    for branch in branch_specs():
        group = sorted((r for r in records if r["branch"] == branch), key=lambda r: r["seed"])
        summaries = [r["summary"] for r in group]
        seeds = [str(r["seed"]) for r in group]
        result[branch] = {
            "H": grouped_summary([s["welfare"]["policy"]["positive_shortfall_rate"] for s in summaries], seeds),
            "benefit_vs_adaptive_or_off_reference": grouped_summary([s["welfare"]["shortfall_reduction"] for s in summaries], seeds),
            "component_counts": {c: sum(s["composite"]["components"][c] for s in summaries) for c in ("broad", "tail", "function")},
            "composite_count": sum(s["composite"]["event"] for s in summaries),
            "mean_cost": sum(s["intervention_cost"] for s in summaries) / len(summaries),
            "active_tactic_days": dict(sum((Counter(s["active_tactic_days"]) for s in summaries), Counter())),
            "mean_executed_notional": sum(s["notional"]["executed_notional"] for s in summaries) / len(summaries),
        }
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_switching_probe(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = yaml.safe_load(config_path.read_text())
    expected = {"schema": "wolfgym-switching-probe-v1", "society_size": 500, "alpha": .02,
                "horizon_days": 30, "seeds": [64100, 64101, 64102, 64103], "workers": 2,
                "initial_budget": 12, "decision_days": [5, 10, 15], "continuation_seed_offset": 1000000,
                "public_rule": {"return_3d_threshold": .03, "volume_zscore_threshold": 1.},
                "maximum_simulated_branches": 66, "release_eligible": False,
                "protocol": "configs/task_protocol_v3.yaml"}
    if config != expected:
        raise ValueError("requires the fixed predeclared v3 development probe")
    output.mkdir(parents=True, exist_ok=False)
    protocol = Path(config["protocol"])
    fingerprint = source_fingerprint()
    protected = [Path("configs/task_protocol_v2.yaml"), *Path("data/harm_metric_probe_v2_20260921").rglob("*")]
    protected_hashes = {str(p): file_hash(p) for p in protected if p.is_file()}
    market_files = [*Path("src/wolfgym/core/env").glob("*.py"), Path("src/wolfgym/core/agents/attackers.py"),
                    *Path("src/wolfgym/core/config").rglob("*.yaml"),
                    *Path("src/wolfgym/core/scenarios").glob("*.py")]
    manifest = {"config": config, "source_sha256": fingerprint, "protocol_sha256": file_hash(protocol),
                "config_sha256": file_hash(config_path), "script_sha256": file_hash("scripts/run_switching_probe.py"),
                "predeclaration_sha256": file_hash("docs/TASK_PROTOCOL_V3_FEASIBILITY.md"),
                "protected_v2_files": protected_hashes,
                "frozen_market_files": {str(p): file_hash(p) for p in market_files},
                "started_unix": time.time(), "release_eligible": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "protocol_at_execution.yaml").write_bytes(protocol.read_bytes())
    (output / "probe_config_at_execution.yaml").write_bytes(config_path.read_bytes())
    (output / "predeclaration_at_execution.md").write_bytes(Path("docs/TASK_PROTOCOL_V3_FEASIBILITY.md").read_bytes())
    records, replays = [], []
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(run_seed, seed, config, str(output), fingerprint, manifest["protocol_sha256"])
                   for seed in config["seeds"]]
        for future in as_completed(futures):
            result = future.result()
            records.extend(result["records"])
            replays.extend(result["replays"])
    records.sort(key=lambda row: (row["seed"], row["branch"]))
    simulated = len(records) + len(replays)
    assert simulated == config["maximum_simulated_branches"]
    assert source_fingerprint() == fingerprint and file_hash(protocol) == manifest["protocol_sha256"]
    assert file_hash(config_path) == manifest["config_sha256"]
    assert all(file_hash(p) == h for p, h in protected_hashes.items())
    (output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    report = {"aggregates": aggregate_switching(records), "replay_checks": replays,
              "simulated_branches": simulated, "scored_branches": len(records),
              "elapsed_seconds": time.time() - manifest["started_unix"],
              "source_verified_at_completion": True, "v2_files_unchanged": True,
              "release_eligible": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
