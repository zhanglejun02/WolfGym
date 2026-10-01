"""Safety-management protocol: targeted rollouts and fail-closed release audit."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
import hashlib
import json
import math
from typing import Mapping, Sequence

from wolfgym.benchmark.generator import BenchmarkGenerator
from wolfgym.benchmark.sampling import matched_continuation_seeds
from wolfgym.benchmark.schema import WolfGymRecord
from wolfgym.counterfactual.evaluator import merge_evaluations
from wolfgym.counterfactual.failure import canonical_scenario_id
from wolfgym.counterfactual.oracle import EffectiveInterventionOracle
from wolfgym.core.metrics.statistics import wilson_interval
from wolfgym.data.splits import SplitName
from wolfgym.observations.serialization import assert_no_hidden_fields
from wolfgym.regulation.actions import RegulatoryAction, RegulatoryActionType


PROTOCOL_VERSION = "wolfgym-safety-management-v3"
RISK_BINS = (0.0, 0.1, 0.3, 0.5, 0.8, 1.0)


def risk_bin(probability: float) -> str:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("risk must be finite and in [0, 1]")
    index = next((i for i in range(5) if probability < RISK_BINS[i + 1]), 4)
    closing = "]" if index == 4 else ")"
    return f"[{RISK_BINS[index]:g},{RISK_BINS[index + 1]:g}{closing}"


def public_candidates(observation: Mapping, *, max_assets: int = 3,
                      max_entities: int = 5) -> tuple[RegulatoryAction, ...]:
    """Rank only realized public activity; never resolve an omitted target."""
    if max_assets < 0 or max_entities < 0:
        raise ValueError("target counts must be non-negative")
    if len(observation.get("assets", ())) > max_assets:
        raise ValueError("action slots cannot silently omit observed assets")
    # ObservationBuilder supplies canonical asset order and public top-k entities;
    # use the same slots as FixedActionCodec instead of a second ranking rule.
    assets = observation.get("assets", ())
    entities = observation.get("entities", ())[:max_entities]
    targets = [item["asset_id"] for item in assets] + [item["entity_id"] for item in entities]
    return (RegulatoryAction(RegulatoryActionType.WAIT), *(
        RegulatoryAction(kind, target)
        for target in targets
        for kind in (RegulatoryActionType.THROTTLE, RegulatoryActionType.FREEZE)
    ))


def strict_label(row: Mapping, threshold: float = 0.20) -> dict:
    return EffectiveInterventionOracle(threshold).label_outcomes(
        row["counterfactuals"]["actions"],
        row.get("observation", {}).get("remaining_budget"),
    )


def trajectory_identity(row: Mapping) -> tuple:
    provenance = row["provenance"]
    trusted = provenance.get("trusted_generation", provenance)
    return (canonical_scenario_id(row["scenario_id"]), row["society_size"],
            float(trusted["alpha"]), int(trusted["trajectory_seed"]))


def _attempt_evidence(sample) -> dict:
    """Audit whether attacker activity actually entered public channels.

    Source attribution remains trusted provenance.  The regulator sees the
    resulting messages, trades, and aggregates, never the attacker identity or
    this audit flag.
    """
    attacker_ids = {agent.agent_id for agent in sample.simulator.society.attackers}
    checkpoint_day = int(sample.state.next_day)
    messages = [
        event for event in sample.simulator.social.message_events
        if int(event["day"]) < checkpoint_day
        and str(event["sender_id"]) in attacker_ids
    ]
    trades = [
        trade for trade in sample.simulator.market.trades
        if int(trade.day) < checkpoint_day
        and (trade.buyer_id in attacker_ids or trade.seller_id in attacker_ids)
    ]
    return {
        "source": "realized_public_channel_activity",
        "public_channel_attempt_observed": bool(messages or trades),
        "message_events": len(messages),
        "message_impressions": sum(int(event["audience_size"]) for event in messages),
        "executed_trade_events": len(trades),
        "executed_trade_quantity": float(sum(trade.quantity for trade in trades)),
        "used_by_oracle": False,
        "exported_as_public_label": False,
    }


def _decision_class(oracle: Mapping, stable: bool) -> str:
    if not stable or bool(oracle.get("ambiguous")):
        return "unresolved"
    if not bool(oracle.get("feasible")):
        return "infeasible"
    primary = oracle.get("primary_action") or {}
    return "wait" if primary.get("type") == "WAIT" else "intervene"


def dual_label_status(
    row: Mapping,
    *,
    interval_is_stable: bool | None = None,
) -> tuple[bool, str]:
    """Validate and summarize trusted attempt/effect label provenance."""
    layers = row["provenance"]["label_layers"]
    attempt = layers["attempted_manipulation_evidence"]
    realized = layers["realized_harm_decision"]
    if attempt.get("source") != "realized_public_channel_activity":
        raise ValueError("attempt evidence must come from realized public channels")
    if attempt.get("used_by_oracle") is not False:
        raise ValueError("attempt evidence must not enter the intervention oracle")
    if attempt.get("exported_as_public_label") is not False:
        raise ValueError("attempt provenance cannot be exported as a public label")
    if realized.get("source") != "matched_counterfactual_rollouts":
        raise ValueError("realized-harm decisions require matched counterfactuals")
    if not math.isclose(
        float(realized["wait_failure_probability"]),
        float(row["risk_probability"]),
        abs_tol=1e-12,
    ):
        raise ValueError("realized-harm provenance disagrees with RF target")
    decision_class = str(realized["decision_class"])
    if decision_class not in {"wait", "intervene", "infeasible", "unresolved"}:
        raise ValueError("unknown realized-harm decision class")
    attempt_observed = attempt["public_channel_attempt_observed"]
    if type(attempt_observed) is not bool:
        raise ValueError("attempt evidence status must be boolean")
    if interval_is_stable is not None and decision_class != _decision_class(
        row["oracle"], interval_is_stable,
    ):
        raise ValueError("realized-harm decision class disagrees with oracle")
    return attempt_observed, decision_class


def freeze_splits(records: Sequence[Mapping], seed: int = 42) -> list[dict]:
    """Freeze 70/10/20 expected train/dev/IID shares by physical trajectory.

    Hashes ignore configuration salts and instance IDs, so relabeling the same
    simulator trajectory cannot move its checkpoints across splits.
    """
    output = []
    for row in records:
        identity = json.dumps((seed, trajectory_identity(row)), separators=(",", ":"))
        bucket = int.from_bytes(hashlib.blake2s(identity.encode(), digest_size=4).digest(), "big") % 10
        split = "train" if bucket < 7 else "dev" if bucket == 7 else "iid_test"
        output.append({**row, "split": split,
                       "provenance": {**row["provenance"], "split_seed": seed}})
    return output


def interval_stable(
    actions: Mapping,
    threshold: float = 0.20,
    remaining_budget: float | None = None,
) -> bool:
    """Conservative pointwise Wilson sensitivity, not a joint confidence claim."""
    bounds = []
    for outcome in actions.values():
        if (
            remaining_budget is not None
            and float(outcome["cost"]) > float(remaining_budget) + 1e-12
        ):
            continue
        trials = outcome["failure_outcomes"]
        low, high = wilson_interval(sum(trials), len(trials))
        bounds.append((low, high, float(outcome["cost"])))
    certain = [cost for low, high, cost in bounds if high <= threshold]
    possible = [cost for low, high, cost in bounds if low <= threshold]
    if not possible:
        return True  # robustly infeasible
    if not certain or min(certain) != min(possible):
        return False
    cost = min(certain)
    return all(not (low <= threshold < high) for low, high, c in bounds if c == cost)


class SafetyManagementGenerator(BenchmarkGenerator):
    """Versioned successor; old frozen pilot generators remain reproducible."""

    def __init__(self, config):
        super().__init__(config)
        if config.target_mode != "observed_targets" or not config.include_asset_targets:
            raise ValueError("safety management requires observed_targets and asset targets")
        if config.safety_threshold != 0.20:
            raise ValueError("safety-management-v3 freezes safety_threshold=0.20")
        self.oracle = EffectiveInterventionOracle(config.safety_threshold, config.ambiguity_margin)

    def _remaining_budget(self, sample) -> float:
        identity = json.dumps((
            canonical_scenario_id(sample.scenario_id),
            sample.society_size,
            float(sample.alpha),
            int(sample.trajectory_seed),
            int(sample.day),
        ), separators=(",", ":")).encode()
        index = int.from_bytes(hashlib.blake2s(identity, digest_size=4).digest(), "big")
        return self.config.remaining_budgets[index % len(self.config.remaining_budgets)]

    def _build_observation(self, sample):
        return self.builder.build(
            sample.simulator,
            sample.state,
            remaining_budget=self._remaining_budget(sample),
        )

    def _evaluate_counterfactuals(self, sample, actions):
        schedule = self.config.rollout_schedule
        seeds = matched_continuation_seeds(sample.instance_id, schedule[-1])
        parts = []
        prior = 0
        for count in schedule:
            parts.append(self.evaluator.evaluate_state(
                simulator=sample.simulator,
                state=sample.state,
                candidate_actions=actions,
                seeds=seeds[prior:count],
                horizon=self.config.rollout_horizon,
            ))
            combined = merge_evaluations(parts)
            prior = count
            # Eight rollouts are screening only. Formal labels use at least 32,
            # and unresolved threshold decisions continue to the configured cap.
            if count >= 32 and interval_stable(
                combined.to_dict()["actions"],
                self.config.safety_threshold,
                self._remaining_budget(sample),
            ):
                return combined
        return combined

    def _oracle_decision(self, result, observation):
        return self.oracle.choose(result, observation.remaining_budget)

    def _candidate_actions(self, observation):
        actions = public_candidates(
            observation.to_dict(), max_entities=self.config.max_action_entities,
        )
        if len(actions) < 5:
            raise ValueError("at least two public intervention targets are required")
        return list(actions)

    def generate(self) -> list[WolfGymRecord]:
        candidates = []
        for index, sample in enumerate(self.config.sampler):
            if index >= self.config.max_evaluated_states:
                break
            candidates.append(self.build_record(sample))
        return self.select_records(candidates)

    def select_records(
        self, candidates: Sequence[WolfGymRecord],
    ) -> list[WolfGymRecord]:
        """Apply the frozen population policy to precomputed candidate states."""
        groups = defaultdict(list)
        for record in candidates:
            action_bucket = (
                record.oracle["primary_action"]["type"]
                if "PI" in record.task_eligibility
                else "NO_IS"
            )
            key = (
                record.provenance["subset"],
                risk_bin(record.risk_probability),
                record.scenario_id,
                action_bucket,
            )
            groups[key].append(record)
        # Give each available main risk bin turns; preserve deficient bins in
        # the audit. S3 gets at most 20% and is never used to fill main quotas.
        selected = []
        stress_count = 0
        trajectory_counts = Counter()
        limit = self.config.target_records
        while len(selected) < limit:
            progress = False
            for key in sorted(groups):
                if key[0] == "stress" and stress_count >= int(limit * 0.20):
                    continue
                while groups[key] and trajectory_counts[groups[key][0].trajectory_id] >= 6:
                    groups[key].pop(0)
                if not groups[key] or len(selected) >= limit:
                    continue
                record = groups[key].pop(0)
                selected.append(record)
                trajectory_counts[record.trajectory_id] += 1
                stress_count += key[0] == "stress"
                progress = True
            if not progress:
                break
        main = [record for record in selected if record.provenance["subset"] == "main"]
        stress = [record for record in selected if record.provenance["subset"] == "stress"]
        # Enforce the fraction on actual output as well as the requested size.
        return main + stress[:len(main) // 4]

    def build_record(self, sample, additional_provenance=None) -> WolfGymRecord:
        if sample.state.horizon_days - sample.state.next_day < self.config.rollout_horizon:
            raise ValueError("snapshot must have the full forecast horizon remaining")
        record = super().build_record(sample, additional_provenance)
        oracle = strict_label(record.to_dict(), self.config.safety_threshold)
        budget = record.observation["remaining_budget"]
        stable = len(record.counterfactuals["seeds"]) >= 32 and interval_stable(
            record.counterfactuals["actions"], self.config.safety_threshold, budget,
        )
        oracle["ambiguous"] = oracle["ambiguous"] or not stable
        tasks = ("RF", "AE", "CR") + (
            ("PI",) if oracle["feasible"] and not oracle["ambiguous"] else ()
        )
        decision_class = _decision_class(oracle, stable)
        return replace(
            record, schema_version="wolfgym-record-4.0", oracle=oracle,
            task_eligibility=tasks,
            provenance={
                **record.provenance, "benchmark_version": PROTOCOL_VERSION,
                "subset": "stress" if canonical_scenario_id(sample.scenario_id) == "s3" else "main",
                "risk_bin": risk_bin(record.risk_probability),
                "interval_stable": stable,
                "adaptive_rollout_schedule": list(self.config.rollout_schedule),
                "rollout_stages_executed": [
                    count for count in self.config.rollout_schedule
                    if count <= len(record.counterfactuals["seeds"])
                ],
                "remaining_budget": budget,
                "risk_target": "monte_carlo_estimate_p_hat_H_K",
                "md_status": "diagnostic_only_scenario_label",
                "label_layers": {
                    "attempted_manipulation_evidence": _attempt_evidence(sample),
                    "realized_harm_decision": {
                        "source": "matched_counterfactual_rollouts",
                        "wait_failure_probability": record.risk_probability,
                        "decision_class": decision_class,
                    },
                },
            },
        )


@dataclass(frozen=True)
class ReleaseRequirements:
    min_states: int = 2000
    min_trajectories: int = 300
    min_scenario_families: int = 5
    min_main_states_per_bin: int = 100
    max_stress_fraction: float = 0.20
    min_feasible_main_states: int = 1500
    min_stable_fraction: float = 0.80
    min_target_effect_fraction: float = 0.10
    target_effect_margin: float = 0.10
    min_manual_reviews: int = 200
    min_final_rollouts: int = 32
    min_budget_constrained_fraction: float = 0.10
    max_wait_optimal_fraction: float = 0.60
    min_is_action_type_fraction: float = 0.10
    min_s1_attempted_states: int = 100
    min_s1_wait_hard_negatives: int = 50
    min_s1_intervention_positives: int = 50


def audit_release(records: Sequence[Mapping], requirements: ReleaseRequirements | None = None,
                  *, shortcut_report: Mapping | None = None, manual_audit: Mapping | None = None) -> dict:
    """Reject insufficient coverage instead of manufacturing quotas or labels."""
    req = requirements or ReleaseRequirements()
    bins = Counter({risk_bin(value): 0 for value in RISK_BINS[:-1]})
    stress_bins = bins.copy()
    trajectories = defaultdict(set)
    physical_splits = defaultdict(set)
    errors = []
    feasible = stable = target_effects = main_count = targeted = 0
    precise = adaptive_complete = budget_constrained = budget_sensitive = 0
    is_action_types = Counter()
    s1_attempted = s1_wait_negatives = s1_intervention_positives = 0
    infeasible = []
    ids = [row.get("instance_id", "<missing>") for row in records]
    for row in records:
        try:
            trajectories[row["trajectory_id"]].add(row["split"])
            physical_splits[trajectory_identity(row)].add(row["split"])
            assert_no_hidden_fields(row["observation"])
            observation_text = str(row.get("observation_text", ""))
            for field in ("trajectory_id", "instance_id", "hidden_state_ref"):
                if str(row[field]) in observation_text:
                    raise ValueError(f"{field} leaked into observation text")
            outcomes = row["counterfactuals"]["actions"]
            seeds = row["counterfactuals"]["seeds"]
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("continuation seeds must be nonempty and unique")
            for name, outcome in outcomes.items():
                trials = outcome["failure_outcomes"]
                if len(trials) != len(seeds) or any(type(value) is not bool for value in trials):
                    raise ValueError(f"invalid matched outcomes for {name}")
                if not math.isclose(sum(trials) / len(trials), outcome["failure_probability"], abs_tol=1e-12):
                    raise ValueError(f"probability disagrees with rollouts for {name}")
            if not math.isclose(outcomes["WAIT"]["failure_probability"], row["risk_probability"], abs_tol=1e-12):
                raise ValueError("RF target disagrees with WAIT rollouts")
            budget = row["observation"].get("remaining_budget")
            if budget is None or not math.isfinite(float(budget)) or float(budget) < 0.0:
                raise ValueError("v3 static observations require a non-negative remaining_budget")
            label = strict_label(row)
            row_stable = len(seeds) >= req.min_final_rollouts and interval_stable(
                outcomes, remaining_budget=float(budget),
            )
            stable += row_stable
            precise += len(seeds) >= req.min_final_rollouts
            adaptive_complete += row_stable or len(seeds) >= 64
            budget_constrained += any(
                float(item["cost"]) > float(budget)
                and float(item["failure_probability"]) <= 0.20
                for item in outcomes.values()
            )
            unlimited = EffectiveInterventionOracle(0.20).label_outcomes(outcomes)
            budget_sensitive += {
                RegulatoryAction(
                    RegulatoryActionType[str(action["type"]).upper()], action.get("target_id")
                ).label
                for action in label["valid_actions"]
            } != {
                RegulatoryAction(
                    RegulatoryActionType[str(action["type"]).upper()], action.get("target_id")
                ).label
                for action in unlimited["valid_actions"]
            }
            if not label["feasible"]:
                infeasible.append(row["instance_id"])
            attempt_observed, decision_class = dual_label_status(
                row, interval_is_stable=row_stable,
            )
            if canonical_scenario_id(row["scenario_id"]) == "s1" and attempt_observed:
                s1_attempted += 1
                s1_wait_negatives += decision_class == "wait"
                s1_intervention_positives += decision_class == "intervene"
            stress = canonical_scenario_id(row["scenario_id"]) == "s3"
            (stress_bins if stress else bins)[risk_bin(row["risk_probability"])] += 1
            if not stress:
                main_count += 1
                eligible = label["feasible"] and row_stable and not label["ambiguous"]
                feasible += eligible
                if eligible:
                    is_action_types[label["primary_action"]["type"]] += 1
                if ("PI" in row["task_eligibility"]) != eligible:
                    raise ValueError("IS eligibility disagrees with feasible stable budgeted oracle")
            visible = {item["asset_id"] for item in row["observation"].get("assets", ())}
            visible.update(item["entity_id"] for item in row["observation"].get("entities", ()))
            interventions = [item for item in outcomes.values() if item["action"]["type"] != "WAIT"]
            targets = {item["action"].get("target_id") for item in interventions}
            targeted += len(targets) >= 2 and None not in targets and targets <= visible
            target_effects += any(
                a["action"]["type"] == b["action"]["type"]
                and a["action"].get("target_id") != b["action"].get("target_id")
                and a["cost"] == b["cost"]
                and abs(a["failure_probability"] - b["failure_probability"]) >= req.target_effect_margin
                for a, b in combinations(interventions, 2)
            )
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"instance_id": row.get("instance_id", "<missing>"), "error": str(exc)})
    n = len(records)
    shortcut_tasks = (shortcut_report or {}).get("tasks", {})
    shortcut_passed = all(
        task in shortcut_tasks and shortcut_tasks[task].get("status") == "diagnostic"
        and max(shortcut_tasks[task][metric] for metric in (
            "logistic_accuracy", "tree_depth3_accuracy", "forest_accuracy"
        )) <= 0.90
        for task in ("RF_risk_bin_diagnostic", "IS_intervention_type_diagnostic")
    )
    reviews = (manual_audit or {}).get("reviews", [])
    reviewed_ids = [review["instance_id"] for review in reviews]
    manual_passed = (
        len(set(reviewed_ids)) >= req.min_manual_reviews
        and len(reviewed_ids) == len(set(reviewed_ids))
        and set(reviewed_ids) <= set(ids)
        and all(review.get("passed") is True and review.get("reviewer") for review in reviews)
    )
    checks = {
        "records_valid": not errors,
        "unique_states": len(set(ids)) == n,
        "trajectory_disjoint_splits": all(len(splits) == 1 for splits in trajectories.values()),
        "physical_trajectory_disjoint_splits": all(len(splits) == 1 for splits in physical_splits.values()),
        "frozen_split_names": n > 0 and all(row["split"] in {name.value for name in SplitName} for row in records),
        "state_count": n >= req.min_states,
        "trajectory_count": len(trajectories) >= req.min_trajectories,
        "scenario_coverage": len({canonical_scenario_id(row["scenario_id"]) for row in records}) >= req.min_scenario_families,
        "main_risk_bin_coverage": all(count >= req.min_main_states_per_bin for count in bins.values()),
        "stress_share": n > 0 and sum(stress_bins.values()) / n <= req.max_stress_fraction,
        "feasible_main_IS": feasible >= req.min_feasible_main_states,
        "formal_rollout_precision": n > 0 and precise == n,
        "adaptive_rollout_complete": n > 0 and adaptive_complete == n,
        "label_stability": n > 0 and stable / n >= req.min_stable_fraction,
        "budget_constraint_exercised": (
            n > 0 and budget_sensitive / n >= req.min_budget_constrained_fraction
        ),
        "IS_action_diversity": (
            feasible > 0
            and is_action_types["WAIT"] / feasible <= req.max_wait_optimal_fraction
            and all(
                is_action_types[action] / feasible >= req.min_is_action_type_fraction
                for action in ("WAIT", "THROTTLE", "FREEZE")
            )
        ),
        "explicit_public_targets": n > 0 and targeted == n,
        "target_dependent_effects": n > 0 and target_effects / n >= req.min_target_effect_fraction,
        "s1_attempt_evidence_coverage": (
            s1_attempted >= req.min_s1_attempted_states
        ),
        "s1_failed_attempt_hard_negatives": (
            s1_wait_negatives >= req.min_s1_wait_hard_negatives
        ),
        "s1_realized_harm_intervention_positives": (
            s1_intervention_positives >= req.min_s1_intervention_positives
        ),
        "shortcut_audit": shortcut_passed,
        "manual_quality_audit": manual_passed,
    }
    return {
        "protocol": PROTOCOL_VERSION, "unique_states": len(set(ids)),
        "unique_trajectories": len(trajectories), "main_states": main_count,
        "main_risk_bins": dict(bins), "stress_risk_bins": dict(stress_bins),
        "feasible_main_IS_states": feasible, "infeasible_diagnostic_ids": infeasible,
        "interval_stable_states": stable, "target_dependent_states": target_effects,
        "budget_constrained_states": budget_constrained,
        "budget_sensitive_states": budget_sensitive,
        "s1_dual_label_counts": {
            "attempted_public_channel_states": s1_attempted,
            "wait_hard_negatives": s1_wait_negatives,
            "intervention_positives": s1_intervention_positives,
        },
        "IS_optimal_action_types": dict(is_action_types),
        "errors": errors, "checks": checks, "passed": all(checks.values()),
        "shortcut_diagnostics": shortcut_report, "manual_review_count": len(reviewed_ids),
        "md_primary_enabled": False,
        "required_followup": "E5 and E6 results are required for paper completion, independently of dataset readiness.",
    }


def diagnostic_shortcuts(records: Sequence[Mapping]) -> dict:
    """Grouped public-feature anchors, including the excluded scenario-label MD."""
    import numpy as np
    from wolfgym.benchmark.shortcut_audit import _cross_validated_accuracy, _fold, _public_features

    if not records:
        return {"status": "insufficient_data"}
    names = ["day", "society_size"] + [
        f"{group}.{key}" for group in ("market", "social")
        for key in sorted(records[0]["observation"][group])
    ]
    tasks = {
        "RF_risk_bin_diagnostic": [(row, risk_bin(row["risk_probability"])) for row in records],
        "MD_scenario_label_diagnostic": [(row, row["mechanism"]) for row in records],
        "IS_intervention_type_diagnostic": [
            (row, label["primary_action"]["type"])
            for row in records
            if "PI" in row.get("task_eligibility", ())
            and (label := strict_label(row))["feasible"]
        ],
    }
    reports = {}
    for name, pairs in tasks.items():
        folds = np.asarray([_fold(row["trajectory_id"]) for row, _ in pairs])
        if len(pairs) < 20 or len(set(folds)) < 2:
            reports[name] = {"status": "insufficient_data", "states": len(pairs)}
            continue
        labels = sorted({label for _, label in pairs})
        y = np.asarray([labels.index(label) for _, label in pairs])
        x = np.asarray([_public_features(row, names) for row, _ in pairs])
        reports[name] = {
            "status": "diagnostic", "states": len(pairs), "classes": labels,
            "majority_accuracy": max(Counter(y).values()) / len(y),
            "logistic_accuracy": _cross_validated_accuracy(x, y, folds, "logistic", len(labels)),
            "tree_depth3_accuracy": _cross_validated_accuracy(x, y, folds, "tree", 3),
            "forest_accuracy": _cross_validated_accuracy(x, y, folds, "forest", len(labels)),
        }
    return {"split_unit": "trajectory_id", "features": names, "tasks": reports,
            "MD_primary_enabled": False,
            "IS_scope": "type-only shortcut diagnostic; main scoring uses type plus target"}
