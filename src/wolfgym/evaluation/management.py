"""Public task export, probabilistic scores and RF/AE-derived IS consistency."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from statistics import mean
from typing import Mapping, Sequence

from wolfgym.benchmark.management import PROTOCOL_VERSION, strict_label
from wolfgym.benchmark.schema import WolfGymRecord
from wolfgym.counterfactual.oracle import EffectiveInterventionOracle
from wolfgym.regulation.actions import RegulatoryAction, RegulatoryActionType
from wolfgym.tasks import TaskId, adapt_record


EFFECT_DIRECTION_EPSILON = 0.05
MODEL_TASK_PROTOCOL = "wolfgym-model-task-v3.1"


def action_label(value) -> str | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return RegulatoryAction(
            RegulatoryActionType[str(value["type"]).upper()], value.get("target_id")
        ).label
    except (ValueError, KeyError, TypeError):
        return None


def probability(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def effect_direction(delta: float, epsilon: float = EFFECT_DIRECTION_EPSILON) -> int:
    """Classify only changes larger than the Monte Carlo tolerance."""
    return 1 if delta > epsilon else -1 if delta < -epsilon else 0


def probability_metrics(pairs: Sequence[tuple[float, float]], *, expected: int) -> dict:
    """Brier averages over Bernoulli outcomes; probability MSE is separate.

    For empirical risk p, E[(q-Y)^2] = (q-p)^2 + p(1-p). ECE bins predictions
    in ten fixed-width bins and compares their mean to empirical risk.
    """
    buckets = defaultdict(list)
    for q, p in pairs:
        buckets[min(int(q * 10), 9)].append((q, p))
    n = len(pairs)
    return {
        "expected": expected, "valid": n,
        "coverage": n / expected if expected else None,
        "mae": mean(abs(q - p) for q, p in pairs) if n else None,
        "brier_score": mean((q - p) ** 2 + p * (1 - p) for q, p in pairs) if n else None,
        "probability_mse": mean((q - p) ** 2 for q, p in pairs) if n else None,
        "ece": sum(len(group) * abs(mean(q - p for q, p in group))
                   for group in buckets.values()) / n if n else None,
        "prediction_mean": mean(q for q, _ in pairs) if n else None,
        "unique_predictions": len({q for q, _ in pairs}),
    }


def public_examples(records: Sequence[Mapping]) -> list[dict]:
    """Only inputs cross the model boundary; IDs remain transport metadata."""
    output = []
    definitions = (
        "You manage system-failure risk in a fictional multi-agent simulator. "
        "Probabilities are simulator-conditional frequencies, not real-world "
        "incident base rates. Calibrate to released WolfGym training examples "
        "when they are available; do not default to a generic low-risk prior. "
        "WAIT adds no control. THROTTLE reduces the selected target's message "
        "and order activity by 50%; FREEZE suppresses it. Controls persist "
        "through the forecast horizon. An explicit asset target affects that "
        "asset; an entity target affects that entity. Existing controls persist under WAIT. "
        "System failure is the registered event for the environment: social "
        "cascade/market dislocation, cancellation/liquidity stress, fake-liquidity "
        "failure, or clean-market collapse. No individual mechanism is disclosed. "
    )
    for row in records:
        if row.get("provenance", {}).get("benchmark_version") != PROTOCOL_VERSION:
            raise ValueError("main evaluation requires safety-management-v3 records; audit old records separately")
        visible = {item["asset_id"] for item in row["observation"].get("assets", ())}
        visible.update(item["entity_id"] for item in row["observation"].get("entities", ()))
        for outcome in row["counterfactuals"]["actions"].values():
            action = outcome["action"]
            if action["type"] != "WAIT" and action.get("target_id") not in visible:
                raise ValueError("v3 interventions require explicit observed public targets")
        record = WolfGymRecord.from_dict(row)
        for task in (TaskId.RISK_FORECASTING, TaskId.ACTION_EFFECT, TaskId.INTERVENTION_SELECTION):
            for example in adapt_record(record, task):
                name = "IS" if task is TaskId.INTERVENTION_SELECTION else task.value
                instruction = (
                    "Compare WAIT and every candidate's estimated conditional "
                    "system-failure risk. Choose any minimum-cost candidate whose "
                    "risk is at most safety_threshold and whose cost is at most "
                    "remaining_budget. WAIT is valid only when its risk also meets "
                    "the threshold. Return JSON "
                    '{"answer": {"type": "...", "target_id": "..."}}. '
                    "Use a null target for WAIT."
                    if name == "IS" else
                    "Estimate system-failure risk within forecast_horizon "
                    + ("under WAIT. " if name == "RF" else "after candidate_action. ")
                    + 'Return JSON {"answer": <number from 0 to 1>}.'
                )
                # Structured observations are the sole model evidence. Avoid a
                # second free-form copy that could disagree or carry metadata.
                inputs = {key: value for key, value in example.inputs.items() if key != "observation_text"}
                output.append({
                    "example_id": example.example_id, "task": name,
                    "task_protocol": MODEL_TASK_PROTOCOL,
                    "prompt": definitions + instruction + "\n" + json.dumps(inputs, sort_keys=True, allow_nan=False),
                })
    return output


def _selection_metrics(rows: Sequence[dict]) -> dict:
    n = len(rows)
    chosen = [row for row in rows if row["recognized"]]
    effective = [row for row in chosen if row["effective"]]
    valid_accuracy = sum(row["valid"] for row in rows) / n if n else None
    action_types = ("WAIT", "THROTTLE", "FREEZE")
    return {
        "states": n, "recognized_actions": len(chosen),
        "valid_action_accuracy": valid_accuracy,
        "invalid_or_missing_rate": (n - len(chosen)) / n if n else None,
        "budget_violation_rate": sum(not row["affordable"] for row in chosen) / len(chosen) if chosen else None,
        "safety_violation_rate": sum(not row["risk_safe"] for row in chosen) / len(chosen) if chosen else None,
        "safety_violation_denominator": len(chosen),
        "cost_regret": mean(row["cost_regret"] for row in effective) if effective else None,
        "cost_regret_denominator": len(effective),
        "prediction_type_distribution": dict(sorted(
            {kind: sum(row["chosen_type"] == kind for row in rows)
             for kind in action_types}.items()
        )),
        "valid_type_distribution": dict(sorted(
            {kind: sum(kind in row["valid_types"] for row in rows)
             for kind in action_types}.items()
        )),
        "best_fixed_action_type_accuracy": (
            max(sum(kind in row["valid_types"] for row in rows)
                for kind in action_types) / n if n else None
        ),
        "single_action_type_collapse": (
            len({row["chosen_type"] for row in chosen}) == 1 if chosen else None
        ),
        "diagnostics": {
            "primary_action_exact_match": (
                sum(row["exact"] for row in rows) / n if n else None
            ),
        },
    }


def score_predictions(records: Sequence[Mapping], predictions: Mapping[str, object]) -> dict:
    expected_ids = {example["example_id"] for example in public_examples(records)}
    unknown = set(predictions) - expected_ids
    if unknown:
        raise ValueError(f"predictions contain unknown examples: {sorted(unknown)[:3]}")
    rf_pairs, ae_pairs, selections, consistency = [], [], [], []
    ae_pairs_by_action = defaultdict(list)
    ae_expected_by_action = defaultdict(int)
    effect_directions = []
    ae_count = 0
    infeasible = ambiguous = 0
    for row in records:
        instance = row["instance_id"]
        q_wait = probability(predictions.get(f"{instance}:RF"))
        if q_wait is not None:
            rf_pairs.append((q_wait, float(row["risk_probability"])))
        predicted_risks = {"WAIT": q_wait}
        outcomes = row["counterfactuals"]["actions"]
        for label, outcome in outcomes.items():
            if label == "WAIT":
                continue
            ae_count += 1
            action_type = str(outcome["action"]["type"]).upper()
            ae_expected_by_action[action_type] += 1
            q = probability(predictions.get(f"{instance}:AE:{label}"))
            predicted_risks[label] = q
            if q is not None:
                ae_pairs.append((q, float(outcome["failure_probability"])))
                ae_pairs_by_action[action_type].append(
                    (q, float(outcome["failure_probability"]))
                )
                if q_wait is not None:
                    true_delta = float(row["risk_probability"]) - float(outcome["failure_probability"])
                    predicted_delta = q_wait - q
                    effect_directions.append(
                        effect_direction(predicted_delta) == effect_direction(true_delta)
                    )
        oracle = strict_label(row)
        if not oracle["feasible"]:
            infeasible += 1
            continue
        if oracle["ambiguous"] or row.get("oracle", {}).get("ambiguous", False):
            ambiguous += 1
            continue
        valid = {action_label(action) for action in oracle["valid_actions"]}
        valid_types = {
            str(action["type"]).upper() for action in oracle["valid_actions"]
        }
        primary = action_label(oracle["primary_action"])
        submitted_action = predictions.get(f"{instance}:IS")
        chosen = action_label(submitted_action)
        chosen_type = (
            str(submitted_action.get("type", "")).upper()
            if isinstance(submitted_action, Mapping) else None
        )
        outcome = outcomes.get(chosen)
        budget = float(row["observation"]["remaining_budget"])
        affordable = outcome is not None and float(outcome["cost"]) <= budget + 1e-12
        risk_safe = outcome is not None and outcome["failure_probability"] <= 0.20
        effective = affordable and risk_safe
        selections.append({
            "recognized": outcome is not None, "affordable": affordable,
            "risk_safe": risk_safe, "effective": effective,
            "valid": chosen in valid, "exact": chosen == primary,
            "chosen_type": chosen_type, "valid_types": valid_types,
            "cost_regret": outcome["cost"] - oracle["oracle_cost"] if effective else None,
        })
        if all(q is not None for q in predicted_risks.values()):
            predicted = {
                label: {**item, "failure_probability": predicted_risks[label]}
                for label, item in outcomes.items()
            }
            derived = EffectiveInterventionOracle().label_outcomes(predicted, budget)
            derived_set = {action_label(action) for action in derived["valid_actions"]}
            consistency.append({
                "derived_valid": action_label(derived["primary_action"]) in valid,
                "direct_valid": chosen in valid,
                "direct_in_derived_set": chosen is not None and chosen in derived_set,
                "predicted_infeasible": not derived["feasible"],
            })
    return {
        "protocol": PROTOCOL_VERSION,
        "model_task_protocol": MODEL_TASK_PROTOCOL,
        "unique_states": len(records),
        "unique_trajectories": len({row["trajectory_id"] for row in records}),
        "decision_instances": len(expected_ids),
        "missing_predictions": len(expected_ids - set(predictions)),
        "RF": probability_metrics(rf_pairs, expected=len(records)),
        "AE": probability_metrics(ae_pairs, expected=ae_count),
        "AE_by_action": {
            action: probability_metrics(
                ae_pairs_by_action[action], expected=ae_expected_by_action[action],
            )
            for action in sorted(ae_expected_by_action)
        },
        "AE_effect_direction": {
            "epsilon": EFFECT_DIRECTION_EPSILON,
            "expected": ae_count,
            "valid": len(effect_directions),
            "accuracy": mean(effect_directions) if effect_directions else None,
        },
        "IS": _selection_metrics(selections),
        "infeasible_diagnostic_states": infeasible,
        "ambiguous_IS_diagnostic_states": ambiguous,
        "cross_task_consistency": {
            "complete_feasible_states": len(consistency),
            **{key: mean(row[key] for row in consistency) if consistency else None
               for key in ("derived_valid", "direct_valid", "direct_in_derived_set", "predicted_infeasible")},
        },
    }
