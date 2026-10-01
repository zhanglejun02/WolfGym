"""Frozen capability/task specification for WolfGym views."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class TaskId(str, Enum):
    RISK_FORECASTING = "RF"
    MECHANISM_DIAGNOSIS = "MD"
    ACTION_EFFECT = "AE"
    # Frozen v1 records serialize this task as PI. New public material calls it
    # Intervention Selection (IS); the old enum name remains an alias only.
    INTERVENTION_SELECTION = "PI"
    PROPORTIONATE_INTERVENTION = "PI"
    CLOSED_LOOP_REGULATION = "CR"


class TaskModality(str, Enum):
    STATIC = "static"
    INTERACTIVE = "interactive"


@dataclass(frozen=True)
class TaskSpec:
    task_id: TaskId
    name: str
    capability: str
    modality: TaskModality
    input_contract: tuple[str, ...]
    output_contract: tuple[str, ...]
    metrics: tuple[str, ...]


TASK_SPECS: Mapping[TaskId, TaskSpec] = MappingProxyType({
    TaskId.RISK_FORECASTING: TaskSpec(
        TaskId.RISK_FORECASTING,
        "Risk Forecasting",
        "understand",
        TaskModality.STATIC,
        ("regulatory_observation", "forecast_horizon"),
        ("failure_probability", "risk_regime"),
        ("brier_score", "mae", "ece", "regime_accuracy"),
    ),
    TaskId.MECHANISM_DIAGNOSIS: TaskSpec(
        TaskId.MECHANISM_DIAGNOSIS,
        "Mechanism Diagnosis",
        "understand",
        TaskModality.STATIC,
        ("regulatory_observation",),
        ("mechanism",),
        ("accuracy", "macro_f1"),
    ),
    TaskId.ACTION_EFFECT: TaskSpec(
        TaskId.ACTION_EFFECT,
        "Action Effect Estimation",
        "intervene",
        TaskModality.STATIC,
        ("regulatory_observation", "candidate_action"),
        ("post_action_failure_probability", "risk_reduction"),
        ("mae", "brier_score", "effect_direction_accuracy"),
    ),
    TaskId.INTERVENTION_SELECTION: TaskSpec(
        TaskId.INTERVENTION_SELECTION,
        "Intervention Selection",
        "intervene",
        TaskModality.STATIC,
        ("regulatory_observation", "candidate_actions", "remaining_budget"),
        ("regulatory_action",),
        ("valid_action_accuracy", "budget_violation_rate", "safety_violation_rate", "cost_regret"),
    ),
    TaskId.CLOSED_LOOP_REGULATION: TaskSpec(
        TaskId.CLOSED_LOOP_REGULATION,
        "Closed-loop Regulation",
        "observe_understand_seek_evidence_intervene_reobserve",
        TaskModality.INTERACTIVE,
        ("wolfgym_env",),
        ("episode_trace",),
        (
            "system_failure",
            "intervention_cost",
            "investigation_cost",
            "clean_utility_loss",
            "oracle_regret",
        ),
    ),
})


def eligible_tasks(mechanism: str, risk_regime: str) -> tuple[TaskId, ...]:
    """Mechanism-aware views; no Cartesian task-coverage assumption."""
    value = mechanism.casefold()
    if "clean" in value:
        return (
            TaskId.RISK_FORECASTING,
            TaskId.MECHANISM_DIAGNOSIS,
            TaskId.INTERVENTION_SELECTION,
            TaskId.CLOSED_LOOP_REGULATION,
        )
    if "spoof" in value:
        # Sharp-boundary states are primarily action-effect/control cases.
        return (
            TaskId.RISK_FORECASTING,
            TaskId.MECHANISM_DIAGNOSIS,
            TaskId.ACTION_EFFECT,
            TaskId.INTERVENTION_SELECTION,
            TaskId.CLOSED_LOOP_REGULATION,
        )
    if "social" in value or "pump" in value:
        return tuple(TaskId)
    # Unaudited mechanisms may enter static diagnosis, but are not silently
    # declared suitable for closed-loop regulation.
    return (
        TaskId.RISK_FORECASTING,
        TaskId.MECHANISM_DIAGNOSIS,
        TaskId.ACTION_EFFECT,
        TaskId.INTERVENTION_SELECTION,
    )
