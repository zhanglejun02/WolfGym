"""Deterministic task views over canonical WolfGym records and traces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from wolfgym.benchmark.schema import WolfGymRecord
from wolfgym.counterfactual.oracle import EffectiveInterventionOracle
from wolfgym.env.trace import EpisodeTrace
from wolfgym.observations.serialization import assert_no_hidden_fields
from wolfgym.tasks.specs import TaskId


@dataclass(frozen=True)
class TaskExample:
    example_id: str
    task_id: TaskId
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    metadata: Mapping[str, Any]


def adapt_record(record: WolfGymRecord, task_id: TaskId) -> tuple[TaskExample, ...]:
    if task_id.value not in record.task_eligibility:
        return ()
    assert_no_hidden_fields(record.observation)
    if any(value in record.observation_text for value in (
        record.instance_id, record.trajectory_id, record.hidden_state_ref,
    )):
        raise ValueError("record metadata leaked into observation text")
    common_inputs = {
        "observation": record.observation,
        "observation_text": record.observation_text,
        "forecast_horizon": record.counterfactuals["horizon"],
    }
    metadata = {
        "instance_id": record.instance_id,
        "trajectory_id": record.trajectory_id,
        "split": record.split,
    }
    if task_id is TaskId.RISK_FORECASTING:
        return (TaskExample(
            f"{record.instance_id}:RF",
            task_id,
            common_inputs,
            {
                "failure_probability": record.risk_probability,
                "risk_regime": record.risk_regime.value,
            },
            metadata,
        ),)
    if task_id is TaskId.MECHANISM_DIAGNOSIS:
        return (TaskExample(
            f"{record.instance_id}:MD",
            task_id,
            common_inputs,
            {"mechanism": record.mechanism},
            metadata,
        ),)
    if task_id is TaskId.ACTION_EFFECT:
        examples: list[TaskExample] = []
        for label, outcome in sorted(record.counterfactuals["actions"].items()):
            if outcome["action"]["type"] == "WAIT":
                continue
            action_risk = float(outcome["failure_probability"])
            examples.append(TaskExample(
                f"{record.instance_id}:AE:{label}",
                task_id,
                {**common_inputs, "candidate_action": outcome["action"]},
                {
                    "post_action_failure_probability": action_risk,
                    "risk_reduction": record.risk_probability - action_risk,
                },
                metadata,
            ))
        return tuple(examples)
    if task_id is TaskId.INTERVENTION_SELECTION:
        oracle = EffectiveInterventionOracle(
            safety_threshold=float(record.oracle.get("safety_threshold", 0.20)),
        ).label_outcomes(
            record.counterfactuals["actions"],
            record.observation.get("remaining_budget"),
        )
        if not oracle["feasible"] or oracle["ambiguous"] or record.oracle.get("ambiguous", False):
            return ()
        oracle["ambiguous"] = oracle["ambiguous"] or bool(record.oracle.get("ambiguous", False))
        return (TaskExample(
            f"{record.instance_id}:IS",
            task_id,
            {
                **common_inputs,
                "candidate_actions": [
                    {**outcome["action"], "cost": outcome["cost"]}
                    for outcome in record.counterfactuals["actions"].values()
                ],
                "safety_threshold": oracle["safety_threshold"],
                "remaining_budget": record.observation.get("remaining_budget"),
            },
            {
                "primary_action": oracle["primary_action"],
                "valid_actions": oracle["valid_actions"],
                "ambiguous": oracle["ambiguous"],
            },
            metadata,
        ),)
    raise ValueError("CR is adapted from EpisodeTrace, not a static record")


def adapt_trace(trace: EpisodeTrace) -> TaskExample:
    return TaskExample(
        example_id=f"{trace.episode_id}:CR",
        task_id=TaskId.CLOSED_LOOP_REGULATION,
        inputs={"initial_observation": trace.initial_observation},
        targets={"episode_trace": trace.to_dict()},
        metadata={"trajectory_id": trace.trajectory_id},
    )
