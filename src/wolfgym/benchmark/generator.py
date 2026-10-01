"""Simulator-only benchmark generation; no model inference belongs here."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from wolfgym.benchmark.sampling import (
    StateSample,
    TrajectoryStateSampler,
    matched_continuation_seeds,
)
from wolfgym.benchmark.schema import BenchmarkRecord, RiskRegime, RiskThresholds
from wolfgym.counterfactual import CounterfactualEvaluator, ProportionateOracle
from wolfgym.core.version import SIMULATOR_VERSION
from wolfgym.observations import InformationLevel, ObservationBuilder, ObservationConfig
from wolfgym.regulation import RegulatoryAction, RegulatoryActionType
from wolfgym.tasks import eligible_tasks


DEFAULT_MECHANISMS = {
    "s0": "clean_control",
    "s1": "social_manipulation",
    "s2": "coordinated_promotion",
    "s3": "spoofing",
    "s4": "wash_trading",
}


@dataclass(frozen=True)
class GenerationConfig:
    output_path: Path
    target_records: int
    max_evaluated_states: int
    split: str
    sampler: TrajectoryStateSampler
    observation: ObservationConfig
    rollout_count: int
    rollout_horizon: int
    safety_threshold: float
    ambiguity_margin: float
    risk_margin: float
    risk_thresholds: RiskThresholds
    target_mode: str
    max_action_entities: int
    include_asset_targets: bool
    rollout_schedule: tuple[int, ...] = ()
    remaining_budgets: tuple[float, ...] = (4.0,)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GenerationConfig":
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("generation config must be a mapping")
        forbidden = _forbidden_inference_keys(raw)
        if forbidden:
            raise ValueError(
                "benchmark generation cannot configure model inference: "
                + ", ".join(sorted(forbidden))
            )
        sampling = _mapping(raw, "sampling")
        observation = _mapping(raw, "observation")
        counterfactual = _mapping(raw, "counterfactual")
        oracle = _mapping(raw, "oracle")
        regimes = _mapping(raw, "risk_regimes")
        scenarios = tuple(sampling["scenarios"])
        mechanisms = dict(DEFAULT_MECHANISMS)
        mechanisms.update(raw.get("mechanisms", {}))
        sampler = TrajectoryStateSampler(
            scenarios=scenarios,
            mechanisms=mechanisms,
            society_sizes=sampling["society_sizes"],
            alpha_by_scenario=sampling["alpha_by_scenario"],
            trajectory_seeds=sampling["trajectory_seeds"],
            sample_days=sampling["sample_days"],
            identity_salt=str(raw.get("schema_version", "wolfgym-1.0")),
        )
        output = Path(str(raw["output_path"]))
        if not output.is_absolute():
            output = (config_path.parent.parent / output).resolve()
        rollout_count = int(counterfactual.get("num_rollouts", 32))
        rollout_schedule = tuple(int(value) for value in counterfactual.get(
            "rollout_schedule", (rollout_count,),
        ))
        remaining_budgets = tuple(float(value) for value in oracle.get(
            "remaining_budgets", (4.0,),
        ))
        result = cls(
            output_path=output,
            target_records=int(raw.get("target_records", 200)),
            max_evaluated_states=int(raw.get("max_evaluated_states", 400)),
            split=str(raw.get("split", "pilot")),
            sampler=sampler,
            observation=ObservationConfig(
                window_days=int(observation.get("window_days", 3)),
                top_k_entities=int(observation.get("top_k_entities", 5)),
                information_level=InformationLevel(int(observation.get("information_level", 2))),
                activity_change_clip=float(observation.get("activity_change_clip", 10.0)),
                include_messages=bool(observation.get("include_messages", False)),
                max_messages=int(observation.get("max_messages", 8)),
            ),
            rollout_count=rollout_count,
            rollout_horizon=int(counterfactual.get("horizon", 10)),
            safety_threshold=float(oracle.get("safety_threshold", 0.20)),
            ambiguity_margin=float(oracle.get("ambiguity_margin", 0.05)),
            risk_margin=float(oracle.get("risk_margin", 0.05)),
            risk_thresholds=RiskThresholds(
                stable_upper=float(regimes.get("stable_upper", 0.10)),
                near_critical_lower=float(regimes.get("near_critical_lower", 0.40)),
                supercritical_lower=float(regimes.get("supercritical_lower", 0.60)),
            ),
            target_mode=str(raw.get("target_mode", "top_entities")),
            max_action_entities=int(raw.get("max_action_entities", 5)),
            include_asset_targets=bool(raw.get("include_asset_targets", False)),
            rollout_schedule=rollout_schedule,
            remaining_budgets=remaining_budgets,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.target_records <= 0 or self.max_evaluated_states <= 0:
            raise ValueError("record limits must be positive")
        if self.max_evaluated_states < self.target_records:
            raise ValueError("max_evaluated_states cannot be below target_records")
        if self.rollout_count <= 0 or self.rollout_horizon <= 0:
            raise ValueError("counterfactual rollout settings must be positive")
        if (
            not self.rollout_schedule
            or any(value <= 0 for value in self.rollout_schedule)
            or tuple(sorted(set(self.rollout_schedule))) != self.rollout_schedule
            or self.rollout_schedule[-1] != self.rollout_count
        ):
            raise ValueError(
                "rollout_schedule must be strictly increasing and end at num_rollouts"
            )
        if not self.remaining_budgets or any(
            not math.isfinite(value) or value < 0.0
            for value in self.remaining_budgets
        ):
            raise ValueError("oracle.remaining_budgets must be finite and non-negative")
        if self.target_mode not in {"aggregate", "top_entities", "observed_targets"}:
            raise ValueError(
                "target_mode must be 'aggregate', 'top_entities', or 'observed_targets'"
            )
        if self.max_action_entities < 0:
            raise ValueError("max_action_entities must be non-negative")


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _forbidden_inference_keys(value: Any) -> set[str]:
    forbidden_names = {"model", "backend", "api_key", "llm"}
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in forbidden_names:
                found.add(str(key))
            found.update(_forbidden_inference_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_inference_keys(item))
    return found


class BenchmarkGenerator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.builder = ObservationBuilder(config.observation)
        self.evaluator = CounterfactualEvaluator()
        self.oracle = ProportionateOracle(
            safety_threshold=config.safety_threshold,
            ambiguity_margin=config.ambiguity_margin,
            risk_margin=config.risk_margin,
        )

    def generate(self) -> list[BenchmarkRecord]:
        candidates: list[BenchmarkRecord] = []
        for index, sample in enumerate(self.config.sampler):
            if index >= self.config.max_evaluated_states:
                break
            candidates.append(self.build_record(sample))
        if len(candidates) < self.config.target_records:
            raise ValueError(
                f"sampling grid yielded {len(candidates)} states, fewer than "
                f"target_records={self.config.target_records}"
            )
        return _balanced_take(candidates, self.config.target_records)

    def write_jsonl(self, output_path: str | Path | None = None) -> list[BenchmarkRecord]:
        records = self.generate()
        path = Path(output_path).resolve() if output_path else self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(record.to_json() + "\n" for record in records)
        path.write_text(content)
        return records

    def build_record(
        self,
        sample: StateSample,
        additional_provenance: Mapping[str, Any] | None = None,
    ) -> BenchmarkRecord:
        observation = self._build_observation(sample)
        actions = self._candidate_actions(observation)
        result = self._evaluate_counterfactuals(sample, actions)
        risk = result.noop_failure_probability
        if risk is None:
            raise RuntimeError("candidate set must contain WAIT")
        oracle = self._oracle_decision(result, observation)
        provenance = {
            "simulator_version": SIMULATOR_VERSION,
            "generator": "simulator_only",
            "trajectory_id": sample.trajectory_id,
            "scenario_id": sample.scenario_id,
            "alpha": sample.alpha,
            "trajectory_seed": sample.trajectory_seed,
            "state_day": sample.day,
            "rollout_count": len(result.seeds),
            "rollout_horizon": result.horizon,
            "matched_continuation_seeds": list(result.seeds),
            "failure_criterion": result.failure_criterion.to_dict(),
        }
        provenance.update(additional_provenance or {})
        record_schema_version = "wolfgym-record-2.0"
        if observation.messages is not None:
            record_schema_version = "wolfgym-record-3.0"
            provenance["observation_schema_version"] = observation.schema_version
        return BenchmarkRecord(
            schema_version=record_schema_version,
            instance_id=sample.instance_id,
            trajectory_id=sample.trajectory_id,
            split=self.config.split,
            scenario_id=sample.scenario_id,
            checkpoint_day=sample.day,
            hidden_state_ref=f"state://{sample.trajectory_id}/{sample.day}",
            observation=observation.to_dict(),
            observation_text=observation.to_text(),
            risk_probability=risk,
            risk_regime=self.config.risk_thresholds.classify(risk),
            mechanism=sample.mechanism,
            society_size=sample.society_size,
            counterfactuals=result.to_dict(),
            failure_criterion=result.failure_criterion.to_dict(),
            wait_outcomes=next(
                outcome.to_dict()
                for outcome in result.actions.values()
                if outcome.action.action_type is RegulatoryActionType.WAIT
            ),
            oracle=oracle.to_dict(),
            task_eligibility=tuple(
                task.value
                for task in eligible_tasks(
                    sample.mechanism,
                    self.config.risk_thresholds.classify(risk).value,
                )
            ),
            provenance=provenance,
        )

    def _build_observation(self, sample: StateSample):
        return self.builder.build(sample.simulator, sample.state)

    def _evaluate_counterfactuals(
        self,
        sample: StateSample,
        actions: Sequence[RegulatoryAction],
    ):
        seeds = matched_continuation_seeds(sample.instance_id, self.config.rollout_count)
        return self.evaluator.evaluate_state(
            simulator=sample.simulator,
            state=sample.state,
            candidate_actions=actions,
            seeds=seeds,
            horizon=self.config.rollout_horizon,
        )

    def _oracle_decision(self, result, observation):
        del observation
        return self.oracle.choose(result)

    def _candidate_actions(self, observation: Any) -> list[RegulatoryAction]:
        actions = [RegulatoryAction(RegulatoryActionType.WAIT)]
        if self.config.target_mode == "aggregate":
            actions.extend([
                RegulatoryAction(RegulatoryActionType.THROTTLE),
                RegulatoryAction(RegulatoryActionType.FREEZE),
            ])
            return actions
        targets = [
            entity.entity_id
            for entity in observation.entities[:self.config.max_action_entities]
        ]
        if self.config.target_mode == "observed_targets" \
                and self.config.include_asset_targets:
            targets.extend(asset.asset_id for asset in observation.assets)
        if not targets:
            targets = [None]
        for target in targets:
            actions.append(RegulatoryAction(RegulatoryActionType.THROTTLE, target))
            actions.append(RegulatoryAction(RegulatoryActionType.FREEZE, target))
        return actions


def _balanced_take(records: Sequence[BenchmarkRecord], count: int) -> list[BenchmarkRecord]:
    """Round-robin across available mechanism/size/risk cells."""
    groups: dict[tuple[str, int, RiskRegime], list[BenchmarkRecord]] = {}
    for record in records:
        key = (record.mechanism, record.society_size, record.risk_regime)
        groups.setdefault(key, []).append(record)
    for values in groups.values():
        values.sort(key=lambda record: record.instance_id)
    regime_order = {
        RiskRegime.NEAR_CRITICAL: 0,
        RiskRegime.SUPERCRITICAL: 1,
        RiskRegime.SUBCRITICAL: 2,
        RiskRegime.STABLE: 3,
    }
    keys = sorted(groups, key=lambda key: (regime_order[key[2]], key[0], key[1]))
    selected: list[BenchmarkRecord] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected
