"""Versioned RL experiment configuration with trajectory-disjoint factories."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from wolfgym.benchmark.schema import stable_instance_id
from wolfgym.env.gymnasium import SimulatorEpisodeFactory
from wolfgym.env.rl import FixedActionCodec, RLRewardConfig, reward_profile


@dataclass(frozen=True)
class SeedRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid trajectory seed range")

    def overlaps(self, other: "SeedRange") -> bool:
        return max(self.start, other.start) <= min(self.end, other.end)


@dataclass(frozen=True)
class RLEpisodeGrid:
    scenarios: tuple[str, ...]
    society_sizes: tuple[int, ...]
    alpha_by_scenario: Mapping[str, tuple[float, ...]]
    checkpoint_days: tuple[int, ...]
    trajectory_seeds: SeedRange

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RLEpisodeGrid":
        seed_range = raw["trajectory_seed_range"]
        result = cls(
            scenarios=tuple(map(str, raw["scenarios"])),
            society_sizes=tuple(map(int, raw["society_sizes"])),
            alpha_by_scenario={
                str(key): tuple(map(float, values))
                for key, values in raw["alpha_by_scenario"].items()
            },
            checkpoint_days=tuple(map(int, raw["checkpoint_days"])),
            trajectory_seeds=SeedRange(int(seed_range["start"]), int(seed_range["end"])),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.scenarios or not self.society_sizes or not self.checkpoint_days:
            raise ValueError("RL episode grid dimensions must be non-empty")
        if any(size <= 0 for size in self.society_sizes):
            raise ValueError("RL society sizes must be positive")
        if any(day < 0 for day in self.checkpoint_days):
            raise ValueError("RL checkpoint days must be non-negative")
        for scenario in self.scenarios:
            values = self.alpha_by_scenario.get(scenario, ())
            if not values or any(not 0.0 <= alpha <= 1.0 for alpha in values):
                raise ValueError(f"invalid alpha grid for {scenario!r}")

    def factories(
        self,
        count: int,
        *,
        initial_budget: float,
        horizon: int,
        top_k_entities: int,
        namespace: str,
    ) -> tuple[SimulatorEpisodeFactory, ...]:
        if count <= 0:
            raise ValueError("factory count must be positive")
        cells = [
            (scenario, size, alpha, day)
            for scenario, size, day in product(
                self.scenarios, self.society_sizes, self.checkpoint_days,
            )
            for alpha in self.alpha_by_scenario[scenario]
        ]
        cells.sort(key=lambda cell: stable_instance_id({
            "namespace": namespace, "cell": cell,
        }))
        seeds_needed = math.ceil(count / len(cells))
        if self.trajectory_seeds.start + seeds_needed - 1 > self.trajectory_seeds.end:
            raise ValueError("trajectory seed range cannot supply the requested factories")
        factories = []
        for index in range(count):
            scenario, size, alpha, day = cells[index % len(cells)]
            seed = self.trajectory_seeds.start + index // len(cells)
            factories.append(SimulatorEpisodeFactory(
                scenario_id=scenario,
                society_size=size,
                alpha=alpha,
                trajectory_seed=seed,
                checkpoint_day=day,
                initial_budget=initial_budget,
                max_interactions=horizon,
                top_k_entities=top_k_entities,
            ))
        return tuple(factories)


@dataclass(frozen=True)
class RLTrainingConfig:
    schema_version: str
    initial_budget: float
    horizon: int
    top_k_entities: int
    action_codec: FixedActionCodec
    reward: RLRewardConfig
    train: RLEpisodeGrid
    iid_evaluation: RLEpisodeGrid
    ppo: Mapping[str, int | float]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RLTrainingConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, Mapping):
            raise ValueError("RL training config must be a mapping")
        environment = raw["environment"]
        slots = environment["action_slots"]
        reward = raw["reward"]
        result = cls(
            schema_version=str(raw["schema_version"]),
            initial_budget=float(environment["initial_budget"]),
            horizon=int(environment["horizon"]),
            top_k_entities=int(environment["top_k_entities"]),
            action_codec=FixedActionCodec(
                max_assets=int(slots["max_assets"]),
                max_entities=int(slots["max_entities"]),
            ),
            reward=RLRewardConfig(
                profile=str(reward["profile"]),
                failure_penalty=float(reward["failure_penalty"]),
                intervention_cost_weight=float(reward["intervention_cost_weight"]),
                investigation_cost_weight=float(reward["investigation_cost_weight"]),
                clean_utility_loss_weight=float(reward["clean_utility_loss_weight"]),
            ),
            train=RLEpisodeGrid.from_mapping(raw["train"]),
            iid_evaluation=RLEpisodeGrid.from_mapping(raw["iid_evaluation"]),
            ppo={str(key): value for key, value in raw["ppo_sanity"].items()},
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != "wolfgym-rl-balanced-v1":
            raise ValueError("unsupported RL training configuration version")
        if not math.isfinite(self.initial_budget) or self.initial_budget <= 0.0:
            raise ValueError("RL initial budget must be finite and positive")
        if self.horizon <= 0 or self.top_k_entities < 0:
            raise ValueError("invalid RL horizon or target count")
        if self.action_codec.max_entities != self.top_k_entities:
            raise ValueError("entity observation and action slot counts must match")
        if self.reward != reward_profile("balanced"):
            raise ValueError("wolfgym-rl-balanced-v1 freezes the balanced reward weights")
        if self.train.trajectory_seeds.overlaps(self.iid_evaluation.trajectory_seeds):
            raise ValueError("RL train and IID evaluation trajectory seeds overlap")
        if int(self.ppo.get("num_envs", 0)) <= 0:
            raise ValueError("PPO num_envs must be positive")
        if int(self.ppo.get("total_timesteps", 0)) <= 0:
            raise ValueError("PPO total_timesteps must be positive")
