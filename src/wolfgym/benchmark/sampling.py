"""Deterministic simulator trajectory/state sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

from wolfgym.benchmark.schema import stable_instance_id
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.env.state import SimulatorState
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.core.version import SIMULATOR_VERSION


@dataclass(frozen=True)
class StateSample:
    instance_id: str
    trajectory_id: str
    simulator: WolfSocietySimulator
    state: SimulatorState
    scenario_id: str
    mechanism: str
    society_size: int
    alpha: float
    trajectory_seed: int
    day: int


EpisodeSpec = tuple[str, int, float, int]


class TrajectoryStateSampler:
    """Yield immutable day-boundary snapshots in a reproducible order."""

    def __init__(
        self,
        scenarios: Sequence[str],
        mechanisms: Mapping[str, str],
        society_sizes: Sequence[int],
        alpha_by_scenario: Mapping[str, Sequence[float]],
        trajectory_seeds: Sequence[int],
        sample_days: Sequence[int],
        identity_salt: str = "wolfgym-v1",
    ) -> None:
        self.scenarios = tuple(scenarios)
        self.mechanisms = dict(mechanisms)
        self.society_sizes = tuple(int(value) for value in society_sizes)
        self.alpha_by_scenario = {
            key: tuple(float(value) for value in values)
            for key, values in alpha_by_scenario.items()
        }
        self.trajectory_seeds = tuple(int(value) for value in trajectory_seeds)
        self.sample_days = tuple(sorted(set(int(value) for value in sample_days)))
        self.identity_salt = str(identity_salt)
        self._validate()

    def _validate(self) -> None:
        if not self.scenarios or not self.society_sizes or not self.trajectory_seeds:
            raise ValueError("sampling grid dimensions must be non-empty")
        if not self.sample_days or any(day <= 0 for day in self.sample_days):
            raise ValueError("sample_days must contain positive day boundaries")
        if any(size <= 0 for size in self.society_sizes):
            raise ValueError("society sizes must be positive")
        if any(seed < 0 for seed in self.trajectory_seeds):
            raise ValueError("trajectory seeds must be non-negative")
        for scenario in self.scenarios:
            if scenario not in self.mechanisms:
                raise ValueError(f"missing mechanism label for {scenario!r}")
            alphas = self.alpha_by_scenario.get(scenario)
            if not alphas:
                raise ValueError(f"missing alpha grid for {scenario!r}")
            if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
                raise ValueError("alpha values must be in [0, 1]")

    def episode_specs(self) -> tuple[EpisodeSpec, ...]:
        """Return the stable trajectory work units without running simulations."""
        episode_specs = [
            (scenario_id, society_size, alpha, seed)
            for scenario_id in self.scenarios
            for society_size in self.society_sizes
            for alpha in self.alpha_by_scenario[scenario_id]
            for seed in self.trajectory_seeds
        ]
        # A generation cap must not silently select only the first scenarios or
        # smallest society size. Hash ordering is deterministic across runs and
        # independent of YAML list order.
        episode_specs.sort(key=lambda spec: stable_instance_id({
            "salt": self.identity_salt,
            "episode": spec,
        }))
        return tuple(episode_specs)

    def episode_jobs(
        self, max_states: int | None = None,
    ) -> tuple[tuple[EpisodeSpec, tuple[int, ...]], ...]:
        """Plan capped work without simulating trajectories."""
        remaining = None if max_states is None else int(max_states)
        if remaining is not None and remaining < 0:
            raise ValueError("max_states must be non-negative")
        jobs = []
        for spec in self.episode_specs():
            if remaining == 0:
                break
            days = self.sample_days if remaining is None else self.sample_days[:remaining]
            jobs.append((spec, tuple(days)))
            if remaining is not None:
                remaining -= len(days)
        return tuple(jobs)

    def trajectory_id_for_episode(self, spec: EpisodeSpec) -> str:
        scenario_id, society_size, alpha, seed = spec
        return "traj_" + stable_instance_id({
            "simulator_version": SIMULATOR_VERSION,
            "salt": self.identity_salt,
            "scenario": scenario_id,
            "society_size": society_size,
            "alpha": alpha,
            "trajectory_seed": seed,
        }).removeprefix("wg_")

    def instance_id_for_episode(self, spec: EpisodeSpec, day: int) -> str:
        scenario_id, society_size, alpha, seed = spec
        return stable_instance_id({
            "simulator_version": SIMULATOR_VERSION,
            "salt": self.identity_salt,
            "scenario": scenario_id,
            "society_size": society_size,
            "alpha": alpha,
            "trajectory_seed": seed,
            "day": int(day),
        })

    def samples_for_episode(
        self,
        spec: EpisodeSpec,
        sample_days: Sequence[int] | None = None,
    ) -> Iterator[StateSample]:
        """Simulate one trajectory once and emit its requested checkpoints."""
        scenario_id, society_size, alpha, seed = spec
        days = tuple(sorted(set(
            self.sample_days if sample_days is None else (int(day) for day in sample_days)
        )))
        if not days:
            return
        if any(day not in self.sample_days for day in days):
            raise ValueError("requested days must belong to the configured sample_days")
        scenario = load_scenario(scenario_id)
        if days[-1] >= scenario.horizon_days:
            raise ValueError(f"sample day must be before horizon for {scenario_id!r}")
        env = WolfSocietySimulator(
            scenario, n_society=society_size, alpha=alpha, seed=seed
        )
        wanted = set(days)
        trajectory_id = self.trajectory_id_for_episode(spec)
        for day in range(1, days[-1] + 1):
            env.step_day()
            if day not in wanted:
                continue
            yield StateSample(
                instance_id=self.instance_id_for_episode(spec, day),
                trajectory_id=trajectory_id,
                simulator=env,
                state=env.get_state(),
                scenario_id=scenario_id,
                mechanism=self.mechanisms[scenario_id],
                society_size=society_size,
                alpha=alpha,
                trajectory_seed=seed,
                day=day,
            )

    def __iter__(self) -> Iterator[StateSample]:
        for spec, days in self.episode_jobs():
            yield from self.samples_for_episode(spec, days)


def matched_continuation_seeds(
    instance_id: str,
    count: int,
    namespace: str = "continuation",
) -> tuple[int, ...]:
    """Derive unique, stable rollout seeds from the state identity."""
    if count <= 0:
        raise ValueError("rollout count must be positive")
    import hashlib

    seeds: list[int] = []
    used: set[int] = set()
    nonce = 0
    while len(seeds) < count:
        digest = hashlib.blake2b(
            f"{instance_id}|{namespace}|{nonce}".encode(), digest_size=8
        ).digest()
        seed = int.from_bytes(digest, "big") & ((1 << 63) - 1)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
        nonce += 1
    return tuple(seeds)
