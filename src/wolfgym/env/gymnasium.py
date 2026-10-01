"""Optional Gymnasium adapter and vector-environment factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from wolfgym.benchmark.schema import stable_instance_id
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.env.core import WolfGymEnv, WolfGymEnvConfig
from wolfgym.env.rl import (
    GLOBAL_FEATURE_NAMES,
    RL_INTERFACE_VERSION,
    TARGET_FEATURE_NAMES,
    FixedActionCodec,
    RLRewardConfig,
    StructuredObservationEncoder,
    reward_profile,
)
from wolfgym.observations import ObservationConfig

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised by the dependency error test
    gym = None
    spaces = None


_GymBase = gym.Env if gym is not None else object


def _require_gymnasium() -> None:
    if gym is None:
        raise ImportError(
            "Gymnasium support requires the RL extra: pip install -e '.[rl]'"
        )


class WolfGymnasiumEnv(_GymBase):
    """Gymnasium-compatible view of :class:`WolfGymEnv` for masked policies."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: WolfGymEnv,
        *,
        action_codec: FixedActionCodec | None = None,
        reward: RLRewardConfig | str = "balanced",
    ) -> None:
        _require_gymnasium()
        super().__init__()
        self.environment = environment
        self.action_codec = action_codec or FixedActionCodec(
            max_assets=len(environment.simulator.market.assets),
            max_entities=environment.config.observation.top_k_entities,
        )
        self.reward_config = reward_profile(reward) if isinstance(reward, str) else reward
        self.encoder = StructuredObservationEncoder(
            action_codec=self.action_codec,
            horizon_days=environment.horizon_days,
            initial_budget=environment.config.initial_budget,
        )
        shapes = self.encoder.shapes
        self.action_space = spaces.Discrete(self.action_codec.size)
        self.observation_space = spaces.Dict({
            "global": spaces.Box(
                low=-self.encoder.value_clip,
                high=self.encoder.value_clip,
                shape=shapes["global"],
                dtype=np.float32,
            ),
            "targets": spaces.Box(
                low=-self.encoder.value_clip,
                high=self.encoder.value_clip,
                shape=shapes["targets"],
                dtype=np.float32,
            ),
            "target_mask": spaces.MultiBinary(shapes["target_mask"][0]),
        })
        self._observation = None

    @property
    def trace(self):
        return self.environment.trace

    def _info(self, info: dict) -> dict:
        # Trusted outcome diagnostics may use account/role information.
        info = {key: value for key, value in info.items()
                if key not in {"financial_outcomes", "episode_id", "trajectory_id"}}
        mask = self.action_codec.action_mask(
            self._observation, self.environment.costs,
        )
        return {
            **info,
            "rl_interface_version": RL_INTERFACE_VERSION,
            "reward_profile": self.reward_config.profile,
            "action_mask": mask.astype(bool, copy=False),
            "action_targets": self.action_codec.targets(self._observation),
            "global_feature_names": GLOBAL_FEATURE_NAMES,
            "target_feature_names": TARGET_FEATURE_NAMES,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        supplied = None if options is None else options.get("continuation_seed")
        continuation_seed = (
            int(supplied)
            if supplied is not None
            else int(self.np_random.integers(0, (1 << 63) - 1))
        )
        self._observation, info = self.environment.reset(seed=continuation_seed)
        info = {**info, "continuation_seed": continuation_seed}
        return self.encoder.encode(self._observation), self._info(info)

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("call reset() before step()")
        typed_action = self.action_codec.decode(
            action, self._observation, self.environment.costs,
        )
        observation, _, terminated, truncated, info = self.environment.step(typed_action)
        self._observation = observation
        reward = self.reward_config.reward(info["reward_components"])
        info = {**info, "typed_action": typed_action.to_dict()}
        return self.encoder.encode(observation), reward, terminated, truncated, self._info(info)

    def action_masks(self) -> np.ndarray:
        """Compatibility hook used by sb3-contrib MaskablePPO."""
        if self._observation is None:
            raise RuntimeError("call reset() before requesting an action mask")
        return self.action_codec.action_mask(
            self._observation, self.environment.costs,
        ).astype(bool, copy=False)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class SimulatorEpisodeFactory:
    """Pickle-friendly builder for trajectory-disjoint RL environments."""

    scenario_id: str
    society_size: int
    alpha: float
    trajectory_seed: int
    checkpoint_day: int
    initial_budget: float = 12.0
    max_interactions: int = 8
    top_k_entities: int = 5

    def __post_init__(self) -> None:
        if self.society_size <= 0 or self.trajectory_seed < 0 or self.checkpoint_day < 0:
            raise ValueError("invalid RL episode factory parameters")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.max_interactions <= 0 or self.top_k_entities < 0:
            raise ValueError("invalid interaction horizon or entity count")
        if not np.isfinite(self.initial_budget) or self.initial_budget <= 0.0:
            raise ValueError("initial budget must be finite and positive")

    def __call__(self) -> WolfGymEnv:
        scenario = load_scenario(self.scenario_id)
        simulator = WolfSocietySimulator(
            scenario,
            n_society=self.society_size,
            alpha=self.alpha,
            seed=self.trajectory_seed,
        )
        for _ in range(self.checkpoint_day):
            simulator.step_day()
        state = simulator.get_state()
        if state.horizon_days - state.next_day < self.max_interactions:
            raise ValueError("RL snapshot lacks the requested interaction horizon")
        identity = {
            "scenario": self.scenario_id,
            "society_size": self.society_size,
            "alpha": self.alpha,
            "trajectory_seed": self.trajectory_seed,
        }
        trajectory_id = "traj_rl_" + stable_instance_id(identity).removeprefix("wg_")
        return WolfGymEnv(
            simulator,
            state,
            config=WolfGymEnvConfig(
                initial_budget=self.initial_budget,
                max_interactions=self.max_interactions,
                observation=ObservationConfig(top_k_entities=self.top_k_entities),
                intervention_duration_steps=1,
            ),
            episode_id=f"rl:{trajectory_id}:{self.checkpoint_day}",
            trajectory_id=trajectory_id,
        )


@dataclass(frozen=True)
class _WrappedFactory:
    factory: Callable[[], WolfGymEnv]
    action_codec: FixedActionCodec | None
    reward: RLRewardConfig | str

    def __call__(self) -> WolfGymnasiumEnv:
        return WolfGymnasiumEnv(
            self.factory(), action_codec=self.action_codec, reward=self.reward,
        )


def make_vector_env(
    environment_factories: Sequence[Callable[[], WolfGymEnv]],
    *,
    asynchronous: bool = False,
    action_codec: FixedActionCodec | None = None,
    reward: RLRewardConfig | str = "balanced",
):
    """Build a standard Gymnasium Sync/AsyncVectorEnv without oracle calls."""
    _require_gymnasium()
    if not environment_factories:
        raise ValueError("at least one environment factory is required")
    factories = [
        _WrappedFactory(factory, action_codec, reward)
        for factory in environment_factories
    ]
    vector_type = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return vector_type(factories)
