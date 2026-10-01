from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from wolfgym.core import WolfSocietySimulator
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.env import (
    FixedActionCodec,
    RLTrainingConfig,
    RewardComponents,
    StructuredObservationEncoder,
    WolfGymEnv,
    WolfGymEnvConfig,
    WolfGymnasiumEnv,
    reward_profile,
)
from wolfgym.observations import ObservationConfig
from wolfgym.regulation import RegulatoryActionType


def base_environment(*, budget=2.0, horizon=3):
    simulator = WolfSocietySimulator(load_scenario("s0"), 20, 0.0, seed=7)
    for _ in range(3):
        simulator.step_day()
    return WolfGymEnv(
        simulator,
        simulator.get_state(),
        config=WolfGymEnvConfig(
            initial_budget=budget,
            max_interactions=horizon,
            observation=ObservationConfig(top_k_entities=2),
            intervention_duration_steps=1,
        ),
    )


def test_realized_reward_profiles_never_require_counterfactual_rollouts():
    components = RewardComponents(
        system_failure=1.0,
        intervention_cost=4.0,
        investigation_cost=0.0,
        clean_utility_loss=4.0,
    )
    assert reward_profile("balanced").reward(components) == -29.0
    assert reward_profile("safety_first").reward(components) == -104.0
    assert reward_profile("cost_sensitive").reward(components) == -14.0
    with pytest.raises(ValueError, match="unknown"):
        reward_profile("oracle_risk")


def test_fixed_action_vocabulary_masks_targets_and_budget():
    environment = base_environment(budget=2.0)
    observation, _ = environment.reset(seed=11)
    codec = FixedActionCodec(max_assets=3, max_entities=2)
    mask = codec.action_mask(observation, environment.costs)
    target_count = len(codec.targets(observation))
    assert codec.size == 11
    assert mask[0] == 1
    assert mask[1:1 + target_count].tolist() == [1] * target_count
    assert not mask[1 + codec.max_targets:].any()
    action = codec.decode(1, observation, environment.costs)
    assert action.action_type is RegulatoryActionType.THROTTLE
    assert codec.encode(action, observation) == 1
    with pytest.raises(ValueError, match="masked"):
        codec.decode(1 + codec.max_targets, observation, environment.costs)


def test_structured_observation_is_fixed_shape_and_float32():
    environment = base_environment()
    observation, _ = environment.reset(seed=11)
    codec = FixedActionCodec(max_assets=3, max_entities=2)
    encoder = StructuredObservationEncoder(
        codec,
        horizon_days=environment.horizon_days,
        initial_budget=environment.config.initial_budget,
    )
    encoded = encoder.encode(observation)
    assert {key: value.shape for key, value in encoded.items()} == encoder.shapes
    assert encoded["global"].dtype == encoded["targets"].dtype == np.float32
    assert encoded["target_mask"].dtype == np.int8
    assert np.isfinite(encoded["global"]).all()
    assert encoded["target_mask"].sum() == len(codec.targets(observation))


def test_rl_training_config_freezes_reward_and_disjoint_seed_ranges():
    config = RLTrainingConfig.from_yaml("tests/fixtures/rl_training.yaml")
    assert config.reward == reward_profile("balanced")
    assert not config.train.trajectory_seeds.overlaps(
        config.iid_evaluation.trajectory_seeds
    )
    factories = config.train.factories(
        4,
        initial_budget=config.initial_budget,
        horizon=config.horizon,
        top_k_entities=config.top_k_entities,
        namespace="test",
    )
    identities = {
        (item.scenario_id, item.society_size, item.alpha,
         item.trajectory_seed, item.checkpoint_day)
        for item in factories
    }
    assert len(identities) == 4


def test_gymnasium_adapter_is_optional_but_has_a_strict_contract():
    environment = base_environment()
    if importlib.util.find_spec("gymnasium") is None:
        with pytest.raises(ImportError, match="RL extra"):
            WolfGymnasiumEnv(environment)
        return
    wrapped = WolfGymnasiumEnv(environment)
    observation, info = wrapped.reset(seed=123)
    assert wrapped.observation_space.contains(observation)
    assert info["action_mask"].shape == (wrapped.action_space.n,)
    next_observation, reward, terminated, truncated, next_info = wrapped.step(0)
    assert wrapped.observation_space.contains(next_observation)
    assert reward == 0.0
    assert not (terminated and truncated)
    assert next_info["typed_action"]["type"] == "WAIT"
