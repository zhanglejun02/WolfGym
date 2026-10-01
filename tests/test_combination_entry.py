from dataclasses import replace
import copy

import numpy as np
import pytest

pytest.importorskip('gymnasium')

from gymnasium.utils.env_checker import check_env
from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.core.scenarios.stock_market import load_stock_market
from wolfgym.env.combination import CombinationGymEnv, register_combination_gym
from wolfgym.env.combination_contract import CombinationSettings
from wolfgym.evaluation.task_validation import run_validation_branch
from wolfgym.evaluation.tracking_prevention import TrackingCampaignFreeze
from wolfgym.evaluation.combination_task import score_combination


def episode(condition='attack'):
    return CombinationEpisode('engineering', 20260925, condition)


@pytest.mark.parametrize('change', [
    {'society_size': 200}, {'alpha': .01}, {'alpha': float('nan')},
    {'initial_budget': 16}, {'horizon_days': 20}, {'initial_stock_share': .4},
    {'daily_notional_multiple': 2.}, {'daily_message_cap': 2.}, {'society_size': 500.},
])
def test_primary_settings_reject_extensions(change):
    with pytest.raises(ValueError):
        CombinationSettings(**change)


@pytest.mark.parametrize('change', ['wealth', 'placement', 'market'])
def test_scenario_override_rejected_before_allocation(monkeypatch, change):
    import wolfgym.env.combination as module
    scenario = copy.deepcopy(load_stock_market('finfluencer'))
    if change == 'wealth':
        scenario = replace(scenario, harmful_wealth_multiplier=1.)
    elif change == 'placement':
        scenario.attackers['finfluencer']['placement'] = 'random'
    else:
        scenario.retail['initial_wealth'] = 20000.
    monkeypatch.setattr(module, 'WolfSocietySimulator', lambda *a, **k: pytest.fail('allocated too early'))
    with pytest.raises(ValueError, match='resources cannot'):
        CombinationGymEnv(episode(), scenario=scenario)


def test_budget_invalid_action_reset_and_public_boundary():
    env = CombinationGymEnv(episode())
    initial, info = env.reset(seed=1)
    assert env.observation_space.contains(initial)
    assert len(info['action_targets']) == 3
    assert set(info) == {'public_observation', 'action_targets', 'action_mask'}
    with pytest.raises(RuntimeError):
        env.evaluation_result()
    for value, exception in [(True, TypeError), (1.5, TypeError), (4, ValueError), (-1, ValueError)]:
        with pytest.raises(exception):
            env.step(value)
        assert env._worlds['policy']._next_day == 0
    for action in (1, 2, 3):
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert reward == 0 and not terminated and not truncated
        assert not any(env._worlds['policy'].persistent_regulatory_controls().values())
    assert list(obs['action_mask']) == [1, 0, 0, 0]
    before = env._worlds['policy'].get_state()
    with pytest.raises(ValueError, match='unaffordable'):
        env.step(1)
    assert env._worlds['policy'].get_state() == before
    with pytest.raises(ValueError, match='overrides'):
        env.reset(options={'alpha': .04})
    restored, _ = env.reset(seed=999)
    for key in initial:
        np.testing.assert_array_equal(initial[key], restored[key])


@pytest.mark.parametrize('condition', ['normal', 'attack'])
def test_full_trajectory_matches_validated_runner_and_stays_public(condition):
    training_reward = condition == 'normal'
    env = CombinationGymEnv(episode(condition), training_reward=training_reward)
    env.reset(seed=12)
    cfg = {'initial_budget': 12, 'horizon_days': 30, 'continuation_seed_offset': 1000000,
           'normal_replica_seed_offset': 2000000}
    args = dict(seed=20260925, config=cfg, pair_id=env._episode.group_id)
    reference, rr = run_validation_branch(env._template, env._initial, name='reference', **args)
    name = 'attack_public' if condition == 'attack' else 'normal_public'
    # Constant positive score asks the same public controller to act at 0,5,10.
    from wolfgym.evaluation.task_validation import public_features
    names = sorted(public_features(env._public))
    model = {'features': names, 'center': [0.] * len(names), 'scale': [1.] * len(names),
             'coefficient': [0.] * len(names), 'intercept': 30.}
    actual, ar = run_validation_branch(
        env._template, env._initial, name=name, model=model,
        frozen={'alarm_threshold': .5}, policy_factory=lambda: TrackingCampaignFreeze(.5),
        tactic_mode='campaign' if condition == 'attack' else None,
        normal_decision_replica=condition == 'normal', **args)
    controller = TrackingCampaignFreeze(.5)
    for day in range(30):
        action = controller(env._public, 1.)
        index = 0 if action.target_id is None else env._targets.index(action.target_id) + 1
        obs, reward, terminated, truncated, info = env.step(index)
        assert terminated == (day == 29) and not truncated
        assert env.observation_space.contains(obs)
        assert set(info) == {'public_observation', 'action_targets', 'action_mask'}
        if day < 29:
            assert reward == 0
    assert env._snapshots['policy'] == actual
    assert env._snapshots['reference'] == reference
    expected = score_combination(reference, actual, rr, ar, .0038562228090846347)
    result = env.evaluation_result()
    assert result['policy'] == expected
    assert reward == (result['H_reduction'] if training_reward else 0.)
    assert result['cost'] == 12.
    with pytest.raises(RuntimeError):
        env.step(0)
    # A full replay must preserve both outcomes and action mapping.
    env.reset(seed=12)
    replay = TrackingCampaignFreeze(.5)
    for _ in range(30):
        action = replay(env._public, 1.)
        env.step(0 if action.target_id is None else env._targets.index(action.target_id) + 1)
    assert env.evaluation_result() == result


def test_gym_checker_and_registration_and_heldout_reward_rejection():
    check_env(CombinationGymEnv(episode()), skip_render_check=True)
    import gymnasium as gym
    name = register_combination_gym()
    with gym.make(name, episode=episode('normal')) as env:
        env.reset(seed=123)
        assert env.step(0)[1] == 0.
    # No held-out market is constructed by this negative test.
    for split, seed in [('validation', 40000000), ('test', 50000000)]:
        with pytest.raises(ValueError, match='forbidden'):
            CombinationGymEnv(CombinationEpisode(split, seed, 'attack'), training_reward=True)
