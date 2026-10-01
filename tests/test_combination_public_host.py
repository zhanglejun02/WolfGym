from dataclasses import replace
from pathlib import Path
import copy

import pytest

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.combination_public import (
    baseline_specs, BaselinePolicy, public_frame, public_target, detector_features,
    PublicHistory, digest, ALARM_THRESHOLD,
)
from wolfgym.evaluation.combination_host import (
    run_episode, load_detector, read_json, PolicyBridge, decode_snapshots,
)
from wolfgym.evaluation.task_validation import public_features, detector_score
from wolfgym.evaluation.tracking_prevention import TrackingCampaignFreeze
from wolfgym.observations.schema import PublicMessageObservation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def detector():
    return load_detector(ROOT / 'data/task_validation_v4_20260923/detector.json')


def test_view_whitelist_precision_and_zero_reach_presence():
    env = CombinationGymEnv(CombinationEpisode('engineering', 20260925, 'attack'))
    try:
        _, info = env.reset()
        info['private_seed'] = 1
        info['public_observation']['true_target'] = 'forbidden'
        info['public_observation']['market']['volume_zscore'] = 123.123456789123
        frame = public_frame(info)
        assert 'private_seed' not in frame and 'true_target' not in frame
        assert frame['market']['volume_zscore'] == 123.123456789123
        target = frame['action_targets'][0]
        info['public_observation']['messages'] = [
            {'asset_id': target, 'audience_reach': 0., 'sender_id': 'unused', 'text': 'unused'}]
        frame = public_frame(info)
        assert public_target(frame) == target
        assert frame['message_summary'][target] == {'count': 1, 'audience_reach': 0., 'present': True}
        history = PublicHistory()
        view = history.observe(frame)
        assert view['history_mask'] == [0, 0, 0, 0, 1]
        view['observations'][0]['market']['liquidity'] = -100
        history.record(frame, 1)
        frame['day'] = 1
        later = history.observe(frame)
        assert later['observations'][0]['market']['liquidity'] != -100
        assert len(later['interventions']) == 1
        with pytest.raises(ValueError, match='exactly one'):
            history.observe(frame)
    finally:
        env.close()


def test_menu_random_budget_and_pairing(detector):
    specs = baseline_specs()
    assert len(specs) == 21 and len({s['id'] for s in specs}) == 21
    for spec in (s for s in specs if s['family'] == 'random'):
        a = BaselinePolicy(spec, detector, randomization_key='paired')
        b = BaselinePolicy(spec, detector, randomization_key='paired')
        assert a.random_actions == b.random_actions
        assert len(a.random_actions) == 3
        assert all(0 <= day < 30 and 1 <= action < 4 for day, action in a.random_actions.items())


@pytest.mark.parametrize('condition', ['normal', 'attack'])
def test_cached_host_matches_native_accounts_public_rules_and_full_score(tmp_path, detector, condition):
    episode = CombinationEpisode('engineering', 20260925, condition)
    specs = [s for s in baseline_specs() if s['id'] in ('wait', 'alarm_refresh', 'date_0_slot_1')]
    directory = tmp_path / condition
    run_episode(episode, specs, detector, directory, 'test-execution')
    for spec in specs:
        artifact = read_json(directory / (spec['id'] + '.json.gz'))
        env = CombinationGymEnv(episode)
        env.reset()
        bridge = PolicyBridge(BaselinePolicy(spec, detector, randomization_key=episode.group_id),
                              replay=artifact['log'])
        old = TrackingCampaignFreeze(ALARM_THRESHOLD)
        try:
            for _ in range(30):
                typed = bridge(env._public)
                frame = bridge.log[-1]['frame']
                assert detector_features(frame) == public_features(env._public)
                if spec['id'] == 'alarm_refresh':
                    assert typed == old(env._public, detector_score(public_features(env._public), detector))
                env.step(bridge.log[-1]['action'])
            expected = env.evaluation_result()
            actual = artifact['result']
            assert all(actual[key] == value for key, value in expected.items())
            assert decode_snapshots(artifact['accounts']) == env._snapshots['policy']
        finally:
            env.close()
    assert run_episode(episode, specs, detector, directory, 'test-execution')['resumed']
    # Detect corruption instead of silently accepting a completed partial study.
    path = directory / 'wait.json.gz'
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        run_episode(episode, specs, detector, directory, 'test-execution')


def test_invalid_action_and_prefix_replay(detector):
    env = CombinationGymEnv(CombinationEpisode('engineering', 20260925, 'normal'))
    try:
        env.reset()
        bridge = PolicyBridge(lambda view: 99)
        assert bridge(env._public).target_id is None
        assert bridge.log[0]['invalid_action'] and bridge.log[0]['action'] == 0
        replay = PolicyBridge(lambda view: 99, replay=bridge.log)
        replay(env._public)
        bad = copy.deepcopy(bridge.log)
        bad[0]['input_sha256'] = 'wrong'
        with pytest.raises(ValueError, match='drift'):
            PolicyBridge(lambda view: 99, replay=bad)(env._public)
    finally:
        env.close()
