import copy
import json

import pytest

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.combination_decision import CombinationDecisionAdapter, parse_response
from wolfgym.evaluation.combination_host import load_detector
from wolfgym.evaluation.combination_public import BaselinePolicy, PublicHistory, public_frame, public_target

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def detector():
    return load_detector(ROOT/'data/task_validation_v4_20260923/detector.json')


@pytest.fixture(scope='module')
def initial_info():
    env = CombinationGymEnv(CombinationEpisode('engineering', 20260925, 'attack'))
    try:
        return env.reset()[1]
    finally:
        env.close()


def day_info(initial, day, budget=12):
    info = copy.deepcopy(initial)
    info['public_observation']['day'] = day
    info['public_observation']['remaining_budget'] = budget
    info['action_mask'] = [1] + [int(budget >= 4)]*3
    return info


def test_rr_matches_frozen_rule_on_actual_policy_trajectory(detector):
    env = CombinationGymEnv(CombinationEpisode('engineering', 20260925, 'attack'))
    adapter = CombinationDecisionAdapter('RR', detector)
    rule = BaselinePolicy({'id': 'alarm_refresh', 'family': 'alarm_refresh'}, detector)
    history = PublicHistory()
    try:
        _, info = env.reset()
        for day in range(30):
            frame = public_frame(info)
            expected = rule(history.observe(frame))
            action = adapter(info)
            assert action == expected
            history.record(frame, action)
            _, reward, done, truncated, info = env.step(action)
            assert reward == 0 and not truncated and done == (day == 29)
        assert all(row['request'] is None for row in adapter.log)
    finally:
        env.close()


def test_rl_asks_only_on_rule_days_and_preserves_target_responsibility(initial_info, detector, monkeypatch):
    monkeypatch.setattr('wolfgym.evaluation.combination_public.detector_score', lambda *args: 1.)
    calls = []
    def request(payload):
        calls.append(payload)
        view = json.loads(payload['user'])
        return json.dumps({'target_id': view['observations'][-1]['action_targets'][1]})
    adapter = CombinationDecisionAdapter('RL', detector, request)
    budget = 12
    for day in range(12):
        action = adapter(day_info(initial_info, day, budget))
        assert action == (2 if day in (0, 5, 10) else 0)
        budget -= 4*bool(action)
    assert len(calls) == 3
    assert len(adapter.history.actions) == 3


def test_lr_receives_no_target_recommendation_and_rule_uses_current_frame(initial_info, detector):
    calls = []
    def request(payload):
        calls.append(payload)
        return '{"action":"FREEZE"}'
    adapter = CombinationDecisionAdapter('LR', detector, request)
    for day in range(4):
        info = day_info(initial_info, day, max(0, 12-4*day))
        target = info['action_targets'][day % 3]
        info['public_observation']['messages'] = [{'asset_id': target, 'audience_reach': 0.}]
        action = adapter(info)
        assert action == (day % 3 + 1 if day < 3 else 0)
    assert len(calls) == 3
    assert all('target_id' not in p['response_schema']['properties'] for p in calls)
    assert set(json.loads(calls[0]['user'])) == {'schema_version', 'window', 'history_mask', 'observations', 'interventions'}


@pytest.mark.parametrize('raw', [
    '```json\n{"action":"WAIT"}\n```',
    '{"action":"WAIT","reason":"extra"}',
    '{"action":"WAIT","action":"FREEZE"}',
    '{"action":"FREEZE","target_id":"hidden_asset"}',
    '{"action":true}',
    '[]',
])
def test_invalid_model_answers_become_logged_waits(raw, initial_info, detector):
    adapter = CombinationDecisionAdapter('LL', detector, lambda _: raw)
    assert adapter(initial_info) == 0
    assert adapter.log[0]['invalid_action'] and adapter.log[0]['cost'] == 0
    assert adapter.history.actions == []


def test_rl_has_no_wait_option_and_mask_is_enforced(initial_info):
    with pytest.raises(ValueError):
        parse_response('{"action":"WAIT"}', 'RL', public_frame(initial_info))
    frame = public_frame(day_info(initial_info, 0, 0))
    with pytest.raises(ValueError, match='unaffordable'):
        parse_response(json.dumps({'target_id': frame['action_targets'][0]}), 'RL', frame)


def test_history_ablation_whitelist_and_episode_isolation(initial_info, detector):
    def response(payload):
        frame = json.loads(payload['user'])['observations'][-1]
        return (json.dumps({'action': 'FREEZE', 'target_id': frame['action_targets'][0]})
                if frame['day'] == 0 else '{"action":"WAIT"}')
    full = CombinationDecisionAdapter('LL', detector, response, window=5)
    current = CombinationDecisionAdapter('LL', detector, response, window=1)
    for day in range(6):
        info = day_info(initial_info, day, 12 if day == 0 else 8)
        info['true_target'] = 'PRIVATE_SENTINEL'
        info['public_observation']['harm'] = 'PRIVATE_SENTINEL'
        for adapter in (full, current):
            assert adapter(info) == (1 if day == 0 else 0)
            assert 'PRIVATE_SENTINEL' not in json.dumps(adapter.log[-1]['request'])
    a, b = [json.loads(x.log[-1]['request']['user']) for x in (full, current)]
    assert len(a['observations']) == 5 and len(b['observations']) == 1
    assert a['observations'][-1] == b['observations'][0]
    assert a['interventions'] == b['interventions'] and len(a['interventions']) == 1
    with pytest.raises(ValueError, match='fresh episode'):
        full(initial_info)
    fresh = CombinationDecisionAdapter('LL', detector, lambda _: '{"action":"WAIT"}')
    fresh(initial_info)
    assert len(json.loads(fresh.log[0]['request']['user'])['observations']) == 1


def test_transport_failure_does_not_consume_history_or_rule_state(initial_info, detector, monkeypatch):
    monkeypatch.setattr('wolfgym.evaluation.combination_public.detector_score', lambda *args: 1.)
    def failed(_):
        raise ConnectionError('offline')
    adapter = CombinationDecisionAdapter('RL', detector, failed)
    with pytest.raises(ConnectionError):
        adapter(initial_info)
    assert adapter.log == [] and adapter.history.frames == [] and adapter.rule.first_alarm is None
    adapter.request = lambda _: json.dumps({'target_id': initial_info['action_targets'][0]})
    assert adapter(initial_info) == 1
    assert adapter.rule.first_alarm == 0
