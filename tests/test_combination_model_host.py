from dataclasses import asdict
import json
from pathlib import Path

import pytest

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.combination_host import load_detector, read_json
from wolfgym.evaluation.combination_model_host import run_engineering_episode
from wolfgym.evaluation.combination_requests import RequestJournal, Reply, RequestLimitError


ROOT = Path(__file__).resolve().parents[1]


def journal(path, max_attempts=30):
    return RequestJournal(path, contract={'model': 'offline-host-mock-v1'}, max_attempts=max_attempts,
                          max_cost_microusd=3000, reservation_microusd=100)


@pytest.fixture(scope='module')
def detector():
    return load_detector(ROOT/'data/task_validation_v4_20260923/detector.json')


@pytest.mark.parametrize('condition', ['normal', 'attack'])
def test_interrupted_step_replays_without_duplicate_calls_and_matches_accounts(tmp_path, detector, monkeypatch, condition):
    episode = CombinationEpisode('engineering', 20260925, condition)
    directory = tmp_path/'episode'
    calls = []
    def transport(payload):
        assert set(payload) == {'version', 'system', 'user', 'response_schema'}
        view = json.loads(payload['user'])
        assert set(view) == {'schema_version', 'window', 'history_mask', 'observations', 'interventions'}
        calls.append(payload)
        return Reply(json.dumps({'action': 'FREEZE',
                                 'target_id': view['observations'][-1]['action_targets'][0]}), 50)

    original = CombinationGymEnv.step
    def interrupted(env, action):
        result = original(env, action)
        if env._worlds['policy']._next_day == 2:
            raise KeyboardInterrupt
        return result
    monkeypatch.setattr(CombinationGymEnv, 'step', interrupted)
    with journal(tmp_path/'requests.sqlite') as j:
        with pytest.raises(KeyboardInterrupt):
            run_engineering_episode(episode, mode='LL', detector=detector, journal=j,
                                    transport=transport, directory=directory)
    assert len(calls) == 2 and not (directory/'complete.json').exists()
    monkeypatch.setattr(CombinationGymEnv, 'step', original)
    with journal(tmp_path/'requests.sqlite') as j:
        result = run_engineering_episode(episode, mode='LL', detector=detector, journal=j,
                                         transport=transport, directory=directory)
        assert len(calls) == 3 and j.totals()['attempts'] == 3
        assert result['model_decisions'] == 3 and result['cost'] == 12 and result['invalid_actions'] == 0
        assert run_engineering_episode(episode, mode='LL', detector=detector, journal=j,
                                       transport=lambda _: pytest.fail('already complete'), directory=directory) == result
    artifact = read_json(directory/'outcome.json.gz')
    assert len(artifact['decisions']) == 30
    env = CombinationGymEnv(episode)
    try:
        env.reset()
        for day in range(30):
            env.step(1 if day < 3 else 0)
        assert all(result[k] == v for k, v in env.evaluation_result().items())
        assert artifact['accounts'] == {k: [asdict(x) for x in v] for k, v in env._snapshots.items()}
    finally:
        env.close()
    # Receipt corruption is rejected instead of being counted as a cached result.
    (directory/'decisions/00.json').write_text('{}')
    with journal(tmp_path/'requests.sqlite') as j:
        with pytest.raises(ValueError, match='completed episode changed'):
            run_engineering_episode(episode, mode='LL', detector=detector, journal=j,
                                    transport=transport, directory=directory)


def test_money_or_request_limit_is_incomplete_not_wait_completion(tmp_path, detector):
    episode = CombinationEpisode('engineering', 20260925, 'normal')
    directory = tmp_path/'episode'
    with journal(tmp_path/'requests.sqlite', max_attempts=1) as j:
        with pytest.raises(RequestLimitError):
            run_engineering_episode(episode, mode='LL', detector=detector, journal=j,
                                    transport=lambda _: Reply('{"action":"WAIT"}', 20), directory=directory)
        assert j.totals()['attempts'] == 1
    assert not (directory/'complete.json').exists()
    assert read_json(directory/'progress.json')['completed_market_days'] == 1
    assert read_json(directory/'last_failure.json')['exception_type'] == 'RequestLimitError'
    # Changing to another empty journal cannot erase the accounting history.
    with journal(tmp_path/'new_requests.sqlite', max_attempts=1) as j:
        with pytest.raises(ValueError, match='configuration drift'):
            run_engineering_episode(episode, mode='LL', detector=detector, journal=j,
                                    transport=lambda _: pytest.fail('must not send'), directory=directory)


@pytest.mark.parametrize('split,seed', [('train',30000000), ('validation',40000000), ('test',50000000)])
def test_research_gate_rejects_before_world_allocation(tmp_path, detector, monkeypatch, split, seed):
    monkeypatch.setattr('wolfgym.evaluation.combination_model_host.CombinationGymEnv',
                        lambda *a, **kw: pytest.fail('must reject before allocating a world'))
    with journal(tmp_path/'requests.sqlite') as j:
        with pytest.raises(ValueError, match='joint model and experiment freeze'):
            run_engineering_episode(CombinationEpisode(split,seed,'normal'), mode='LL', detector=detector,
                                    journal=j, transport=lambda _: pytest.fail('must not call'), directory=tmp_path/'episode')
    assert not (tmp_path/'episode').exists()
