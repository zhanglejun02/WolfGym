from dataclasses import asdict
import json
from pathlib import Path

import pytest

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.combination_host import load_detector, read_json
from wolfgym.evaluation.combination_presented_host import run_presented_engineering_episode
from wolfgym.evaluation.combination_presentation import AssetPresentation
from wolfgym.evaluation.combination_requests import RequestJournal, Reply, RequestLimitError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def detector():
    return load_detector(ROOT/'data/task_validation_v4_20260923/detector.json')


def journal(path, attempts=30):
    return RequestJournal(path, contract={'model':'presentation-offline-mock-v1'},
                          max_attempts=attempts, max_cost_microusd=3000, reservation_microusd=100)


@pytest.mark.parametrize('condition', ['normal','attack'])
def test_presented_requests_resume_and_match_physical_world(tmp_path, detector, monkeypatch, condition):
    episode = CombinationEpisode('engineering', 20260925, condition)
    env = CombinationGymEnv(episode)
    try:
        _, info = env.reset()
        mapping = AssetPresentation(info['action_targets'], episode.group_id)
        native_targets = info['action_targets'][:]
        target = mapping.aliases[native_targets[0]]
        for day in range(30):
            env.step(1 if day in (0,5,10) else 0)
        expected = env.evaluation_result()
        expected_accounts = {k: [asdict(x) for x in v] for k,v in env._snapshots.items()}
        expected_financial = env._rows
    finally:
        env.close()
    calls = []
    def transport(payload):
        assert set(payload) == {'version','system','user','response_schema'}
        assert not any(native in json.dumps(payload) for native in native_targets)
        view = json.loads(payload['user'])
        assert set(view) == {'schema_version','window','history_mask','observations','interventions'}
        frame = view['observations'][-1]
        day = frame['day']
        calls.append(day)
        for f in view['observations']:
            assert set(f['action_targets']) == set(mapping.aliases.values())
        assert all(r['target_id'] == target for r in view['interventions'])
        if day == 10:
            assert len(view['observations']) == 5 and len(view['interventions']) == 2
        response = {'action':'FREEZE','target_id':target} if day in (0,5,10) else {'action':'WAIT'}
        return Reply(json.dumps(response), 50)

    original = CombinationGymEnv.step
    def interrupted(env, action):
        result = original(env, action)
        if env._worlds['policy']._next_day == 6:
            raise KeyboardInterrupt
        return result
    monkeypatch.setattr(CombinationGymEnv, 'step', interrupted)
    directory = tmp_path/'episode'
    with journal(tmp_path/'journal.sqlite') as j:
        with pytest.raises(KeyboardInterrupt):
            run_presented_engineering_episode(episode, detector=detector, journal=j,
                                              transport=transport, directory=directory)
    assert calls == list(range(6)) and not (directory/'complete.json').exists()
    monkeypatch.setattr(CombinationGymEnv, 'step', original)
    with journal(tmp_path/'journal.sqlite') as j:
        result = run_presented_engineering_episode(episode, detector=detector, journal=j,
                                                  transport=transport, directory=directory)
        assert calls == list(range(11)) and j.totals()['attempts'] == 11
        assert result['model_decisions'] == 11 and result['invalid_actions'] == 0
        assert all(result[k] == v for k,v in expected.items())
        assert run_presented_engineering_episode(episode, detector=detector, journal=j,
            transport=lambda _: pytest.fail('already complete'), directory=directory) == result
    artifact = read_json(directory/'outcome.json.gz')
    assert artifact['accounts'] == expected_accounts
    assert artifact['financial_rows'] == expected_financial
    assert artifact['presentation'] == mapping.metadata()
    assert [row['native_action'] for row in artifact['decisions']] == [1 if d in (0,5,10) else 0 for d in range(30)]
    assert all(type(row['target_rule_tie']) is bool for row in artifact['decisions'])
    # A receipt with one missing daily record must not be accepted as complete.
    receipt = read_json(directory/'complete.json')
    del receipt['sha256']['decisions/29.json']
    (directory/'complete.json').write_text(json.dumps(receipt))
    with journal(tmp_path/'journal.sqlite') as j:
        with pytest.raises(ValueError, match='completion receipt'):
            run_presented_engineering_episode(episode, detector=detector, journal=j,
                transport=lambda _: pytest.fail('must not call'), directory=directory)


@pytest.mark.parametrize('split,seed', [('train',30000000),('validation',40000000),('test',50000000)])
def test_research_stays_closed_before_world_allocation(tmp_path, detector, monkeypatch, split, seed):
    monkeypatch.setattr('wolfgym.evaluation.combination_presented_host.CombinationGymEnv',
                        lambda *a, **k: pytest.fail('must not allocate'))
    with journal(tmp_path/'journal.sqlite') as j:
        with pytest.raises(ValueError, match='joint model and experiment freeze'):
            run_presented_engineering_episode(CombinationEpisode(split,seed,'normal'),
                detector=detector, journal=j, transport=lambda _: pytest.fail('must not call'), directory=tmp_path/'episode')
    assert not (tmp_path/'episode').exists()


def test_spend_stop_is_incomplete_and_mapping_or_ledger_drift_rejected(tmp_path, detector):
    episode = CombinationEpisode('engineering',20260925,'normal')
    directory = tmp_path/'episode'
    with journal(tmp_path/'journal.sqlite', attempts=1) as j:
        with pytest.raises(RequestLimitError):
            run_presented_engineering_episode(episode,detector=detector,journal=j,
                transport=lambda _: Reply('{"action":"FREEZE","target_id":"unavailable-native-id"}',50), directory=directory)
        assert j.totals()['attempts'] == 1
    row = read_json(directory/'decisions/00.json')
    assert row['invalid_action'] and row['action'] == row['native_action'] == 0
    assert not (directory/'complete.json').exists()
    assert read_json(directory/'progress.json')['completed_market_days'] == 1
    metadata = read_json(directory/'presentation.json')
    metadata['display_order'].reverse()
    (directory/'presentation.json').write_text(json.dumps(metadata))
    with journal(tmp_path/'journal.sqlite', attempts=1) as j:
        with pytest.raises(ValueError, match='mapping drift'):
            run_presented_engineering_episode(episode,detector=detector,journal=j,
                transport=lambda _: pytest.fail('must not call'), directory=directory)
    with journal(tmp_path/'new-journal.sqlite', attempts=1) as j:
        with pytest.raises(ValueError, match='configuration drift'):
            run_presented_engineering_episode(episode,detector=detector,journal=j,
                transport=lambda _: pytest.fail('must not call'), directory=directory)
