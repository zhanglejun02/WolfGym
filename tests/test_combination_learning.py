from pathlib import Path
import copy
import json

import numpy as np
import pytest

pytest.importorskip('gymnasium')

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination_learning import (
    CombinationSampleEnv, encode_public_view, VECTOR_WIDTH, FRAME_WIDTH, ID_BYTES,
)


def engineering():
    return [CombinationEpisode('engineering', 20260925, c).to_dict() for c in ('normal', 'attack')]


def test_sampler_changes_world_and_keeps_signed_terminal_reward(tmp_path):
    logfile = tmp_path / 'terminal.jsonl'
    env = CombinationSampleEnv(engineering(), terminal_log=logfile, audit_directory=tmp_path/'raw')
    try:
        for index in range(2):
            obs, info = env.reset(seed=11)
            assert env.env._episode.condition == ('normal', 'attack')[index]
            assert obs.shape == (VECTOR_WIDTH,) and env.observation_space.contains(obs)
            assert info == {} and env.current_view['interventions'] == []
            with pytest.raises(RuntimeError, match='unfinished'):
                env.reset(seed=999)
            before = env.env._worlds['policy']._next_day
            with pytest.raises(TypeError):
                env.step(True)
            assert env.env._worlds['policy']._next_day == before
            for day in range(30):
                action = 1 if day < 3 else 0
                obs, reward, done, truncated, info = env.step(action)
                assert env.observation_space.contains(obs) and not truncated
                assert done == (day == 29)
                if not done:
                    assert reward == 0 and info == {}
                else:
                    assert reward == 1000*env.env.evaluation_result()['H_reduction']
                    assert set(info) == {'episode'}
                if day == 2:
                    np.testing.assert_array_equal(env.action_masks(), [1, 0, 0, 0])
                    with pytest.raises(ValueError, match='masked'):
                        env.step(1)
            with pytest.raises(RuntimeError, match='terminal'):
                env.step(0)
        obs, info = env.reset()
        assert env.exhausted and not obs.any() and info == {}
        with pytest.raises(RuntimeError, match='exhausted'):
            env.step(0)
        rows = [json.loads(x) for x in logfile.read_text().splitlines()]
        assert len(rows) == 2 and all(len(x['actions']) == 30 for x in rows)
        assert {x['result']['condition'] for x in rows} == {'normal', 'attack'}
        from wolfgym.evaluation.combination_host import read_json
        from wolfgym.evaluation.combination_public import PublicHistory, digest
        import hashlib
        for path in (tmp_path/'raw').glob('*.json.gz'):
            saved = read_json(path)
            assert len(saved['policy_accounts']) == len(saved['public_log']) == 30
            history = PublicHistory()
            for row in saved['public_log']:
                view = history.observe(row['frame'])
                assert digest(view) == row['input_sha256']
                assert hashlib.sha256(encode_public_view(view).tobytes()).hexdigest() == row['vector_sha256']
                history.record(row['frame'], row['action'])
    finally:
        env.close()


def test_validation_training_rejected_without_allocation():
    heldout = [CombinationEpisode('validation', 40000000, 'normal').to_dict()]
    with pytest.raises(ValueError, match='held-out'):
        CombinationSampleEnv(heldout, training=True)
    with pytest.raises(ValueError, match='repeat'):
        CombinationSampleEnv(engineering()*2)


def test_vector_entity_alignment_identity_roundtrip_and_no_clipping():
    env = CombinationSampleEnv(engineering())
    try:
        env.reset()
        view = copy.deepcopy(env.current_view)
        entity = dict(entity_id='public_abcdef', message_reach=50., trading_concentration=.1,
                      cancellation_rate=.2, social_influence=.3, activity_change=.4)
        view['observations'][0]['entities'] = [entity]
        second = copy.deepcopy(view['observations'][0])
        second['day'] = 1
        second['entities'] = [dict(entity, entity_id='public_a'), entity]
        view['observations'].append(second)
        view['history_mask'] = [0, 0, 0, 1, 1]
        encoded = encode_public_view(view)
        matrix = encoded[:FRAME_WIDTH*5].reshape(5, FRAME_WIDTH)
        # First entity block begins after metadata, market/social and assets.
        assert matrix[3, 45] == 0 and matrix[3, 51] == 1
        assert matrix[4, 45] == 1 and matrix[4, 51] == 1
        assert np.isclose(np.expm1(matrix[3, 52]), 50.)
        ids = encoded[FRAME_WIDTH*5+5:FRAME_WIDTH*5+5+28*ID_BYTES].reshape(28, ID_BYTES)
        decoded = bytes(np.rint(ids[4]*255).astype(np.uint8)).rstrip(b'\0').decode()
        assert decoded == 'public_abcdef'
        # Field magnitudes above the old +/-10 clipping remain distinguishable.
        different = copy.deepcopy(view)
        different['observations'][-1]['entities'][1]['message_reach'] = 100.
        assert not np.array_equal(encoded, encode_public_view(different))
    finally:
        env.close()
