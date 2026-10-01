import copy
from dataclasses import asdict

import numpy as np
import pytest

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.combination_presentation import AssetPresentation, public_target_has_tie
from wolfgym.evaluation.combination_public import PublicHistory, public_frame, public_target


@pytest.mark.parametrize('condition', ['normal','attack'])
def test_action_replay_preserves_all_world_accounts_and_scores(condition):
    episode = CombinationEpisode('engineering', 20260925, condition)
    original, changed = CombinationGymEnv(episode), CombinationGymEnv(episode)
    try:
        _, before = original.reset()
        _, after = changed.reset()
        presentation = AssetPresentation(before['action_targets'], episode.group_id)
        a, b = PublicHistory(), PublicHistory()
        for day in range(30):
            frame = public_frame(before)
            view = a.observe(frame)
            displayed = presentation.info(after)
            transformed = public_frame(displayed)
            assert transformed == presentation.frame(frame)
            assert b.observe(transformed) == presentation.view(view)
            action = {3:1, 8:2, 13:3}.get(day,0)
            shown_action = presentation.to_display_action(action, after)
            native_action = presentation.to_native_action(shown_action, after)
            assert native_action == action
            a.record(frame, action)
            b.record(transformed, shown_action)
            _, _, _, _, before = original.step(action)
            _, _, _, _, after = changed.step(native_action)
        assert original.evaluation_result() == changed.evaluation_result()
        for world in ('reference','wait','policy'):
            assert [asdict(x) for x in original._snapshots[world]] == [asdict(x) for x in changed._snapshots[world]]
    finally:
        original.close()
        changed.close()


def test_transform_is_pair_stable_and_does_not_consume_global_rng():
    np.random.seed(12)
    expected = np.random.random(5)
    np.random.seed(12)
    p = AssetPresentation(['a','b','c'], 'same-society')
    q = AssetPresentation(['a','b','c'], 'same-society')
    np.testing.assert_array_equal(np.random.random(5), expected)
    assert p.metadata() == q.metadata()
    assert set(p.aliases) == {'a','b','c'} and len(set(p.aliases.values())) == 3
    assert p.metadata() != AssetPresentation(['a','b','c'], 'another-society').metadata()


def test_unique_public_rule_is_equivariant_and_ties_are_marked():
    frame = {'action_targets':['a','b','c'], 'action_mask':[1,1,1,1],
             'assets':[{'asset_id':a,'return_3d':i,'volume_zscore':0} for i,a in enumerate('abc')],
             'message_summary':{a:{'present':False,'audience_reach':0,'count':0} for a in 'abc'}}
    p = AssetPresentation(frame['action_targets'],'society')
    assert not public_target_has_tie(frame)
    assert public_target(p.frame(frame)) == p.aliases[public_target(frame)]
    frame['message_summary']['b'] = {'present':True,'audience_reach':0,'count':1}
    assert not public_target_has_tie(frame)  # Zero-reach message differs from absence.
    assert public_target(p.frame(frame)) == p.aliases['b']
    frame['message_summary']['a'] = {'present':True,'audience_reach':0,'count':1}
    assert public_target_has_tie(frame) and public_target_has_tie(p.frame(frame))
    frame['message_summary'] = {a:{'present':False,'audience_reach':0,'count':0} for a in 'abc'}
    for row in frame['assets']:
        row['return_3d'] = 0
    assert public_target_has_tie(frame)


def test_invalid_action_masks_and_asset_drift_are_rejected():
    p = AssetPresentation(['a','b','c'],'society')
    info = {'action_targets':['a','b','c'], 'action_mask':[1,0,0,0]}
    assert p.to_native_action(0,info) == p.to_display_action(0,info) == 0
    for action in (True,1,4,-1):
        with pytest.raises(ValueError):
            p.to_native_action(action,info)
    info['action_targets'] = ['a','b','foreign']
    with pytest.raises(ValueError,match='asset set'):
        p.to_display_action(0,info)
