from types import SimpleNamespace
import pickle

import pytest

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.env.combination_fast import FastCombinationGymEnv
from wolfgym.observations import ObservationBuilder, ObservationConfig
from wolfgym.observations.fast_messages import FastMessageObservationBuilder


class FakeWorld:
    def __init__(self):
        self._next_day=3;self.n_society=500;self.snapshots=0
        self._public_entity_ids={'retail_1':'entity_public'};self._public_asset_ids={'asset_1':'asset_public'}
        self.graph=SimpleNamespace(id_to_node={'retail_1':0});self.market=SimpleNamespace(assets={'asset_1':object()})
        self.social=SimpleNamespace(message_events=[])
    def public_entity_id(self,key):return self._public_entity_ids[key]
    def public_asset_id(self,key):return self._public_asset_ids[key]
    def get_state(self):self.snapshots+=1;return SimpleNamespace(next_day=self._next_day)


def event(day,text='retail_1 asset_1',message_id='tie'):
    return {'day':day,'text':text,'sender_id':'retail_1','asset':'asset_1','message_id':message_id,'audience_size':25}


def builders():
    config=ObservationConfig(include_messages=True,max_messages=20)
    return ObservationBuilder(config),FastMessageObservationBuilder(config)


def test_fast_day_does_not_snapshot_and_preserves_stable_message_ties():
    world=FakeWorld();old,fast=builders()
    world.social.message_events=[event(2,'first'),event(-9,'old'),event(9,'future'),event(2,'second'),event(2)]
    expected=old._build_messages(world);assert world.snapshots==1
    assert fast._build_messages(world)==expected and world.snapshots==1
    assert [m.text for m in expected if m.day==2]==['first','second','entity_public asset_public']
    world.social.message_events.append(event(2,'appended'))
    assert fast._build_messages(world)==old._build_messages(world)


def test_cache_replacement_truncation_rewind_and_identity_restore():
    world=FakeWorld();old,fast=builders();world.social.message_events=[event(i) for i in range(3)]
    assert fast._build_messages(world)==old._build_messages(world)
    world._next_day=8;world.social.message_events.append(event(7))
    assert fast._build_messages(world)==old._build_messages(world)
    world._next_day=2;world.social.message_events=world.social.message_events[:2]
    world._public_entity_ids={'retail_1':'entity_restored'}
    assert fast._build_messages(world)==old._build_messages(world)
    world.social.message_events[:]=[event(1,'replacement')]
    assert fast._build_messages(world)==old._build_messages(world)
    world.social.message_events.clear()
    assert fast._build_messages(world)==old._build_messages(world)==()


def test_cache_is_world_local_and_past_text_edits_are_visible():
    first=FakeWorld();second=FakeWorld();old,fast=builders()
    first.social.message_events=[event(1)];second.social.message_events=[event(1,'other')]
    assert fast._build_messages(first)==old._build_messages(first)
    assert fast._build_messages(second)==old._build_messages(second)
    second.social.message_events[0]['text']='changed'
    assert fast._build_messages(second)==old._build_messages(second)


@pytest.mark.parametrize('condition',['normal','attack'])
def test_full30day_outputs_accounts_rewards_and_reset_are_bitwise_equal(condition):
    episode=CombinationEpisode('engineering',20260925,condition)
    old=CombinationGymEnv(episode,training_reward=True)
    fast=FastCombinationGymEnv(episode,training_reward=True)
    try:
        original=old.reset();optimized=fast.reset()
        assert pickle.dumps(original,protocol=5)==pickle.dumps(optimized,protocol=5)
        for day in range(30):
            action={2:1,17:2,29:3}.get(day,0)
            assert pickle.dumps(old.step(action),protocol=5)==pickle.dumps(fast.step(action),protocol=5)
            assert old._snapshots==fast._snapshots and old._rows==fast._rows
        assert old.evaluation_result()==fast.evaluation_result()
        assert pickle.dumps(old.reset(),protocol=5)==pickle.dumps(fast.reset(),protocol=5)
        fast.close()
        assert fast._message_builder._events is None
        assert pickle.dumps(old.reset(),protocol=5)==pickle.dumps(fast.reset(),protocol=5)
    finally:old.close();fast.close()


def test_invalid_mode_and_resource_override_fail_closed():
    e=CombinationEpisode('engineering',20260925,'attack')
    with pytest.raises(ValueError,match='message_mode'):FastCombinationGymEnv(e,message_mode='invalid')
    with pytest.raises(TypeError):FastCombinationGymEnv(e,settings={'society_size':50})


def test_training_sampler_keeps_vectors_rewards_episode_order_and_logs(tmp_path):
    from wolfgym.env.combination_learning import CombinationSampleEnv
    from wolfgym.env.combination_learning_fast import FastCombinationSampleEnv
    from wolfgym.evaluation.combination_host import read_json
    episodes=[CombinationEpisode('engineering',20260925,c).to_dict() for c in ('normal','attack')]
    old=CombinationSampleEnv(episodes,terminal_log=tmp_path/'old.jsonl',audit_directory=tmp_path/'old')
    fast=FastCombinationSampleEnv(episodes,terminal_log=tmp_path/'fast.jsonl',audit_directory=tmp_path/'fast')
    try:
        for e in episodes:
            assert pickle.dumps(old.reset(seed=7),protocol=5)==pickle.dumps(fast.reset(seed=7),protocol=5)
            with pytest.raises(RuntimeError,match='unfinished'):fast.reset()
            for day in range(30):
                assert old.action_masks().tobytes()==fast.action_masks().tobytes()
                action={2:1,17:2,29:3}.get(day,0)
                assert pickle.dumps(old.step(action),protocol=5)==pickle.dumps(fast.step(action),protocol=5)
            name=f"{e['trajectory_seed']}_{e['condition']}.json.gz"
            assert read_json(tmp_path/'old'/name)==read_json(tmp_path/'fast'/name)
        assert (tmp_path/'old.jsonl').read_bytes()==(tmp_path/'fast.jsonl').read_bytes()
        assert pickle.dumps(old.reset(),protocol=5)==pickle.dumps(fast.reset(),protocol=5)
        assert old.exhausted and fast.exhausted
        with pytest.raises(RuntimeError,match='exhausted'):fast.step(0)
    finally:old.close();fast.close()
