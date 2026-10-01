"""Versioned, opt-in observation optimization; old task entry remains replayable."""
from dataclasses import replace

import numpy as np

from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.task_validation import public_observation
from wolfgym.observations import ObservationConfig
from wolfgym.observations.fast_messages import FastMessageObservationBuilder


FAST_RUNTIME_VERSION = 'combination-observation-fast-v1'


class FastCombinationGymEnv(CombinationGymEnv):
    """Same worlds, actions, RNG, accounts and rewards; faster public messages.

    Explicit opt-in for future declarations. day/ids/indexed permit measuring A1,
    A1+A2 and A1+A2+A3 without changing any financial or learning parameters.
    This synchronous entry assumes frozen identities and append-only events.
    """
    def __init__(self, episode, *, message_mode='indexed', **kwargs):
        if message_mode not in ('day', 'ids', 'indexed'):
            raise ValueError('message_mode must be day, ids or indexed')
        self.message_mode = message_mode
        self._message_builder = FastMessageObservationBuilder(
            ObservationConfig(include_messages=True),
            cache_ids=message_mode != 'day', index_events=message_mode == 'indexed')
        super().__init__(episode, **kwargs)

    def reset(self, *, seed=None, options=None):
        self._message_builder.clear_cache()
        return super().reset(seed=seed, options=options)

    def _public_output(self):
        sim = self._worlds['policy']
        obs = public_observation(sim, self._remaining)
        messages = self._message_builder._build_messages(sim)
        self._public = replace(obs, messages=messages)
        encoded = self._encoder.encode(self._public)
        self._targets = tuple(a.asset_id for a in self._public.assets)
        reach = {asset: 0. for asset in self._targets}
        for message in messages:
            if message.asset_id in reach:
                reach[message.asset_id] += message.audience_reach
        encoded['message_reach'] = np.asarray([reach[a] for a in self._targets], dtype=np.float32)
        encoded['action_mask'] = self.action_masks()
        return encoded, {'public_observation': self._public.to_dict(),
                         'action_targets': self._targets, 'action_mask': self.action_masks()}

    def close(self):
        self._message_builder.clear_cache()
        super().close()
