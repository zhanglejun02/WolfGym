"""Train-list episode sampling and an explicit encoding of the shared public view.

This is an adapter outside the frozen market contract. Its own version and hash
belong to the learning run, not to historical market acceptance.
"""
from dataclasses import fields, asdict
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.combination_public import PublicHistory, public_frame, digest
from wolfgym.evaluation.combination_host import write_json
from wolfgym.observations.schema import MarketObservation, SocialObservation, AssetObservation, EntityObservation


ENCODING_VERSION = 'public-structured-learning-v1'
ID_BYTES = 32
MAX_ENTITIES = 25  # At most five distinct public entities in each of five frames.
FRAME_WIDTH = 195
VECTOR_WIDTH = 1895


def _numeric(values):
    # Fixed invertible compression, not a fit on validation/test observations.
    return [math.copysign(math.log1p(abs(float(x))), float(x)) for x in values]


def _identity(value):
    data = value.encode('utf-8')
    if not data or len(data) > ID_BYTES or 0 in data:
        raise ValueError('public ID does not fit the declared byte encoding')
    return list(data) + [0] * (ID_BYTES-len(data))


def encode_public_view(view):
    """Preserve all allowed numeric fields and opaque identities; no raw text.

    Entity slots align by public identity across this window only. Exact UTF-8
    bytes retain the IDs rather than giving LLMs additional identity information.
    Float32 conversion is an explicit learning-algorithm implementation choice.
    """
    observations = view['observations']
    if view['window'] not in (1, 5) or not 1 <= len(observations) <= view['window']:
        raise ValueError('invalid history window')
    assets = observations[-1]['action_targets']
    entities = sorted({row['entity_id'] for f in observations for row in f['entities']})
    if len(entities) > MAX_ENTITIES:
        raise ValueError('too many public identities in this window')
    matrix = np.zeros((5, FRAME_WIDTH), dtype=np.float32)
    for i, f in enumerate(observations, 5-len(observations)):
        if set(f['action_targets']) != set(assets):
            raise ValueError('asset universe changed within the window')
        values = [f['day']/30., f['society_size']/500., f['information_level']/3.,
                  f['remaining_budget']/12.]
        values += _numeric(f['market'][x.name] for x in fields(MarketObservation))
        values += _numeric(f['social'][x.name] for x in fields(SocialObservation))
        values += [f['action_mask'][0]] + [f['action_mask'][f['action_targets'].index(a)+1] for a in assets]
        by_asset = {row['asset_id']: row for row in f['assets']}
        for a in assets:
            values += _numeric(by_asset[a][x.name] for x in fields(AssetObservation) if x.name != 'asset_id')
            msg = f['message_summary'][a]
            values += _numeric((msg['count'], msg['audience_reach'])) + [float(msg['present'])]
        by_entity = {row['entity_id']: row for row in f['entities']}
        for identity in entities:
            row = by_entity.get(identity)
            values += ([1.] + _numeric(row[x.name] for x in fields(EntityObservation) if x.name != 'entity_id')
                       if row else [0.] * 6)
        values += [0.] * ((MAX_ENTITIES-len(entities))*6)
        if len(values) != FRAME_WIDTH:
            raise ValueError('learning frame shape drift')
        matrix[i] = values
    ids = np.zeros((3+MAX_ENTITIES, ID_BYTES), dtype=np.float32)
    for i, identity in enumerate(assets + entities):
        ids[i] = np.asarray(_identity(identity), dtype=np.float32) / 255.
    interventions = view['interventions']
    if len(interventions) > 3:
        raise ValueError('intervention count violates primary budget')
    action_rows = np.zeros((3, 6), dtype=np.float32)
    for i, action in enumerate(interventions):
        action_rows[i, 0] = 1.
        action_rows[i, 1] = action['day']/30.
        action_rows[i, 2+assets.index(action['target_id'])] = 1.
        action_rows[i, 5] = action['cost']/4.
    mask = [0.]*(5-len(observations)) + [1.]*len(observations)
    vector = np.concatenate((matrix.ravel(), np.asarray(mask, dtype=np.float32), ids.ravel(),
                             action_rows.ravel(), np.asarray([view['window']/5.], dtype=np.float32)))
    if vector.shape != (VECTOR_WIDTH,) or not np.isfinite(vector).all():
        raise ValueError('invalid learning vector')
    return vector


class CombinationSampleEnv(gym.Env):
    """Consume each assigned manifest episode once, with a private terminal log."""
    metadata = {'render_modes': []}

    def __init__(self, episodes, *, training=True, reward_scale=1000., terminal_log=None, audit_directory=None):
        if not episodes:
            raise ValueError('nonempty episode plan required')
        self.episodes = [CombinationEpisode(e['split'], e['trajectory_seed'], e['condition']) for e in episodes]
        if len({e.to_dict()['episode_id'] for e in self.episodes}) != len(self.episodes):
            raise ValueError('sampler must not silently repeat an episode')
        if training and any(e.split not in ('train', 'engineering') for e in self.episodes):
            raise ValueError('held-out training is forbidden')
        self.training = training
        self.reward_scale = float(reward_scale)
        if self.reward_scale != 1000.:
            raise ValueError('frozen learning reward scale is 1000')
        self.terminal_log = Path(terminal_log) if terminal_log else None
        self.audit_directory = Path(audit_directory) if audit_directory else None
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (VECTOR_WIDTH,), dtype=np.float32)
        self.cursor = 0
        self.env = None
        self.exhausted = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('sampling is controlled by the frozen manifest')
        if self.env is not None:
            if not self.env._terminated:
                raise RuntimeError('cannot discard an unfinished training episode')
            self.env.close()
            self.env = None
        if self.cursor == len(self.episodes):
            # Vector environments auto-reset after the last terminal transition.
            # Return an inert sentinel; further stepping is an error, not replay.
            self.exhausted = True
            return np.zeros(VECTOR_WIDTH, dtype=np.float32), {}
        self.env = CombinationGymEnv(self.episodes[self.cursor], training_reward=self.training)
        self.cursor += 1
        _, info = self.env.reset()
        self.history = PublicHistory()
        self.frame = public_frame(info)
        self.current_view = self.history.observe(self.frame)
        self.actions = []
        self.public_log = []
        return encode_public_view(self.current_view), {}

    def action_masks(self):
        if self.env is None and not self.exhausted:
            raise RuntimeError('reset before requesting action masks')
        return np.array([True, False, False, False] if self.exhausted else self.frame['action_mask'], dtype=bool)

    def step(self, action):
        if self.env is None or self.exhausted:
            raise RuntimeError('episode sampler exhausted or not reset')
        if self.env._terminated:
            raise RuntimeError('reset after the terminal transition')
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError('action must be an integer')
        action = int(action)
        if not 0 <= action < 4 or not self.action_masks()[action]:
            raise ValueError('masked action is not allowed')
        if self.audit_directory:
            self.public_log.append({'frame': self.frame, 'action': action,
                                   'input_sha256': digest(self.current_view),
                                   'vector_sha256': hashlib.sha256(encode_public_view(self.current_view).tobytes()).hexdigest()})
        self.history.record(self.frame, action)
        self.actions.append(action)
        _, reward, terminated, truncated, info = self.env.step(action)
        self.frame = public_frame(info)
        self.current_view = self.history.observe(self.frame)
        output = {}
        if terminated:
            result = self.env.evaluation_result()
            output['episode'] = {'r': float(reward*self.reward_scale), 'l': 30}
            if self.audit_directory:
                path = self.audit_directory / f"{result['trajectory_seed']}_{result['condition']}.json.gz"
                write_json(path, {'result': result, 'actions': self.actions, 'public_log': self.public_log,
                                  'policy_accounts': [asdict(x) for x in self.env._snapshots['policy']],
                                  'policy_rows': self.env._rows['policy']})
            if self.terminal_log:
                self.terminal_log.parent.mkdir(parents=True, exist_ok=True)
                with self.terminal_log.open('a') as stream:
                    stream.write(json.dumps({'result': result, 'actions': self.actions},
                                            sort_keys=True, allow_nan=False)+'\n')
        return encode_public_view(self.current_view), float(reward*self.reward_scale), terminated, truncated, output

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
