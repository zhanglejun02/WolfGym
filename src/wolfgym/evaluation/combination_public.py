"""Versioned public-only view and the predeclared combination baseline menu."""
import copy
from dataclasses import fields
import hashlib
import json
import math

import numpy as np

from wolfgym.observations.schema import (
    AssetObservation, EntityObservation, InformationLevel, MarketObservation,
    RegulatoryObservation, SocialObservation,
)
from wolfgym.observations.serialization import assert_no_hidden_fields
from wolfgym.evaluation.task_validation import public_features, detector_score


VIEW_VERSION = 'public-structured-v1'
ALARM_THRESHOLD = 0.6739093390044616
DETECTOR_SHA256 = 'c1896c6edae20201af3523e56059c64cded31a028255c481c8453ecdd2faaa90'
SCHEDULES = ((0, 5, 10), (3, 8, 13), (5, 10, 15), (8, 13, 18))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _numbers(value, schema, identity=None):
    result = {}
    for field in fields(schema):
        if field.name == identity:
            if not isinstance(value[field.name], str):
                raise ValueError('public identity must be a string')
            result[field.name] = value[field.name]
        else:
            number = float(value[field.name])
            if not math.isfinite(number):
                raise ValueError('nonfinite public number')
            result[field.name] = number
    return result


def public_frame(info):
    """Whitelist only; extra host metadata never reaches a policy."""
    obs = info['public_observation']
    targets = list(info['action_targets'])
    mask = [int(x) for x in info['action_mask']]
    assets = [_numbers(x, AssetObservation, 'asset_id') for x in obs['assets']]
    if len(targets) != 3 or len(set(targets)) != 3 or targets != [x['asset_id'] for x in assets]:
        raise ValueError('all three asset slots must match public action targets')
    budget = float(obs['remaining_budget'])
    if mask != [1, *([int(budget >= 4)] * 3)]:
        raise ValueError('public budget/mask mismatch')
    if len(obs['entities']) > 5:
        raise ValueError('unexpected entity truncation contract')
    messages = {key: {'count': 0, 'audience_reach': 0., 'present': False} for key in targets}
    for message in obs['messages']:
        if message['asset_id'] in messages:
            row = messages[message['asset_id']]
            reach = float(message['audience_reach'])
            if not math.isfinite(reach) or not 0 <= reach <= 1:
                raise ValueError('invalid public reach')
            row['count'] += 1
            # Same summation order as the frozen public controller.
            row['audience_reach'] += reach
            row['present'] = True
    frame = {
        'day': int(obs['day']), 'society_size': int(obs['society_size']),
        'information_level': int(obs['information_level']), 'remaining_budget': budget,
        'market': _numbers(obs['market'], MarketObservation),
        'social': _numbers(obs['social'], SocialObservation), 'assets': assets,
        'entities': [_numbers(x, EntityObservation, 'entity_id') for x in obs['entities']],
        'message_summary': messages, 'action_targets': targets, 'action_mask': mask,
    }
    assert_no_hidden_fields(frame)
    canonical_json(frame)
    return frame


def detector_features(frame):
    """Reuse the original detector feature function without the full JSON view."""
    observation = RegulatoryObservation(
        day=frame['day'], society_size=frame['society_size'],
        information_level=InformationLevel(frame['information_level']),
        remaining_budget=frame['remaining_budget'],
        market=MarketObservation(**frame['market']), social=SocialObservation(**frame['social']),
        assets=tuple(AssetObservation(**row) for row in frame['assets']),
        entities=tuple(EntityObservation(**row) for row in frame['entities']),
    )
    return public_features(observation)


def public_target(frame):
    reached = {key: row['audience_reach'] for key, row in frame['message_summary'].items()
               if row['present']}
    if reached:
        return max(reached, key=lambda key: (reached[key], key))
    return max(frame['assets'], key=lambda row: (
        row['return_3d'], row['volume_zscore'], row['asset_id']))['asset_id']


class PublicHistory:
    def __init__(self, window=5):
        if window not in (1, 5):
            raise ValueError('only the predeclared windows are supported')
        self.window = window
        self.frames = []
        self.actions = []

    def observe(self, frame):
        if self.frames and frame['day'] != self.frames[-1]['day'] + 1:
            raise ValueError('history must advance exactly one day')
        self.frames.append(copy.deepcopy(frame))
        selected = self.frames[-self.window:]
        return {'schema_version': VIEW_VERSION, 'window': self.window,
                'history_mask': [0] * (self.window-len(selected)) + [1] * len(selected),
                'observations': copy.deepcopy(selected), 'interventions': copy.deepcopy(self.actions)}

    def record(self, frame, action):
        if type(action) is not int or not 0 <= action < 4 or not frame['action_mask'][action]:
            raise ValueError('invalid executed action')
        if action:
            self.actions.append({'day': frame['day'], 'target_id': frame['action_targets'][action-1],
                                 'action': 'FREEZE', 'cost': 4.})


def baseline_specs():
    specs = [{'id': 'wait', 'family': 'wait'}]
    specs += [{'id': f'random_{seed}', 'family': 'random', 'rng_seed': seed} for seed in (11, 22, 33)]
    specs += [{'id': f'date_{days[0]}_slot_{slot}', 'family': 'date', 'days': list(days), 'slot': slot}
              for days in SCHEDULES for slot in range(3)]
    specs += [{'id': f'early_lock_{days[0]}', 'family': 'early_lock', 'days': list(days)}
              for days in SCHEDULES[1:]]
    specs += [{'id': 'alarm_lock', 'family': 'alarm_lock'},
              {'id': 'alarm_refresh', 'family': 'alarm_refresh'}]
    return specs


class BaselinePolicy:
    """A policy gets the same bounded public view as a future LLM adapter."""
    def __init__(self, spec, detector, *, randomization_key='engineering'):
        if spec not in baseline_specs():
            raise ValueError('unknown predeclared baseline')
        self.spec = copy.deepcopy(spec)
        self.detector = detector
        self.target = None
        self.first_alarm = None
        self.random_actions = {}
        if spec['family'] == 'random':
            # Host-owned independent policy RNG, paired across conditions.
            seed = int(digest([randomization_key, spec['rng_seed']])[:16], 16)
            rng = np.random.default_rng(seed)
            days = sorted(int(x) for x in rng.choice(30, 3, replace=False))
            self.random_actions = dict(zip(days, (int(x) for x in rng.integers(1, 4, 3))))

    def __call__(self, view):
        f = view['observations'][-1]
        day = f['day']
        family = self.spec['family']
        if family == 'wait':
            return 0
        if family == 'date' and self.target is None:
            self.target = f['action_targets'][self.spec['slot']]
        if family == 'early_lock' and day == 3:
            self.target = public_target(f)
        if family.startswith('alarm'):
            score = detector_score(detector_features(f), self.detector)
            if self.first_alarm is None and score > ALARM_THRESHOLD:
                self.first_alarm = day
                self.target = public_target(f)
            scheduled = self.first_alarm is not None and day-self.first_alarm in (0, 5, 10)
            if scheduled and family == 'alarm_refresh':
                self.target = public_target(f)
        elif family == 'random':
            return self.random_actions.get(day, 0) if f['remaining_budget'] >= 4 else 0
        else:
            scheduled = day in self.spec['days']
        if scheduled and f['remaining_budget'] >= 4:
            return f['action_targets'].index(self.target) + 1
        return 0
