"""Trusted resumable baseline host; shared R/WAIT never enter policy inputs.

The optimized path reuses the already validated branch runner and scorer.
It is parity-tested against CombinationGymEnv; no frozen implementation changes.
"""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.combination_public import (
    BaselinePolicy, PublicHistory, public_frame, canonical_json, digest, DETECTOR_SHA256,
)
from wolfgym.evaluation.counterfactual_welfare import WelfareSnapshot, InvestorAccount
from wolfgym.evaluation.task_validation import run_validation_branch
from wolfgym.evaluation.combination_task import score_combination
from wolfgym.regulation import RegulatoryAction, RegulatoryActionType


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if path.suffix == '.gz':
        with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as stream:
            stream.write(canonical_json(value))
    else:
        temporary.write_text(canonical_json(value) + '\n')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_detector(path):
    if file_hash(path) != DETECTOR_SHA256:
        raise ValueError('frozen public detector changed')
    return read_json(path)


def decode_snapshots(rows):
    return [WelfareSnapshot(row['pair_id'], row['day'], row['fundamentals'],
                           {key: InvestorAccount(**value) for key, value in row['accounts'].items()})
            for row in rows]


class PolicyBridge:
    def __init__(self, policy, *, replay=None):
        self.policy = policy
        self.history = PublicHistory()
        self.log = []
        self.replay = replay or []

    def __call__(self, observation, unused_score=None):
        info = {'public_observation': observation.to_dict(),
                'action_targets': [a.asset_id for a in observation.assets],
                'action_mask': [1, *([int(observation.remaining_budget >= 4)] * 3)]}
        frame = public_frame(info)
        view = self.history.observe(frame)
        proposed = self.policy(view)
        valid = type(proposed) is int and 0 <= proposed < 4 and frame['action_mask'][proposed]
        action = proposed if valid else 0
        entry = {'day': frame['day'], 'frame': frame, 'input_sha256': digest(view),
                 'proposed_action': proposed, 'action': action, 'invalid_action': not bool(valid)}
        if len(self.log) < len(self.replay) and entry != self.replay[len(self.log)]:
            raise ValueError('resume/replay input or action drift')
        self.history.record(frame, action)
        self.log.append(entry)
        return (RegulatoryAction(RegulatoryActionType.FREEZE, frame['action_targets'][action-1])
                if action else RegulatoryAction(RegulatoryActionType.WAIT))


def aggregate_financial(rows):
    import math
    return {key: math.fsum(row['financial_outcomes'][key] for row in rows)
            for key in ('requested_notional', 'blocked_notional', 'unexecuted_notional', 'executed_notional')}


def run_episode(episode, specs, detector, directory, execution_id):
    """One complete society condition. Individual policies commit atomically."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'complete.json'
    if marker.exists():
        completed = read_json(marker)
        if completed['execution_id'] != execution_id or completed['episode'] != episode.to_dict():
            raise ValueError('episode completion provenance mismatch')
        if any(file_hash(directory / name) != sha for name, sha in completed['sha256'].items()):
            raise ValueError('completed artifact changed')
        return {'episode_id': episode.to_dict()['episode_id'], 'resumed': True}
    env = CombinationGymEnv(episode)
    env.reset()
    cfg = env._contract
    args = dict(seed=episode.trajectory_seed, config=cfg, pair_id=episode.group_id)
    try:
        cache_path = directory / 'reference_wait.json.gz'
        if cache_path.exists():
            cache = read_json(cache_path)
            if cache['execution_id'] != execution_id or cache['episode'] != episode.to_dict():
                raise ValueError('R/WAIT cache identity mismatch')
            reference = decode_snapshots(cache['reference'])
            wait = decode_snapshots(cache['wait'])
            rr, wr = cache['reference_rows'], cache['wait_rows']
        else:
            reference, rr = run_validation_branch(env._template, env._initial, name='reference', **args)
            wait, wr = run_validation_branch(
                env._template, env._initial,
                name='attack' if episode.condition == 'attack' else 'normal_replica',
                tactic_mode='campaign', **args)
            cache = {'execution_id': execution_id, 'episode': episode.to_dict(),
                     'reference': [asdict(x) for x in reference], 'wait': [asdict(x) for x in wait],
                     'reference_rows': rr, 'wait_rows': wr}
            write_json(cache_path, cache)
        wait_score = score_combination(reference, wait, rr, wr, cfg['harm_threshold'])
        for spec in specs:
            artifact_path = directory / (spec['id'] + '.json.gz')
            if artifact_path.exists():
                artifact = read_json(artifact_path)
                if (artifact['execution_id'] != execution_id or artifact['spec'] != spec
                        or artifact['result']['episode_id'] != episode.to_dict()['episode_id']):
                    raise ValueError('policy artifact provenance mismatch')
                continue
            bridge = PolicyBridge(BaselinePolicy(spec, detector, randomization_key=episode.group_id))
            actual, rows = run_validation_branch(
                env._template, env._initial,
                name='attack_public' if episode.condition == 'attack' else 'normal_public',
                model=detector, frozen={'alarm_threshold': 0.6739093390044616},
                policy_factory=lambda: bridge, tactic_mode='campaign',
                normal_decision_replica=episode.condition == 'normal', **args)
            score = score_combination(reference, actual, rr, rows, cfg['harm_threshold'])
            result = {**episode.to_dict(), 'policy_id': spec['id'],
                      'H_reduction': wait_score['damage']['H'][-1] - score['damage']['H'][-1],
                      'cost': sum(row['cost'] for row in rows), 'wait': wait_score, 'policy': score,
                      'invalid_actions': sum(row['invalid_action'] for row in bridge.log),
                      'financial_outcomes': {'reference': aggregate_financial(rr),
                                             'wait': aggregate_financial(wr), 'policy': aggregate_financial(rows)}}
            if spec['family'] == 'wait' and (actual != wait or result['H_reduction'] != 0):
                raise ValueError('WAIT policy does not reproduce paired control')
            write_json(artifact_path, {'execution_id': execution_id, 'spec': spec, 'result': result,
                                      'log': bridge.log, 'rows': rows,
                                      'accounts': [asdict(x) for x in actual]})
        paths = [cache_path] + [directory / (s['id'] + '.json.gz') for s in specs]
        write_json(marker, {'execution_id': execution_id, 'episode': episode.to_dict(),
                            'sha256': {path.name: file_hash(path) for path in paths}})
        return {'episode_id': episode.to_dict()['episode_id'], 'resumed': False}
    finally:
        env.close()
