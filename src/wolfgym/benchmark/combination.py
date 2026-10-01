"""Frozen, paired population plan. No simulation or outcome-based selection."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def benchmark_contract():
    return json.loads(Path(__file__).with_name('combination_splits_v1.json').read_text())


def split_for_seed(seed):
    if type(seed) is not int:
        raise ValueError('trajectory_seed must be an integer')
    contract = benchmark_contract()
    for split, block in contract['splits'].items():
        if block['start'] <= seed < block['start'] + block['count']:
            return split
    raise ValueError('seed is outside the frozen benchmark partitions')


@dataclass(frozen=True)
class CombinationEpisode:
    split: str
    trajectory_seed: int
    condition: str

    def __post_init__(self):
        contract = benchmark_contract()
        if self.condition not in contract['conditions']:
            raise ValueError('condition must be normal or attack')
        if self.split == 'engineering':
            if type(self.trajectory_seed) is not int or self.trajectory_seed != contract['engineering_seed']:
                raise ValueError('engineering episodes must use the retired engineering seed')
        elif split_for_seed(self.trajectory_seed) != self.split:
            raise ValueError('trajectory belongs to another split')

    @property
    def group_id(self):
        # Deliberately independent of condition, policy, checkpoint, continuation.
        raw = f"{benchmark_contract()['version']}:{self.trajectory_seed}"
        return 'group_' + hashlib.sha256(raw.encode()).hexdigest()[:24]

    def to_dict(self):
        return {'split': self.split, 'trajectory_seed': self.trajectory_seed,
                'condition': self.condition, 'group_id': self.group_id,
                'episode_id': f'{self.group_id}:{self.condition}'}

    def descendant_metadata(self, checkpoint_day, continuation_seed):
        if type(checkpoint_day) is not int or not 0 <= checkpoint_day < 30:
            raise ValueError('checkpoint must be before day 30')
        if type(continuation_seed) is not int or continuation_seed < 0:
            raise ValueError('continuation_seed must be a nonnegative integer')
        return {**self.to_dict(), 'checkpoint_day': checkpoint_day,
                'continuation_seed': continuation_seed}


def episode_plan(split):
    contract = benchmark_contract()
    if split not in contract['splits']:
        raise ValueError('unknown benchmark split')
    block = contract['splits'][split]
    episodes = [CombinationEpisode(split, seed, condition)
                for seed in range(block['start'], block['start'] + block['count'])
                for condition in contract['conditions']]
    # Fixed order independent of the normal/attack label; no scenario-prefix cap.
    return tuple(sorted(episodes, key=lambda e: hashlib.sha256(
        e.to_dict()['episode_id'].encode()).hexdigest()))


def validate_episode_manifest(rows, *, split, complete=True):
    expected = {e.to_dict()['episode_id']: e.to_dict() for e in episode_plan(split)}
    seen = set()
    for row in rows:
        episode = CombinationEpisode(row['split'], row['trajectory_seed'], row['condition'])
        identity = episode.to_dict()
        key = identity['episode_id']
        if row != identity or key not in expected or key in seen:
            raise ValueError('duplicate, foreign-split, altered or noncanonical episode')
        seen.add(key)
    if complete and seen != set(expected):
        raise ValueError('incomplete split: both conditions of every initial society are required')
    return True


def paired_protection_summary(rows, *, split):
    """One policy's complete split; cluster normal/attack within initial society."""
    from wolfgym.evaluation.closed_loop import grouped_summary

    identities = [{key: row[key] for key in ('split', 'trajectory_seed', 'condition', 'group_id', 'episode_id')}
                  for row in rows]
    validate_episode_manifest(identities, split=split)
    groups = {}
    for row in rows:
        import math
        value = float(row['H_reduction'])
        if not math.isfinite(value):
            raise ValueError('nonfinite protection benefit')
        groups.setdefault(row['group_id'], {})[row['condition']] = value
    # Never count two related conditions as two independent societies.
    values = [.5 * (pair['normal'] + pair['attack']) for pair in groups.values()]
    labels = list(groups)
    return {'balanced': grouped_summary(values, labels),
            'normal': grouped_summary([p['normal'] for p in groups.values()], labels),
            'attack': grouped_summary([p['attack'] for p in groups.values()], labels)}
