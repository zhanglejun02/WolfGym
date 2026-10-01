"""Temporary, independently simulated R/WAIT records for offline host tests."""
import copy
from dataclasses import asdict

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv


def reference_wait_artifact(identity):
    episode = CombinationEpisode(identity['split'], identity['trajectory_seed'], identity['condition'])
    env = CombinationGymEnv(episode)
    try:
        env.reset()
        for _ in range(30):
            env.step(0)
        return {
            'accounts': {world: [asdict(row) for row in env._snapshots[world]]
                         for world in ('reference', 'wait')},
            'financial_rows': {world: copy.deepcopy(env._rows[world])
                               for world in ('reference', 'wait')},
        }
    finally:
        env.close()
