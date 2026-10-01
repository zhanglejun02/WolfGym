"""Drop-in opt-in sampler; same episode order, encoding, reward and terminal log."""
import gymnasium as gym
import numpy as np

from wolfgym.env.combination_fast import FastCombinationGymEnv
from wolfgym.env.combination_learning import CombinationSampleEnv, VECTOR_WIDTH, encode_public_view
from wolfgym.evaluation.combination_public import PublicHistory, public_frame


class FastCombinationSampleEnv(CombinationSampleEnv):
    def __init__(self, episodes, *, message_mode='indexed', **kwargs):
        if message_mode not in ('day', 'ids', 'indexed'):
            raise ValueError('message_mode must be day, ids or indexed')
        self.message_mode = message_mode
        super().__init__(episodes, **kwargs)

    def reset(self, *, seed=None, options=None):
        # Preserve the frozen sampler reset contract while selecting a new env
        # factory locally. No process-global monkey patch or old source edit.
        gym.Env.reset(self, seed=seed)
        if options:
            raise ValueError('sampling is controlled by the frozen manifest')
        if self.env is not None:
            if not self.env._terminated:
                raise RuntimeError('cannot discard an unfinished training episode')
            self.env.close()
            self.env = None
        if self.cursor == len(self.episodes):
            self.exhausted = True
            return np.zeros(VECTOR_WIDTH, dtype=np.float32), {}
        self.env = FastCombinationGymEnv(self.episodes[self.cursor],
            training_reward=self.training, message_mode=self.message_mode)
        self.cursor += 1
        _, info = self.env.reset()
        self.history = PublicHistory()
        self.frame = public_frame(info)
        self.current_view = self.history.observe(self.frame)
        self.actions = []
        self.public_log = []
        return encode_public_view(self.current_view), {}
