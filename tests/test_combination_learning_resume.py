"""Checkpoint/RNG continuation test on a cheap masked finite-horizon world."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
gym = pytest.importorskip('gymnasium')
torch = pytest.importorskip('torch')
pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

spec = importlib.util.spec_from_file_location(
    'learning_runner', Path(__file__).resolve().parents[1]/'scripts/train_combination_policy.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Toy(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = self.spent = 0
        return np.zeros(2, dtype=np.float32), {}

    def action_masks(self):
        return np.array([1, self.spent < 3, self.spent < 3, self.spent < 3], dtype=bool)

    def step(self, action):
        assert self.action_masks()[action]
        self.spent += int(action != 0)
        self.day += 1
        return np.array([self.day/30, self.spent/3], dtype=np.float32), float(action == 2), self.day == 30, False, {}


def test_checkpoint_rng_replay_preserves_next_update(tmp_path):
    torch.set_num_threads(1)
    vec = DummyVecEnv([Toy, Toy])
    model = MaskablePPO('MlpPolicy', vec, n_steps=30, batch_size=60, n_epochs=2,
                        gamma=1., gae_lambda=1., seed=11, device='cpu', policy_kwargs={'net_arch':[16,16]})
    model.learn(60)
    model.save(tmp_path/'model.zip')
    runner.save_rng(tmp_path/'rng.pt')
    states = []
    for _ in range(2):
        loaded = MaskablePPO.load(tmp_path/'model.zip', env=DummyVecEnv([Toy,Toy]), device='cpu')
        runner.restore_rng(tmp_path/'rng.pt')
        loaded.learn(60, reset_num_timesteps=False)
        assert loaded.num_timesteps == 120
        states.append({k:v.clone() for k,v in loaded.policy.state_dict().items()})
        loaded.env.close()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    assert any(not torch.equal(states[0][key], model.policy.state_dict()[key]) for key in states[0])
    vec.close()
