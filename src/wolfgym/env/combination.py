"""Primary combination Gym: frozen population, public feedback, paired scoring.

One environment represents a manifest-owned initial trajectory and condition.
reset(seed=...) resets Gym's RNG, but cannot replace the manifest's world seed.
Training code selects new episodes from the train plan, never from held-out roots.
"""
import copy
from dataclasses import asdict, replace
import math

import numpy as np

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.core.agents.switching import install_switching_cohort, prepare_tactic
from wolfgym.core.env.environment import WolfSocietySimulator
from wolfgym.env.combination_contract import (
    CombinationSettings, primary_contract, validate_primary_scenario, validate_primary_population,
)
from wolfgym.env.gymnasium import _GymBase, _require_gymnasium, spaces
from wolfgym.env.rl import FixedActionCodec, StructuredObservationEncoder
from wolfgym.evaluation.combination_task import score_combination
from wolfgym.evaluation.counterfactual_welfare import capture_welfare_snapshot
from wolfgym.evaluation.switching_probe import check_wallets, account_totals
from wolfgym.evaluation.task_validation import public_observation, reseed_normal_decisions
from wolfgym.observations import ObservationBuilder, ObservationConfig
from wolfgym.regulation import RegulatoryAction, RegulatoryActionType, RegulatoryActionExecutor


class CombinationGymEnv(_GymBase):
    """Actions: 0=WAIT, 1..3=one-day FREEZE of the corresponding public asset.

    step/reset return only public feedback. A trusted evaluator may call
    evaluation_result() after day 30; never give that API to a test policy.
    Training rewards are opt-in and rejected on validation/test episodes.
    """

    metadata = {'render_modes': []}
    render_mode = None

    def __init__(self, episode: CombinationEpisode, *, settings=None, scenario=None,
                 training_reward=False):
        _require_gymnasium()
        super().__init__()
        if type(episode) is not CombinationEpisode:
            raise TypeError('episode must be a validated CombinationEpisode')
        self._episode = CombinationEpisode(episode.split, episode.trajectory_seed, episode.condition)
        if settings is not None and type(settings) is not CombinationSettings:
            raise TypeError('settings must be CombinationSettings')
        self.settings = CombinationSettings(**asdict(settings)) if settings is not None else CombinationSettings()
        if type(training_reward) is not bool:
            raise TypeError('training_reward must be boolean')
        if training_reward and episode.split not in {'train', 'engineering'}:
            raise ValueError('hidden training reward is forbidden on validation/test')
        self._training_reward = training_reward
        self._contract = primary_contract()
        self._freeze_cost = float(self._contract['freeze_cost'])
        if not math.isfinite(self._freeze_cost) or self._freeze_cost <= 0:
            raise ValueError('contract freeze_cost must be finite and positive')
        # Resource/scenario drift fails before simulator construction.
        canonical = validate_primary_scenario(scenario)
        self._build_template(canonical)
        self._encoder = StructuredObservationEncoder(
            action_codec=FixedActionCodec(max_assets=3, max_entities=5),
            horizon_days=30, initial_budget=12.)
        self.action_space = spaces.Discrete(4)
        shapes = self._encoder.shapes
        self.observation_space = spaces.Dict({
            'global': spaces.Box(-10., 10., shape=shapes['global'], dtype=np.float32),
            'targets': spaces.Box(-10., 10., shape=shapes['targets'], dtype=np.float32),
            'target_mask': spaces.MultiBinary(shapes['target_mask'][0]),
            # At most eight public messages, each with reach clipped to [0,1].
            'message_reach': spaces.Box(0., 8., shape=(3,), dtype=np.float32),
            'action_mask': spaces.MultiBinary(4),
        })
        self._executor = RegulatoryActionExecutor()
        self._ready = False

    def _build_template(self, scenario):
        self._template = install_switching_cohort(WolfSocietySimulator(
            scenario, self.settings.society_size, self.settings.alpha,
            seed=self._episode.trajectory_seed), initial_inventory_share=self.settings.initial_stock_share)
        validate_primary_population(self._template)
        self._initial = self._template.get_state()

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('primary reset accepts no population/seed/checkpoint overrides')
        super().reset(seed=seed)
        # Never permit reset options to bypass the constructor's resource lock.
        canonical = validate_primary_scenario()
        if self._template is None:
            self._build_template(canonical)
        self._worlds = {key: self._template.fork_from_state(self._initial)
                        for key in ('reference', 'wait', 'policy')}
        for key, sim in self._worlds.items():
            validate_primary_population(sim)
            sim.set_continuation_seed(self._episode.trajectory_seed + self._contract['continuation_seed_offset'])
            if self._episode.condition == 'normal' and key != 'reference':
                reseed_normal_decisions(sim, self._episode.trajectory_seed + self._contract['normal_replica_seed_offset'])
        self._totals = {key: account_totals(sim) for key, sim in self._worlds.items()}
        self._snapshots = {key: [] for key in self._worlds}
        self._rows = {key: [] for key in self._worlds}
        self._remaining = self.settings.initial_budget
        self._terminated = False
        self._ready = True
        self._result = None
        return self._public_output()

    def _public_output(self):
        sim = self._worlds['policy']
        obs = public_observation(sim, self._remaining)
        messages = ObservationBuilder(ObservationConfig(include_messages=True))._build_messages(sim)
        self._public = replace(obs, messages=messages)
        encoded = self._encoder.encode(self._public)
        self._targets = tuple(a.asset_id for a in self._public.assets)
        reach = {asset: 0. for asset in self._targets}
        for message in messages:
            if message.asset_id in reach:
                reach[message.asset_id] += message.audience_reach
        encoded['message_reach'] = np.asarray([reach[a] for a in self._targets], dtype=np.float32)
        encoded['action_mask'] = self.action_masks()
        # No condition, root seed, true target, damage, reference accounts or
        # reward components. Public views for LLMs and tensor policies coexist.
        return encoded, {'public_observation': self._public.to_dict(),
                         'action_targets': self._targets, 'action_mask': self.action_masks()}

    def action_masks(self):
        if not self._ready:
            raise RuntimeError('call reset before requesting actions')
        return np.asarray([1, *([int(self._remaining >= self._freeze_cost)] * 3)], dtype=np.int8)

    def step(self, action):
        if not self._ready or self._terminated:
            raise RuntimeError('call reset before stepping a new episode')
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError('action must be an integer in 0..3')
        if not 0 <= int(action) < 4 or not self.action_masks()[int(action)]:
            raise ValueError('invalid or unaffordable action; state was not advanced')
        cost = self._freeze_cost if action else 0.
        typed = (RegulatoryAction(RegulatoryActionType.FREEZE, self._targets[int(action) - 1])
                 if action else RegulatoryAction(RegulatoryActionType.WAIT))
        for key, sim in self._worlds.items():
            mode = 'campaign' if key != 'reference' and self._episode.condition == 'attack' else 'off'
            prepare_tactic(sim, mode=mode,
                          selection_seed=self._episode.trajectory_seed + self._contract['continuation_seed_offset'])
            selected = typed if key == 'policy' else RegulatoryAction(RegulatoryActionType.WAIT)
            entry = sim.step_day(pre_step=lambda world, day: self._executor.apply(world, selected, day))
            if key == 'policy' and cost:
                self._executor.release(sim, selected.target_id)
            check_wallets(sim)
            totals = account_totals(sim)
            if any(not math.isclose(totals[k], v, rel_tol=1e-12, abs_tol=1e-5)
                   for k, v in self._totals[key].items()):
                raise ValueError('cash/share conservation failed')
            for agent in sim.society.attackers:
                if (agent.last_activity['submitted_notional'] > agent.portfolio.initial_wealth + 1e-7
                        or agent.last_activity['message_intensity'] > 1.8 + 1e-10):
                    raise ValueError('attack resource cap exceeded')
            self._snapshots[key].append(capture_welfare_snapshot(sim, pair_id=self._episode.group_id))
            self._rows[key].append({'day': sim._next_day, 'financial_outcomes': entry['retail_outcomes']})
        self._remaining -= cost
        # A damage event never ends the trajectory early.
        self._terminated = self._worlds['policy']._next_day == 30
        reward = 0.
        if self._terminated:
            self._score_terminal()
            if self._training_reward:
                reward = self._result['H_reduction']
        observation, info = self._public_output()
        return observation, float(reward), self._terminated, False, info

    def _score_terminal(self):
        scores = {key: score_combination(
            self._snapshots['reference'], self._snapshots[key],
            self._rows['reference'], self._rows[key], self._contract['harm_threshold'])
            for key in ('wait', 'policy')}
        self._result = {**self._episode.to_dict(),
                        'H_reduction': scores['wait']['damage']['H'][-1] - scores['policy']['damage']['H'][-1],
                        'cost': self.settings.initial_budget - self._remaining,
                        'wait': scores['wait'], 'policy': scores['policy'],
                        'financial_outcomes': {key: {
                            metric: math.fsum(row['financial_outcomes'][metric] for row in rows)
                            for metric in ('requested_notional', 'blocked_notional', 'unexecuted_notional', 'executed_notional')}
                            for key, rows in self._rows.items()}}

    def evaluation_result(self):
        """Trusted terminal scorer only; exclude this method from policy tools."""
        if not self._ready or not self._terminated:
            raise RuntimeError('trusted evaluation is available only after the full 30 days')
        return copy.deepcopy(self._result)

    def close(self):
        """Release simulator graphs and histories, even when the env is retained.

        Repeated close is harmless. A later reset rebuilds the same fixed episode
        from its manifest identity rather than retaining the large initial state.
        """
        self._ready = False
        self._terminated = False
        self._worlds = {}
        self._snapshots = {}
        self._rows = {}
        self._totals = {}
        self._result = None
        self._public = None
        self._targets = ()
        self._template = None
        self._initial = None


def make_combination_env(episode, **kwargs):
    return CombinationGymEnv(episode, **kwargs)


def register_combination_gym():
    """Explicit opt-in; legacy environments and failure registry stay unchanged."""
    _require_gymnasium()
    import gymnasium as gym
    name = 'WolfGym-Combination-v0'
    if name not in gym.registry:
        gym.register(name, entry_point='wolfgym.env.combination:CombinationGymEnv')
    return name
