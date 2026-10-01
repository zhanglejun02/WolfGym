"""Episode-consistent asset aliases/order with invertible action references.

Representation randomness is independent of market RNG and shared by the paired
normal/attack episodes via the host-owned society key. No market is mutated.
"""
import copy
import numpy as np

from wolfgym.evaluation.combination_public import digest


VERSION = 'combination-asset-presentation-v1'


class AssetPresentation:
    def __init__(self, initial_targets, society_key):
        self.original = list(initial_targets)
        if len(self.original) != 3 or len(set(self.original)) != 3 or not all(type(x) is str for x in self.original):
            raise ValueError('three distinct public assets required')
        rng = np.random.default_rng(int(digest([VERSION, society_key])[:16], 16))
        self.order = [self.original[int(i)] for i in rng.permutation(3)]
        self.aliases = {asset: 'instrument_'+str(int(rng.integers(0, 2**63))) for asset in self.original}
        if len(set(self.aliases.values())) != 3:
            raise ValueError('presentation alias collision')
        self.reverse = {alias: asset for asset, alias in self.aliases.items()}

    def metadata(self):
        return {'version': VERSION, 'original_targets': self.original[:],
                'display_order': self.order[:], 'aliases': dict(self.aliases)}

    def _targets_mask(self, targets, mask):
        if len(targets) != 3 or set(targets) != set(self.original) or len(mask) != 4:
            raise ValueError('presentation asset set or mask changed')
        return ([self.aliases[a] for a in self.order],
                [mask[0], *[mask[targets.index(a)+1] for a in self.order]])

    def frame(self, frame):
        result = copy.deepcopy(frame)
        targets = list(frame['action_targets'])
        result['action_targets'], result['action_mask'] = self._targets_mask(targets, frame['action_mask'])
        assets = {r['asset_id']: r for r in frame['assets']}
        result['assets'] = [{**assets[a], 'asset_id': self.aliases[a]} for a in self.order]
        result['message_summary'] = {self.aliases[a]: copy.deepcopy(frame['message_summary'][a]) for a in self.order}
        return result

    def view(self, view):
        result = copy.deepcopy(view)
        result['observations'] = [self.frame(f) for f in view['observations']]
        for intervention in result['interventions']:
            intervention['target_id'] = self.aliases[intervention['target_id']]
        return result

    def info(self, info):
        result = copy.deepcopy(info)
        targets = list(info['action_targets'])
        result['action_targets'], result['action_mask'] = self._targets_mask(targets, info['action_mask'])
        observation = result['public_observation']
        assets = {r['asset_id']: r for r in observation['assets']}
        observation['assets'] = [{**assets[a], 'asset_id': self.aliases[a]} for a in self.order]
        for message in observation['messages']:
            if message['asset_id'] in self.aliases:
                message['asset_id'] = self.aliases[message['asset_id']]
        return result

    def to_native_action(self, action, native_info):
        targets = list(native_info['action_targets'])
        _, mask = self._targets_mask(targets, native_info['action_mask'])
        if type(action) is not int or not 0 <= action < 4 or not mask[action]:
            raise ValueError('invalid display action')
        return targets.index(self.order[action-1])+1 if action else 0

    def to_display_action(self, action, native_info):
        targets = list(native_info['action_targets'])
        self._targets_mask(targets, native_info['action_mask'])
        if type(action) is not int or not 0 <= action < 4 or not native_info['action_mask'][action]:
            raise ValueError('invalid native action')
        return self.order.index(targets[action-1])+1 if action else 0


def public_target_has_tie(frame):
    """ID-based tie breaks can change under renaming; report them separately."""
    reached = [r['audience_reach'] for r in frame['message_summary'].values() if r['present']]
    if reached:
        return reached.count(max(reached)) > 1
    values = [(r['return_3d'], r['volume_zscore']) for r in frame['assets']]
    return values.count(max(values)) > 1
