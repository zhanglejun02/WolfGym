import gc
from pathlib import Path
import weakref

import numpy as np
import pytest

pytest.importorskip('gymnasium')

import wolfgym.env.combination as entry
from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination_contract import primary_contract


def episode():
    return CombinationEpisode('engineering', 20260925, 'normal')


@pytest.mark.parametrize('relative', ['env/combination.py', 'env/combination_contract.py'])
def test_entry_file_drift_rejected_before_market_allocation(monkeypatch, relative):
    assert relative in primary_contract()['source_files']
    root = Path(entry.__file__).resolve().parents[1]
    original_read = Path.read_bytes

    def changed_read(path):
        content = original_read(path)
        return content + b'\n# unexpected drift\n' if path == root / relative else content

    monkeypatch.setattr(Path, 'read_bytes', changed_read)
    monkeypatch.setattr(entry, 'WolfSocietySimulator', lambda *a, **k: pytest.fail('market allocated before rejection'))
    with pytest.raises(ValueError, match='contract changed'):
        entry.CombinationGymEnv(episode())


def test_action_mask_and_charge_read_same_contract_cost(monkeypatch):
    # Dependency substitution exercises fee plumbing, not an allowed new primary
    # task. The on-disk production contract stays fixed at four.
    contract = primary_contract()
    assert contract['freeze_cost'] == 4.
    contract['freeze_cost'] = 5.
    monkeypatch.setattr(entry, 'primary_contract', lambda: contract)
    env = entry.CombinationGymEnv(episode())
    env.reset()
    try:
        env.step(1)
        assert env._remaining == 7.
        env.step(2)
        assert env._remaining == 2.
        assert list(env.action_masks()) == [1, 0, 0, 0]
        with pytest.raises(ValueError, match='unaffordable'):
            env.step(3)
        assert all(world._next_day == 2 for world in env._worlds.values())
    finally:
        env.close()


@pytest.mark.parametrize('advance', [False, True])
def test_close_releases_worlds_and_snapshot_and_can_rebuild(advance):
    env = entry.CombinationGymEnv(episode())
    template_ref = weakref.ref(env._template)
    world_refs = []
    initial = None
    if advance:
        initial, _ = env.reset(seed=42)
        world_refs = [weakref.ref(world) for world in env._worlds.values()]
        env.step(1)
        env.step(0)
        assert env._snapshots['policy'] and env._rows['policy']
    env.close()
    env.close()
    gc.collect()
    assert template_ref() is None and all(ref() is None for ref in world_refs)
    assert env._initial is None and env._template is None
    assert not env._worlds and not env._snapshots and not env._rows and not env._totals
    assert env._public is None and env._result is None
    with pytest.raises(RuntimeError):
        env.step(0)
    with pytest.raises(RuntimeError):
        env.evaluation_result()
    rebuilt, _ = env.reset(seed=42)
    if initial is not None:
        for key in initial:
            np.testing.assert_array_equal(initial[key], rebuilt[key])
    assert env._remaining == 12 and env._worlds['policy']._next_day == 0
    env.step(0)
    env.close()
