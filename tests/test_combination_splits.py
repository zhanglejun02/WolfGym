import pytest

from wolfgym.benchmark.combination import (
    CombinationEpisode, benchmark_contract, episode_plan,
    validate_episode_manifest, paired_protection_summary,
)


def test_exact_balance_disjoint_societies_and_descendant_inheritance():
    seen = set()
    for split, count in [('train', 1000), ('validation', 128), ('test', 256)]:
        plan = episode_plan(split)
        assert len(plan) == 2 * count
        assert sum(e.condition == 'normal' for e in plan) == count
        groups = {e.group_id for e in plan}
        assert len(groups) == count and not groups & seen
        seen |= groups
        assert validate_episode_manifest([e.to_dict() for e in plan], split=split)
        e = plan[0]
        for day, continuation in [(0, 42), (10, 314), (29, 50000000)]:
            child = e.descendant_metadata(day, continuation)
            assert child['group_id'] == e.group_id and child['split'] == split
        counterpart = CombinationEpisode(split, e.trajectory_seed,
                                          'normal' if e.condition == 'attack' else 'attack')
        assert counterpart.group_id == e.group_id


def test_reject_research_seeds_cross_split_duplicate_and_missing_pairs():
    for seed in [1, 70000, 73000, 74255, 20260924, 20260925]:
        with pytest.raises(ValueError):
            CombinationEpisode('train', seed, 'attack')
    with pytest.raises(ValueError):
        CombinationEpisode('train', 50000000, 'normal')
    rows = [e.to_dict() for e in episode_plan('test')]
    for bad in (rows[:-1], rows + [rows[0]], [{**rows[0], 'group_id': 'forged'}, *rows[1:]]):
        with pytest.raises(ValueError):
            validate_episode_manifest(bad, split='test')


def test_balanced_score_counts_roots_not_two_related_episodes():
    rows = [{**e.to_dict(), 'H_reduction': .01 if e.condition == 'attack' else -.02}
            for e in episode_plan('test')]
    out = paired_protection_summary(rows, split='test')
    assert out['balanced']['independent_groups'] == 256
    assert out['balanced']['mean'] == pytest.approx(-.005)
    assert out['normal']['mean'] == pytest.approx(-.02)
    assert out['attack']['mean'] == pytest.approx(.01)
    assert benchmark_contract()['condition_weights'] == {'normal': .5, 'attack': .5}
