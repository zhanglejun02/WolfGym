"""Write the frozen 1:1 trajectory-disjoint plan without running any simulator."""
import argparse
import hashlib
import json
from pathlib import Path

from wolfgym.benchmark.combination import benchmark_contract, episode_plan, validate_episode_manifest
from wolfgym.env.combination_contract import primary_contract, validate_primary_scenario


def main(output):
    validate_primary_scenario()
    contract = benchmark_contract()
    plans = {split: [episode.to_dict() for episode in episode_plan(split)]
             for split in contract['splits']}
    groups = set()
    for split, rows in plans.items():
        validate_episode_manifest(rows, split=split)
        current = {r['group_id'] for r in rows}
        if groups & current:
            raise ValueError('initial society leaked across partitions')
        groups |= current
    # Audit known local raw-trajectory names as well as the reserved seed floor.
    old_seeds = set()
    for path in Path('data').glob('**/*.json.gz'):
        prefix = path.name.split('_')[0]
        if prefix.isdigit():
            old_seeds.add(int(prefix))
    all_seeds = {r['trajectory_seed'] for rows in plans.values() for r in rows}
    if all_seeds & old_seeds or any(s < contract['reserved_below'] for s in all_seeds):
        raise ValueError('planned trajectory overlaps prior research')
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split, rows in plans.items():
        (output / f'{split}.json').write_text(json.dumps(rows, indent=2) + '\n')
        counts[split] = {'initial_societies': len(rows) // 2, 'episodes': len(rows),
                         'normal': len(rows) // 2, 'attack': len(rows) // 2}
    paths = [Path('src/wolfgym/env/combination_v51.json'),
             Path('src/wolfgym/benchmark/combination_splits_v1.json'),
             Path('src/wolfgym/benchmark/combination.py'),
             Path('src/wolfgym/env/combination_contract.py'),
             Path('src/wolfgym/env/combination.py'), Path(__file__), *output.glob('*.json')]
    receipt = {'protocol': contract, 'primary_task': primary_contract()['version'],
               'counts': counts, 'simulations_run': 0, 'prior_raw_seed_count_checked': len(old_seeds),
               'prior_raw_seed_overlap': [], 'release_eligible': False,
               'source_and_plan_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'counts': counts, 'simulations_run': 0, 'prior_seed_overlap': []}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
