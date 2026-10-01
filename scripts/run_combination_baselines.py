#!/usr/bin/env python
"""Execute only the predeclared train/validation baseline study, resumably."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import time

from wolfgym.benchmark.combination import CombinationEpisode, episode_plan
from wolfgym.env.combination_contract import validate_primary_scenario, primary_contract
from wolfgym.evaluation.combination_public import baseline_specs, digest
from wolfgym.evaluation.combination_host import read_json, write_json, file_hash, load_detector, run_episode


ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / 'data/task_validation_v4_20260923/detector.json'


def selected_episodes(split, count):
    plan = episode_plan(split)
    groups = list(dict.fromkeys(e.group_id for e in plan))[:count]
    return [e for e in plan if e.group_id in groups]


def worker(job):
    episode_dict, directory, execution_id = job
    episode = CombinationEpisode(episode_dict['split'], episode_dict['trajectory_seed'], episode_dict['condition'])
    return run_episode(episode, baseline_specs(), load_detector(DETECTOR), directory, execution_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('workers must be between one and six')
    validate_primary_scenario()
    load_detector(DETECTOR)
    episodes = selected_episodes('train', 64) + selected_episodes('validation', 128)
    sources = [ROOT / 'src/wolfgym/evaluation/combination_public.py',
               ROOT / 'src/wolfgym/evaluation/combination_host.py', Path(__file__),
               ROOT / 'src/wolfgym/env/combination_v51.json', DETECTOR,
               ROOT/'docs/EXPERIMENT_PROTOCOL.md']
    declaration = {'version': 'combination-baselines-20260926-v1',
                   'protocol': 'docs/EXPERIMENT_PROTOCOL.md',
                   'task_contract': primary_contract(), 'specs': baseline_specs(),
                   'episodes': [e.to_dict() for e in episodes],
                   'source_sha256': {str(p.resolve()): file_hash(p) for p in sources},
                   'formal_test_allowed': False, 'llm_calls': 0,
                   'statistics': {'bootstrap_repetitions': 20000, 'rng_seed': 20260926,
                                  'shortcut_margin_h': .0002}}
    execution_id = digest(declaration)
    manifest = {**declaration, 'execution_id': execution_id}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError('existing manifest differs; never mix versions or populations')
    else:
        write_json(manifest_path, manifest)
    jobs = [(e.to_dict(), str(args.output / e.split / str(e.trajectory_seed) / e.condition), execution_id)
            for e in episodes]
    started = time.monotonic()
    completed = {'train': 0, 'validation': 0}
    failures = []
    # Spawn avoids inheriting simulator and numerical-library state.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            identity = futures[future]
            try:
                result = future.result()
                completed[identity['split']] += 1
                print(f"completed {sum(completed.values())}/{len(jobs)} {identity['split']} "
                      f"{identity['trajectory_seed']} {identity['condition']} resumed={result['resumed']}", flush=True)
            except Exception as exc:
                failures.append({'episode': identity, 'error': repr(exc)})
                print(f"FAILED {identity['episode_id']} {exc!r}", flush=True)
            write_json(args.output / 'progress.json', {
                'execution_id': execution_id, 'completed_episodes': completed,
                'expected_episodes': {'train': 128, 'validation': 256},
                'failures': failures, 'elapsed_seconds': time.monotonic()-started,
                'complete': sum(completed.values()) == len(jobs) and not failures})
    if failures:
        raise RuntimeError(f'{len(failures)} episodes failed; artifacts retained for diagnosis/resume')


if __name__ == '__main__':
    main()
