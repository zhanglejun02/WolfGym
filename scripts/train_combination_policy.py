#!/usr/bin/env python
"""Bounded MaskablePPO study: three seeds, one train pass, four validation checkpoints."""
import argparse
from functools import partial
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np

from wolfgym.benchmark.combination import CombinationEpisode, episode_plan, validate_episode_manifest
from wolfgym.env.combination_learning import CombinationSampleEnv, ENCODING_VERSION
from wolfgym.env.combination_contract import validate_primary_scenario
from wolfgym.evaluation.combination_host import write_json, read_json, file_hash
from wolfgym.evaluation.combination_public import digest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'seeds': [11, 22, 33], 'episodes_per_seed': 2000, 'checkpoints': [500, 1000, 1500, 2000],
          'num_envs': 4, 'n_steps': 30, 'batch_size': 120, 'n_epochs': 10,
          'learning_rate': .0003, 'gamma': 1., 'gae_lambda': 1., 'clip_range': .2,
          'ent_coef': .01, 'vf_coef': .5, 'max_grad_norm': .5, 'net_arch': [128, 128],
          'reward_scale': 1000., 'initial_action_prior': [.9, 1/30, 1/30, 1/30],
          'evaluation_deterministic': True, 'device': 'cpu'}


def read_logs(directory, expected):
    rows = []
    for path in sorted(Path(directory).glob('worker_*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    identities = [r['result']['episode_id'] for r in rows]
    if len(set(identities)) != len(identities) or set(identities) != {e['episode_id'] for e in expected}:
        raise ValueError('missing, duplicate or foreign logged episodes')
    for row in rows:
        if len(row['actions']) != 30 or 4*sum(a != 0 for a in row['actions']) != row['result']['cost']:
            raise ValueError('invalid training/evaluation action log')
    return rows


def make_vector(episodes, directory, training, count):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    directory = Path(directory)
    if len(episodes) % count:
        raise ValueError('vector lanes must exhaust at the same boundary')
    factories = [partial(CombinationSampleEnv, episodes[i::count], training=training,
                         terminal_log=str(directory / f'worker_{i}.jsonl'),
                         audit_directory=None if training else str(directory/'raw'))
                 for i in range(count)]
    return SubprocVecEnv(factories, start_method='spawn')


def save_rng(path):
    import torch
    torch.save({'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state()}, path)


def restore_rng(path):
    import torch
    # Only self-produced local checkpoints, verified against their receipt.
    state = torch.load(path, weights_only=False, map_location='cpu')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def check_receipt(path, execution_id):
    receipt = read_json(path)
    if receipt['execution_id'] != execution_id:
        raise ValueError('checkpoint belongs to another execution')
    for filename, sha in receipt['sha256'].items():
        if file_hash(Path(path).parent/filename) != sha:
            raise ValueError('checkpoint artifact changed')
    return receipt


def run_training_chunk(episodes, directory, execution_id, seed, checkpoint, previous, output, *, smoke=False):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
    torch.set_num_threads(1)
    count = 2 if smoke else CONFIG['num_envs']
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory/'training_complete.json'
    if receipt_path.exists():
        return check_receipt(receipt_path, execution_id)
    attempt = directory / ('train_attempt_'+uuid.uuid4().hex[:12])
    attempt.mkdir()
    vec = make_vector(episodes, attempt, True, count)
    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % (30*count) == 0:
                write_json(output/'progress.json', {'stage': 'training', 'seed': seed,
                    'target_checkpoint': checkpoint, 'seed_timesteps': self.num_timesteps,
                    'seed_episodes': self.num_timesteps//30, 'execution_id': execution_id,
                    'complete': False, 'updated_at_unix': time.time(), 'pid': os.getpid()})
            return True
    try:
        if previous:
            model = MaskablePPO.load(directory.parent/previous['model'], env=vec, device='cpu')
            restore_rng(directory.parent/previous['rng'])
        else:
            parameters = {key: CONFIG[key] for key in (
                'n_steps', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range',
                'ent_coef', 'vf_coef', 'max_grad_norm')}
            model = MaskablePPO('MlpPolicy', vec, seed=seed, device='cpu', verbose=0,
                                batch_size=count*30, policy_kwargs={'net_arch': CONFIG['net_arch']},
                                **parameters)
            with torch.no_grad():
                model.policy.action_net.bias.copy_(torch.log(torch.tensor(CONFIG['initial_action_prior'])))
        model.learn(total_timesteps=len(episodes)*30, reset_num_timesteps=not bool(previous), callback=Progress())
        if model.num_timesteps != checkpoint*30:
            raise ValueError('PPO exceeded the exact episode budget')
        logs = read_logs(attempt, episodes)
        model.save(attempt/'model.zip')
        save_rng(attempt/'rng.pt')
        paths = [attempt/'model.zip', attempt/'rng.pt', *sorted(attempt.glob('worker_*.jsonl'))]
        receipt = {'execution_id': execution_id, 'seed': seed, 'checkpoint': checkpoint,
                   'episode_count': len(logs), 'seed_timesteps': model.num_timesteps,
                   'model': str((attempt/'model.zip').relative_to(directory.parent)),
                   'rng': str((attempt/'rng.pt').relative_to(directory.parent)),
                   'logs': str(attempt.relative_to(directory.parent)),
                   'sha256': {str(p.relative_to(directory)): file_hash(p) for p in paths}}
        write_json(receipt_path, receipt)
        return receipt
    finally:
        vec.close()


def run_validation(episodes, directory, execution_id, model_path, output, *, smoke=False):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    import torch
    torch.set_num_threads(1)
    receipt_path = directory/'validation_complete.json'
    if receipt_path.exists():
        receipt = check_receipt(receipt_path, execution_id)
        if receipt['model_sha256'] != file_hash(model_path):
            raise ValueError('validation result belongs to another model')
        return receipt
    attempt = directory / ('validation_attempt_'+uuid.uuid4().hex[:12])
    attempt.mkdir()
    count = 2 if smoke else CONFIG['num_envs']
    vec = make_vector(episodes, attempt, False, count)
    try:
        model = MaskablePPO.load(model_path, device='cpu')
        obs = vec.reset()
        completed = 0
        for _ in range(len(episodes)//count*30):
            actions, _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(vec))
            obs, rewards, dones, infos = vec.step(actions)
            if np.any(rewards != 0):
                raise ValueError('held-out reward leaked into evaluation')
            completed += int(dones.sum())
            if dones.any():
                write_json(output/'progress.json', {'stage': 'validation', 'checkpoint_path': str(directory),
                    'completed_validation_episodes': completed, 'expected_validation_episodes': len(episodes),
                    'execution_id': execution_id, 'complete': False, 'updated_at_unix': time.time(), 'pid': os.getpid()})
        logs = read_logs(attempt, episodes)
        results = [x['result'] for x in logs]
        if not smoke:
            validate_episode_manifest([{k:r[k] for k in ('split','trajectory_seed','condition','group_id','episode_id')}
                                       for r in results], split='validation')
        pairs = {}
        for row in results:
            pairs.setdefault(row['group_id'], {})[row['condition']] = row['H_reduction']
        if any(set(p) != {'normal', 'attack'} for p in pairs.values()):
            raise ValueError('evaluation requires complete society pairs')
        balanced = float(np.mean([.5*(p['normal']+p['attack']) for p in pairs.values()]))
        write_json(attempt/'results.json', results)
        paths = [attempt/'results.json', *sorted(attempt.glob('worker_*.jsonl')), *sorted((attempt/'raw').glob('*.json.gz'))]
        receipt = {'execution_id': execution_id, 'episodes': len(results), 'societies': len(pairs),
                   'balanced_mean': balanced, 'results': str((attempt/'results.json').relative_to(directory)),
                   'raw': str((attempt/'raw').relative_to(directory)),
                   'model_sha256': file_hash(model_path),
                   'sha256': {str(p.relative_to(directory)):file_hash(p) for p in paths}}
        write_json(receipt_path, receipt)
        return receipt
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='retired engineering pair only; never research results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'.run.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('this study already has an active runner')
    validate_primary_scenario()
    train = [e.to_dict() for e in episode_plan('train')]
    validation = [e.to_dict() for e in episode_plan('validation')]
    if args.smoke:
        train = validation = [CombinationEpisode('engineering',20260925,c).to_dict() for c in ('normal','attack')]
    sources = [Path(__file__), ROOT/'src/wolfgym/env/combination_learning.py',
               ROOT/'src/wolfgym/evaluation/combination_public.py', ROOT/'src/wolfgym/evaluation/combination_host.py',
               ROOT/'src/wolfgym/env/combination_v51.json', ROOT/'docs/EXPERIMENT_PROTOCOL.md']
    declaration = {'version':'combination-learning-v1', 'encoding':ENCODING_VERSION, 'config':CONFIG,
                   'engineering_smoke':args.smoke, 'train':train, 'validation':validation,
                   'packages':{name:importlib.metadata.version(name) for name in
                               ('numpy','torch','gymnasium','stable-baselines3','sb3-contrib')},
                   'source_sha256':{str(p.resolve()):file_hash(p) for p in sources}, 'formal_test_allowed':False}
    execution_id = digest(declaration)
    manifest = {**declaration, 'execution_id':execution_id}
    if (args.output/'manifest.json').exists() and read_json(args.output/'manifest.json') != manifest:
        raise ValueError('frozen learning configuration/source changed')
    write_json(args.output/'manifest.json', manifest)
    selected = {}
    for seed in ([11] if args.smoke else CONFIG['seeds']):
        shuffled = [train[int(i)] for i in np.random.default_rng(seed).permutation(len(train))]
        seed_dir = args.output/f'seed_{seed}'
        seed_dir.mkdir(exist_ok=True)
        previous = None
        start = 0
        candidates = []
        for checkpoint in ([2] if args.smoke else CONFIG['checkpoints']):
            directory = seed_dir/f'checkpoint_{checkpoint}'
            receipt = run_training_chunk(shuffled[start:checkpoint], directory, execution_id,
                                         seed, checkpoint, previous, args.output, smoke=args.smoke)
            print(f'seed={seed} trained={checkpoint} checkpoint saved', flush=True)
            evaluation = run_validation(validation, directory, execution_id, seed_dir/receipt['model'],
                                        args.output, smoke=args.smoke)
            print(f"seed={seed} checkpoint={checkpoint} validation_balanced={evaluation['balanced_mean']:.9f}", flush=True)
            candidates.append({'checkpoint':checkpoint, 'balanced_mean':evaluation['balanced_mean'], 'model':receipt['model']})
            previous, start = receipt, checkpoint
        chosen = min(candidates, key=lambda x:(-x['balanced_mean'],x['checkpoint']))
        selected[str(seed)] = {'chosen':chosen, 'learning_curve':candidates}
        write_json(seed_dir/'selection.json', selected[str(seed)])
    write_json(args.output/'selection.json', selected)
    write_json(args.output/'progress.json', {'execution_id':execution_id, 'stage':'training_and_validation_finished',
                                           'complete':True, 'audit_complete':False, 'formal_test_episodes':0,
                                           'updated_at_unix':time.time()})
    print('training and validation complete; independent audit still required', flush=True)


if __name__ == '__main__':
    main()
