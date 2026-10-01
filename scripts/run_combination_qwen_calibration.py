"""Bounded second prompt and one-model factorial on the accepted eight train societies."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import os
from pathlib import Path

from dotenv import dotenv_values
from run_combination_qwen_pilot import episodes
from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination_fast import FastCombinationGymEnv
from wolfgym.env.combination_contract import validate_primary_scenario
from wolfgym.evaluation.combination_host import read_json, file_hash, load_detector
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_calibration import VERSION, DICTIONARY_PATH, CalibratedAdapter, calibrated_request
from wolfgym.evaluation.combination_qwen_pilot import SETTINGS, RESERVATION, QwenTransport, wire_body
from wolfgym.evaluation.combination_requests import RequestJournal

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'data/combination_qwen_pilot_20260928'
DETECTOR = ROOT/'data/task_validation_v4_20260923/detector.json'
JOBS = ([{'id': f'LL_{i}', 'mode': 'LL', 'shard': i, 'max_requests': 120, 'max_microusd': 3250000} for i in range(4)]
        + [{'id': f'LR_{i}', 'mode': 'LR', 'shard': i, 'max_requests': 120, 'max_microusd': 3500000} for i in range(4)]
        + [{'id': 'RL_0', 'mode': 'RL', 'shard': 0, 'max_requests': 48, 'max_microusd': 3000000}])


def verify_sources(manifest):
    for path, sha in manifest['source_sha256'].items():
        if file_hash(path) != sha:
            raise ValueError('frozen source/input drift: '+path)


def freeze(output):
    validate_primary_scenario()
    old = read_json(OLD/'manifest.json')
    verify_sources(old)
    if old['episodes'] != episodes():
        raise ValueError('original pilot sample mismatch')
    source_paths = [Path(x) for x in old['source_sha256']]
    source_paths += [Path(__file__), ROOT/'scripts/report_combination_qwen_calibration.py',
                     ROOT/'tests/test_combination_qwen_calibration.py',
                     ROOT/'src/wolfgym/evaluation/combination_qwen_calibration.py', DICTIONARY_PATH,
                     ROOT/'docs/EXPERIMENT_PROTOCOL.md', DETECTOR,
                     OLD/'manifest.json', OLD/'final_acceptance.json', OLD/'outcome_review.json',
                     OLD/'outcome_review_receipt.json',
                     ROOT/'data/combination_action_diagnostic_20260928/acceptance.json']
    # Bind all reused pilot and RR inputs. There is no fresh baseline study.
    for base in (OLD/'episodes', ROOT/'data/combination_action_diagnostic_20260928'):
        for identity in episodes():
            d = base/identity['episode_id'].replace(':', '_')
            marker = read_json(d/'complete.json')
            source_paths.append(d/'complete.json')
            for name, sha in marker['sha256'].items():
                p = d/name
                if file_hash(p) != sha: raise ValueError('historical artifact drift')
                if base == OLD/'episodes' or name in ('rule.json.gz','reference_wait.json.gz'):
                    source_paths.append(p)
    max_body = 0
    for p in (OLD/'episodes').glob('*/decisions/*.json'):
        row = read_json(p)
        if row['request'] is None: continue
        import json
        view = json.loads(row['request']['user'])
        for mode in ('LL','LR','RL'):
            max_body = max(max_body, len(wire_body(calibrated_request(view, mode))))
    manifest = {'version': VERSION, 'episodes': episodes(), 'settings': SETTINGS, 'jobs': JOBS,
                'source_sha256': {str(p.resolve()): file_hash(p) for p in source_paths},
                'max_requests_total': sum(j['max_requests'] for j in JOBS),
                'max_cost_microusd_total': sum(j['max_microusd'] for j in JOBS),
                'reservation_microusd': RESERVATION, 'retries': 0,
                'sequence': ['LL', 'LL_audit', 'LR_and_RL', 'full_audit'],
                'formal_test_allowed': False, 'dictionary_sha256': file_hash(DICTIONARY_PATH),
                'old_frame_max_wire_bytes': max_body}
    path = output/'manifest.json'
    if path.exists() and read_json(path) != manifest: raise ValueError('calibration manifest drift')
    durable_json(path, manifest)
    return manifest


def selected_episodes(manifest, job):
    if job not in manifest['jobs']: raise ValueError('undeclared job')
    if job['mode'] == 'RL': return manifest['episodes']
    groups = list(dict.fromkeys(x['group_id'] for x in manifest['episodes']))
    groups = groups[job['shard']*2:job['shard']*2+2]
    return [x for x in manifest['episodes'] if x['group_id'] in groups]


def run_one(output, identity, manifest, job, journal, key, stop_after=None):
    if identity not in selected_episodes(manifest, job) or identity['split'] != 'train':
        raise ValueError('only frozen train grid may run')
    directory = output/job['mode']/'episodes'/identity['episode_id'].replace(':', '_')
    directory.mkdir(parents=True, exist_ok=True)
    scope = digest([digest(manifest), job['mode'], identity])
    marker = directory/'complete.json'
    if marker.exists():
        saved = read_json(marker)
        if saved['scope'] != scope or any(file_hash(directory/p) != sha for p,sha in saved['sha256'].items()):
            raise ValueError('completed episode drift')
        return read_json(directory/'outcome.json.gz')['result']
    transport = QwenTransport(key, directory/'responses')
    def request(payload): return journal.request(scope, len(adapter.log), payload, transport)
    adapter = CalibratedAdapter(job['mode'], load_detector(DETECTOR), request)
    env = FastCombinationGymEnv(CombinationEpisode(identity['split'],identity['trajectory_seed'],identity['condition']), message_mode='indexed')
    try:
        _,info = env.reset()
        for day in range(30):
            action = adapter(info); row = adapter.log[-1]
            path = directory/'decisions'/f'{day:02}.json'
            if path.exists() and read_json(path) != row: raise ValueError('decision replay drift')
            durable_json(path, row)
            _,reward,done,truncated,info = env.step(action)
            if reward != 0 or truncated or done != (day==29): raise ValueError('feedback/termination violation')
            durable_json(directory/'progress.json', {'completed_days':day+1,'complete':False, **journal.totals()})
            print(f"{job['mode']} {identity['episode_id']} day={day+1} action={action} calls={journal.totals()['attempts']}", flush=True)
            if stop_after == day+1: return None
        result = env.evaluation_result()
        result.update(invalid_actions=sum(x['invalid_action'] for x in adapter.log),
                      model_decisions=sum(x['request'] is not None for x in adapter.log))
        durable_json(directory/'outcome.json.gz', {'scope':scope,'result':result,'decisions':adapter.log,
            'accounts':{k:[asdict(x) for x in v] for k,v in env._snapshots.items()}, 'financial_rows':env._rows})
        paths = [directory/'outcome.json.gz',*sorted((directory/'decisions').glob('*.json')),
                 *sorted((directory/'responses').glob('*.json'))]
        durable_json(marker, {'scope':scope,'sha256':{str(p.relative_to(directory)):file_hash(p) for p in paths}})
        durable_json(directory/'progress.json', {'complete':True,'completed_days':30})
        return result
    finally:
        env.close()


def run_job(args):
    output,manifest,job = args; output = Path(output)
    verify_sources(manifest)
    key = dotenv_values(ROOT/'.env').get('OPENROUTER_API_KEY')
    if not key: raise ValueError('missing local provider credential')
    with RequestJournal(output/(job['id']+'.sqlite'), contract={'manifest_sha256':digest(manifest),'job':job['id']},
                        max_attempts=job['max_requests'],max_cost_microusd=job['max_microusd'],
                        reservation_microusd=RESERVATION,retries=0) as journal:
        for identity in selected_episodes(manifest,job): run_one(output,identity,manifest,job,journal,key)
        value = {'job':job['id'],'complete':True,'totals':journal.totals()}
        durable_json(output/(job['id']+'_complete.json'),value)
        return value


def main(output, phase):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = freeze(output)
        if phase == 'freeze': return
        if (output/'acceptance.json').exists(): raise ValueError('already accepted; do not repeat')
        from report_combination_qwen_calibration import audit
        for modes in (('LL',),('LR','RL')):
            durable_json(output/'progress.json', {'pid':os.getpid(),'complete':False,'stage':'+'.join(modes)})
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(run_job,(str(output),manifest,j)) for j in JOBS if j['mode'] in modes]
                for future in as_completed(futures): future.result()
            verify_sources(manifest)
            audit(output, manifest, ('LL',) if modes==('LL',) else ('LL','LR','RL'))
        durable_json(output/'progress.json', {'pid':os.getpid(),'complete':True,'audit_complete':True})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('freeze','run'),required=True)
    args=parser.parse_args(); main(args.output.resolve(),args.phase)
