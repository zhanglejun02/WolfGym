"""Engineering-only resumable model host. Research/test entry remains closed.

Replay rebuilds the world from its fixed initial state and committed responses.
Only the final durable artifact is a complete episode; spend exhaustion is an
incomplete run, never an artificial series of WAIT decisions.
"""
from dataclasses import asdict
import fcntl
import os
from pathlib import Path

from wolfgym.env.combination import CombinationGymEnv
from wolfgym.env.combination_contract import validate_primary_scenario
from wolfgym.evaluation.combination_decision import CombinationDecisionAdapter
from wolfgym.evaluation.combination_host import read_json, write_json, file_hash
from wolfgym.evaluation.combination_public import digest


ROOT = Path(__file__).resolve().parents[1]


def durable_json(path, value):
    path = Path(path)
    write_json(path, value)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_engineering_episode(episode, *, mode, detector, journal, transport, directory, window=5):
    # No permissive flag can turn this smoke host into a formal test runner.
    if episode.split != 'engineering':
        raise ValueError('research/test closed until joint model and experiment freeze')
    validate_primary_scenario()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.episode.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [Path(__file__), Path(__file__).with_name('combination_requests.py'),
                   Path(__file__).with_name('combination_decision.py'),
                   Path(__file__).with_name('combination_public.py'), ROOT/'env/combination.py',
                   ROOT/'env/combination_v51.json']
        declaration = {'version': 'combination-model-engineering-v1', 'episode': episode.to_dict(),
                       'mode': mode, 'window': window, 'detector_sha256': digest(detector),
                       'request_contract': journal.declaration,
                       'request_journal_id': journal.identity,
                       'source_sha256': {str(p): file_hash(p) for p in sources},
                       'formal_test_allowed': False}
        manifest_path = directory/'manifest.json'
        if manifest_path.exists() and read_json(manifest_path) != declaration:
            raise ValueError('episode source, model or request configuration drift')
        durable_json(manifest_path, declaration)
        scope = digest(declaration)
        completed_path = directory/'complete.json'
        if completed_path.exists():
            receipt = read_json(completed_path)
            if receipt['scope'] != scope or any(file_hash(directory/k) != v for k, v in receipt['sha256'].items()):
                raise ValueError('completed episode changed')
            return read_json(directory/'outcome.json.gz')['result']

        def request(payload):
            day = len(adapter.log)
            return journal.request(scope, day, payload, transport)

        adapter = CombinationDecisionAdapter(mode, detector, request, window=window)
        env = CombinationGymEnv(episode, training_reward=False)
        try:
            _, info = env.reset()
            for day in range(30):
                action = adapter(info)
                row = adapter.log[-1]
                path = directory/'decisions'/f'{day:02d}.json'
                if path.exists() and read_json(path) != row:
                    raise ValueError('public input, response or action changed during replay')
                # Commit chosen action before simulating it. A crash in step
                # replays this decision from the same initial world next time.
                durable_json(path, row)
                _, reward, terminated, truncated, info = env.step(action)
                if reward != 0 or truncated or terminated != (day == 29):
                    raise ValueError('evaluation reward/termination contract violated')
                durable_json(directory/'progress.json', {'complete': False, 'scope': scope,
                                                        'completed_market_days': day+1})
            result = env.evaluation_result()
            result['invalid_actions'] = sum(row['invalid_action'] for row in adapter.log)
            result['model_decisions'] = sum(row['request'] is not None for row in adapter.log)
            artifact = {'scope': scope, 'result': result, 'decisions': adapter.log,
                        'accounts': {k: [asdict(x) for x in v] for k, v in env._snapshots.items()},
                        'financial_rows': env._rows}
            durable_json(directory/'outcome.json.gz', artifact)
            paths = [manifest_path, directory/'outcome.json.gz', *sorted((directory/'decisions').glob('*.json'))]
            durable_json(completed_path, {'scope': scope, 'sha256': {
                str(p.relative_to(directory)): file_hash(p) for p in paths}})
            durable_json(directory/'progress.json', {'complete': True, 'scope': scope, 'completed_market_days': 30})
            return result
        except BaseException as exc:
            durable_json(directory/'last_failure.json', {'scope': scope, 'exception_type': type(exc).__name__,
                                                        'committed_decisions': len(list((directory/'decisions').glob('*.json')))})
            raise
        finally:
            env.close()
