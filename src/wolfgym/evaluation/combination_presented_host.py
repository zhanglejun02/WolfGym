"""Versioned engineering host for the predeclared LL presentation experiment.

Uses the accepted decision adapter, presentation mapping, request journal and
native Gym. No research population, live transport, or release gate is enabled.
"""
from dataclasses import asdict
import fcntl
from pathlib import Path

from wolfgym.env.combination import CombinationGymEnv
from wolfgym.env.combination_contract import validate_primary_scenario
from wolfgym.evaluation.combination_decision import CombinationDecisionAdapter
from wolfgym.evaluation.combination_host import file_hash, read_json
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_presentation import AssetPresentation, public_target_has_tie
from wolfgym.evaluation.combination_public import digest, public_frame

VERSION = 'combination-presented-engineering-v1'
ROOT = Path(__file__).resolve().parents[1]
FILES = ['manifest.json', 'presentation.json', 'outcome.json.gz',
         *[f'decisions/{day:02d}.json' for day in range(30)]]


def _completed(directory, declaration):
    receipt = read_json(directory/'complete.json')
    if receipt['scope'] != digest(declaration) or set(receipt['sha256']) != set(FILES):
        raise ValueError('incomplete or foreign completion receipt')
    if any(file_hash(directory/name) != sha for name, sha in receipt['sha256'].items()):
        raise ValueError('completed file fingerprint changed')
    metadata = read_json(directory/'presentation.json')
    mapping = AssetPresentation(metadata['original_targets'], declaration['episode']['group_id'])
    if mapping.metadata() != metadata:
        raise ValueError('presentation mapping drift')
    artifact = read_json(directory/'outcome.json.gz')
    if artifact['scope'] != receipt['scope'] or artifact['presentation'] != metadata or len(artifact['decisions']) != 30:
        raise ValueError('completed artifact identity changed')
    for day, row in enumerate(artifact['decisions']):
        if row != read_json(directory/f'decisions/{day:02d}.json') or row['day'] != day:
            raise ValueError('decision log mismatch')
        if mapping.frame(row['native_frame']) != row['frame']:
            raise ValueError('recorded presentation differs from native public frame')
        if mapping.to_native_action(row['action'], row['native_frame']) != row['native_action']:
            raise ValueError('recorded physical action mapping changed')
        if row['target_rule_tie'] != public_target_has_tie(row['native_frame']):
            raise ValueError('public target tie diagnostic changed')
    return artifact['result']


def run_presented_engineering_episode(episode, *, detector, journal, transport, directory):
    """Full five-frame LL episode, fresh controller and society-stable aliases."""
    if episode.split != 'engineering':
        raise ValueError('research/test closed until joint model and experiment freeze')
    validate_primary_scenario()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.episode.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [Path(__file__), *[Path(__file__).with_name(name) for name in (
            'combination_requests.py', 'combination_decision.py', 'combination_presentation.py',
            'combination_public.py', 'combination_host.py', 'combination_model_host.py')],
            ROOT/'env/combination.py', ROOT/'env/combination_v51.json']
        declaration = {'version': VERSION, 'episode': episode.to_dict(), 'mode': 'LL', 'window': 5,
                       'presentation': 'society_stable_alias_and_order',
                       'detector_sha256': digest(detector), 'request_contract': journal.declaration,
                       'request_journal_id': journal.identity,
                       'source_sha256': {str(p): file_hash(p) for p in sources}, 'formal_test_allowed': False}
        manifest_path = directory/'manifest.json'
        if manifest_path.exists() and read_json(manifest_path) != declaration:
            raise ValueError('episode source, model or request configuration drift')
        durable_json(manifest_path, declaration)
        if (directory/'complete.json').exists():
            return _completed(directory, declaration)
        scope = digest(declaration)

        def request(payload):
            return journal.request(scope, len(adapter.log), payload, transport)

        adapter = CombinationDecisionAdapter('LL', detector, request, window=5)
        env = CombinationGymEnv(episode, training_reward=False)
        decisions = []
        try:
            _, info = env.reset()
            mapping = AssetPresentation(info['action_targets'], episode.group_id)
            metadata_path = directory/'presentation.json'
            if metadata_path.exists() and read_json(metadata_path) != mapping.metadata():
                raise ValueError('presentation mapping drift')
            durable_json(metadata_path, mapping.metadata())
            for day in range(30):
                display_action = adapter(mapping.info(info))
                native_action = mapping.to_native_action(display_action, info)
                frame = public_frame(info)
                row = {**adapter.log[-1], 'native_frame': frame, 'native_action': native_action,
                       'target_rule_tie': public_target_has_tie(frame)}
                path = directory/f'decisions/{day:02d}.json'
                if path.exists() and read_json(path) != row:
                    raise ValueError('public input, response or action changed during replay')
                durable_json(path, row)
                decisions.append(row)
                _, reward, terminated, truncated, info = env.step(native_action)
                if reward != 0 or truncated or terminated != (day == 29):
                    raise ValueError('evaluation reward/termination contract violated')
                durable_json(directory/'progress.json', {'complete': False, 'scope': scope,
                                                        'completed_market_days': day+1})
            result = env.evaluation_result()
            result['invalid_actions'] = sum(row['invalid_action'] for row in decisions)
            result['model_decisions'] = sum(row['request'] is not None for row in decisions)
            artifact = {'scope': scope, 'presentation': mapping.metadata(), 'result': result,
                        'decisions': decisions,
                        'accounts': {k: [asdict(x) for x in v] for k, v in env._snapshots.items()},
                        'financial_rows': env._rows}
            durable_json(directory/'outcome.json.gz', artifact)
            durable_json(directory/'complete.json', {'scope': scope, 'sha256': {
                name: file_hash(directory/name) for name in FILES}})
            durable_json(directory/'progress.json', {'complete': True, 'scope': scope, 'completed_market_days': 30})
            return result
        except BaseException as exc:
            durable_json(directory/'last_failure.json', {'scope': scope, 'exception_type': type(exc).__name__,
                'committed_decisions': len(list((directory/'decisions').glob('*.json')))})
            raise
        finally:
            env.close()
