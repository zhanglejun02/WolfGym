"""One-model, train-only 16-episode pilot; four disjoint bounded request journals."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import sqlite3

from dotenv import dotenv_values
from report_combination_baselines import wealth_table, independent_score, check_score
from wolfgym.benchmark.combination import CombinationEpisode, episode_plan
from wolfgym.env.combination_fast import FastCombinationGymEnv
from wolfgym.env.combination_contract import primary_contract, validate_primary_scenario
from wolfgym.evaluation.combination_host import read_json, file_hash, aggregate_financial
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_requests import RequestJournal
from wolfgym.evaluation.combination_qwen_pilot import (
    MODEL, CANONICAL, VERSION, SETTINGS, RESERVATION, PilotAdapter, QwenTransport, decode_reply,
)
ROOT = Path(__file__).resolve().parents[1]


def episodes():
    groups = list(dict.fromkeys(e.group_id for e in episode_plan('train')))[:8]
    return [e.to_dict() for g in groups for e in episode_plan('train') if e.group_id == g]


def freeze(output):
    validate_primary_scenario()
    catalog = read_json(output/'model_catalog.json')
    matches = [x for x in catalog if x['id'] == MODEL]
    if len(matches) != 1 or matches[0]['canonical_slug'] != CANONICAL:
        raise ValueError('model catalog identity mismatch')
    endpoints = read_json(output/'model_endpoints.json')['data']
    candidates = [x for x in endpoints['endpoints'] if x['tag'] == 'alibaba']
    if endpoints['id'] != MODEL or len(candidates) != 1:
        raise ValueError('provider metadata mismatch')
    from decimal import Decimal
    pricing = candidates[0]['pricing']
    for name, limit in [('prompt', '0.000003'), ('completion', '0.000006'), ('input_cache_write', '0.000003')]:
        if Decimal(str(pricing.get(name, 0))) > Decimal(limit):
            raise ValueError('metadata price exceeds declared bound')
    sources = [Path(__file__), ROOT/'scripts/report_combination_baselines.py',
               ROOT/'docs/EXPERIMENT_PROTOCOL.md', ROOT/'tests/test_combination_qwen_pilot.py']
    sources += [ROOT/'src/wolfgym'/p for p in ('evaluation/combination_qwen_pilot.py',
        'evaluation/combination_requests.py', 'evaluation/combination_model_host.py',
        'evaluation/combination_decision.py', 'evaluation/combination_public.py', 'evaluation/combination_host.py',
        'env/combination.py', 'env/combination_fast.py', 'observations/fast_messages.py', 'regulation/executor.py',
        'env/combination_contract.py', 'env/combination_v51.json', 'benchmark/combination.py', 'benchmark/combination_splits_v1.json')]
    sources += [ROOT/'src/wolfgym'/p for p in primary_contract()['source_files']]
    sources += [output/'model_catalog.json', output/'model_endpoints.json']
    manifest = {'version': VERSION, 'episodes': episodes(), 'model': MODEL, 'canonical_slug': CANONICAL,
                'settings': SETTINGS, 'source_sha256': {str(p.resolve()): file_hash(p) for p in sources},
                'max_requests_total': 480, 'max_cost_microusd_total': 10000000,
                'shards': 4, 'per_shard_requests': 120, 'per_shard_microusd': 2500000,
                'reservation_microusd': RESERVATION, 'retries': 0, 'formal_test_allowed': False}
    path = output/'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('pilot declaration/source drift')
    durable_json(path, manifest)
    return manifest


def run_one(output, identity, manifest, journal, key, stop_after=None):
    if identity not in manifest['episodes'] or identity['split'] != 'train':
        raise ValueError('only the declared 16 train episodes may run')
    e = CombinationEpisode(identity['split'], identity['trajectory_seed'], identity['condition'])
    directory = output/'episodes'/identity['episode_id'].replace(':', '_'); directory.mkdir(parents=True, exist_ok=True)
    scope = digest([digest(manifest), identity])
    complete = directory/'complete.json'
    if complete.exists():
        c = read_json(complete)
        if c['scope'] != scope or any(file_hash(directory/p) != sha for p, sha in c['sha256'].items()):
            raise ValueError('completed pilot episode drift')
        return read_json(directory/'outcome.json.gz')['result']
    transport = QwenTransport(key, directory/'responses')
    def request(payload): return journal.request(scope, len(adapter.log), payload, transport)
    adapter = PilotAdapter(request); env = FastCombinationGymEnv(e, message_mode='indexed')
    try:
        _, info = env.reset()
        for day in range(30):
            action = adapter(info); log = adapter.log[-1]
            p = directory/'decisions'/f'{day:02}.json'
            if p.exists() and read_json(p) != log: raise ValueError('decision replay drift')
            durable_json(p, log)
            _, reward, done, truncated, info = env.step(action)
            if reward != 0 or truncated or done != (day == 29): raise ValueError('hidden feedback violation')
            durable_json(directory/'progress.json', {'completed_days': day+1, 'complete': False, **journal.totals()})
            print(f"{identity['episode_id']} day={day+1} action={action} calls={journal.totals()['attempts']}", flush=True)
            if stop_after == day+1: return None
        result = env.evaluation_result()
        result['invalid_actions'] = sum(x['invalid_action'] for x in adapter.log)
        result['model_decisions'] = sum(x['request'] is not None for x in adapter.log)
        artifact = {'scope': scope, 'result': result, 'decisions': adapter.log,
                    'accounts': {k: [asdict(x) for x in v] for k, v in env._snapshots.items()}, 'financial_rows': env._rows}
        durable_json(directory/'outcome.json.gz', artifact)
        paths = [directory/'outcome.json.gz', *sorted((directory/'decisions').glob('*.json')),
                 *sorted((directory/'responses').glob('*.json'))]
        durable_json(complete, {'scope': scope, 'sha256': {str(p.relative_to(directory)): file_hash(p) for p in paths}})
        durable_json(directory/'progress.json', {'completed_days': 30, 'complete': True})
        return result
    finally:
        env.close()


def shard(job):
    output, manifest, index, probe = job; output = Path(output)
    groups = list(dict.fromkeys(e['group_id'] for e in manifest['episodes']))
    selected = groups[index*2:index*2+2]
    identities = [e for e in manifest['episodes'] if e['group_id'] in selected]
    key = dotenv_values(ROOT/'.env').get('OPENROUTER_API_KEY')
    if not key: raise ValueError('local OpenRouter credential missing')
    with RequestJournal(output/f'requests_{index}.sqlite', contract={'manifest_sha256': digest(manifest), 'shard': index},
                        max_attempts=120, max_cost_microusd=2500000, reservation_microusd=RESERVATION, retries=0) as journal:
        results = []
        for e in identities[:1] if probe else identities:
            results.append(run_one(output, e, manifest, journal, key, stop_after=1 if probe else None))
        return {'shard': index, 'totals': journal.totals(), 'results': results}


def audit(output, manifest):
    from math import isclose
    rows = []; replay_calls = 0; response_cost = 0; input_tokens = output_tokens = reasoning_tokens = 0
    baseline_dir = ROOT/'data/combination_action_diagnostic_20260928'
    baseline_acceptance = read_json(baseline_dir/'acceptance.json')
    for p, sha in baseline_acceptance['outputs_sha256'].items():
        if file_hash(baseline_dir/p) != sha: raise ValueError('historical comparator drift')
    for identity in manifest['episodes']:
        d = output/'episodes'/identity['episode_id'].replace(':', '_')
        marker = read_json(d/'complete.json')
        if marker['scope'] != digest([digest(manifest), identity]): raise ValueError('scope mismatch')
        for p, sha in marker['sha256'].items():
            if file_hash(d/p) != sha: raise ValueError('artifact integrity mismatch')
        a = read_json(d/'outcome.json.gz'); r = a['result']; accounts = a['accounts']; fin = a['financial_rows']
        if any(r[k] != v for k, v in identity.items()): raise ValueError('identity mismatch')
        bdir = baseline_dir/identity['episode_id'].replace(':', '_'); bmarker = read_json(bdir/'complete.json')
        for name in ('reference_wait.json.gz', 'rule.json.gz'):
            if file_hash(bdir/name) != bmarker['sha256'][name]: raise ValueError('baseline account drift')
        old = read_json(bdir/'reference_wait.json.gz')
        if any(accounts[k] != old['accounts'][k] or fin[k] != old['financial_rows'][k] for k in ('reference', 'wait')):
            raise ValueError('historical R/WAIT pairing failed')
        ref, initial = wealth_table(accounts['reference'], accounts['reference']); scores = {}
        for world in ('wait', 'policy'):
            values, _ = wealth_table(accounts[world], accounts['reference'])
            scores[world] = independent_score(ref, values, initial, fin['reference'], fin[world], primary_contract()['harm_threshold'])
            check_score(scores[world], r[world])
        benefit = float(scores['wait']['H'][-1]-scores['policy']['H'][-1])
        if not isclose(benefit, r['H_reduction'], abs_tol=1e-12): raise ValueError('benefit mismatch')
        remaining = 12
        def reply(payload):
            nonlocal replay_calls, response_cost, input_tokens, output_tokens, reasoning_tokens
            raw = read_json(d/'responses'/f'{digest(payload)}.json'); decoded = decode_reply(raw)
            replay_calls += 1; response_cost += decoded.cost_microusd
            input_tokens += decoded.usage['input_tokens']; output_tokens += decoded.usage['output_tokens']
            reasoning_tokens += decoded.usage.get('reasoning_tokens', 0)
            return decoded.text
        if len(a['decisions']) != 30: raise ValueError('incomplete decisions')
        action_days = []
        for day, log in enumerate(a['decisions']):
            f = log['frame']
            if f['day'] != day or f['remaining_budget'] != remaining or f['action_mask'] != [1, *([int(remaining >= 4)]*3)]:
                raise ValueError('budget/day mismatch')
            # Adapter consumes the original public observation schema; rebuild an equivalent
            # info from the source-of-truth saved request view is unnecessary: replay parser,
            # request, public history and legality directly on the canonical saved frame.
            from wolfgym.evaluation.combination_public import PublicHistory
            from wolfgym.evaluation.combination_qwen_pilot import pilot_request
            from wolfgym.evaluation.combination_decision import parse_response
            if day == 0: history = PublicHistory(5)
            view = history.observe(f); request = pilot_request(view) if remaining >= 4 else None
            if request != log['request'] or digest(view) != log['input_sha256']: raise ValueError('public request mismatch')
            action = 0; error = None
            if request is not None:
                text = reply(request)
                if text != log['response']: raise ValueError('provider reply mismatch')
                try: action = parse_response(text, 'LL', f)
                except (ValueError, TypeError, RecursionError) as exc: error = type(exc).__name__+': '+str(exc)
            if action != log['action'] or error != log['error'] or bool(error) != log['invalid_action'] or log['cost'] != 4*bool(action):
                raise ValueError('model action replay mismatch')
            history.record(f, action)
            if action: remaining -= 4; action_days.append(day)
        if r['cost'] != 12-remaining: raise ValueError('admin cost mismatch')
        if r['invalid_actions'] != sum(x['invalid_action'] for x in a['decisions']) or r['model_decisions'] != sum(x['request'] is not None for x in a['decisions']):
            raise ValueError('decision count mismatch')
        financial = {k: aggregate_financial(v) for k, v in fin.items()}
        if financial != r['financial_outcomes']: raise ValueError('financial mismatch')
        teacher = read_json(bdir/'rule.json.gz')['result']
        rows.append({**identity, 'benefit_pp': 100*benefit, 'teacher_pp': 100*teacher['H_reduction'],
            'H_wait_pp': 100*float(scores['wait']['H'][-1]), 'H_policy_pp': 100*float(scores['policy']['H'][-1]),
            'event_wait': scores['wait']['event'], 'event_policy': scores['policy']['event'],
            'cost': r['cost'], 'action_days': action_days, 'invalid_actions': r['invalid_actions'],
            'model_decisions': r['model_decisions'], 'financial_delta': {k: financial['policy'][k]-financial['wait'][k] for k in financial['policy']}})
    totals = []; latency = []
    for i in range(4):
        db = sqlite3.connect(f'file:{output}/requests_{i}.sqlite?mode=ro', uri=True); db.row_factory = sqlite3.Row
        xs = [dict(x) for x in db.execute('SELECT * FROM attempts')]; db.close()
        if any(x['state'] != 'success' for x in xs): raise ValueError('unresolved provider attempts')
        totals += xs; latency += [x['elapsed_seconds'] for x in xs]
    if len(totals) != replay_calls or sum(x['charged'] for x in totals) != response_cost:
        raise ValueError('response/journal cost mismatch')
    for x in totals:
        import json
        saved = json.loads(x['response']); payload = json.loads(x['payload'])
        matches = list((output/'episodes').glob(f'*/responses/{digest(payload)}.json'))
        if not any(decode_reply(read_json(p)).text == saved['text'] and decode_reply(read_json(p)).cost_microusd == saved['cost_microusd'] for p in matches):
            raise ValueError('journal not matched to provider response')
    import numpy as np
    summary = {'balanced_benefit_pp': float(np.mean([x['benefit_pp'] for x in rows])),
               'teacher_balanced_pp': float(np.mean([x['teacher_pp'] for x in rows]))}
    for condition in ('normal', 'attack'):
        cr = [x for x in rows if x['condition'] == condition]
        summary[condition] = {'benefit_pp': float(np.mean([x['benefit_pp'] for x in cr])),
            'teacher_pp': float(np.mean([x['teacher_pp'] for x in cr])),
            'events_wait': sum(x['event_wait'] for x in cr), 'events_policy': sum(x['event_policy'] for x in cr),
            'prevented': sum(x['event_wait'] and not x['event_policy'] for x in cr),
            'added': sum(not x['event_wait'] and x['event_policy'] for x in cr),
            'intervened_episodes': sum(bool(x['action_days']) for x in cr),
            'mean_admin_cost': float(np.mean([x['cost'] for x in cr])),
            'mean_financial_delta': {k: float(np.mean([x['financial_delta'][k] for x in cr])) for k in cr[0]['financial_delta']}}
    report = {'model': MODEL, 'scope': 'train cost/format pilot; 8 paired societies, no formal ranking or independent test',
              'summary': summary, 'usage': {'requests': replay_calls, 'cost_usd_rounded_up_per_request': response_cost/1e6,
                'input_tokens': input_tokens, 'output_tokens': output_tokens, 'reasoning_tokens': reasoning_tokens,
                'latency_median_seconds': float(np.median(latency)), 'latency_p95_seconds': float(np.quantile(latency,.95)),
                'invalid_actions': sum(x['invalid_actions'] for x in rows)},
              'audit': {'episodes': 16, 'account_closes': 16*3*30*490, 'decisions': 480, 'historical_pairs': 16}, 'rows': rows}
    durable_json(output/'report.json', report)
    durable_json(output/'acceptance.json', {'manifest_sha256': file_hash(output/'manifest.json'),
                 'report_sha256': file_hash(output/'report.json'), 'episode_receipts': {
                 str(p.relative_to(output)): file_hash(p) for p in (output/'episodes').glob('*/complete.json')}})
    return report


def main(output, phase):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = freeze(output)
        if (output/'acceptance.json').exists(): raise ValueError('pilot already accepted; do not repeat')
        if phase == 'probe':
            shard((str(output), manifest, 0, True)); return
        durable_json(output/'progress.json', {'pid': os.getpid(), 'complete': False, 'stage': 'running'})
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(shard, (str(output), manifest, i, False)) for i in range(4)]
            for future in as_completed(futures):
                result = future.result(); durable_json(output/f'shard_{result["shard"]}_complete.json', result)
        for p, sha in manifest['source_sha256'].items():
            if file_hash(p) != sha: raise ValueError('source drift during execution')
        report = audit(output, manifest)
        durable_json(output/'progress.json', {'complete': True, 'audit_complete': True, 'requests': report['usage']['requests']})
        print('Pilot complete and audited', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('probe','run'), required=True); a = p.parse_args(); main(a.output.resolve(), a.phase)
