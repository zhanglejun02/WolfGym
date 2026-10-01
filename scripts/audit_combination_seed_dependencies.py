"""Read-only seed dataflow audit plus fixed validation-only CI sensitivity.

No simulator construction, training, test outcomes, or existing report writes.
The chain bootstrap is a sensitivity calculation with only 13 clusters, not a
certificate of coverage or a replacement independent confirmation experiment.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPLICATES = 20000
ANALYSIS_SEED = 20260927


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def surviving_edges(roots):
    roots = set(roots)
    return [(s, s+13) for s in sorted(roots) if s+13 in roots]


def chain_sensitivity(seeds, values):
    seeds, values = np.asarray(seeds), np.asarray(values, dtype=float)
    if len(seeds) != len(values) or len(set(seeds.tolist())) != len(seeds) or not np.all(np.isfinite(values)):
        raise ValueError('one finite paired value per unique root required')
    order = np.argsort(seeds)
    seeds, values = seeds[order], values[order]
    if len(seeds) < 26 or np.any(np.diff(seeds) != 1):
        raise ValueError('sensitivity requires a complete consecutive root block')
    groups = [values[seeds % 13 == r] for r in range(13)]
    totals = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(ANALYSIS_SEED)
    indices = rng.integers(0, 13, size=(REPLICATES, 13))
    # Resample complete dependency chains, keeping all conditions/policies within
    # each already-paired societal value. Unequal chain sizes retain root weights.
    samples = totals[indices].sum(axis=1)/counts[indices].sum(axis=1)
    iid_indices = np.random.default_rng(ANALYSIS_SEED).integers(0,len(values),size=(REPLICATES,len(values)))
    iid = values[iid_indices].mean(axis=1)
    lag_corr = None if np.std(values[:-13]) == 0 or np.std(values[13:]) == 0 else float(np.corrcoef(values[:-13],values[13:])[0,1])
    return {'mean_pp': float(values.mean()), 'root_count':len(values), 'chain_count':13,
            'chain_sizes': counts.tolist(), 'seed_lag_13_correlation_descriptive':lag_corr,
            'chain_bootstrap_ci95_pp': np.quantile(samples,[.025,.975]).tolist(),
            'iid_bootstrap_sensitivity_ci95_pp': np.quantile(iid,[.025,.975]).tolist(),
            'replicates':REPLICATES,'analysis_seed':ANALYSIS_SEED,
            'coverage_certified':False}


def audit(output):
    output.mkdir(parents=True,exist_ok=False)
    splits_path=ROOT/'src/wolfgym/benchmark/combination_splits_v1.json'
    splits=json.loads(splits_path.read_text())['splits']
    address_counts={}
    stream_equality={}
    for name,block in splits.items():
        roots=range(block['start'],block['start']+block['count'])
        edges=surviving_edges(roots)
        address_counts[name]=len(edges)
        # Mirrors split_rng's first integer and SocialGraph's integer draw only;
        # reads test seed metadata, never creates a test world or reads outcomes.
        stream_equality[name]=all(
            int(np.random.default_rng(s+13).integers(0,2**31-1)) ==
            int(np.random.default_rng(t).integers(0,2**31-1,size=3)[0]) for s,t in edges)
    input_path=ROOT/'data/combination_baselines_20260926/audited_rows.json.gz'
    acceptance=ROOT/'data/combination_baselines_20260926/acceptance.json'
    expected=json.loads(acceptance.read_text())['outputs_sha256'][str(input_path.relative_to(ROOT))]
    if fingerprint(input_path)!=expected:
        raise ValueError('accepted baseline rows changed')
    with gzip.open(input_path,'rt') as stream:
        rows=json.load(stream)
    selected=defaultdict(dict)
    for row in rows:
        if row['split']=='validation' and row['policy_id'] in ('alarm_refresh','date_8_slot_1'):
            key=row['trajectory_seed'],row['policy_id']
            if row['condition'] in selected[key]:raise ValueError('duplicate paired result')
            selected[key][row['condition']]=row['benefit']
    block=splits['validation']
    seeds=list(range(block['start'],block['start']+block['count']))
    vectors={}
    for policy in ('alarm_refresh','date_8_slot_1'):
        pairs=[selected[s,policy] for s in seeds]
        if any(set(p)!= {'normal','attack'} for p in pairs):raise ValueError('incomplete validation pairs')
        vectors[policy]=np.array([50*(p['normal']+p['attack']) for p in pairs])
    vectors['alarm_refresh_minus_date_8_slot_1']=vectors['alarm_refresh']-vectors['date_8_slot_1']
    sources=[splits_path,input_path,acceptance,Path(__file__),*[ROOT/'src/wolfgym'/p for p in (
        'core/env/environment.py','core/env/market.py','core/env/social.py',
        'core/scenarios/society.py','core/utils/seeds.py','core/agents/retail.py',
        'core/agents/switching.py','env/combination.py','evaluation/task_validation.py')]]
    result={'version':'seed-dependency-readonly-audit-v1',
            'retained_cross_root_graph_retail_edges':address_counts,
            'shared_31bit_seed_verified':stream_equality,
            'discarded_direct_roles':{'7':'market initial state deterministic; generator replaced before day 1',
                                     '23':'initial event key unused before continuation/normal reseed'},
            'retained_dependency':'root s graph seed equals root s+13 retail initializer seed; different graph/retail algorithms',
            'validation_sensitivity':{p:chain_sensitivity(seeds,v) for p,v in vectors.items()},
            'source_sha256':{str(p.relative_to(ROOT)):fingerprint(p) for p in sources},
            'market_days_run':0,'test_outcomes_read':False,'training_rerun':False,
            'limitations':['Address equality does not measure outcome dependence.',
                          '13-chain bootstrap has few clusters; no coverage guarantee.',
                          'Validation-selected policies are not independent confirmation.',
                          'No inference of independence from a small observed lag correlation.'],
            'decision':'retain primary market and roots; carry dependency disclosure and chain sensitivity into pre-test analysis addendum'}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('retained_cross_root_graph_retail_edges','shared_31bit_seed_verified','validation_sensitivity')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
