"""Fixed Qwen and prompt, 32 new-to-Qwen train societies, paired public baselines."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import os
from pathlib import Path
from dotenv import dotenv_values
from run_combination_qwen_calibration import run_one,selected_episodes,verify_sources,ROOT
from wolfgym.benchmark.combination import episode_plan
from wolfgym.env.combination_contract import validate_primary_scenario
from wolfgym.evaluation.combination_host import read_json,file_hash
from wolfgym.evaluation.combination_public import digest,baseline_specs
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_qwen_pilot import SETTINGS,RESERVATION
from wolfgym.evaluation.combination_qwen_calibration import DICTIONARY_PATH
from wolfgym.evaluation.combination_qwen_dev32 import VERSION,DevelopmentJournal
from wolfgym.evaluation.combination_qwen_recovery import pooled_totals

OLD=ROOT/'data/combination_qwen_calibration_20260928'
BASE=ROOT/'data/combination_baselines_20260926'
POLICIES=('wait','date_8_slot_1','alarm_refresh')


def episode_selection():
    old=read_json(OLD/'manifest.json')
    excluded={e['group_id'] for e in old['episodes']}
    plan=episode_plan('train')
    groups=[g for g in dict.fromkeys(e.group_id for e in plan) if g not in excluded][:32]
    return [e.to_dict() for g in groups for e in plan if e.group_id==g]


def baseline_directory(identity):
    return BASE/identity['split']/str(identity['trajectory_seed'])/identity['condition']


def freeze(output):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'manifest.json'
    if path.exists():
        manifest=read_json(path);verify_sources(manifest)
        if manifest['episodes']!=episode_selection():raise ValueError('frozen sample drift')
        return manifest
    validate_primary_scenario();old=read_json(OLD/'manifest.json');verify_sources(old)
    rows=episode_selection()
    if len(rows)!=64 or len({x['group_id'] for x in rows})!=32:raise ValueError('sample grid')
    selection=read_json(BASE/'report.json')['results']['validation']['selection']
    if selection['date']!='date_8_slot_1':raise ValueError('historical date selection drift')
    sources={**old['source_sha256']}
    paths=[OLD/'manifest.json',OLD/'LL_RECOVERY_REPORT_acceptance.json',BASE/'acceptance.json',BASE/'manifest.json',BASE/'report.json',
        Path(__file__),ROOT/'scripts/report_combination_qwen_dev32.py',
        ROOT/'src/wolfgym/evaluation/combination_qwen_dev32.py',
        ROOT/'src/wolfgym/evaluation/combination_qwen_recovery.py',
        ROOT/'src/wolfgym/evaluation/combination_qwen_incomplete_recovery.py',
        ROOT/'tests/test_combination_qwen_dev32.py',ROOT/'docs/EXPERIMENT_PROTOCOL.md']
    for identity in rows:
        d=baseline_directory(identity);marker=read_json(d/'complete.json')
        if marker['episode']!=identity:raise ValueError('baseline identity mismatch')
        paths.append(d/'complete.json')
        for name in ('reference_wait.json.gz',*[p+'.json.gz' for p in POLICIES]):
            if file_hash(d/name)!=marker['sha256'][name]:raise ValueError('historical comparator drift')
            paths.append(d/name)
    sources.update({str(p.resolve()):file_hash(p) for p in paths})
    jobs=[{'id':f'LL_{i}','mode':'LL','shard':i} for i in range(16)]
    manifest={'version':VERSION,'episodes':rows,'excluded_qwen_groups':sorted({e['group_id'] for e in old['episodes']}),
        'settings':SETTINGS,'dictionary_sha256':file_hash(DICTIONARY_PATH),'jobs':jobs,'source_sha256':sources,
        'policies':POLICIES,'max_requests_total':2048,'max_cost_microusd_total':60000000,
        'reservation_microusd':RESERVATION,'retries':2,'max_concurrent_posts':2,
        'statistics':{'replicates':20000,'society_bootstrap_seed':20260926,'chain_sensitivity_seed':20260927,
            'chain_grouping':'root seed modulo 13, occupied chains; exploratory sensitivity only',
            'comparisons':['LL minus wait','LL minus date_8_slot_1','LL minus alarm_refresh']},
        'scope':'new-to-Qwen train societies; previously used baseline development worlds; not independent test',
        'formal_test_allowed':False,'existing_exhausted_requests_excluded':True,
        'user_authorization':'2026-09-30 user selected the proposed fixed-Qwen 32 new development societies/64 episodes with WAIT/date/public rule; proposed new cap USD60 applies'}
    # Normalize tuples to JSON before computing execution identities.
    import json
    manifest=json.loads(json.dumps(manifest));durable_json(path,manifest);return manifest


def run_job(args):
    output,manifest,job=args;output=Path(output);verify_sources(manifest)
    key=dotenv_values(ROOT/'.env').get('OPENROUTER_API_KEY')
    if not key:raise ValueError('missing provider credential')
    with DevelopmentJournal(output,manifest,job) as journal:
        for identity in selected_episodes(manifest,job):run_one(output,identity,manifest,job,journal,key)
        durable_json(output/(job['id']+'_complete.json'),{'complete':True,'totals':journal.totals()})


def main(output,phase):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);manifest=freeze(output)
        if phase=='freeze':return
        if (output/'acceptance.json').exists():raise ValueError('already accepted, do not rerun')
        if (output/'incomplete_recovery_stop.json').exists():raise ValueError('stopped incident requires diagnosis')
        try:
            durable_json(output/'progress.json',{'pid':os.getpid(),'stage':'Qwen_LL','complete':False})
            with ProcessPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run_job,(str(output),manifest,j)) for j in manifest['jobs']]
                for f in as_completed(futures):
                    f.result()
                    durable_json(output/'progress.json',{'pid':os.getpid(),'stage':'Qwen_LL','complete':False,
                        'completed_episodes':len(list((output/'LL/episodes').glob('*/complete.json'))),**pooled_totals(output,manifest)})
            from report_combination_qwen_dev32 import audit
            durable_json(output/'progress.json',{'pid':os.getpid(),'stage':'full_audit','complete':False,**pooled_totals(output,manifest)})
            audit(output,manifest)
            durable_json(output/'progress.json',{'pid':os.getpid(),'complete':True,'audit_complete':True,**pooled_totals(output,manifest)})
        except BaseException as exc:
            durable_json(output/'progress.json',{'pid':os.getpid(),'stage':'stopped','complete':False,
                'exception_type':type(exc).__name__,**pooled_totals(output,manifest)})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('freeze','run'),required=True);a=p.parse_args();main(a.output.resolve(),a.phase)
