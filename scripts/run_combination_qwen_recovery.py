"""Resume calibrated LL then LR/RL under the user's explicit bounded retry amendment."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path

from dotenv import dotenv_values
from run_combination_qwen_calibration import ROOT, run_one, selected_episodes, verify_sources
from wolfgym.evaluation.combination_host import read_json, file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_recovery import (
    VERSION, RecoveryJournal, ledger_snapshot, pooled_totals, verify_prefix)


def verify_amendment(output,manifest,amendment):
    if (amendment['version']!=VERSION or amendment['manifest_sha256']!=file_hash(output/'manifest.json') or
        amendment['max_attempts_total']!=1008 or amendment['max_cost_microusd_total']!=30000000 or
        amendment['supplemental_attempts_per_failed_request']!=2 or
        manifest['max_requests_total']!=1008 or manifest['max_cost_microusd_total']!=30000000):
        raise ValueError('recovery authority/limits mismatch')
    for p,sha in amendment['source_sha256'].items():
        if file_hash(p)!=sha: raise ValueError('recovery source drift: '+p)
    verify_prefix(output,amendment)


def freeze(output,manifest):
    path=output/'recovery_amendment.json'
    if path.exists():
        amendment=read_json(path);verify_amendment(output,manifest,amendment);return amendment
    paths=[Path(__file__),ROOT/'scripts/report_combination_qwen_recovery.py',
        ROOT/'src/wolfgym/evaluation/combination_qwen_recovery.py',ROOT/'tests/test_combination_qwen_recovery.py',
        ROOT/'docs/EXPERIMENT_PROTOCOL.md']
    old={j['id']:ledger_snapshot(output/(j['id']+'.sqlite')) for j in manifest['jobs'] if (output/(j['id']+'.sqlite')).exists()}
    if any(r['state'] not in ('success','fatal') for x in old.values() for r in x['attempts']):
        raise ValueError('unresolved pre-amendment state')
    amendment={'version':VERSION,'manifest_sha256':file_hash(output/'manifest.json'),
        'user_authorization':'2026-09-29: failed requests may be supplemented at most twice within original $30 and total request limit; never reissue success; fully reserve unknown charges',
        'max_attempts_total':1008,'max_cost_microusd_total':30000000,
        'supplemental_attempts_per_failed_request':2,'max_concurrent_posts':2,'retry_delays_seconds':[5,15],
        'allocation':'original shard allocations become historical; all nine jobs share the unchanged global pool, counting every original and supplemental POST',
        'original_ledgers':old,'starting_totals':pooled_totals(output,manifest),
        'source_sha256':{str(p.resolve()):file_hash(p) for p in paths},'formal_test_allowed':False}
    durable_json(path,amendment);verify_amendment(output,manifest,amendment);return amendment


def run_job(args):
    output,manifest,job,amendment=args;output=Path(output)
    verify_sources(manifest);verify_amendment(output,manifest,amendment)
    key=dotenv_values(ROOT/'.env').get('OPENROUTER_API_KEY')
    if not key: raise ValueError('missing provider credential')
    with RecoveryJournal(output,manifest,job,amendment) as journal:
        for identity in selected_episodes(manifest,job):
            run_one(output,identity,manifest,job,journal,key)
        value={'job':job['id'],'complete':True,'totals':journal.totals(),'amendment':digest(amendment)}
        durable_json(output/(job['id']+'_recovery_complete.json'),value)
    return value


def main(output,phase):
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=read_json(output/'manifest.json');verify_sources(manifest)
        amendment=freeze(output,manifest)
        if phase=='freeze':return
        if (output/'RECOVERY_REPORT_acceptance.json').exists():raise ValueError('already accepted; do not repeat')
        from report_combination_qwen_recovery import audit
        try:
            for modes in (('LL',),('LR','RL')):
                durable_json(output/'progress.json',{'pid':os.getpid(),'complete':False,'stage':'recovery_'+'+'.join(modes),
                    'recovery_amendment_sha256':file_hash(output/'recovery_amendment.json'),**pooled_totals(output,manifest)})
                with ProcessPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(run_job,(str(output),manifest,j,amendment)) for j in manifest['jobs'] if j['mode'] in modes]
                    for future in as_completed(futures):future.result()
                audited_modes=('LL',) if modes==('LL',) else ('LL','LR','RL')
                report=audit(output,manifest,amendment,audited_modes)
                print('STAGE_AUDITED '+str(audited_modes)+' '+str(report['summary']),flush=True)
            durable_json(output/'progress.json',{'pid':os.getpid(),'complete':True,'audit_complete':True,**pooled_totals(output,manifest)})
        except BaseException as exc:
            durable_json(output/'progress.json',{'pid':os.getpid(),'complete':False,'audit_complete':False,
                'stage':'recovery_stopped','exception_type':type(exc).__name__,**pooled_totals(output,manifest)})
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('freeze','run'),required=True)
    args=parser.parse_args();main(args.output.resolve(),args.phase)
