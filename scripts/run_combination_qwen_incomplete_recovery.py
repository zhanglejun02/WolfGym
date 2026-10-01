"""Continue LR/RL after the diagnosed empty-response incident; no LL rerun."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path

from dotenv import dotenv_values
from run_combination_qwen_calibration import ROOT,run_one,selected_episodes,verify_sources
from run_combination_qwen_recovery import verify_amendment
from wolfgym.evaluation.combination_host import read_json,file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_recovery import ledger_snapshot,verify_prefix,pooled_totals
from wolfgym.evaluation.combination_qwen_incomplete_recovery import VERSION,IncompleteRecoveryJournal,incomplete_receipt


def verify_supplement(output,manifest,amendment,supplement):
    verify_amendment(output,manifest,amendment)
    if (supplement['version']!=VERSION or supplement['max_attempts_total']!=1008 or
        supplement['max_cost_microusd_total']!=30000000 or supplement['max_attempts_per_request']!=3 or
        supplement['original_amendment_sha256']!=file_hash(output/'recovery_amendment.json') or
        supplement['acknowledged_stop_sha256']!=file_hash(output/'recovery_stop.json')):
        raise ValueError('supplement authority/incident drift')
    for p,h in supplement['source_sha256'].items():
        if file_hash(p)!=h:raise ValueError('supplement source/evidence drift: '+p)
    verify_prefix(output,supplement)


def freeze(output,manifest,amendment):
    p=output/'incomplete_recovery_supplement.json'
    if p.exists():
        supplement=read_json(p);verify_supplement(output,manifest,amendment,supplement);return supplement
    stop=read_json(output/'recovery_stop.json')
    if stop!={'job':'LR_1','reason':'non-network/provider validation failure; reservation retained'}:
        raise ValueError('not the diagnosed incident')
    old={j['id']:ledger_snapshot(output/(j['id']+'.sqlite')) for j in manifest['jobs']}
    if any(r['state'] not in ('success','fatal') for x in old.values() for r in x['attempts']):
        raise ValueError('pending/overrun requires separate reconciliation')
    incident=next(r for r in old['LR_1']['attempts'] if r['id']==82)
    evidence=output/'recovery_attempts/LR_1/82'/f'{digest(__import__("json").loads(incident["payload"]))}.json'
    if incident['state']!='fatal' or incident['charged']!=110000 or not incomplete_receipt(evidence):
        raise ValueError('incident evidence mismatch')
    acceptance=read_json(output/'LL_RECOVERY_REPORT_acceptance.json')
    if not acceptance['complete'] or acceptance['new_episodes']!=16:raise ValueError('LL not accepted')
    for n,h in acceptance['outputs_sha256'].items():
        if file_hash(output/n)!=h:raise ValueError('LL receipt drift')
    paths=[Path(__file__),ROOT/'scripts/report_combination_qwen_incomplete_recovery.py',
        ROOT/'src/wolfgym/evaluation/combination_qwen_incomplete_recovery.py',
        ROOT/'tests/test_combination_qwen_incomplete_recovery.py',
        ROOT/'docs/EXPERIMENT_PROTOCOL.md',
        output/'recovery_stop.json',evidence,output/'provider_incident_LR_1_82_lookup.json',
        output/'LL_RECOVERY_REPORT_acceptance.json']
    supplement={'version':VERSION,'original_amendment_sha256':file_hash(output/'recovery_amendment.json'),
        'acknowledged_stop_sha256':file_hash(output/'recovery_stop.json'),
        'authorization':'Existing user authorization: failed requests at most two supplemental attempts within original $30/1008, success never reissued, unknown fees reserved. User asks to fix missing usage/body response.',
        'max_attempts_total':1008,'max_cost_microusd_total':30000000,'max_attempts_per_request':3,
        'old_stop_retained':True,'original_ledgers':old,'starting_totals':pooled_totals(output,manifest),
        'source_sha256':{str(p.resolve()):file_hash(p) for p in paths},'formal_test_allowed':False}
    durable_json(p,supplement);verify_supplement(output,manifest,amendment,supplement);return supplement


def run_job(args):
    output,manifest,job,amendment,supplement=args;output=Path(output)
    verify_sources(manifest);verify_supplement(output,manifest,amendment,supplement)
    key=dotenv_values(ROOT/'.env').get('OPENROUTER_API_KEY')
    if not key:raise ValueError('missing provider credential')
    with IncompleteRecoveryJournal(output,manifest,job,amendment,supplement) as journal:
        for identity in selected_episodes(manifest,job):run_one(output,identity,manifest,job,journal,key)
        durable_json(output/(job['id']+'_incomplete_recovery_complete.json'),
                     {'complete':True,'job':job['id'],'totals':journal.totals(),'supplement':digest(supplement)})


def main(output,phase):
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=read_json(output/'manifest.json');amendment=read_json(output/'recovery_amendment.json')
        verify_sources(manifest);supplement=freeze(output,manifest,amendment)
        if phase=='freeze':return
        if (output/'INCOMPLETE_RECOVERY_REPORT_acceptance.json').exists():raise ValueError('already accepted')
        if (output/'incomplete_recovery_stop.json').exists():raise ValueError('new incident stop requires diagnosis')
        try:
            durable_json(output/'progress.json',{'pid':os.getpid(),'stage':'incomplete_recovery_LR_RL',
                'complete':False,'LL_audit_complete':True,**pooled_totals(output,manifest)})
            with ProcessPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run_job,(str(output),manifest,j,amendment,supplement))
                         for j in manifest['jobs'] if j['mode'] in ('LR','RL')]
                for future in as_completed(futures):future.result()
            from report_combination_qwen_incomplete_recovery import audit
            audit(output,manifest,amendment,supplement,('LL','LR','RL'))
            durable_json(output/'progress.json',{'pid':os.getpid(),'complete':True,'audit_complete':True,**pooled_totals(output,manifest)})
        except BaseException as exc:
            durable_json(output/'progress.json',{'pid':os.getpid(),'stage':'incomplete_recovery_stopped',
                'complete':False,'audit_complete':False,'exception_type':type(exc).__name__,**pooled_totals(output,manifest)})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('freeze','run'),required=True);args=p.parse_args()
    main(args.output.resolve(),args.phase)
