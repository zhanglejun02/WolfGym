"""Continue the original 64 episodes after two bounded diagnostic requests succeed."""
import argparse
import fcntl
import os
from pathlib import Path
from dotenv import dotenv_values
from run_combination_qwen_calibration import ROOT,verify_sources,selected_episodes,run_one
from wolfgym.evaluation.combination_host import read_json,file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_recovery import pooled_totals,verify_prefix
from wolfgym.evaluation.combination_qwen_network_retry import NetworkRetryJournal
from report_combination_qwen_network_continuation import audit


def main(output):
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read_json(output/'manifest.json');s=read_json(output/'network_retry_supplement.json')
        verify_sources(m);verify_sources(s);verify_prefix(output,s)
        probes=read_json(output/'network_retry_result.json')['probes']
        if len(probes)!=2 or not all(p['success'] for p in probes):raise ValueError('diagnostic probes not successful')
        p=output/'network_continuation_manifest.json'
        if not p.exists():
            files=[Path(__file__).resolve(),ROOT/'scripts/report_combination_qwen_network_continuation.py',
                   ROOT/'tests/test_combination_qwen_network_continuation.py']
            durable_json(p,{'original_manifest':digest(m),'supplement_sha256':file_hash(output/'network_retry_supplement.json'),
                'source_sha256':{str(f):file_hash(f) for f in files},'workers':1,
                'scope':'resume original fixed 64 episodes, serial requests; unchanged global budget and count',
                'formal_test_allowed':False})
        c=read_json(p);verify_sources(c)
        if c['original_manifest']!=digest(m) or c['supplement_sha256']!=file_hash(output/'network_retry_supplement.json'):raise ValueError('continuation contract drift')
        if (output/'network_retry_stop.json').exists():raise ValueError('diagnostic continuation stopped')
        if (output/'acceptance.json').exists():raise ValueError('already accepted')
        key=dotenv_values(ROOT/'.env').get('OPENROUTER_API_KEY')
        if not key:raise ValueError('missing credential')
        def progress(stage):
            durable_json(output/'progress.json',{'pid':os.getpid(),'stage':stage,'complete':False,
                'completed_episodes':len(list((output/'LL/episodes').glob('*/complete.json'))),**pooled_totals(output,m)})
        try:
            progress('Qwen_LL_serial_network_recovery')
            for job in m['jobs']:
                with NetworkRetryJournal(output,m,job,s) as journal:
                    for identity in selected_episodes(m,job):
                        run_one(output,identity,m,job,journal,key);progress('Qwen_LL_serial_network_recovery')
            progress('full_audit');audit(output,m)
            durable_json(output/'progress.json',{'pid':os.getpid(),'complete':True,'audit_complete':True,**pooled_totals(output,m)})
        except BaseException as exc:
            progress('stopped');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
