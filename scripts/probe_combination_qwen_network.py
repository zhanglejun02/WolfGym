"""Retry exactly the two diagnosed exhausted requests, once each, preserving history."""
import argparse
import fcntl
import json
from pathlib import Path
from dotenv import dotenv_values
from run_combination_qwen_calibration import ROOT,verify_sources,selected_episodes,run_one
from wolfgym.evaluation.combination_host import read_json,file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_recovery import ledger_snapshot,pooled_totals,verify_prefix
from wolfgym.evaluation.combination_qwen_network_retry import NetworkRetryJournal


def main(output):
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read_json(output/'manifest.json');verify_sources(m)
        path=output/'network_retry_supplement.json'
        if path.exists():s=read_json(path)
        else:
            status=read_json(output/'status_check_20261001.json')
            if pooled_totals(output,m)!=status['totals']:raise ValueError('ledger changed before authorization binding')
            exhausted=status['exhausted_requests']
            if len(exhausted)!=2:raise ValueError('expected exactly two failures')
            files=[Path(__file__).resolve(),ROOT/'src/wolfgym/evaluation/combination_qwen_network_retry.py',
                   ROOT/'src/wolfgym/evaluation/combination_qwen_network_diagnostic.py',ROOT/'tests/test_combination_qwen_network_retry.py']
            s={'version':'qwen-dev32-network-diagnostic-v1','manifest_sha256':digest(m),
               'authorization':'2026-10-01 user: 重试一下，检查一下问题出在哪。One additional attempt per two exhausted requests; all global limits unchanged.',
               'acknowledged_stop_sha256':file_hash(output/'incomplete_recovery_stop.json'),
               'original_ledgers':{j['id']:ledger_snapshot(output/(j['id']+'.sqlite')) for j in m['jobs'] if (output/(j['id']+'.sqlite')).exists()},
               'one_extra_attempt_keys':{x['job']:[x['request_key']] for x in exhausted},
               'source_sha256':{str(p):file_hash(p) for p in files},'max_new_posts':2,
               'max_requests_total':m['max_requests_total'],'max_cost_microusd_total':m['max_cost_microusd_total']}
            durable_json(path,s)
        if s['manifest_sha256']!=digest(m):raise ValueError('manifest drift')
        verify_sources(s);verify_prefix(output,s)
        key=dotenv_values(ROOT/'.env').get('OPENROUTER_API_KEY')
        if not key:raise ValueError('missing credential')
        results=[]
        for jobid,keys in s['one_extra_attempt_keys'].items():
            job=next(j for j in m['jobs'] if j['id']==jobid)
            matches=[(e,day) for e in selected_episodes(m,job) for day in range(30)
                     if digest([digest([digest(m),job['mode'],e]),day]) in keys]
            if len(matches)!=1:raise ValueError('scope identity mismatch')
            identity,day=matches[0]
            try:
                with NetworkRetryJournal(output,m,job,s) as journal:
                    run_one(output,identity,m,job,journal,key,stop_after=day+1)
                results.append({'job':jobid,'day':day,'success':True})
            except Exception as exc:
                results.append({'job':jobid,'day':day,'success':False,'error_type':type(exc).__name__})
            durable_json(output/'network_retry_result.json',{'probes':results,'totals':pooled_totals(output,m),
                         'scope':'two request diagnosis only; not completed episodes or scientific results'})
            if not results[-1]['success']:break
        verify_prefix(output,s)
        print(json.dumps(read_json(output/'network_retry_result.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
