"""Recovery-aware audit: reuse frozen account replay, separately bound unknown fees."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

from report_combination_qwen_calibration import (
    audit_episode, result_row, summarize, paired_differences, BASE, OLD)
from run_combination_qwen_calibration import verify_sources, selected_episodes
from wolfgym.evaluation.combination_host import read_json, file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_pilot import decode_reply, RESERVATION, wire_body
from wolfgym.evaluation.combination_qwen_recovery import (
    ledger_snapshot, pooled_totals, verify_prefix, retryable_failure)


def audit_ledgers(output, manifest, amendment, modes):
    verify_prefix(output,amendment)
    receipts, usage = {}, {}
    for job in manifest['jobs']:
        if job['mode'] not in modes: continue
        snapshot = ledger_snapshot(output/(job['id']+'.sqlite'))
        expected = {'version':'combination-request-journal-v1',
                    'contract':{'manifest_sha256':digest(manifest),'job':job['id']},
                    'max_attempts':job['max_requests'],'max_cost_microusd':job['max_microusd'],
                    'reservation_microusd':RESERVATION,'retries':0}
        if json.loads(snapshot['metadata'][0]['declaration'])!=expected: raise ValueError('historical declaration drift')
        request_map = {}
        for identity in selected_episodes(manifest,job):
            d = output/job['mode']/'episodes'/identity['episode_id'].replace(':','_')
            scope = digest([digest(manifest),job['mode'],identity])
            for day in range(30):
                log = read_json(d/'decisions'/f'{day:02}.json')
                if log['request'] is not None: request_map[digest([scope,day])] = (d,log)
        grouped = {}
        for r in snapshot['attempts']: grouped.setdefault(r['request_key'],[]).append(r)
        if set(grouped)!=set(request_map): raise ValueError('logical request grid mismatch')
        old_count = len(amendment['original_ledgers'].get(job['id'],{}).get('attempts',[]))
        for key, attempts in grouped.items():
            d,log = request_map[key]; payload = log['request']; wire_body(payload)
            if not 1<=len(attempts)<=3 or any(json.loads(x['payload'])!=payload for x in attempts):
                raise ValueError('retry bound/payload mismatch')
            if attempts[-1]['state']!='success' or any(x['state']=='success' for x in attempts[:-1]):
                raise ValueError('incomplete request or successful request repeated')
            for row in attempts:
                evidence = output/'recovery_attempts'/job['id']/str(row['id'])
                if row['state']!='success':
                    path = (d/'responses' if row['id']<=old_count else evidence)/f'{digest(payload)}.failure.json'
                    if row['state']!='fatal' or row['charged']!=RESERVATION or not retryable_failure(path,payload):
                        raise ValueError('unknown charge reservation/evidence mismatch')
                    receipts[str(path.relative_to(output))]=file_hash(path)
                    continue
                p = d/'responses'/f'{digest(payload)}.json'; reply = decode_reply(read_json(p))
                if asdict(reply)!=json.loads(row['response']) or reply.text!=log['response'] or reply.cost_microusd!=row['charged']:
                    raise ValueError('provider fee/content mismatch')
                receipts[str(p.relative_to(output))]=file_hash(p)
                if row['id']>old_count:
                    attempt_receipt=evidence/p.name
                    if read_json(attempt_receipt)!=read_json(p): raise ValueError('per-attempt receipt mismatch')
                    receipts[str(attempt_receipt.relative_to(output))]=file_hash(attempt_receipt)
        successes=[x for x in snapshot['attempts'] if x['state']=='success']
        usage[job['id']]={'attempts':len(snapshot['attempts']),'logical_requests':len(grouped),
            'confirmed_usd':sum(x['charged'] for x in successes)/1e6,
            'unknown_reserved_usd':sum(x['charged'] for x in snapshot['attempts'] if x['state']!='success')/1e6}
        receipts[job['id']+'.sqlite']=file_hash(output/(job['id']+'.sqlite'))
    totals=pooled_totals(output,manifest)
    if totals['overrun'] or totals['attempts']>manifest['max_requests_total'] or totals['charged_or_reserved_microusd']>manifest['max_cost_microusd_total']:
        raise ValueError('pooled limit exceeded')
    return usage,receipts,totals


def audit(output, manifest, amendment, modes):
    from run_combination_qwen_recovery import verify_amendment
    verify_sources(manifest); verify_amendment(output,manifest,amendment)
    stem='RECOVERY_REPORT' if len(modes)==3 else 'LL_RECOVERY_REPORT'
    marker=output/(stem+'_acceptance.json')
    if marker.exists():
        receipt=read_json(marker)
        for p,sha in receipt['outputs_sha256'].items():
            if file_hash(output/p)!=sha: raise ValueError('accepted recovery report drift')
        return read_json(output/(stem+'.json'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(audit_episode,[(str(output),manifest,m,e) for m in modes for e in manifest['episodes']]))
    usage,receipts,totals=audit_ledgers(output,manifest,amendment,modes)
    for identity in manifest['episodes']:
        name=identity['episode_id'].replace(':','_')
        for mode,path in [('RR',BASE/name/'rule.json.gz'),('old_LL',OLD/'episodes'/name/'outcome.json.gz')]:
            value=read_json(path); logs=value.get('decisions',value.get('log',[])) or value['actions']
            actions=[x['action'] if isinstance(x,dict) else x for x in logs]
            if len(actions)!=30: raise ValueError('historical action length mismatch')
            rows.append(result_row(identity,mode,value['result'],actions))
    report={'scope':'8 exposed train societies; descriptive paired development comparison; no independent test',
        'summary':summarize(rows),'paired_differences':paired_differences(rows),'usage':usage,'rows':rows,
        'pooled_totals_at_audit':totals,'new_episodes':16*len(modes),'account_closes':16*len(modes)*3*30*490,
        'new_decisions':16*len(modes)*30,'audit_api_calls':0,'formal_test_allowed':False,
        'unknown_fees_reconciled':False,'recovery_amendment_sha256':file_hash(output/'recovery_amendment.json')}
    durable_json(output/(stem+'.json'),report)
    lines=['# 输入校准与四格开发对照（显式网络恢复）','',report['scope'],'',
        '|条件|等权减损pp|正常减损pp|攻击减损pp|正常冻结|攻击冻结|正常新增事件|攻击防止事件|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,s in report['summary'].items():
        n,a=s['normal'],s['attack']
        lines.append(f'|{mode}|{s["balanced_pp"]:.6f}|{n["benefit_pp"]:.6f}|{a["benefit_pp"]:.6f}|{n["freezes"]}|{a["freezes"]}|{n["added"]}|{a["prevented"]}|')
    lines += ['',f'全批截至审计：{totals["attempts"]}次尝试，确认费用${totals["confirmed_microusd"]/1e6:.6f}，未知预留${totals["unknown_reserved_microusd"]/1e6:.6f}。未知预留不是确认收费。',
        '失败请求最多两次补发，成功请求不重发；原失败账本及预留保留。',
        'LL相对旧LL仅追加说明，网络恢复不改模型/输入；供应商随机性与日期不同仍限制因果归因。',
        'LR/RL在自己的闭环状态决策；规则负责时机时正常零干预不算LLM辨别能力。',
        '规则开发信息与模型不同；只报告模块替换后的系统效果，不拆成独立纯能力贡献。',
        '这是开发集8社会描述，未进行正式test，不作显著性结论；完整正常财富、交易功能、事件及逐社会差值见JSON。']
    (output/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    for p in (output/(stem+'.json'),output/(stem+'.md'),output/'recovery_amendment.json'):
        receipts[str(p.relative_to(output))]=file_hash(p)
    for mode in modes:
        for p in (output/mode/'episodes').glob('*/complete.json'):receipts[str(p.relative_to(output))]=file_hash(p)
    durable_json(marker,{'manifest_sha256':file_hash(output/'manifest.json'),'outputs_sha256':receipts,
        'new_episodes':report['new_episodes'],'account_closes':report['account_closes'],
        'modes':list(modes),'complete':True,'official_release_authorized':False,'unknown_fees_reconciled':False})
    return report
