"""Original full account audit with the two explicitly authorized fourth attempts."""
from report_combination_qwen_dev32 import *
from wolfgym.evaluation.combination_qwen_recovery import verify_prefix

def audit_ledgers(output, manifest):
    supplement=read_json(output/'network_retry_supplement.json')
    verify_sources(supplement);verify_prefix(output,supplement)
    if supplement['manifest_sha256']!=digest(manifest):raise ValueError('supplement manifest drift')
    modes=('LL',); amendment={'original_ledgers':{}}
    receipts, usage = {}, {}
    for job in manifest['jobs']:
        if job['mode'] not in modes: continue
        snapshot = ledger_snapshot(output/(job['id']+'.sqlite'))
        expected = {'version':'combination-request-journal-v1',
                    'contract':{'manifest_sha256':digest(manifest),'job':job['id'], 'version':VERSION,'limits_scope':'all_jobs_pooled'},
                    'max_attempts':manifest['max_requests_total'],'max_cost_microusd':manifest['max_cost_microusd_total'],
                    'reservation_microusd':RESERVATION,'retries':2}
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
            cap=4 if key in supplement['one_extra_attempt_keys'].get(job['id'],[]) else 3
            if not 1<=len(attempts)<=cap or any(json.loads(x['payload'])!=payload for x in attempts):
                raise ValueError('retry bound/payload mismatch')
            if attempts[-1]['state']!='success' or any(x['state']=='success' for x in attempts[:-1]):
                raise ValueError('incomplete request or successful request repeated')
            for row in attempts:
                evidence = output/'recovery_attempts'/job['id']/str(row['id'])
                if row['state']!='success':
                    path = (d/'responses' if row['id']<=old_count else evidence)/f'{digest(payload)}.failure.json'
                    if row['state']!='fatal' or row['charged']!=RESERVATION or not eligible_failure(path,payload):
                        raise ValueError('unknown charge reservation/evidence mismatch')
                    if not path.exists(): path=path.with_name(digest(payload)+'.json')
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


def audit(output,manifest):
    verify_sources(manifest)
    verify_sources(read_json(output/'network_continuation_manifest.json'))
    if (output/'acceptance.json').exists():
        for p,h in read_json(output/'acceptance.json')['outputs_sha256'].items():
            if file_hash(output/p)!=h:raise ValueError('accepted report drift')
        return
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(audit_episode,[(str(output),manifest,'LL',e) for e in manifest['episodes']]))
        for baseline_rows in pool.map(audit_baselines,manifest['episodes']):rows.extend(baseline_rows)
    usage,receipts,totals=audit_ledgers(output,manifest)
    report={'model':manifest['settings']['model'],'scope':manifest['scope'],'summary':summarize(rows),
        'paired_comparisons':paired_statistics(rows),'rows':rows,'usage':usage,'totals':totals,
        'new_qwen_episodes':64,'reused_baseline_episodes':192,'audited_account_closes':64*6*30*490,
        'full_Qwen_market_replay_decisions':1920,'replayed_baseline_actions':5760,
        'invalid_actions':sum(r['invalid_actions'] for r in rows),'audit_API_calls':0,'formal_test_allowed':False,
        'interval_note':'Pointwise exploratory intervals, not a formal confirmatory family; modulo13 sensitivity has few clusters and no coverage guarantee'}
    durable_json(output/'report.json',report)
    lines=['# 固定 Qwen：32社会开发评测','',report['scope'],'',
        '|策略|正常减损pp|攻击减损pp|等权减损pp|正常冻结|攻击冻结|正常新增事件|攻击防止事件|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy,s in report['summary'].items():
        n,a=s['normal'],s['attack']
        lines.append(f'|{policy}|{n["benefit_pp"]:.6f}|{a["benefit_pp"]:.6f}|{s["balanced_pp"]:.6f}|{n["freezes"]}|{a["freezes"]}|{n["added"]}|{a["prevented"]}|')
    lines += ['',f'成功{totals["successful_attempts"]}次/全部{totals["attempts"]}次；确认${totals["confirmed_microusd"]/1e6:.6f}，未知预留${totals["unknown_reserved_microusd"]/1e6:.6f}。',
        '新样本仅指此前Qwen未访问；这些训练社会已用于非LLM开发，不是独立test。日期和公开规则按历史选择固定，没有在本32社会重新挑选。',
        '配对区间以社会为单位，正常/攻击等权；三项比较为开发描述，不据此宣布正式显著性。lag13敏感性按占用模13链分组，少簇不保证覆盖。',
        '完整财富/事件/交易功能/API及行政成本、无效动作、社会差值见JSON。重大事件预防和连续减损分开解释。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for p in (output/'network_retry_supplement.json',output/'network_continuation_manifest.json',output/'report.json',output/'REPORT.md',output/'manifest.json'):receipts[str(p.relative_to(output))]=file_hash(p)
    for p in (output/'LL/episodes').glob('*/complete.json'):receipts[str(p.relative_to(output))]=file_hash(p)
    verify_sources(manifest)
    durable_json(output/'acceptance.json',{'complete':True,'manifest_sha256':file_hash(output/'manifest.json'),
        'outputs_sha256':receipts,'new_qwen_episodes':64,'paired_societies':32,'official_release_authorized':False})
