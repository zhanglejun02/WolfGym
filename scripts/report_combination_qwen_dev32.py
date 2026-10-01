"""Complete 32-society grid: market/action/account audit and paired development report."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from report_combination_baselines import wealth_table,independent_score,check_score,bootstrap
from report_combination_qwen_calibration import result_row
from run_combination_qwen_dev32 import ROOT,BASE,POLICIES,baseline_directory
from run_combination_qwen_calibration import verify_sources,selected_episodes,DETECTOR
from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination_fast import FastCombinationGymEnv
from wolfgym.env.combination_contract import primary_contract
from wolfgym.evaluation.combination_host import read_json,file_hash,load_detector,aggregate_financial
from wolfgym.evaluation.combination_public import digest,BaselinePolicy,PublicHistory,baseline_specs
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_qwen_calibration import CalibratedAdapter
from wolfgym.evaluation.combination_qwen_pilot import decode_reply,wire_body,RESERVATION
from wolfgym.evaluation.combination_qwen_recovery import ledger_snapshot,pooled_totals
from wolfgym.evaluation.combination_qwen_incomplete_recovery import eligible_failure
from wolfgym.evaluation.combination_qwen_dev32 import VERSION


def audit_episode(args):
    output,manifest,mode,identity=args; output=Path(output)
    directory=output/mode/'episodes'/identity['episode_id'].replace(':','_')
    marker=read_json(directory/'complete.json')
    if marker['scope']!=digest([digest(manifest),mode,identity]): raise ValueError('scope mismatch')
    for p,sha in marker['sha256'].items():
        if file_hash(directory/p)!=sha: raise ValueError('episode artifact drift')
    saved=read_json(directory/'outcome.json.gz'); result=saved['result']; logs=saved['decisions']
    if len(logs)!=30 or any(result[k]!=v for k,v in identity.items()): raise ValueError('incomplete/wrong identity')
    def reply(payload):
        wire_body(payload)
        return decode_reply(read_json(directory/'responses'/f'{digest(payload)}.json')).text
    adapter=CalibratedAdapter(mode,load_detector(DETECTOR),reply)
    env=FastCombinationGymEnv(CombinationEpisode(identity['split'],identity['trajectory_seed'],identity['condition']),message_mode='indexed')
    try:
        _,info=env.reset()
        for day,log in enumerate(logs):
            if read_json(directory/'decisions'/f'{day:02}.json')!=log: raise ValueError('separate decision mismatch')
            action=adapter(info)
            if adapter.log[-1]!=log: raise ValueError('public request/response/model action mismatch')
            _,reward,done,truncated,info=env.step(action)
            if reward or truncated or done!=(day==29): raise ValueError('termination/feedback mismatch')
        if any(result[k]!=v for k,v in env.evaluation_result().items()): raise ValueError('full replay result mismatch')
        if saved['accounts']!={k:[asdict(x) for x in v] for k,v in env._snapshots.items()}: raise ValueError('full account replay mismatch')
        if saved['financial_rows']!=env._rows: raise ValueError('full financial replay mismatch')
    finally: env.close()
    old=read_json(baseline_directory(identity)/'reference_wait.json.gz')
    for k in ('reference','wait'):
        if saved['accounts'][k]!=old[k]:raise ValueError('historical R/WAIT account mismatch')
        # Old branch diagnostics and Gym telemetry have different nonfinancial keys.
        paired=lambda rows:[{'day':r['day'],'financial_outcomes':r['financial_outcomes']} for r in rows]
        if paired(saved['financial_rows'][k])!=paired(old[k+'_rows']):
            raise ValueError('historical R/WAIT financial mismatch')
    ref,initial=wealth_table(saved['accounts']['reference'],saved['accounts']['reference'])
    scores={}
    for world in ('wait','policy'):
        actual,_=wealth_table(saved['accounts'][world],saved['accounts']['reference'])
        scores[world]=independent_score(ref,actual,initial,saved['financial_rows']['reference'],saved['financial_rows'][world],primary_contract()['harm_threshold'])
        check_score(scores[world],result[world])
    if not np.isclose(scores['wait']['H'][-1]-scores['policy']['H'][-1],result['H_reduction'],atol=1e-12,rtol=1e-10):
        raise ValueError('independent benefit mismatch')
    if {k:aggregate_financial(v) for k,v in saved['financial_rows'].items()}!=result['financial_outcomes']:
        raise ValueError('independent financial mismatch')
    if result['cost']!=sum(x['cost'] for x in logs) or result['invalid_actions']!=sum(x['invalid_action'] for x in logs):
        raise ValueError('cost/error count mismatch')
    if result['model_decisions']!=sum(x['request'] is not None for x in logs): raise ValueError('request count mismatch')
    row=result_row(identity,mode,result,[x['action'] for x in logs])
    row.update(invalid_actions=result['invalid_actions'],model_decisions=result['model_decisions'])
    return row


def summarize(rows):
    result={}
    for mode in sorted({x['mode'] for x in rows}):
        selected=[x for x in rows if x['mode']==mode]
        if len(selected)!=64 or len({x['group_id'] for x in selected})!=32:
            raise ValueError('incomplete factorial grid')
        result[mode]={'balanced_pp':float(np.mean([x['benefit_pp'] for x in selected]))}
        for condition in ('normal','attack'):
            rs=[x for x in selected if x['condition']==condition]
            if len(rs)!=32 or len({x['group_id'] for x in rs})!=32: raise ValueError('unpaired condition grid')
            result[mode][condition]={'benefit_pp':float(np.mean([x['benefit_pp'] for x in rs])),
                'intervened_episodes':sum(bool(x['action_days']) for x in rs),
                'freezes':sum(len(x['action_days']) for x in rs),
                'mean_admin_cost':float(np.mean([x['cost'] for x in rs])),
                'events_wait':sum(x['event_wait'] for x in rs),'events_policy':sum(x['event_policy'] for x in rs),
                'prevented':sum(x['event_wait'] and not x['event_policy'] for x in rs),
                'added':sum(not x['event_wait'] and x['event_policy'] for x in rs),
                'mean_financial_delta':{k:float(np.mean([x['financial_delta'][k] for x in rs])) for k in rs[0]['financial_delta']}}
    return result


def audit_ledgers(output, manifest):
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
            if not 1<=len(attempts)<=3 or any(json.loads(x['payload'])!=payload for x in attempts):
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


def audit_baselines(identity):
    directory=baseline_directory(identity);marker=read_json(directory/'complete.json')
    if marker['episode']!=identity:raise ValueError('baseline identity')
    cache=read_json(directory/'reference_wait.json.gz')
    ref,initial=wealth_table(cache['reference'],cache['reference'])
    wait,_=wealth_table(cache['wait'],cache['reference'])
    wait_score=independent_score(ref,wait,initial,cache['reference_rows'],cache['wait_rows'],primary_contract()['harm_threshold'])
    output=[]
    for policy in POLICIES:
        p=directory/(policy+'.json.gz')
        if file_hash(p)!=marker['sha256'][p.name]:raise ValueError('baseline file drift')
        artifact=read_json(p);result=artifact['result']
        spec=next(s for s in baseline_specs() if s['id']==policy)
        if artifact['spec']!=spec or any(result[k]!=v for k,v in identity.items()):raise ValueError('baseline provenance')
        actual,_=wealth_table(artifact['accounts'],cache['reference'])
        score=independent_score(ref,actual,initial,cache['reference_rows'],artifact['rows'],primary_contract()['harm_threshold'])
        check_score(wait_score,result['wait']);check_score(score,result['policy'])
        if not np.isclose(wait_score['H'][-1]-score['H'][-1],result['H_reduction'],atol=1e-12,rtol=1e-10):raise ValueError('baseline H reduction')
        policy_fn=BaselinePolicy(spec,load_detector(DETECTOR),randomization_key=identity['group_id']);history=PublicHistory(5)
        logs=artifact['log'];actions=[]
        if len(logs)!=30:raise ValueError('baseline horizon')
        for day,log in enumerate(logs):
            f=log['frame'];view=history.observe(f);action=policy_fn(view)
            if f['day']!=day or log['input_sha256']!=digest(view) or action!=log['action'] or log['invalid_action']:
                raise ValueError('baseline public action replay')
            history.record(f,action);actions.append(action)
        expected={'reference':aggregate_financial(cache['reference_rows']),'wait':aggregate_financial(cache['wait_rows']),
                  'policy':aggregate_financial(artifact['rows'])}
        if result['financial_outcomes']!=expected or result['cost']!=sum(bool(x)*4 for x in actions):raise ValueError('baseline finance/cost')
        row=result_row(identity,policy,result,actions);row.update(invalid_actions=0,model_decisions=0);output.append(row)
    return output


def paired_statistics(rows):
    groups=sorted({r['group_id'] for r in rows});index={(r['mode'],r['group_id'],r['condition']):r for r in rows}
    if len(groups)!=32 or len(rows)!=256 or len(index)!=256:raise ValueError('complete 32 x 2 x 4 grid required')
    rng=np.random.default_rng(20260926);indices=rng.integers(0,32,size=(20000,32))
    roots=np.array([index['LL',g,'normal']['trajectory_seed'] for g in groups])
    chains=sorted(set((roots%13).tolist()))
    cluster_ids=np.random.default_rng(20260927).integers(0,len(chains),size=(20000,len(chains)))
    sizes=np.array([sum(roots%13==r) for r in chains]);output={}
    for other in POLICIES:
        normal=np.array([index['LL',g,'normal']['benefit_pp']-index[other,g,'normal']['benefit_pp'] for g in groups])
        attack=np.array([index['LL',g,'attack']['benefit_pp']-index[other,g,'attack']['benefit_pp'] for g in groups])
        values=.5*(normal+attack);totals=np.array([values[roots%13==r].sum() for r in chains])
        samples=totals[cluster_ids].sum(axis=1)/sizes[cluster_ids].sum(axis=1)
        output['LL minus '+other]={'balanced_pp':bootstrap(values,indices),'normal_pp':bootstrap(normal,indices),
            'attack_pp':bootstrap(attack,indices),'society_differences_pp':dict(zip(groups,values.tolist())),
            'lag13_chain_sensitivity_ci95_pp':np.quantile(samples,[.025,.975]).tolist(),
            'occupied_chains':len(chains),'chain_sizes':sizes.tolist(),'coverage_certified':False}
    return output


def audit(output,manifest):
    verify_sources(manifest)
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
    for p in (output/'report.json',output/'REPORT.md',output/'manifest.json'):receipts[str(p.relative_to(output))]=file_hash(p)
    for p in (output/'LL/episodes').glob('*/complete.json'):receipts[str(p.relative_to(output))]=file_hash(p)
    verify_sources(manifest)
    durable_json(output/'acceptance.json',{'complete':True,'manifest_sha256':file_hash(output/'manifest.json'),
        'outputs_sha256':receipts,'new_qwen_episodes':64,'paired_societies':32,'official_release_authorized':False})
