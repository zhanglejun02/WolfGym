"""Independent account, full-market replay and provider-ledger audit; no API."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import numpy as np
from report_combination_baselines import wealth_table, independent_score, check_score
from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination_fast import FastCombinationGymEnv
from wolfgym.env.combination_contract import primary_contract
from wolfgym.evaluation.combination_host import read_json, file_hash, load_detector, aggregate_financial
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_calibration import CalibratedAdapter
from wolfgym.evaluation.combination_qwen_pilot import decode_reply, wire_body, RESERVATION

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/combination_action_diagnostic_20260928'
OLD=ROOT/'data/combination_qwen_pilot_20260928'
DETECTOR=ROOT/'data/task_validation_v4_20260923/detector.json'


def result_row(identity, mode, result, actions):
    financial=result['financial_outcomes']
    return {**identity,'mode':mode,'benefit_pp':100*result['H_reduction'],
            'event_wait':result['wait']['event'],'event_policy':result['policy']['event'],
            'H_wait_pp':100*result['wait']['damage']['H'][-1], 'H_policy_pp':100*result['policy']['damage']['H'][-1],
            'G_wait_pp':100*result['wait']['damage']['G'][-1], 'G_policy_pp':100*result['policy']['damage']['G'][-1],
            'components_wait':result['wait']['components'],'components_policy':result['policy']['components'],
            'first_event_wait':result['wait']['first_event_day'],'first_event_policy':result['policy']['first_event_day'],
            'cost':result['cost'],'action_days':[i for i,a in enumerate(actions) if a],
            'actions':actions,'financial_delta':{k:financial['policy'][k]-financial['wait'][k] for k in financial['policy']}}


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
    old=read_json(BASE/identity['episode_id'].replace(':','_')/'reference_wait.json.gz')
    for k in ('reference','wait'):
        if saved['accounts'][k]!=old['accounts'][k] or saved['financial_rows'][k]!=old['financial_rows'][k]:
            raise ValueError('historical R/WAIT mismatch')
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
    return result_row(identity,mode,result,[x['action'] for x in logs])


def summarize(rows):
    result={}
    for mode in sorted({x['mode'] for x in rows}):
        selected=[x for x in rows if x['mode']==mode]
        if len(selected)!=16 or len({x['group_id'] for x in selected})!=8:
            raise ValueError('incomplete factorial grid')
        result[mode]={'balanced_pp':float(np.mean([x['benefit_pp'] for x in selected]))}
        for condition in ('normal','attack'):
            rs=[x for x in selected if x['condition']==condition]
            if len(rs)!=8 or len({x['group_id'] for x in rs})!=8: raise ValueError('unpaired condition grid')
            result[mode][condition]={'benefit_pp':float(np.mean([x['benefit_pp'] for x in rs])),
                'intervened_episodes':sum(bool(x['action_days']) for x in rs),
                'freezes':sum(len(x['action_days']) for x in rs),
                'mean_admin_cost':float(np.mean([x['cost'] for x in rs])),
                'events_wait':sum(x['event_wait'] for x in rs),'events_policy':sum(x['event_policy'] for x in rs),
                'prevented':sum(x['event_wait'] and not x['event_policy'] for x in rs),
                'added':sum(not x['event_wait'] and x['event_policy'] for x in rs),
                'mean_financial_delta':{k:float(np.mean([x['financial_delta'][k] for x in rs])) for k in rs[0]['financial_delta']}}
    return result


def paired_differences(rows):
    index={(x['mode'],x['group_id'],x['condition']):x for x in rows}
    modes={x['mode'] for x in rows}; groups=sorted({x['group_id'] for x in rows})
    pairs=[('LL','old_LL'),('LL','RR'),('RL','RR'),('LR','LL'),('RL','LL'),('LR','RR')]
    result={}
    for left,right in pairs:
        if left not in modes or right not in modes: continue
        by_group={g:sum(.5*(index[left,g,c]['benefit_pp']-index[right,g,c]['benefit_pp']) for c in ('normal','attack')) for g in groups}
        result[left+' minus '+right]={'mean_pp':float(np.mean(list(by_group.values()))),'society_differences_pp':by_group}
    return result


def audit(output,manifest,modes):
    from run_combination_qwen_calibration import verify_sources, selected_episodes
    verify_sources(manifest)
    marker=output/('acceptance.json' if len(modes)==3 else 'LL_acceptance.json')
    if marker.exists():
        receipt=read_json(marker)
        for p,sha in receipt['outputs_sha256'].items():
            if file_hash(output/p)!=sha: raise ValueError('accepted report drift')
        return
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(audit_episode,[(str(output),manifest,m,e) for m in modes for e in manifest['episodes']]))
    usage={}; attempt_keys=set(); receipts={}
    for job in manifest['jobs']:
        if job['mode'] not in modes: continue
        db=sqlite3.connect(f'file:{output}/{job["id"]}.sqlite?mode=ro',uri=True); db.row_factory=sqlite3.Row
        declaration=json.loads(db.execute('select declaration from metadata where id=1').fetchone()[0])
        attempts=[dict(x) for x in db.execute('select * from attempts order by id')];db.close()
        expected={'version':'combination-request-journal-v1','contract':{'manifest_sha256':digest(manifest),'job':job['id']},
                  'max_attempts':job['max_requests'],'max_cost_microusd':job['max_microusd'],'reservation_microusd':RESERVATION,'retries':0}
        if declaration!=expected or len(attempts)>job['max_requests'] or sum(x['charged'] for x in attempts)>job['max_microusd']:
            raise ValueError('ledger declaration/limit mismatch')
        request_map={}
        for identity in selected_episodes(manifest,job):
            d=output/job['mode']/'episodes'/identity['episode_id'].replace(':','_')
            scope=digest([digest(manifest),job['mode'],identity])
            for day in range(30):
                log=read_json(d/'decisions'/f'{day:02}.json')
                if log['request'] is not None: request_map[digest([scope,day])]=(d,log)
        if len(request_map)!=len(attempts): raise ValueError('request grid mismatch')
        for x in attempts:
            if x['state']!='success' or x['request_key'] in attempt_keys: raise ValueError('unknown/duplicate attempt')
            attempt_keys.add(x['request_key']);d,log=request_map[x['request_key']]
            if json.loads(x['payload'])!=log['request']: raise ValueError('ledger payload mismatch')
            p=d/'responses'/f'{digest(log["request"])}.json';reply=decode_reply(read_json(p))
            if asdict(reply)!=json.loads(x['response']) or reply.text!=log['response'] or reply.cost_microusd!=x['charged']:
                raise ValueError('provider fee/content mismatch')
            receipts[str(p.relative_to(output))]=file_hash(p)
        usage[job['id']]={'attempts':len(attempts),'cost_usd':sum(x['charged'] for x in attempts)/1e6,
            'input_tokens':sum(json.loads(x['response'])['usage']['input_tokens'] for x in attempts),
            'output_tokens':sum(json.loads(x['response'])['usage']['output_tokens'] for x in attempts)}
    for identity in manifest['episodes']:
        name=identity['episode_id'].replace(':','_')
        for mode,path in [('RR',BASE/name/'rule.json.gz'),('old_LL',OLD/'episodes'/name/'outcome.json.gz')]:
            value=read_json(path)
            logs=value.get('decisions',value.get('log',[]))
            if not logs: logs=value['actions']
            actions=[x['action'] if isinstance(x,dict) else x for x in logs]
            if len(actions)!=30: raise ValueError('historical action length mismatch')
            rows.append(result_row(identity,mode,value['result'],actions))
    summary=summarize(rows); differences=paired_differences(rows)
    report={'scope':'8 exposed train societies; paired descriptive development comparisons; no independent test',
            'summary':summary,'paired_differences':differences,'usage':usage,'rows':rows,
            'api_cost_usd':sum(x['cost_usd'] for x in usage.values()),
            'new_episodes':16*len(modes),'account_closes':16*len(modes)*3*30*490,
            'new_decisions':16*len(modes)*30,'audit_api_calls':0,
            'formal_test_allowed':False}
    stem='REPORT' if len(modes)==3 else 'LL_REPORT'
    durable_json(output/(stem+'.json'),report)
    lines=['# 输入校准与四格开发对照', '',report['scope'],'',
           '|条件|等权减损pp|正常减损pp|攻击减损pp|正常冻结|攻击冻结|正常新增事件|攻击防止事件|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,s in summary.items():
        n,a=s['normal'],s['attack']
        lines.append(f'|{m}|{s["balanced_pp"]:.6f}|{n["benefit_pp"]:.6f}|{a["benefit_pp"]:.6f}|{n["freezes"]}|{a["freezes"]}|{n["added"]}|{a["prevented"]}|')
    lines += ['',f'新增实际费用 ${report["api_cost_usd"]:.6f}。每社会差值、费用、金融功能和完整动作见同名JSON。',
              '', '旧LL→新LL只追加固定语义说明；供应商即使temperature=0仍可能非确定，本轮每条件一次，不把差值全归因于提示。',
              'RR复用已验收同样本结果。RL/LR在自己的状态上运行；RR/RL的正常不行动不能算LLM的辨别能力。',
              '规则接受过开发数据，BC使用教师，Qwen本轮无示范/训练/报警分数；成绩不是纯推理能力排名。',
              '事件预防与连续减损分别报告；没有新阈值或市场修改。此为8社会开发描述，不作正式显著性/13链精确覆盖声明。']
    (output/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    outputs={str(p.relative_to(output)):file_hash(p) for p in (output/(stem+'.json'),output/(stem+'.md'))}
    outputs.update(receipts)
    for mode in modes:
        for p in (output/mode/'episodes').glob('*/complete.json'):outputs[str(p.relative_to(output))]=file_hash(p)
    durable_json(marker,{'manifest_sha256':file_hash(output/'manifest.json'),'outputs_sha256':outputs,
                         'new_episodes':report['new_episodes'],'account_closes':report['account_closes'],
                         'modes':list(modes),'complete':True,'official_release_authorized':False})
