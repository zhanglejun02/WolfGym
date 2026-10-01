"""Read-only diagnosis of the accepted Qwen pilot; no simulator or provider calls."""
from pathlib import Path
import json
import numpy as np
from wolfgym.evaluation.combination_host import read_json,file_hash,load_detector,write_json
from wolfgym.evaluation.combination_public import (
    ALARM_THRESHOLD,detector_features,public_target,BaselinePolicy,PublicHistory,digest)
from wolfgym.evaluation.task_validation import detector_score
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'data/combination_qwen_pilot_20260928'
BASE=ROOT/'data/combination_action_diagnostic_20260928'
OUT=ROOT/'data/qwen_decision_diagnostic_20260928'


def first_crossing(scores,threshold=ALARM_THRESHOLD):
    return next((day for day,score in enumerate(scores) if score>threshold),None)


def action_timing(day,event_close):
    # Decision t sees closes through t and acts during close t+1.
    if event_close is None:return 'no_wait_event'
    return 'before_wait_event_close' if day<event_close else 'after_wait_event_observable'


def rank_fraction(value,values):
    return sum(x<=value for x in values)/len(values)


def signals(frame,target=None):
    assets=frame['assets'];entities=frame['entities'];summary=frame['message_summary']
    result={'asset_max_volume_zscore':max(x['volume_zscore'] for x in assets),
            'any_asset_volume_clipped':any(abs(x['volume_zscore'])>=10 for x in assets),
            'entity_max_activity_change':max((x['activity_change'] for x in entities),default=0),
            'entity_max_cancel':max((x['cancellation_rate'] for x in entities),default=0),
            'cascade_reach':frame['social']['cascade_reach'],'consensus':frame['social']['consensus_concentration'],
            'market_cancel':frame['market']['cancellation_rate'],'market_return_3d':frame['market']['return_3d'],
            'liquidity':frame['market']['liquidity']}
    if target is not None:
        asset=next(x for x in assets if x['asset_id']==target)
        result.update({'chosen_'+k:v for k,v in asset.items() if k!='asset_id'})
        result['chosen_message_reach_sum']=summary[target]['audience_reach']
        result['chosen_reach_rank_fraction']=rank_fraction(summary[target]['audience_reach'],[x['audience_reach'] for x in summary.values()])
        result['chosen_volume_rank_fraction']=rank_fraction(asset['volume_zscore'],[x['volume_zscore'] for x in assets])
    return result


def main():
    # Verify accepted receipts, not rerun the prior accounts/market audit.
    accepted=read_json(PILOT/'final_acceptance.json')
    for p,sha in accepted['sha256'].items():assert file_hash(ROOT/p)==sha
    review=read_json(PILOT/'outcome_review.json');manifest=read_json(PILOT/'manifest.json')
    assert review['full_grid_complete'] and len(manifest['episodes'])==16
    for p,sha in manifest['source_sha256'].items():assert file_hash(p)==sha
    detector_path=ROOT/'data/task_validation_v4_20260923/detector.json'
    detector=load_detector(detector_path)
    bound={str(PILOT/'final_acceptance.json'):file_hash(PILOT/'final_acceptance.json'),str(detector_path):file_hash(detector_path)}
    rows=[];frames=[]
    for identity in manifest['episodes']:
        d=PILOT/'episodes'/identity['episode_id'].replace(':','_');marker=read_json(d/'complete.json')
        for p,sha in marker['sha256'].items():assert file_hash(d/p)==sha
        bound[str(d/'complete.json')]=file_hash(d/'complete.json')
        a=read_json(d/'outcome.json.gz');decisions=a['decisions'];assert len(decisions)==30
        b=BASE/identity['episode_id'].replace(':','_');bm=read_json(b/'complete.json')
        assert file_hash(b/'rule.json.gz')==bm['sha256']['rule.json.gz']
        bound[str(b/'rule.json.gz')]=file_hash(b/'rule.json.gz')
        rule=read_json(b/'rule.json.gz')
        own_scores=[float(detector_score(detector_features(x['frame']),detector)) for x in decisions]
        q_days=[x['day'] for x in decisions if x['action']]
        rule_days=[x['day'] for x in rule['decisions'] if x['action']]
        first=first_crossing(own_scores);shadow=BaselinePolicy({'id':'alarm_refresh','family':'alarm_refresh'},detector);history=PublicHistory(5)
        event=a['result']['wait']['first_event_day'];interventions=[];shadow_days=[]
        for log,score in zip(decisions,own_scores):
            f=log['frame'];view=history.observe(f);shadow_action=shadow(view)
            # Counterfactual recommendation only: record ACTUAL Qwen action/history and budget.
            history.record(f,log['action'])
            if shadow_action:shadow_days.append(f['day'])
            target=f['action_targets'][log['action']-1] if log['action'] else None
            frame={'episode_id':identity['episode_id'],'condition':identity['condition'],'day':f['day'],
                   'eligible':f['remaining_budget']>=4,'action':log['action'],'score':score,
                   'above_alarm':score>ALARM_THRESHOLD,'shadow_action':shadow_action,
                   'pre_or_at_first_freeze':f['day']<=q_days[0],**signals(f,target)}
            frames.append(frame)
            if target:
                interventions.append({**frame,'target':target,'public_target':public_target(f),
                    'target_agrees_public':target==public_target(f),'wait_event_timing':action_timing(f['day'],event)})
        # Until either policy first intervenes their public views must be identical.
        shared_until=min(q_days[0],rule_days[0] if rule_days else 30)
        assert all(decisions[t]['frame']==rule['decisions'][t]['frame'] for t in range(shared_until+1))
        rows.append({**identity,'benefit_pp':100*a['result']['H_reduction'],
              'rule_benefit_pp':100*rule['result']['H_reduction'],'first_alarm_day0_on_qwen_path':first,
              'first_freeze_day0':q_days[0],'first_delay_from_alarm':q_days[0]-first if first is not None else None,
              'qwen_days0':q_days,'original_rule_days0':rule_days,'same_state_shadow_days0':shadow_days,
              'wait_first_event_close':event,'policy_first_event_close':a['result']['policy']['first_event_day'],
              'budget_left_terminal':12-4*len(q_days),'interventions':interventions,
              'shared_prefix_frames':shared_until+1})
    summary={}
    for condition in ('normal','attack'):
        fs=[f for f in frames if f['condition']==condition];rs=[r for r in rows if r['condition']==condition]
        interventions=[i for r in rs for i in r['interventions']]
        groups={'eligible_wait':[f for f in fs if f['eligible'] and not f['action']],
                'freeze':[f for f in fs if f['action']],
                'pre_first_freeze_wait':[f for f in fs if f['pre_or_at_first_freeze'] and not f['action']]}
        summaries={}
        keys=list(signals(next(x for x in decisions)['frame']).keys())
        for name,xs in groups.items():
            summaries[name]={'n':len(xs),'above_alarm':sum(x['above_alarm'] for x in xs),
                'volume_clip_days':sum(x['any_asset_volume_clipped'] for x in xs),
                'entity_activity_at_clip_days':sum(x['entity_max_activity_change']>=10 for x in xs),
                'entity_cancel_one_days':sum(x['entity_max_cancel']>=1 for x in xs),
                'medians':{k:float(np.median([x[k] for x in xs])) for k in keys} if xs else {}}
        summary[condition]={'episodes':len(rs),'alarm_episodes':sum(r['first_alarm_day0_on_qwen_path'] is not None for r in rs),
            'freezes':len(interventions),'freezes_above_alarm':sum(x['above_alarm'] for x in interventions),
            'target_agreements':sum(x['target_agrees_public'] for x in interventions),
            'chosen_volume_max':sum(x['chosen_volume_rank_fraction']==1 for x in interventions),
            'first_delays':[r['first_delay_from_alarm'] for r in rs],
            'post_wait_event_freezes':sum(x['wait_event_timing']=='after_wait_event_observable' for x in interventions),
            'terminal_budget_left':[r['budget_left_terminal'] for r in rs],
            'feature_groups':summaries}
    report={'scope':'offline descriptive diagnosis only; no API, market rollouts, fitting or threshold search',
            'alarm_threshold':ALARM_THRESHOLD,'rows':rows,'frames':frames,'summary':summary,
            'limits':['Detector is a frozen comparator, not an optimal-action label or calibrated harm probability.',
                      'Only public signals enter shadow recommendations; WAIT events used retrospectively only.',
                      'Shadow suggestions on Qwen states do not estimate a new closed-loop policy outcome.',
                      'Provider rationale was not saved; co-occurrence does not establish model motivation.',
                      'Repeated frames within 8 societies are dependent; no significance or generalization claims.']}
    OUT.mkdir(exist_ok=True);write_json(OUT/'diagnostic.json',report)
    bound[str(Path(__file__))]=file_hash(Path(__file__))
    bound[str(ROOT/'src/wolfgym/observations/builder.py')]=file_hash(ROOT/'src/wolfgym/observations/builder.py')
    write_json(OUT/'receipt.json',{'input_and_analysis_sha256':bound,'report_sha256':file_hash(OUT/'diagnostic.json'),
                                 'episodes':16,'frames':480,'freezes':24,'api_calls':0,'market_rollouts':0})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
