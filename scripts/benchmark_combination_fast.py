"""Bounded engineering-only equality, profiling and A1/A2/A3 timing comparison."""
import argparse
import cProfile
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import pickle
import pstats
import time

import numpy as np

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.env.combination_fast import FastCombinationGymEnv, FAST_RUNTIME_VERSION
from wolfgym.evaluation.combination_host import file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest

ROOT=Path(__file__).resolve().parents[1]
MODES=('baseline','day','ids','indexed')


def rng_state(sim):
    return {**{name:getattr(sim,name).rng.bit_generator.state for name in ('market','social','graph')},
            'simulator':sim.rng.bit_generator.state,
            'agents':{a.agent_id:a.rng.bit_generator.state for a in sim.society.all_agents if getattr(a,'rng',None) is not None}}


def run(mode,condition,profile_path=None):
    gc.collect()
    profiler=cProfile.Profile() if profile_path else None
    if profiler:profiler.enable()
    start=time.perf_counter()
    episode=CombinationEpisode('engineering',20260925,condition)
    env=(CombinationGymEnv(episode,training_reward=True) if mode=='baseline' else
         FastCombinationGymEnv(episode,training_reward=True,message_mode=mode))
    construction=time.perf_counter()-start
    start=time.perf_counter();outputs=[env.reset()];reset=time.perf_counter()-start;durations=[]
    try:
        for day in range(30):
            action={2:1,17:2,29:3}.get(day,0)
            start=time.perf_counter();outputs.append(env.step(action));durations.append(time.perf_counter()-start)
        if profiler:
            profiler.disable();profiler.dump_stats(str(profile_path))
        # Exclude audit serialization and artifacts from env timing, explicitly.
        equality={
            'public_outputs_reward_flags_bytes':hashlib.sha256(pickle.dumps(outputs,protocol=5)).hexdigest(),
            'terminal_result':digest(env.evaluation_result()),
            'all_accounts':digest({k:[asdict(s) for s in v] for k,v in env._snapshots.items()}),
            'financial_rows':digest(env._rows),
            'all_world_component_agent_rng':digest({k:rng_state(s) for k,s in env._worlds.items()}),
            'raw_message_events':digest({k:s.social.message_events for k,s in env._worlds.items()})}
        return {'mode':mode,'condition':condition,'construction_seconds':construction,'reset_seconds':reset,
                'step_seconds':durations,'total_env_seconds':construction+reset+sum(durations),
                'last5_step_seconds':sum(durations[-5:]),'equality':equality}
    finally:env.close()


def profile_summary(path):
    stats=pstats.Stats(str(path));rows=[]
    for (filename,line,name),(primitive,calls,total,cumulative,callers) in stats.stats.items():
        if name in ('get_state','_build_messages','observable_entity_history','_build_entities','deepcopy','_public_output'):
            rows.append({'file':filename,'line':line,'function':name,'calls':calls,
                         'self_seconds':total,'cumulative_seconds':cumulative})
    return sorted(rows,key=lambda r:-r['cumulative_seconds'])


def main(output,repeats):
    if repeats!=3:raise ValueError('this bounded benchmark fixes three repeats')
    if output.exists():raise ValueError('use a new output directory; do not overwrite a performance declaration')
    output.mkdir(parents=True)
    sources=[Path(__file__),ROOT/'src/wolfgym/observations/fast_messages.py',ROOT/'src/wolfgym/env/combination_fast.py',
             ROOT/'src/wolfgym/observations/builder.py',ROOT/'src/wolfgym/env/combination.py',
             ROOT/'src/wolfgym/env/combination_v51.json']
    manifest={'version':FAST_RUNTIME_VERSION,'repeats':repeats,'modes':list(MODES),'conditions':['normal','attack'],
              'seed':20260925,'split':'engineering','training_reward':True,'actions':{'2':1,'17':2,'29':3},
              'source_sha256':{str(p.resolve()):file_hash(p) for p in sources},
              'scope':'single-process env construction+reset+30 steps, excluding model optimization and artifact I/O',
              'timing_order':'cyclic rotation of four modes across repeats; two conditions per mode',
              'formal_train_validation_test_runs':0}
    durable_json(output/'manifest.json',manifest)
    profiles={}
    # Profiling establishes hotspots; these instrumented runs are NOT timing samples.
    for mode in ('baseline','indexed'):
        path=output/f'{mode}.prof';row=run(mode,'attack',path)
        profiles[mode]={'timing_instrumented':row,'functions':profile_summary(path)}
    if profiles['baseline']['timing_instrumented']['equality']!=profiles['indexed']['timing_instrumented']['equality']:
        raise ValueError('profiled execution differs')
    records=[];expected={}
    for repeat in range(repeats):
        order=MODES[repeat:]+MODES[:repeat]
        for mode in order:
            for condition in ('normal','attack'):
                row=run(mode,condition);row['repeat']=repeat
                if condition in expected and row['equality']!=expected[condition]:raise ValueError('bitwise episode equality failed')
                expected[condition]=row['equality'];records.append(row)
                durable_json(output/'progress.json',{'completed':len(records),'total':repeats*8,'complete':False})
                print(f"{len(records)}/24 {mode} {condition}: {row['total_env_seconds']:.3f}s",flush=True)
    summary={}
    for mode in MODES:
        times=[r['total_env_seconds'] for r in records if r['mode']==mode]
        late=[r['last5_step_seconds'] for r in records if r['mode']==mode]
        summary[mode]={'median_seconds':float(np.median(times)),'min_seconds':min(times),'max_seconds':max(times),
                       'median_last5_step_seconds':float(np.median(late))}
    for mode,s in summary.items():s['speedup_vs_baseline']=summary['baseline']['median_seconds']/s['median_seconds']
    report={'manifest_sha256':file_hash(output/'manifest.json'),'summary':summary,'runs':records,'profiles':profiles,
            'bitwise_checks':['31 public outputs including tensor bytes/dtypes, masks, reward, flags','terminal result all fields',
                              'three-world 30-day ordinary accounts','financial rows','all component and agent RNG states','raw messages'],
            'all_checks_passed':True,'scope':manifest['scope'],'research_or_hyperparameter_changes':False}
    durable_json(output/'report.json',report)
    lines=['# 观测基础设施优化：工程等价与耗时','',manifest['scope'],'',
           '| 入口 | 中位耗时 s | 基线/耗时 | 最后5步中位 s |','| --- | ---: | ---: | ---: |']
    for mode,s in summary.items():lines.append(f"| {mode} | {s['median_seconds']:.4f} | {s['speedup_vs_baseline']:.3f}x | {s['median_last5_step_seconds']:.4f} |")
    lines+=['','2条退役engineering正常/攻击轨迹，3次重复、轮换执行顺序；不是独立性能样本或正式金融实验。',
            '全部观测/张量字节、奖励、终止、终值、三世界账户/金融功能及组件/agent RNG逐位一致。',
            '未计模型API/优化更新/日志压缩写盘，不能把环境加速倍数直接称为完整训练加速倍数。',
            '保留旧入口和全部历史源指纹；新入口须由未来执行manifest显式选择。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    durable_json(output/'acceptance.json',{'manifest_sha256':file_hash(output/'manifest.json'),
        'outputs_sha256':{name:file_hash(output/name) for name in ('report.json','REPORT.md','baseline.prof','indexed.prof')},
        'bitwise_equivalence':True})
    durable_json(output/'progress.json',{'completed':24,'total':24,'complete':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);a=p.parse_args();main(a.output.resolve(),a.repeats)
