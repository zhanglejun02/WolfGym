"""Two fixed BC loss treatments, persistent curves and complete train-monitor rollouts.

No test, external API, market changes or outcome-based extra training.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.env.combination_learning import encode_public_view
from wolfgym.evaluation.combination_bc import network, masked_logits, classification_metrics, BCPolicy
from wolfgym.evaluation.combination_host import read_json,file_hash,load_detector
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import PublicHistory,public_frame,digest,BaselinePolicy

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'data/combination_imitation_preparation_20260927'


def load_samples():
    with gzip.open(PREP/'demonstrations.jsonl.gz','rt') as stream:
        rows=[json.loads(line) for line in stream]
    rows=[r for r in rows if r['view']['observations'][-1]['remaining_budget']>=4]
    x=torch.from_numpy(np.stack([encode_public_view(r['view']) for r in rows]))
    y=torch.tensor([r['action'] for r in rows],dtype=torch.long)
    masks=torch.tensor([r['view']['observations'][-1]['action_mask'] for r in rows],dtype=torch.bool)
    if len(rows)!=2891 or int((y!=0).sum())!=192:raise ValueError('declared class counts changed')
    return x,y,masks


def train_one(output,config,x,y,masks,execution_id):
    directory=output/config['id']; directory.mkdir(exist_ok=True)
    marker=directory/'training_complete.json'
    if marker.exists():
        receipt=read_json(marker)
        if receipt['execution_id']!=execution_id or any(file_hash(directory/k)!=v for k,v in receipt['sha256'].items()):
            raise ValueError('committed training changed')
        return
    torch.manual_seed(11)
    model=network();optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    batches=torch.Generator().manual_seed(112)
    weights=torch.tensor(config['class_weights'],dtype=torch.float32)
    curves=[];start=0
    state_path=directory/'latest.pt'
    if state_path.exists():
        state=torch.load(state_path,weights_only=True,map_location='cpu')
        if state['execution_id']!=execution_id:raise ValueError('foreign state')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        batches.set_state(state['batch_rng']);start=state['update'];curves=state['curves']
    for update in range(start,1001):
        if update>start:
            model.train()
            idx=torch.randint(len(y),(128,),generator=batches)
            logits=masked_logits(model,x[idx],masks[idx])
            loss=torch.nn.functional.cross_entropy(logits,y[idx],weight=weights)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
        if update%25==0:
            model.eval()
            with torch.no_grad():
                logits=masked_logits(model,x,masks)
                row={'update':update,**classification_metrics(logits,y),
                    'weighted_nll':float(torch.nn.functional.cross_entropy(logits,y,weight=weights))}
            if curves and curves[-1]['update']==update:curves[-1]=row
            else:curves.append(row)
            state={'execution_id':execution_id,'update':update,'model':model.state_dict(),
                   'optimizer':optimizer.state_dict(),'batch_rng':batches.get_state(),'curves':curves}
            temporary=directory/'latest.tmp';torch.save(state,temporary);temporary.replace(state_path)
            durable_json(directory/'curve.json',curves)
            durable_json(output/'progress.json',{'stage':'offline_training','configuration':config['id'],
                'update':update,'complete':False,'pid':os.getpid(),'updated_unix':time.time()})
            if update in (250,500,1000):
                torch.save({'model':model.state_dict(),'update':update,'execution_id':execution_id},directory/f'model_{update}.pt')
    names=['curve.json','latest.pt','model_250.pt','model_500.pt','model_1000.pt']
    durable_json(marker,{'execution_id':execution_id,'updates':1000,'sha256':{n:file_hash(directory/n) for n in names}})
    print(f"trained {config['id']} freeze_recall={curves[-1]['freeze_recall']:.4f} precision={curves[-1]['freeze_precision']}",flush=True)


def plot_curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    panels=[('unweighted_nll','Unweighted NLL'),('freeze_precision','FREEZE precision'),('freeze_recall','FREEZE recall'),
            ('correct_freeze_recall','Correct asset + FREEZE recall'),('wait_recall','WAIT recall'),('predicted_freeze_rate','Predicted FREEZE fraction')]
    for name in ('bc_unweighted','bc_balanced_wait_freeze'):
        rows=read_json(output/name/'curve.json')
        for ax,(metric,title) in zip(axes.ravel(),panels):
            values=[np.nan if r[metric] is None else r[metric] for r in rows]
            ax.plot([r['update'] for r in rows],values,label=name.replace('bc_',''),linewidth=1.7)
            ax.set(title=title,xlabel='Optimizer updates');ax.grid(alpha=.2)
            if metric!='unweighted_nll':ax.set_ylim(-.03,1.03)
    axes[1,2].axhline(192/2891,color='gray',linestyle='--',label='Teacher FREEZE frequency')
    axes[0,0].legend(fontsize=8);axes[1,2].legend(fontsize=8)
    fig.suptitle('Round 1 BC: training-set diagnostics only (not closed-loop protection)',fontsize=14)
    fig.savefig(output/'round1_training_curves.png',dpi=160)
    fig.savefig(output/'round1_training_curves.pdf')
    plt.close(fig)


def monitor_job(episode,policy_id,checkpoint,output,execution_id):
    torch.set_num_threads(1)
    episode=CombinationEpisode(**{k:episode[k] for k in ('split','trajectory_seed','condition')})
    if episode.split!='train':raise ValueError('monitor must stay inside train')
    directory=Path(output)/'monitor'/policy_id/episode.to_dict()['episode_id'].replace(':','_')
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'complete.json'
    if marker.exists():
        r=read_json(marker)
        if r['execution_id']!=execution_id or file_hash(directory/'artifact.json.gz')!=r['sha256']:raise ValueError('monitor artifact changed')
        return read_json(directory/'artifact.json.gz')['result']
    policy=(BCPolicy(checkpoint) if checkpoint else BaselinePolicy({'id':'alarm_refresh','family':'alarm_refresh'},
        load_detector(ROOT/'data/task_validation_v4_20260923/detector.json')))
    env=CombinationGymEnv(episode);history=PublicHistory();log=[]
    try:
        _,info=env.reset()
        for day in range(30):
            frame=public_frame(info);view=history.observe(frame);action=policy(view)
            if not frame['action_mask'][action]:raise ValueError('illegal BC action')
            log.append({'day':day,'frame':frame,'input_sha256':digest(view),'action':action})
            history.record(frame,action)
            _,reward,done,truncated,info=env.step(action)
            if reward!=0 or truncated or done!=(day==29):raise ValueError('evaluation contract changed')
        result=env.evaluation_result()
        artifact={'execution_id':execution_id,'policy_id':policy_id,'result':result,'decisions':log,
                  'accounts':{k:[asdict(s) for s in v] for k,v in env._snapshots.items()},'financial_rows':env._rows}
        durable_json(directory/'artifact.json.gz',artifact)
        durable_json(marker,{'execution_id':execution_id,'sha256':file_hash(directory/'artifact.json.gz')})
        return result
    finally:env.close()


def main(output,workers):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        candidate=read_json(PREP/'bc_candidate.json')
        accepted=read_json(PREP/'acceptance.json')
        for p,h in accepted['outputs_sha256'].items():
            if file_hash(ROOT/p)!=h:raise ValueError('prepared data/configuration changed')
        sources=[Path(__file__),ROOT/'src/wolfgym/evaluation/combination_bc.py',
            ROOT/'src/wolfgym/env/combination_learning.py',ROOT/'src/wolfgym/evaluation/combination_public.py',
            ROOT/'docs/EXPERIMENT_PROTOCOL.md']
        declaration={'version':'combination-bc-round1-v1','candidate':candidate,'data_receipt':file_hash(PREP/'acceptance.json'),
            'source_sha256':{str(p):file_hash(p) for p in sources},'torch':torch.__version__,
            'numpy':np.__version__,'device':'cpu','workers':workers,'test_allowed':False,'api_allowed':False,
            'monitor':read_json(PREP/'partitions.json')['development_monitor'],
            'loss_smoothing':0,'sampling':'uniform eligible frames with replacement; common batch_rng112',
            'checkpoint_selection':'fixed final1000, no best training-loss selection'}
        execution_id=digest(declaration)
        path=output/'manifest.json'
        if path.exists() and read_json(path)!=declaration:raise ValueError('execution configuration drift')
        durable_json(path,declaration)
        torch.set_num_threads(1)
        x,y,masks=load_samples()
        for config in candidate['configurations']:train_one(output,config,x,y,masks,execution_id)
        plot_curves(output)
        print('offline training and curves complete; starting train-monitor rollouts',flush=True)
        jobs=[(ep,p,str(output/p/'model_1000.pt') if p!='alarm_refresh' else None,str(output),execution_id)
              for p in ('bc_unweighted','bc_balanced_wait_freeze','alarm_refresh') for ep in declaration['monitor']]
        completed=0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(monitor_job,*job) for job in jobs]
            for future in as_completed(futures):
                future.result();completed+=1
                durable_json(output/'progress.json',{'stage':'train_monitor','completed':completed,'total':len(jobs),
                    'complete':False,'pid':os.getpid(),'updated_unix':time.time()})
                if completed%16==0:print(f'monitor {completed}/{len(jobs)}',flush=True)
        durable_json(output/'progress.json',{'stage':'monitor_complete_audit_pending','complete':True,
            'audit_complete':False,'completed':completed,'total':len(jobs)})
        print('monitor complete; independent account/action audit required before selection',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.output.resolve(),a.workers)
