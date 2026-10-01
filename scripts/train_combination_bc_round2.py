"""Matched-update continuation of fixed-data versus aggregated-data BC."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch

from run_combination_bc_pilot import load_samples, monitor_job
from wolfgym.env.combination_learning import encode_public_view
from wolfgym.evaluation.combination_bc import network, masked_logits, classification_metrics
from wolfgym.evaluation.combination_host import read_json, file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest

ROOT=Path(__file__).resolve().parents[1]
FIRST=ROOT/'data/combination_bc_round1_20260927'
PREP=ROOT/'data/combination_imitation_preparation_20260927'
BRANCHES=('bc_fixed_data','bc_aggregated_data')


def verify_receipt(directory,name):
    receipt=read_json(directory/name)
    for path,sha in receipt['outputs_sha256'].items():
        if file_hash(directory/path)!=sha:raise ValueError('receipt artifact drift')
    return receipt


def load_datasets(output):
    original=load_samples()
    rows=read_json(output/'student_demonstrations.json.gz')
    if len(rows)!=7680 or any(set(r)!={'sample_id','view','action'} for r in rows):
        raise ValueError('student dataset completeness/private fields')
    rows=[r for r in rows if r['view']['observations'][-1]['remaining_budget']>=4]
    student=(torch.from_numpy(np.stack([encode_public_view(r['view']) for r in rows])),
             torch.tensor([r['action'] for r in rows],dtype=torch.long),
             torch.tensor([r['view']['observations'][-1]['action_mask'] for r in rows],dtype=torch.bool))
    if len(rows)!=6529 or int((student[1]!=0).sum())!=353:raise ValueError('audited student class counts drift')
    return {'original':original,'student':student,
            'aggregated':tuple(torch.cat([a,b]) for a,b in zip(original,student))}


def initialize(parent):
    state=torch.load(parent,weights_only=True,map_location='cpu')
    if state['update']!=1000:raise ValueError('parent update count drift')
    model=network();model.load_state_dict(state['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    optimizer.load_state_dict(state['optimizer'])
    generator=torch.Generator();generator.set_state(state['batch_rng'])
    return model,optimizer,generator


def describe(model,data):
    with torch.no_grad():
        x,y,mask=data
        return classification_metrics(masked_logits(model,x,mask),y)


def train_branch(output,branch,datasets,execution):
    directory=output/branch;directory.mkdir(exist_ok=True)
    marker=directory/'training_complete.json'
    if marker.exists():
        r=read_json(marker)
        if r['execution_id']!=execution or r['additional_updates']!=1000:raise ValueError('foreign training receipt')
        for n,h in r['sha256'].items():
            if file_hash(directory/n)!=h:raise ValueError('committed model/curve changed')
        return
    model,optimizer,generator=initialize(FIRST/'bc_unweighted/latest.pt')
    start=0;curve=[]
    state_path=directory/'latest.pt'
    if state_path.exists():
        state=torch.load(state_path,weights_only=True,map_location='cpu')
        if state['execution_id']!=execution:raise ValueError('foreign continuation state')
        start=state['additional_update'];curve=state['curves']
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);generator.set_state(state['batch_rng'])
    fit_key='original' if branch=='bc_fixed_data' else 'aggregated'
    x,y,masks=datasets[fit_key]
    for update in range(start,1001):
        if update>start:
            model.train();indices=torch.randint(len(y),(128,),generator=generator)
            logits=masked_logits(model,x[indices],masks[indices])
            loss=torch.nn.functional.cross_entropy(logits,y[indices])
            if not torch.isfinite(loss):raise ValueError('nonfinite optimization loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
        if update%25==0:
            model.eval()
            measures={k:describe(model,v) for k,v in datasets.items()}
            row={'additional_update':update,'total_update':1000+update,'fit_dataset':fit_key,
                 'fit':measures[fit_key],'original':measures['original'],'student':measures['student']}
            if curve and curve[-1]['additional_update']==update:curve[-1]=row
            else:curve.append(row)
            state={'execution_id':execution,'additional_update':update,'total_update':1000+update,
                   'model':model.state_dict(),'optimizer':optimizer.state_dict(),'batch_rng':generator.get_state(),'curves':curve}
            temporary=directory/'latest.tmp';torch.save(state,temporary);temporary.replace(state_path)
            durable_json(directory/'curve.json',curve)
            durable_json(output/'training_progress.json',{'stage':'round2_offline_training','branch':branch,
                'additional_update':update,'complete':False,'pid':os.getpid(),'updated_unix':time.time()})
            if update in (250,500,1000):
                torch.save({'model':model.state_dict(),'update':1000+update,'additional_update':update,
                            'execution_id':execution},directory/f'model_{update}.pt')
    names=['curve.json','latest.pt','model_250.pt','model_500.pt','model_1000.pt']
    durable_json(marker,{'execution_id':execution,'additional_updates':1000,'total_updates':2000,
                        'sha256':{n:file_hash(directory/n) for n in names}})
    print(f'trained {branch}; additional_updates=1000; final metrics {curve[-1]}',flush=True)


def plot_curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=[('unweighted_nll','NLL'),('freeze_precision','FREEZE precision'),
             ('freeze_recall','FREEZE recall'),('correct_freeze_recall','Correct asset + FREEZE recall'),
             ('wait_recall','WAIT recall')]
    fig,axes=plt.subplots(2,5,figsize=(18,7),constrained_layout=True)
    for branch in BRANCHES:
        rows=read_json(output/branch/'curve.json')
        for i,dataset in enumerate(('original','student')):
            for j,(metric,title) in enumerate(metrics):
                ax=axes[i,j]
                ax.plot([r['total_update'] for r in rows],
                        [np.nan if r[dataset][metric] is None else r[dataset][metric] for r in rows],
                        label=branch.removeprefix('bc_'),linewidth=1.7)
                ax.set(title=f'{dataset}: {title}',xlabel='Total optimizer updates');ax.grid(alpha=.2)
                if metric!='unweighted_nll':ax.set_ylim(-.03,1.03)
    axes[0,0].legend();axes[1,0].legend()
    fig.suptitle('Round 2: matched +1000 updates, common train-data probes (NOT validation or closed-loop benefit)')
    fig.savefig(output/'round2_training_curves.png',dpi=160);fig.savefig(output/'round2_training_curves.pdf');plt.close(fig)


def declaration(output,workers):
    receipt=verify_receipt(output,'collection_acceptance.json')
    collection=read_json(output/'collection_manifest.json')
    if not receipt['audit_complete'] or digest(collection)!=receipt['execution_id']:raise ValueError('collection not audited')
    for path,sha in collection['source_sha256'].items():
        if file_hash(path)!=sha:raise ValueError('collection source drift')
    verify_receipt(FIRST,'acceptance.json')
    if file_hash(FIRST/'acceptance.json')!=collection['parent_acceptance_sha256']:
        raise ValueError('first-round receipt changed')
    if file_hash(FIRST/'bc_unweighted/latest.pt')!=collection['parent_optimizer_state_sha256']:
        raise ValueError('parent optimizer state changed')
    if file_hash(PREP/'acceptance.json')!=collection['data_acceptance_sha256']:raise ValueError('preparation receipt changed')
    for p,h in read_json(PREP/'acceptance.json')['outputs_sha256'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('original dataset drift')
    sources=[Path(__file__),ROOT/'scripts/run_combination_bc_pilot.py',ROOT/'src/wolfgym/evaluation/combination_bc.py',
             ROOT/'src/wolfgym/env/combination_learning.py',ROOT/'src/wolfgym/evaluation/combination_public.py',
             ROOT/'docs/EXPERIMENT_PROTOCOL.md']
    return {'version':'combination-bc-round2-training-v1','collection_acceptance_sha256':file_hash(output/'collection_acceptance.json'),
        'collection_execution_id':receipt['execution_id'],'source_sha256':{str(p.resolve()):file_hash(p) for p in sources},
        'parent_state_sha256':collection['parent_optimizer_state_sha256'],'branches':list(BRANCHES),
        'additional_updates_each':1000,'batch_size':128,'learning_rate':.0003,'class_weights':[1.,1.,1.,1.],
        'initialization':'same saved first-round model, Adam state and batch RNG for both branches',
        'sampling':'uniform eligible frames with replacement; dataset-specific ranges',
        'curve_schedule':list(range(0,1001,25)),'checkpoint_selection':'fixed final +1000; no best-checkpoint selection',
        'monitor':read_json(PREP/'partitions.json')['development_monitor'],
        'workers':workers,'torch':torch.__version__,'numpy':np.__version__,'api_allowed':False,'test_allowed':False}


def main(output,workers):
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=declaration(output,workers);execution=digest(manifest);path=output/'training_manifest.json'
        if path.exists() and read_json(path)!=manifest:raise ValueError('training configuration drift')
        durable_json(path,manifest);torch.set_num_threads(1)
        datasets=load_datasets(output)
        for branch in BRANCHES:train_branch(output,branch,datasets,execution)
        plot_curves(output)
        print('Both continuation branches trained; curves ready; starting 256 monitor episodes',flush=True)
        jobs=[(ep,branch,str(output/branch/'model_1000.pt'),str(output),execution)
              for branch in BRANCHES for ep in manifest['monitor']]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(monitor_job,*job) for job in jobs]
            for count,f in enumerate(as_completed(futures),1):
                f.result()
                durable_json(output/'training_progress.json',{'stage':'round2_train_monitor','completed':count,'total':256,
                    'complete':False,'pid':os.getpid(),'updated_unix':time.time()})
                if count%16==0:print(f'monitor {count}/256',flush=True)
        durable_json(output/'training_progress.json',{'stage':'round2_monitor_complete_audit_pending','complete':True,
                    'audit_complete':False,'completed':256,'total':256})
        print('Round2 monitor complete; final account/action audit pending',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.output.resolve(),a.workers)
