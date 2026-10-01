"""Frozen selected BC method: three optimization seeds, one validation per seed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch

from run_combination_bc_pilot import load_samples
from train_combination_bc_round2 import verify_receipt
from wolfgym.benchmark.combination import CombinationEpisode, episode_plan
from wolfgym.env.combination import CombinationGymEnv
from wolfgym.evaluation.combination_bc import network, masked_logits, classification_metrics, BCPolicy
from wolfgym.evaluation.combination_host import read_json, file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import PublicHistory, public_frame, digest

ROOT=Path(__file__).resolve().parents[1]
FIRST=ROOT/'data/combination_bc_round1_20260927'
SECOND=ROOT/'data/combination_bc_round2_20260927'
PREP=ROOT/'data/combination_imitation_preparation_20260927'
SEEDS=(11,22,33)


def initial_state(seed):
    torch.manual_seed(seed);model=network()
    return model,torch.optim.Adam(model.parameters(),lr=.0003),torch.Generator().manual_seed(seed*10+2)


def train_one(output,seed,data,execution):
    directory=output/f'bc_seed_{seed}';directory.mkdir(exist_ok=True)
    marker=directory/'training_complete.json'
    if marker.exists():
        r=read_json(marker)
        if r['execution_id']!=execution or r['total_updates']!=2000:raise ValueError('foreign training receipt')
        for n,h in r['sha256'].items():
            if file_hash(directory/n)!=h:raise ValueError('committed training drift')
        return
    if seed==11:
        state=torch.load(SECOND/'bc_fixed_data/latest.pt',weights_only=True,map_location='cpu')
        if state['total_update']!=2000:raise ValueError('reuse update mismatch')
        curve=read_json(FIRST/'bc_unweighted/curve.json')
        curve += [{'update':r['total_update'],**r['original']} for r in read_json(SECOND/'bc_fixed_data/curve.json') if r['additional_update']]
        durable_json(directory/'curve.json',curve)
        torch.save({'execution_id':execution,'update':2000,'model':state['model'],
                    'reused_from':str(SECOND/'bc_fixed_data/latest.pt')},directory/'model_1000.pt')
        names=['curve.json','model_1000.pt']
    else:
        x,y,masks=data;model,optimizer,generator=initial_state(seed);start=0;curve=[]
        state_path=directory/'latest.pt'
        if state_path.exists():
            state=torch.load(state_path,weights_only=True,map_location='cpu')
            if state['execution_id']!=execution:raise ValueError('foreign training state')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            generator.set_state(state['batch_rng']);start=state['update'];curve=state['curves']
        for update in range(start,2001):
            if update>start:
                model.train();indices=torch.randint(len(y),(128,),generator=generator)
                loss=torch.nn.functional.cross_entropy(masked_logits(model,x[indices],masks[indices]),y[indices])
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                optimizer.zero_grad();loss.backward();optimizer.step()
            if update%25==0:
                model.eval()
                with torch.no_grad():row={'update':update,**classification_metrics(masked_logits(model,x,masks),y)}
                if curve and curve[-1]['update']==update:curve[-1]=row
                else:curve.append(row)
                state={'execution_id':execution,'update':update,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                       'batch_rng':generator.get_state(),'curves':curve}
                temporary=directory/'latest.tmp';torch.save(state,temporary);temporary.replace(state_path)
                durable_json(directory/'curve.json',curve)
                durable_json(output/'progress.json',{'stage':'final_offline_training','seed':seed,'update':update,
                    'complete':False,'pid':os.getpid(),'updated_unix':time.time()})
                if update in (500,1000,1500,2000):
                    torch.save({'execution_id':execution,'update':update,'model':model.state_dict()},directory/f'checkpoint_{update}.pt')
        # Keep the previous audited host's checkpoint basename; internal update is 2000.
        (directory/'model_1000.pt').write_bytes((directory/'checkpoint_2000.pt').read_bytes())
        names=['curve.json','latest.pt','model_1000.pt',*[f'checkpoint_{s}.pt' for s in (500,1000,1500,2000)]]
    if [r['update'] for r in read_json(directory/'curve.json')]!=list(range(0,2001,25)):
        raise ValueError('incomplete training curve')
    durable_json(marker,{'execution_id':execution,'seed':seed,'total_updates':2000,'reused_seed11':seed==11,
                        'sha256':{n:file_hash(directory/n) for n in names}})
    print(f'final seed {seed} ready; reused={seed==11}',flush=True)


def validation_job(identity,policy_id,output,execution):
    if identity['split']!='validation':raise ValueError('final host permits only frozen validation episodes')
    episode=CombinationEpisode(**{k:identity[k] for k in ('split','trajectory_seed','condition')})
    if episode.to_dict()!=identity:raise ValueError('identity mismatch')
    torch.set_num_threads(1);output=Path(output)
    directory=output/'monitor'/policy_id/identity['episode_id'].replace(':','_');directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'complete.json'
    if marker.exists():
        r=read_json(marker)
        if r!={'execution_id':execution,'sha256':file_hash(directory/'artifact.json.gz')}:raise ValueError('validation artifact drift')
        return
    policy=BCPolicy(output/policy_id/'model_1000.pt');env=CombinationGymEnv(episode);history=PublicHistory();log=[]
    try:
        _,info=env.reset()
        for day in range(30):
            frame=public_frame(info);view=history.observe(frame);action=policy(view)
            if not frame['action_mask'][action]:raise ValueError('illegal model action')
            log.append({'day':day,'frame':frame,'input_sha256':digest(view),'action':action});history.record(frame,action)
            _,reward,done,truncated,info=env.step(action)
            if reward!=0 or truncated or done!=(day==29):raise ValueError('validation contract drift')
        artifact={'execution_id':execution,'policy_id':policy_id,'result':env.evaluation_result(),'decisions':log,
                  'accounts':{k:[asdict(s) for s in v] for k,v in env._snapshots.items()},'financial_rows':env._rows}
        durable_json(directory/'artifact.json.gz',artifact)
        durable_json(marker,{'execution_id':execution,'sha256':file_hash(directory/'artifact.json.gz')})
    finally:env.close()


def declaration(workers):
    verify_receipt(FIRST,'acceptance.json');verify_receipt(SECOND,'training_acceptance.json')
    for directory,name in ((FIRST,'manifest.json'),(SECOND,'training_manifest.json'),(SECOND,'collection_manifest.json')):
        for path,sha in read_json(directory/name)['source_sha256'].items():
            if file_hash(path)!=sha:raise ValueError('ancestor source drift')
    for path,sha in read_json(PREP/'acceptance.json')['outputs_sha256'].items():
        if file_hash(ROOT/path)!=sha:raise ValueError('original data drift')
    for directory in (FIRST/'bc_unweighted',SECOND/'bc_fixed_data'):
        for n,h in read_json(directory/'training_complete.json')['sha256'].items():
            if file_hash(directory/n)!=h:raise ValueError('reuse training source changed')
    results=read_json(SECOND/'training_report.json')['results']
    candidates=[p for p in results if p!='alarm_refresh']
    selected=min(candidates,key=lambda p:(-results[p]['balanced_benefit_pp'],results[p]['mean_cost'],p))
    if selected!='bc_fixed_data':raise ValueError('frozen method not the declared selected candidate')
    sources=[Path(__file__),ROOT/'scripts/run_combination_bc_pilot.py',ROOT/'scripts/train_combination_bc_round2.py',
             ROOT/'src/wolfgym/evaluation/combination_bc.py',ROOT/'src/wolfgym/env/combination_learning.py',
             ROOT/'src/wolfgym/env/combination.py',ROOT/'src/wolfgym/evaluation/combination_public.py',
             ROOT/'src/wolfgym/env/combination_v51.json',ROOT/'docs/EXPERIMENT_PROTOCOL.md']
    return {'version':'combination-bc-final-three-seed-v1','selected_method':'bc_fixed_data','seeds':list(SEEDS),
        'data_sha256':file_hash(PREP/'demonstrations.jsonl.gz'),'preparation_acceptance_sha256':file_hash(PREP/'acceptance.json'),
        'parent_training_acceptance_sha256':file_hash(SECOND/'training_acceptance.json'),
        'reused_seed11_state_sha256':file_hash(SECOND/'bc_fixed_data/latest.pt'),
        'source_sha256':{str(p.resolve()):file_hash(p) for p in sources},
        'total_updates':2000,'batch_size':128,'learning_rate':.0003,'weights':[1,1,1,1],
        'batch_rng':'seed * 10 + 2','evaluation_checkpoint':'fixed final 2000, no checkpoint selection',
        'validation':[e.to_dict() for e in episode_plan('validation')],
        'workers':workers,'torch':torch.__version__,'numpy':np.__version__,'api_allowed':False,'test_allowed':False}


def plot_curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    panels=[('unweighted_nll','NLL'),('freeze_precision','FREEZE precision'),('freeze_recall','FREEZE recall'),
            ('correct_freeze_recall','Correct asset + FREEZE recall'),('wait_recall','WAIT recall'),('entropy','Policy entropy')]
    for seed in SEEDS:
        rows=read_json(output/f'bc_seed_{seed}'/'curve.json')
        for ax,(metric,title) in zip(axes.ravel(),panels):
            ax.plot([r['update'] for r in rows],[np.nan if r[metric] is None else r[metric] for r in rows],label=f'seed {seed}')
            ax.set(title=title,xlabel='Optimizer updates');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Fixed selected BC: all three seeds, training-set diagnostics (not validation outcomes)')
    fig.savefig(output/'final_three_seed_curves.png',dpi=160);fig.savefig(output/'final_three_seed_curves.pdf');plt.close(fig)


def main(output,workers):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=declaration(workers);execution=digest(manifest);path=output/'manifest.json'
        if path.exists() and read_json(path)!=manifest:raise ValueError('final configuration drift')
        durable_json(path,manifest);torch.set_num_threads(1);data=load_samples()
        for seed in SEEDS:train_one(output,seed,data,execution)
        plot_curves(output);print('All final models ready; starting one complete validation per seed',flush=True)
        jobs=[(ep,f'bc_seed_{seed}',str(output),execution) for seed in SEEDS for ep in manifest['validation']]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(validation_job,*job) for job in jobs]
            for count,f in enumerate(as_completed(futures),1):
                f.result();durable_json(output/'progress.json',{'stage':'final_validation','completed':count,'total':768,
                    'complete':False,'pid':os.getpid(),'updated_unix':time.time()})
                if count%32==0:print(f'validated {count}/768',flush=True)
        durable_json(output/'progress.json',{'stage':'final_validation_complete_audit_pending','completed':768,'total':768,
                    'complete':True,'audit_complete':False})
        print('All validation complete; full account/model/statistics audit pending',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.output.resolve(),a.workers)
