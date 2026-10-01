"""Sequentially transfer unused shard allowance; preserve every request and $10 total."""
import fcntl,json,sqlite3
from pathlib import Path
from dotenv import dotenv_values
from run_combination_qwen_pilot import ROOT,run_one
from wolfgym.evaluation.combination_requests import RequestJournal
from wolfgym.evaluation.combination_host import read_json,file_hash
from wolfgym.evaluation.combination_public import digest,canonical_json
from wolfgym.evaluation.combination_model_host import durable_json
OUT=ROOT/'data/combination_qwen_pilot_20260928'


def allocation(charged,index,total=10000000):
    if len(charged)!=4 or any(type(x) is not int or x<0 for x in charged) or sum(charged)>total:
        raise ValueError('invalid project accounting')
    result=list(charged);result[index]+=total-sum(charged)
    assert sum(result)==total and all(a>=b for a,b in zip(result,charged))
    return result


def snapshot(path):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    meta=json.loads(db.execute('select declaration from metadata where id=1').fetchone()[0])
    identity=db.execute('select declaration from metadata where id=2').fetchone()[0]
    attempts=[dict(r) for r in db.execute('select * from attempts order by id')];db.close()
    if any(r['state']!='success' for r in attempts):raise ValueError('unresolved requests; no settlement')
    return {'declaration':meta,'identity':identity,'attempts_sha256':digest(attempts),
            'attempts':len(attempts),'charged':sum(r['charged'] for r in attempts)}


def main():
    with (OUT/'.run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read_json(OUT/'manifest.json')
        for p,sha in m['source_sha256'].items():assert file_hash(p)==sha
        sources=[Path(__file__),ROOT/'docs/EXPERIMENT_PROTOCOL.md',ROOT/'tests/test_qwen_budget_settlement.py']
        amendment={'original_manifest_sha256':file_hash(OUT/'manifest.json'),'total_limit_microusd':10000000,
                   'change':'sequential transfer of unused shard allowances only; all previous attempts and episode scopes retained',
                   'source_sha256':{str(p):file_hash(p) for p in sources}}
        ap=OUT/'budget_settlement.json'
        if ap.exists():assert read_json(ap)==amendment
        else:durable_json(ap,amendment)
        groups=list(dict.fromkeys(e['group_id'] for e in m['episodes']))
        key=dotenv_values(ROOT/'.env')['OPENROUTER_API_KEY']
        for i in range(4):
            es=[e for e in m['episodes'] if e['group_id'] in groups[i*2:i*2+2]]
            if all((OUT/'episodes'/e['episode_id'].replace(':','_')/'complete.json').exists() for e in es):continue
            stage=OUT/f'budget_settlement_{i}.json'
            if stage.exists():raise ValueError('existing settlement stage requires review, not automatic repeated migration')
            paths=[OUT/f'requests_{j}.sqlite' for j in range(4)]
            locks=[]
            try:
                for p in paths:
                    f=p.with_suffix(p.suffix+'.lock').open('a+');locks.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                before=[snapshot(p) for p in paths];limits=allocation([x['charged'] for x in before],i)
                after=[]
                for j,x in enumerate(before):
                    d=dict(x['declaration'])
                    assert d['contract']=={'manifest_sha256':digest(m),'shard':j}
                    assert d['max_attempts']==120 and d['reservation_microusd']==110000 and d['retries']==0
                    d['max_cost_microusd']=limits[j];after.append(d)
                receipt={'target_shard':i,'before':before,'after_declarations':after,'applied':False}
                durable_json(stage,receipt)
                for p,d in zip(paths,after):
                    db=sqlite3.connect(p);db.execute('PRAGMA synchronous=FULL')
                    with db:db.execute('update metadata set declaration=? where id=1',(canonical_json(d),))
                    db.close()
                for p,old,d in zip(paths,before,after):
                    new=snapshot(p)
                    assert new['declaration']==d
                    assert all(new[k]==old[k] for k in ('identity','attempts_sha256','attempts','charged'))
                receipt['applied']=True;durable_json(stage,receipt)
            finally:
                for f in locks:f.close()
            d=after[i]
            with RequestJournal(paths[i],contract=d['contract'],max_attempts=120,
                    max_cost_microusd=d['max_cost_microusd'],reservation_microusd=110000,retries=0) as journal:
                for e in es:run_one(OUT,e,m,journal,key)
            assert sum(snapshot(p)['charged'] for p in paths)<=10000000
        for p,sha in m['source_sha256'].items():assert file_hash(p)==sha
        durable_json(OUT/'budget_settlement_complete.json',{'complete_episodes':16,'total_microusd':sum(snapshot(OUT/f'requests_{i}.sqlite')['charged'] for i in range(4))})
        print('Sequential settlement completed all 16 episodes inside original $10 total',flush=True)


if __name__=='__main__':main()
