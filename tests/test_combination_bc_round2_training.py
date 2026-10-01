"""Same model/optimizer/RNG starts for both second-round branches."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_combination_bc_round2 import initialize, load_datasets, declaration


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def test_starts_reuse_parent_model_optimizer_and_generator():
    path=ROOT/'data/combination_bc_round1_20260927/bc_unweighted/latest.pt'
    if not path.exists():pytest.skip('parent fixture unavailable')
    parent=torch.load(path,weights_only=True,map_location='cpu')
    a,oa,ga=initialize(path);b,ob,gb=initialize(path)
    assert same(a.state_dict(),parent['model']) and same(a.state_dict(),b.state_dict())
    assert same(oa.state_dict(),parent['optimizer']) and same(oa.state_dict(),ob.state_dict())
    assert torch.equal(ga.get_state(),parent['batch_rng']) and torch.equal(ga.get_state(),gb.get_state())
    assert torch.equal(torch.randint(2891,(128,),generator=ga),torch.randint(2891,(128,),generator=gb))
    # A branch's update must not mutate its peer's parameters or optimizer tensors.
    old=copy.deepcopy(ob.state_dict())
    oa.zero_grad();a(torch.zeros(1,1895)).sum().backward();oa.step()
    assert same(old,ob.state_dict())
    assert not same(a.state_dict(),b.state_dict())


def test_data_composition_and_equal_update_contract():
    output=ROOT/'data/combination_bc_round2_20260927'
    if not (output/'collection_acceptance.json').exists():pytest.skip('collection fixture unavailable')
    d=load_datasets(output)
    assert [len(d[k][1]) for k in ('original','student','aggregated')]==[2891,6529,9420]
    assert int((d['aggregated'][1]!=0).sum())==545
    assert all(torch.equal(d['original'][i],d['aggregated'][i][:2891]) for i in range(3))
    assert all(torch.equal(d['student'][i],d['aggregated'][i][2891:]) for i in range(3))
    m=declaration(output,4)
    assert m['additional_updates_each']==1000 and m['class_weights']==[1,1,1,1]
    assert len(m['monitor'])==128 and all(e['split']=='train' for e in m['monitor'])
