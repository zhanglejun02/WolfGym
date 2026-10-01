from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_combination_bc_final import declaration, initial_state, train_one, validation_job
from wolfgym.evaluation.combination_host import read_json


def test_complete_validation_plan_and_no_test():
    if not (ROOT/'data/combination_bc_round2_20260927/training_acceptance.json').exists():pytest.skip('audit fixture unavailable')
    m=declaration(4)
    assert m['seeds']==[11,22,33] and m['total_updates']==2000
    assert len(m['validation'])==256 and len({e['group_id'] for e in m['validation']})==128
    assert all(e['split']=='validation' for e in m['validation']) and not m['test_allowed']


def test_seed11_reuses_full_curve_and_model_without_training(tmp_path):
    parent=ROOT/'data/combination_bc_round2_20260927/bc_fixed_data/latest.pt'
    if not parent.exists():pytest.skip('parent fixture unavailable')
    train_one(tmp_path,11,None,'fixture-execution')
    curve=read_json(tmp_path/'bc_seed_11/curve.json')
    assert [r['update'] for r in curve]==list(range(0,2001,25))
    saved=torch.load(tmp_path/'bc_seed_11/model_1000.pt',weights_only=True,map_location='cpu')
    old=torch.load(parent,weights_only=True,map_location='cpu')
    assert saved['update']==2000
    assert all(torch.equal(v,old['model'][k]) for k,v in saved['model'].items())
    train_one(tmp_path,11,None,'fixture-execution')


def test_optimization_seed_is_reproducible_and_validation_gate(tmp_path):
    a,_,ga=initial_state(22);b,_,gb=initial_state(22);c,_,_=initial_state(33)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert not all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    assert torch.equal(ga.get_state(),gb.get_state())
    for split in ('train','test','engineering'):
        with pytest.raises(ValueError,match='only frozen validation'):validation_job({'split':split},'bc_seed_11',tmp_path,'x')
