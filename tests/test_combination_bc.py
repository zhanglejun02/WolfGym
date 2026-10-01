import pytest
import torch

from wolfgym.evaluation.combination_bc import classification_metrics,masked_logits


def test_always_wait_cannot_pass_freeze_metrics():
    labels=torch.tensor([0]*19+[2])
    logits=torch.tensor([[9.,0.,0.,0.]]*20)
    m=classification_metrics(logits,labels)
    assert m['overall_accuracy']==pytest.approx(.95)
    assert m['freeze_recall']==0 and m['correct_freeze_recall']==0
    assert m['freeze_precision'] is None and m['binary_balanced_accuracy']==.5


def test_wrong_asset_does_not_get_correct_target_credit():
    labels=torch.tensor([0,1,2,3])
    logits=torch.tensor([[9.,0.,0.,0.],[0.,0.,9.,0.],[0.,0.,0.,9.],[0.,9.,0.,0.]])
    m=classification_metrics(logits,labels)
    assert m['freeze_precision']==m['freeze_recall']==1
    assert m['correct_freeze_recall']==m['target_accuracy_when_both_freeze']==0


def test_mask_prevents_spending_exhausted_budget():
    model=torch.nn.Identity()
    x=torch.tensor([[0.,9.,8.,7.]])
    logits=masked_logits(model,x,torch.tensor([[True,False,False,False]]))
    assert logits.argmax(-1).item()==0
    with pytest.raises(ValueError):masked_logits(model,x,torch.tensor([[False,True,True,True]]))


def test_binary_balancing_gives_equal_total_mass_without_asset_rebalancing():
    weights=torch.tensor([2891/(2*2699)]+[2891/(2*192)]*3)
    assert (weights[0]*2699).item()==pytest.approx((weights[1]*192).item())
    counts=torch.tensor([2699,79,54,59])
    assert (weights*counts)[1]/(weights*counts)[2]==pytest.approx(79/54)
