"""Bounded behavior cloning helpers on the unchanged public learning encoding."""
import numpy as np
import torch
from torch import nn

from wolfgym.env.combination_learning import VECTOR_WIDTH, encode_public_view


def network():
    return nn.Sequential(nn.Linear(VECTOR_WIDTH,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,4))


def masked_logits(model, x, masks):
    if not torch.all(masks[:,0]) or not torch.all(masks.any(dim=1)):
        raise ValueError('WAIT must remain legal')
    return model(x).masked_fill(~masks,-1e9)


def classification_metrics(logits, labels):
    probabilities=logits.softmax(-1)
    predicted=logits.argmax(-1)
    true_freeze=labels!=0; predicted_freeze=predicted!=0
    tp=int((true_freeze & predicted_freeze).sum())
    fp=int((~true_freeze & predicted_freeze).sum())
    fn=int((true_freeze & ~predicted_freeze).sum())
    tn=int((~true_freeze & ~predicted_freeze).sum())
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    wait_recall=tn/(tn+fp) if tn+fp else None
    both=true_freeze & predicted_freeze
    return {'unweighted_nll':float(nn.functional.cross_entropy(logits,labels)),
        'overall_accuracy':float((predicted==labels).float().mean()),
        'freeze_precision':precision,'freeze_recall':recall,
        'freeze_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'wait_recall':wait_recall,
        'binary_balanced_accuracy':(recall+wait_recall)/2 if recall is not None and wait_recall is not None else None,
        'target_accuracy_when_both_freeze':float((predicted[both]==labels[both]).float().mean()) if both.any() else None,
        'correct_freeze_recall':float((predicted[true_freeze]==labels[true_freeze]).float().mean()) if true_freeze.any() else None,
        'predicted_freeze_rate':float(predicted_freeze.float().mean()),
        'entropy':float(-(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1).mean()),
        'confusion_WAIT_FREEZE':{'tp':tp,'fp':fp,'fn':fn,'tn':tn}}


class BCPolicy:
    def __init__(self, checkpoint):
        self.model=network()
        self.model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['model'])
        self.model.eval()

    def __call__(self, view):
        x=torch.from_numpy(encode_public_view(view)).unsqueeze(0)
        mask=torch.tensor([view['observations'][-1]['action_mask']],dtype=torch.bool)
        with torch.no_grad():
            return int(masked_logits(self.model,x,mask).argmax(-1).item())
