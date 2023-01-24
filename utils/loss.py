import torch
from torch import nn
from torch.nn import functional as F


class BCEWeightedLoss(nn.Module):
    def __init__(self, logits=True):
        super(BCEWeightedLoss, self).__init__()
        self.logits = logits

    def forward(self, input, target, weight=None):
        if self.logits:
            return F.binary_cross_entropy_with_logits(
                input, target, weight, reduction='mean')
        else:
            return F.binary_cross_entropy(
                input, target, weight, reduction='mean')
