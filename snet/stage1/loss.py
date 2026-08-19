"""Stage 1 loss function — weighted MSE."""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """MSE with higher weight for spine regions (pos_weight=10 by default)."""

    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        weight_mask = torch.where(target > 0.01, self.pos_weight, 1.0)
        return (weight_mask * ((pred - target) ** 2)).mean()
