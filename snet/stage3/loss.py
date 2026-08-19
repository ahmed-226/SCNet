"""Stage 3 loss function: combined BCE + Dice loss."""

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy + Dice loss."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        smooth = 1e-6
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
