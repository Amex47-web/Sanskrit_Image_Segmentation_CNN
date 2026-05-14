"""
losses.py — Loss Functions for Binary Segmentation
=====================================================

Implements:
    - DiceLoss:     Region-based loss for imbalanced segmentation
    - BCEDiceLoss:  Combined Binary Cross-Entropy + Dice loss

Why Dice Loss matters for Sanskrit text segmentation:
    Text pixels typically occupy only 10–30% of a manuscript page.
    Standard BCE treats each pixel equally, so the model can achieve
    low loss by predicting "background" everywhere.  Dice Loss directly
    optimizes the overlap between predicted and ground-truth regions,
    forcing the model to correctly segment the minority (text) class.

    The combined BCE+Dice loss gets the best of both worlds:
    - BCE provides stable pixel-level gradients
    - Dice ensures high region-level overlap
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice = 2 × |pred ∩ target| / (|pred| + |target|)
    Loss = 1 - Dice

    A smooth term prevents division by zero when both pred and target
    are empty (e.g., a patch with no text).

    Args:
        smooth: Smoothing constant to avoid zero division.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.

        Args:
            pred:   Predicted probabilities, shape (B, 1, H, W), range [0, 1].
            target: Ground truth mask, shape (B, 1, H, W), values {0, 1}.

        Returns:
            Scalar Dice loss value.
        """
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice Loss.

    Loss = bce_weight × BCE + dice_weight × Dice

    This combination provides:
    - BCE: stable per-pixel gradients, good for learning boundaries
    - Dice: region-level overlap optimization, good for sparse classes

    Args:
        bce_weight:  Weight for the BCE component.
        dice_weight: Weight for the Dice component.
        smooth:      Smoothing constant for Dice.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined BCE + Dice loss.

        Args:
            pred:   Predicted probabilities, shape (B, 1, H, W).
            target: Ground truth mask, shape (B, 1, H, W).

        Returns:
            Scalar combined loss value.
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create a loss function from config.

    Args:
        config: Loss configuration dict with keys:
                'type' (str): 'bce', 'dice', or 'bce_dice'
                'bce_weight' (float): Weight for BCE component
                'dice_weight' (float): Weight for Dice component

    Returns:
        Configured loss module.
    """
    loss_type = config.get("type", "bce_dice")

    if loss_type == "bce":
        return nn.BCELoss()
    elif loss_type == "dice":
        return DiceLoss(smooth=config.get("smooth", 1.0))
    elif loss_type == "bce_dice":
        return BCEDiceLoss(
            bce_weight=config.get("bce_weight", 0.5),
            dice_weight=config.get("dice_weight", 0.5),
            smooth=config.get("smooth", 1.0),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
