"""
metrics.py — Segmentation Metrics
====================================

Computes per-batch and per-epoch metrics for binary segmentation:
    - IoU (Intersection over Union / Jaccard Index)
    - Dice Coefficient
    - Precision
    - Recall
    - F1 Score
    - Pixel Accuracy
"""

from __future__ import annotations

from typing import Dict

import torch
import torchmetrics


class SegmentationMetrics:
    """
    Accumulates and computes segmentation metrics across batches.

    Uses TorchMetrics for numerically stable, GPU-compatible computation.

    Args:
        threshold: Probability threshold for binarizing predictions.
        device:    Device to place metric tensors on.
    """

    def __init__(self, threshold: float = 0.5, device: str = "cpu") -> None:
        self.threshold = threshold
        self.device = device

        # TorchMetrics instances
        # Note: torchmetrics.Dice was removed in v1.9+. F1Score for binary
        # classification is mathematically equivalent to Dice coefficient.
        self.iou = torchmetrics.JaccardIndex(task="binary").to(device)
        self.dice_f1 = torchmetrics.F1Score(task="binary", threshold=threshold).to(device)
        self.precision_metric = torchmetrics.Precision(task="binary", threshold=threshold).to(device)
        self.recall_metric = torchmetrics.Recall(task="binary", threshold=threshold).to(device)
        self.f1 = torchmetrics.F1Score(task="binary", threshold=threshold).to(device)
        self.accuracy = torchmetrics.Accuracy(task="binary", threshold=threshold).to(device)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Update metrics with a batch of predictions and targets.

        Args:
            pred:   Predicted probabilities, shape (B, 1, H, W).
            target: Ground truth mask, shape (B, 1, H, W).

        Returns:
            Dict of per-batch metric values.
        """
        # Flatten spatial dims and remove channel dim
        pred_flat = (pred > self.threshold).long().view(-1)
        target_flat = target.long().view(-1)

        self.iou.update(pred_flat, target_flat)
        self.dice_f1.update(pred_flat, target_flat)
        self.precision_metric.update(pred_flat, target_flat)
        self.recall_metric.update(pred_flat, target_flat)
        self.f1.update(pred_flat, target_flat)
        self.accuracy.update(pred_flat, target_flat)

        # Return batch-level metrics (instantaneous)
        return self._compute_batch(pred_flat, target_flat)

    def _compute_batch(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute metrics for a single batch (not accumulated)."""
        intersection = (pred * target).sum().float()
        union = ((pred + target) > 0).sum().float()

        batch_iou = (intersection / (union + 1e-8)).item()
        batch_dice = (2 * intersection / (pred.sum() + target.sum() + 1e-8)).item()

        tp = (pred * target).sum().float()
        fp = (pred * (1 - target)).sum().float()
        fn = ((1 - pred) * target).sum().float()

        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-8))
        accuracy = ((pred == target).sum().float() / pred.numel()).item()

        return {
            "batch_iou": batch_iou,
            "batch_dice": batch_dice,
            "batch_precision": precision,
            "batch_recall": recall,
            "batch_f1": f1,
            "batch_accuracy": accuracy,
        }

    def compute_epoch(self) -> Dict[str, float]:
        """
        Compute accumulated epoch-level metrics and reset.

        Returns:
            Dict of epoch-level metric values.
        """
        metrics = {
            "iou": self.iou.compute().item(),
            "dice": self.dice_f1.compute().item(),
            "precision": self.precision_metric.compute().item(),
            "recall": self.recall_metric.compute().item(),
            "f1": self.f1.compute().item(),
            "pixel_accuracy": self.accuracy.compute().item(),
        }

        self.reset()
        return metrics

    def reset(self) -> None:
        """Reset all metric accumulators."""
        self.iou.reset()
        self.dice_f1.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1.reset()
        self.accuracy.reset()
