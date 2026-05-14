"""
validate.py — Validation Loop
================================

Runs one complete validation epoch: forward pass on all validation
batches, accumulates metrics, and returns epoch-level results.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import SegmentationMetrics


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: SegmentationMetrics,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Run one validation epoch.

    Args:
        model:      The segmentation model.
        val_loader: Validation DataLoader.
        criterion:  Loss function.
        device:     Device to run on.
        metrics:    SegmentationMetrics instance.
        use_amp:    Whether to use automatic mixed precision.

    Returns:
        Dict with 'val_loss' and all epoch-level metrics.
    """
    model.eval()
    metrics.reset()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)

            running_loss += loss.item()
            num_batches += 1

            # Update metrics
            metrics.update(outputs, masks)

    # Compute epoch metrics
    epoch_metrics = metrics.compute_epoch()
    epoch_metrics["val_loss"] = running_loss / max(num_batches, 1)

    return epoch_metrics
