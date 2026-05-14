"""
checkpoint.py — Model Checkpoint Utilities
=============================================

Save and load model checkpoints with full training state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str | Path,
    extra: Optional[dict] = None,
) -> Path:
    """
    Save a training checkpoint.

    Saves model weights, optimizer state, epoch number, and metrics.

    Args:
        model:     The model to save.
        optimizer: The optimizer to save.
        epoch:     Current epoch number.
        metrics:   Dict of metric values.
        path:      Output file path (.pth).
        extra:     Optional extra data to include.

    Returns:
        Path to the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if extra:
        state.update(extra)

    torch.save(state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, Dict]:
    """
    Load a training checkpoint.

    Args:
        path:      Path to the .pth checkpoint file.
        model:     Model to load weights into.
        optimizer: Optional optimizer to restore state.
        device:    Device to map tensors to.

    Returns:
        Tuple of (model, optimizer, epoch, metrics).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"Loaded checkpoint from epoch {epoch} → {path}")
    return model, optimizer, epoch, metrics
