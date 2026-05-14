"""
callbacks.py — Training Callbacks
====================================

Modular callback system for the training loop:
    - EarlyStopping:     Stop training when validation metric plateaus
    - CheckpointCallback: Save best and periodic model checkpoints
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience:  Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode:      'min' (for loss) or 'max' (for accuracy/dice).
        verbose:   Print early stopping messages.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = False
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (best={self.best_value:.6f})")

        self.should_stop = self.counter >= self.patience
        return self.should_stop


class CheckpointCallback:
    """
    Save model checkpoints during training.

    Saves:
        - Best model (by monitored metric)
        - Periodic checkpoints every N epochs

    Args:
        checkpoint_dir:   Directory to save checkpoints.
        save_best_only:   If True, only save when metric improves.
        save_every_n:     Save a checkpoint every N epochs (0 = disabled).
        mode:             'min' or 'max' for the monitored metric.
        verbose:          Print checkpoint messages.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_best_only: bool = True,
        save_every_n: int = 10,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.save_every_n = save_every_n
        self.mode = mode
        self.verbose = verbose

        self.best_value: Optional[float] = None

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        value: float,
        metrics: dict,
    ) -> Optional[Path]:
        """
        Save checkpoint if appropriate.

        Args:
            model:     The model to save.
            optimizer: The optimizer to save.
            epoch:     Current epoch number.
            value:     Current metric value.
            metrics:   Full metrics dict.

        Returns:
            Path to saved checkpoint, or None.
        """
        saved_path = None

        # Check if this is the best model
        is_best = False
        if self.best_value is None:
            is_best = True
        elif self.mode == "min" and value < self.best_value:
            is_best = True
        elif self.mode == "max" and value > self.best_value:
            is_best = True

        if is_best:
            self.best_value = value
            path = self.checkpoint_dir / "best_model.pth"
            self._save(model, optimizer, epoch, metrics, path)
            saved_path = path
            if self.verbose:
                print(f"  ✓ Best model saved (val={value:.6f}) → {path}")

        # Periodic save
        if self.save_every_n > 0 and (epoch + 1) % self.save_every_n == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pth"
            self._save(model, optimizer, epoch, metrics, path)
            if self.verbose:
                print(f"  ✓ Periodic checkpoint → {path}")

        return saved_path

    @staticmethod
    def _save(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        path: Path,
    ) -> None:
        """Save model state dict, optimizer state, epoch, and metrics."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
