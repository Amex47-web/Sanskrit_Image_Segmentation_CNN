"""
trainer.py — Complete Training Pipeline
==========================================

Orchestrates the full training workflow:
    - Model & optimizer initialization
    - Training loop with AMP, gradient clipping
    - Validation at each epoch
    - TensorBoard logging
    - Callback dispatch (early stopping, checkpointing, LR scheduling)
    - Seed reproducibility
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.mini_unet import MiniUNet
from ..models.losses import get_loss_function
from ..utils.seed import set_seed
from .callbacks import EarlyStopping, CheckpointCallback
from .metrics import SegmentationMetrics
from .validate import validate_one_epoch


def get_device(config: dict) -> torch.device:
    """
    Select the best available device.

    Args:
        config: Hardware config with 'device' key.

    Returns:
        torch.device instance.
    """
    device_str = config.get("device", "auto")

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device_str)


class Trainer:
    """
    Full training pipeline for Sanskrit text segmentation.

    Args:
        model:        MiniUNet model instance.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        config:       Full training configuration dict.
        project_root: Root directory of the project.
    """

    def __init__(
        self,
        model: MiniUNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        project_root: str | Path = ".",
    ) -> None:
        self.config = config
        self.project_root = Path(project_root)

        # Hardware
        hw_config = config.get("hardware", {})
        self.device = get_device(hw_config)
        self.use_amp = hw_config.get("mixed_precision", True) and self.device.type == "cuda"
        set_seed(hw_config.get("seed", 42))

        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")

        # Model
        self.model = model.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.grad_clip = train_cfg.get("gradient_clip_value", 1.0)

        # Loss
        self.criterion = get_loss_function(train_cfg.get("loss", {}))

        # Optimizer
        lr = train_cfg.get("learning_rate", 1e-4)
        wd = train_cfg.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd,
        )

        # Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 5),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )

        # AMP scaler
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # Metrics
        self.train_metrics = SegmentationMetrics(device=str(self.device))
        self.val_metrics = SegmentationMetrics(device=str(self.device))

        # Callbacks
        paths_cfg = config.get("paths", {})
        checkpoint_dir = self.project_root / paths_cfg.get("checkpoint_dir", "outputs/checkpoints")

        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 15),
            mode=es_cfg.get("mode", "min"),
        )

        self.checkpoint_cb = CheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            save_best_only=train_cfg.get("save_best_only", True),
            save_every_n=train_cfg.get("save_every_n_epochs", 10),
            mode="min",
        )

        # TensorBoard
        log_cfg = config.get("logging", {})
        log_dir = self.project_root / paths_cfg.get("log_dir", "outputs/logs")
        self.writer = SummaryWriter(log_dir=str(log_dir)) if log_cfg.get("tensorboard", True) else None
        self.log_every = log_cfg.get("log_every_n_steps", 10)

        # History
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "train_iou": [], "val_iou": [],
            "lr": [],
        }

    def train(self) -> Dict[str, list]:
        """
        Execute the full training loop.

        Returns:
            Training history dict.
        """
        print(f"\n{'='*60}")
        print(f"Starting training: {self.epochs} epochs")
        print(f"{'='*60}\n")

        global_step = 0

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # --- Training epoch ---
            train_loss, train_metrics, global_step = self._train_one_epoch(epoch, global_step)

            # --- Validation epoch ---
            val_metrics = validate_one_epoch(
                self.model, self.val_loader, self.criterion,
                self.device, self.val_metrics, self.use_amp,
            )

            val_loss = val_metrics["val_loss"]
            epoch_time = time.time() - epoch_start

            # --- Logging ---
            lr = self.optimizer.param_groups[0]["lr"]
            self._log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, lr, epoch_time)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_dice"].append(train_metrics.get("dice", 0))
            self.history["val_dice"].append(val_metrics.get("dice", 0))
            self.history["train_iou"].append(train_metrics.get("iou", 0))
            self.history["val_iou"].append(val_metrics.get("iou", 0))
            self.history["lr"].append(lr)

            # --- Callbacks ---
            self.scheduler.step(val_loss)

            self.checkpoint_cb(
                self.model, self.optimizer, epoch, val_loss, val_metrics,
            )

            if self.early_stopping(val_loss):
                print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
                break

        if self.writer:
            self.writer.close()

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best val loss: {self.early_stopping.best_value:.6f}")
        print(f"{'='*60}")

        return self.history

    def _train_one_epoch(
        self, epoch: int, global_step: int,
    ) -> Tuple[float, Dict[str, float], int]:
        """Run one training epoch."""
        self.model.train()
        self.train_metrics.reset()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass with AMP
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update metrics
            with torch.no_grad():
                batch_metrics = self.train_metrics.update(outputs, masks)

            # Progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_metrics['batch_dice']:.4f}")

            # TensorBoard step logging
            if self.writer and global_step % self.log_every == 0:
                self.writer.add_scalar("step/train_loss", loss.item(), global_step)

        epoch_metrics = self.train_metrics.compute_epoch()
        avg_loss = running_loss / max(num_batches, 1)

        return avg_loss, epoch_metrics, global_step

    def _log_epoch(
        self, epoch: int, train_loss: float, val_loss: float,
        train_metrics: dict, val_metrics: dict, lr: float, epoch_time: float,
    ) -> None:
        """Log epoch results to console and TensorBoard."""
        print(
            f"\nEpoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_metrics.get('dice', 0):.4f} | "
            f"Val IoU: {val_metrics.get('iou', 0):.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        if self.writer:
            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
            self.writer.add_scalar("epoch/learning_rate", lr, epoch)
            for key, val in val_metrics.items():
                if key != "val_loss":
                    self.writer.add_scalar(f"epoch/val_{key}", val, epoch)
            for key, val in train_metrics.items():
                self.writer.add_scalar(f"epoch/train_{key}", val, epoch)
