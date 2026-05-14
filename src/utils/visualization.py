"""
visualization.py — Training & Prediction Visualization
=========================================================

Generate publication-quality plots for:
    - Training/validation loss and metric curves
    - Prediction comparisons (original / GT / prediction / cleaned)
    - Overlay visualizations
    - Error heatmaps
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plot training and validation loss/metric curves.

    Args:
        history: Dict with keys like 'train_loss', 'val_loss',
                 'train_dice', 'val_dice', etc.
        save_path: Path to save the figure (None = show interactively).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # Loss
    ax = axes[0, 0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss", color="#2196F3")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss", color="#F44336")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dice
    ax = axes[0, 1]
    if "train_dice" in history:
        ax.plot(history["train_dice"], label="Train Dice", color="#4CAF50")
    if "val_dice" in history:
        ax.plot(history["val_dice"], label="Val Dice", color="#FF9800")
    ax.set_title("Dice Coefficient")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IoU
    ax = axes[1, 0]
    if "train_iou" in history:
        ax.plot(history["train_iou"], label="Train IoU", color="#9C27B0")
    if "val_iou" in history:
        ax.plot(history["val_iou"], label="Val IoU", color="#00BCD4")
    ax.set_title("IoU (Jaccard Index)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    if "lr" in history:
        ax.plot(history["lr"], label="Learning Rate", color="#607D8B")
        ax.set_yscale("log")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prediction_comparison(
    original: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    cleaned: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Prediction Comparison",
) -> None:
    """
    4-panel comparison: Original | Ground Truth | Prediction | Cleaned.

    Args:
        original:     Original grayscale image.
        ground_truth: Ground truth binary mask.
        prediction:   Raw model prediction (probability or binary).
        cleaned:      Post-processed binary mask (optional).
        save_path:    Path to save (None = show).
        title:        Figure title.
    """
    n_panels = 4 if cleaned is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if cleaned is not None:
        axes[3].imshow(cleaned, cmap="gray")
        axes[3].set_title("Cleaned")
        axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    save_path: Optional[str | Path] = None,
    title: str = "Mask Overlay",
) -> None:
    """
    Plot the image with the mask overlaid in green.

    Args:
        image:     Grayscale or BGR image.
        mask:      Binary mask.
        alpha:     Overlay opacity.
        save_path: Path to save (None = show).
        title:     Figure title.
    """
    if len(image.shape) == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay = display.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    overlay[mask > 0] = green

    blended = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(blended)
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_error_heatmap(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str | Path] = None,
    title: str = "Error Heatmap",
) -> None:
    """
    Visualize prediction errors as a heatmap.

    Color coding:
        - Green:  True Positive  (correctly predicted text)
        - Red:    False Positive (noise predicted as text)
        - Blue:   False Negative (missed text)
        - Black:  True Negative  (correct background)

    Args:
        ground_truth: Binary GT mask, values {0, 1} or {0, 255}.
        prediction:   Binary prediction mask.
        save_path:    Path to save.
        title:        Figure title.
    """
    gt = (ground_truth > 0).astype(np.uint8)
    pred = (prediction > 0).astype(np.uint8)

    # Create RGB error map
    error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)

    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)

    error_map[tp] = [0, 200, 0]    # Green — correct text
    error_map[fp] = [200, 0, 0]    # Red — false positive
    error_map[fn] = [0, 0, 200]    # Blue — false negative

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(error_map)
    ax.set_title(title)
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", label="True Positive"),
        Patch(facecolor="red", label="False Positive"),
        Patch(facecolor="blue", label="False Negative"),
        Patch(facecolor="black", label="True Negative"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
