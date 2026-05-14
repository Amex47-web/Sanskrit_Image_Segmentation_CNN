"""
mask_utils.py — Binary Mask Creation, Validation & Visualization
=================================================================

Utilities for creating ground-truth binary masks from annotations
(polygons, bounding boxes, raster images) and validating them.

Mask convention:
    1 (or 255) = Sanskrit text
    0          = Background / decorative / non-text
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


def create_mask_from_polygons(
    image_shape: Tuple[int, int],
    polygons: List[np.ndarray],
) -> np.ndarray:
    """
    Create a binary mask from polygon annotations.

    Args:
        image_shape: (height, width) of the target mask.
        polygons:    List of polygon arrays, each shape (N, 2) with (x, y).

    Returns:
        Binary mask, shape (H, W), dtype uint8, values {0, 255}.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for poly in polygons:
        pts = poly.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=255)
    return mask


def create_mask_from_bboxes(
    image_shape: Tuple[int, int],
    bboxes: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Create a binary mask from bounding box annotations.

    Args:
        image_shape: (height, width) of the target mask.
        bboxes:      List of (x1, y1, x2, y2) bounding boxes.

    Returns:
        Binary mask, dtype uint8, values {0, 255}.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
    return mask


def load_raster_mask(
    mask_path: str | Path,
    threshold: int = 128,
) -> np.ndarray:
    """
    Load a raster mask image and binarize it.

    Args:
        mask_path: Path to the mask image file.
        threshold: Pixel value threshold for binarization.

    Returns:
        Binary mask, dtype uint8, values {0, 255}.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize mask to {0, 1} float32 for training.

    Args:
        mask: Binary mask with values {0, 255} or {0, 1}.

    Returns:
        Mask with values {0.0, 1.0}, dtype float32.
    """
    if mask.max() > 1:
        return (mask / 255.0).astype(np.float32)
    return mask.astype(np.float32)


def validate_mask(
    mask: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
    min_text_ratio: float = 0.01,
    max_text_ratio: float = 0.95,
) -> dict:
    """
    Validate a binary mask for common issues.

    Args:
        mask:           Binary mask to validate.
        image_shape:    Expected (H, W) shape, or None to skip shape check.
        min_text_ratio: Minimum acceptable ratio of text pixels.
        max_text_ratio: Maximum acceptable ratio of text pixels.

    Returns:
        Dict with keys: 'valid' (bool), 'issues' (list of strings),
        'text_ratio' (float).
    """
    issues: List[str] = []

    # Shape check
    if image_shape is not None and mask.shape[:2] != image_shape[:2]:
        issues.append(
            f"Shape mismatch: mask {mask.shape[:2]} vs expected {image_shape[:2]}"
        )

    # Value check
    unique = np.unique(mask)
    valid_vals = {0, 1, 255}
    if not set(unique.tolist()).issubset(valid_vals):
        issues.append(f"Unexpected values: {unique.tolist()}")

    # Coverage check
    binary = mask > 0
    text_ratio = binary.sum() / binary.size
    if text_ratio < min_text_ratio:
        issues.append(f"Text ratio too low: {text_ratio:.4f}")
    if text_ratio > max_text_ratio:
        issues.append(f"Text ratio too high: {text_ratio:.4f}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "text_ratio": float(text_ratio),
    }


def visualize_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Overlay a binary mask on an image for visual inspection.

    Args:
        image: Grayscale or BGR image.
        mask:  Binary mask, same spatial dimensions as image.
        alpha: Overlay transparency (0=invisible, 1=opaque).
        color: BGR color for the overlay.

    Returns:
        BGR overlay image.
    """
    if len(image.shape) == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display = image.copy()

    # Normalize mask
    binary = (mask > 0).astype(np.uint8)

    overlay = display.copy()
    overlay[binary == 1] = color

    return cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)


def correct_mask(
    mask: np.ndarray,
    opening_kernel: int = 3,
    closing_kernel: int = 5,
) -> np.ndarray:
    """
    Apply morphological corrections to clean up a noisy mask.

    Args:
        mask:           Binary mask, values {0, 255}.
        opening_kernel: Kernel size for opening (removes small noise).
        closing_kernel: Kernel size for closing (fills small gaps).

    Returns:
        Cleaned binary mask.
    """
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kern_close)

    return cleaned
