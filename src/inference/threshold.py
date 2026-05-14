"""
threshold.py — Probability Map Thresholding
=============================================

Converts probability maps (float [0,1]) to binary masks using
various thresholding strategies.
"""

from __future__ import annotations

import cv2
import numpy as np


def fixed_threshold(
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply a fixed threshold to create a binary mask.

    Args:
        prob_map:  Probability map, shape (H, W), range [0, 1].
        threshold: Threshold value.

    Returns:
        Binary mask, dtype uint8, values {0, 255}.
    """
    return ((prob_map > threshold) * 255).astype(np.uint8)


def otsu_threshold(prob_map: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's automatic thresholding.

    Otsu's method finds the optimal threshold by minimizing
    intra-class variance. Works well when text and background
    have bimodal intensity distributions.

    Args:
        prob_map: Probability map, shape (H, W), range [0, 1].

    Returns:
        Binary mask, dtype uint8, values {0, 255}.
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    _, binary = cv2.threshold(prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(
    prob_map: np.ndarray,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """
    Apply adaptive thresholding using local neighborhood.

    Useful when probability values vary across different regions
    of the page (e.g., darker text areas vs faded margins).

    Args:
        prob_map:   Probability map, shape (H, W), range [0, 1].
        block_size: Size of the local neighborhood (must be odd).
        c:          Constant subtracted from the local mean.

    Returns:
        Binary mask, dtype uint8, values {0, 255}.
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    return cv2.adaptiveThreshold(
        prob_uint8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )


def apply_threshold(
    prob_map: np.ndarray,
    method: str = "fixed",
    **kwargs,
) -> np.ndarray:
    """
    Factory function: apply thresholding by method name.

    Args:
        prob_map: Probability map, shape (H, W), range [0, 1].
        method:   'fixed', 'otsu', or 'adaptive'.
        **kwargs: Additional arguments for the selected method.

    Returns:
        Binary mask, dtype uint8, values {0, 255}.
    """
    if method == "fixed":
        return fixed_threshold(prob_map, threshold=kwargs.get("value", 0.5))
    elif method == "otsu":
        return otsu_threshold(prob_map)
    elif method == "adaptive":
        return adaptive_threshold(
            prob_map,
            block_size=kwargs.get("adaptive_block_size", 11),
            c=kwargs.get("adaptive_c", 2),
        )
    else:
        raise ValueError(f"Unknown threshold method: {method}")
