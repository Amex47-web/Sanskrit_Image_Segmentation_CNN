"""
postprocess.py — Morphological Postprocessing
================================================

Cleans up raw binary predictions using morphological operations:
    - Opening:  removes small false-positive noise
    - Closing:  reconnects broken shirorekha and character strokes
    - Connected component filtering: removes tiny artifacts

These operations are specifically tuned for Sanskrit/Devanagari text
where the shirorekha (headline) commonly breaks at character boundaries
in noisy predictions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def morphological_opening(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological opening (erosion → dilation).

    Removes small noise spots that are false positives —
    e.g., scanner artifacts mistaken for text pixels.

    Args:
        mask:        Binary mask, dtype uint8, values {0, 255}.
        kernel_size: Size of the structuring element.
        iterations:  Number of times to apply the operation.

    Returns:
        Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morphological_closing(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 2,
) -> np.ndarray:
    """
    Apply morphological closing (dilation → erosion).

    Reconnects broken text elements — critical for Sanskrit where:
        - Shirorekha breaks between characters
        - Thin strokes fragment in noisy predictions
        - Character components get disconnected

    Using a larger kernel (5×5) and 2 iterations specifically to
    handle the horizontal shirorekha line.

    Args:
        mask:        Binary mask, dtype uint8, values {0, 255}.
        kernel_size: Size of the structuring element.
        iterations:  Number of times to apply the operation.

    Returns:
        Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 50,
) -> np.ndarray:
    """
    Remove connected components smaller than a minimum area.

    Filters out tiny noise artifacts that survive morphological
    operations. Text regions are typically larger than 50 pixels.

    Args:
        mask:     Binary mask, dtype uint8, values {0, 255}.
        min_area: Minimum component area in pixels.

    Returns:
        Filtered binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def extract_text_region(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Extract only the text regions from an image using the binary mask.

    Args:
        image: Original grayscale image, shape (H, W).
        mask:  Binary mask, shape (H, W), values {0, 255}.

    Returns:
        Image with non-text regions set to white (255).
    """
    result = np.full_like(image, 255)
    text_pixels = mask > 0
    result[text_pixels] = image[text_pixels]
    return result


def get_bounding_boxes(
    mask: np.ndarray,
    min_area: int = 50,
) -> list:
    """
    Extract bounding boxes for all text regions.

    Args:
        mask:     Binary mask, dtype uint8, values {0, 255}.
        min_area: Minimum component area to include.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, w, h))

    return boxes


def full_postprocess(
    mask: np.ndarray,
    config: Optional[dict] = None,
) -> np.ndarray:
    """
    Run the complete postprocessing pipeline.

    Pipeline: Opening → Closing → Connected Component Filtering

    Args:
        mask:   Raw binary mask from thresholding, values {0, 255}.
        config: Postprocessing config dict from inference.yaml.

    Returns:
        Cleaned binary mask.
    """
    if config is None:
        config = {}

    # Opening
    open_cfg = config.get("opening", {})
    if open_cfg.get("enabled", True):
        mask = morphological_opening(
            mask,
            kernel_size=open_cfg.get("kernel_size", 3),
            iterations=open_cfg.get("iterations", 1),
        )

    # Closing
    close_cfg = config.get("closing", {})
    if close_cfg.get("enabled", True):
        mask = morphological_closing(
            mask,
            kernel_size=close_cfg.get("kernel_size", 5),
            iterations=close_cfg.get("iterations", 2),
        )

    # Connected component filtering
    min_area = config.get("min_component_area", 50)
    mask = remove_small_components(mask, min_area=min_area)

    return mask
