"""
stitch.py — Patch Stitching with Overlap Blending
====================================================

Reconstructs full-resolution prediction maps from overlapping patches.
Uses weighted averaging in overlap regions to prevent visible seams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


class PatchStitcher:
    """
    Reconstructs a full image from overlapping patch predictions.

    Uses a weight map with Gaussian-like falloff at patch edges to
    smoothly blend overlapping regions.

    Args:
        image_height: Height of the original full image.
        image_width:  Width of the original full image.
        patch_size:   Side length of square patches.
        overlap:      Overlap in pixels between adjacent patches.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int = 256,
        overlap: int = 64,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.overlap = overlap

        # Accumulators for weighted averaging
        self.prediction_map = np.zeros((image_height, image_width), dtype=np.float64)
        self.weight_map = np.zeros((image_height, image_width), dtype=np.float64)

        # Pre-compute patch weight mask (higher weight in center)
        self._patch_weight = self._create_weight_mask(patch_size)

    @staticmethod
    def _create_weight_mask(size: int) -> np.ndarray:
        """
        Create a 2D weight mask with smooth falloff at edges.

        Center pixels get weight ~1.0, edge pixels get lower weight.
        This produces seamless blending in overlap regions.
        """
        # 1D linear ramp
        ramp = np.linspace(0, 1, size // 2)
        ramp = np.concatenate([ramp, ramp[::-1]])
        if len(ramp) < size:
            ramp = np.append(ramp, ramp[-1])
        ramp = ramp[:size]

        # 2D weight = outer product of two ramps
        weight = np.outer(ramp, ramp)
        # Ensure minimum weight to avoid zero-division
        weight = np.clip(weight, 0.01, 1.0)
        return weight

    def add_patch(
        self,
        patch: np.ndarray,
        row_start: int,
        col_start: int,
        patch_height: int,
        patch_width: int,
    ) -> None:
        """
        Add a predicted patch to the accumulator.

        Args:
            patch:        Predicted probability patch, shape (patch_size, patch_size).
            row_start:    Top-left row in the original image.
            col_start:    Top-left column in the original image.
            patch_height: Actual height (before padding).
            patch_width:  Actual width (before padding).
        """
        # Crop to actual (non-padded) region
        p = patch[:patch_height, :patch_width]
        w = self._patch_weight[:patch_height, :patch_width]

        row_end = row_start + patch_height
        col_end = col_start + patch_width

        self.prediction_map[row_start:row_end, col_start:col_end] += p * w
        self.weight_map[row_start:row_end, col_start:col_end] += w

    def get_result(self) -> np.ndarray:
        """
        Compute the final stitched prediction map.

        Returns:
            Probability map, shape (H, W), dtype float32, range [0, 1].
        """
        # Avoid division by zero
        safe_weights = np.maximum(self.weight_map, 1e-8)
        result = (self.prediction_map / safe_weights).astype(np.float32)
        return np.clip(result, 0.0, 1.0)

    @classmethod
    def from_metadata(cls, metadata_path: str | Path) -> "PatchStitcher":
        """
        Create a PatchStitcher from saved metadata JSON.

        Args:
            metadata_path: Path to the metadata JSON file.

        Returns:
            Configured PatchStitcher instance.
        """
        with open(metadata_path) as f:
            meta = json.load(f)

        return cls(
            image_height=meta["image_height"],
            image_width=meta["image_width"],
            patch_size=meta["patch_size"],
            overlap=meta["overlap"],
        )
