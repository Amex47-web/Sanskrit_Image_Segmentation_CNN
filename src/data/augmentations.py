"""
augmentations.py — Data Augmentation Pipeline
===============================================

Provides Albumentations-based augmentation pipelines for training and
validation. All augmentations are applied synchronously to both the
image and the mask to maintain spatial correspondence.

Design notes:
    - Elastic deformation simulates ink bleed and paper warping common
      in historical manuscripts.
    - Rotation/affine transforms improve generalization to tilted scans.
    - Gaussian blur/noise simulate scanner artifacts and degraded print.
    - Brightness/contrast adjustments handle faded or overexposed pages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_augmentations(
    patch_size: int = 256,
    rotation_limit: int = 15,
    scale_limit: float = 0.1,
    shift_limit: float = 0.05,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    blur_limit: tuple = (3, 7),
    noise_var_limit: tuple = (10.0, 50.0),
    elastic_alpha: float = 120.0,
    elastic_sigma: float = 6.0,
    probability: float = 0.5,
) -> A.Compose:
    """
    Build the training augmentation pipeline.

    All spatial transforms are applied identically to image and mask.
    Pixel-level transforms (blur, noise, brightness) are applied to
    the image only.

    Args:
        patch_size:       Expected input patch size.
        rotation_limit:   Max rotation in degrees.
        scale_limit:      Max scale factor deviation.
        shift_limit:      Max shift as fraction of image size.
        brightness_limit: Max brightness change.
        contrast_limit:   Max contrast change.
        blur_limit:       Gaussian blur kernel size range.
        noise_var_limit:  Gaussian noise variance range.
        elastic_alpha:    Elastic deformation intensity.
        elastic_sigma:    Elastic deformation smoothness.
        probability:      Per-augmentation probability.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        # --- Spatial transforms (applied to image + mask) ---
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotation_limit,
            border_mode=0,  # Zero-padding at borders
            p=probability,
        ),
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            border_mode=0,
            p=probability * 0.6,  # Lower probability — aggressive transform
        ),
        A.Affine(
            shear=(-5, 5),
            mode=0,
            p=probability * 0.4,
        ),
        A.HorizontalFlip(p=0.3),

        # --- Pixel-level transforms (image only) ---
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=probability,
        ),
        A.GaussianBlur(
            blur_limit=blur_limit,
            p=probability * 0.5,
        ),
        A.GaussNoise(
            var_limit=noise_var_limit,
            p=probability * 0.5,
        ),

        # --- Final: convert to tensor ---
        ToTensorV2(),
    ])


def get_val_augmentations() -> A.Compose:
    """
    Build the validation augmentation pipeline.

    Only normalization and tensor conversion — no data augmentation
    during validation to ensure consistent metric computation.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        ToTensorV2(),
    ])


def apply_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    transform: A.Compose,
) -> Dict[str, Any]:
    """
    Apply an augmentation pipeline to an image-mask pair.

    Args:
        image: Grayscale image, shape (H, W), dtype float32 [0, 1].
        mask:  Binary mask, shape (H, W), dtype float32 {0, 1}.
        transform: Albumentations Compose pipeline.

    Returns:
        Dict with keys 'image' (tensor) and 'mask' (tensor).
    """
    # Albumentations expects (H, W, C) — add channel dim if needed
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    result = transform(image=image, mask=mask)

    return {
        "image": result["image"].float(),
        "mask": result["mask"].float(),
    }
