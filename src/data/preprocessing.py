"""
preprocessing.py — Image Preprocessing Pipeline for Sanskrit Manuscripts
=========================================================================

Converts raw scanned manuscript images into normalized grayscale images
suitable for patch extraction and CNN input.

Pipeline stages:
    1. Grayscale conversion
    2. Resize to target dimensions
    3. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
    4. Adaptive thresholding (optional)
    5. Noise reduction (Non-local means denoising)
    6. Contrast normalization (0-1 range)

Design notes:
    - CLAHE is critical for scanned manuscripts with uneven lighting and
      faded ink — it equalizes contrast locally rather than globally.
    - Non-local means denoising preserves edge details (important for thin
      Devanagari strokes and shirorekha) while removing scanner noise.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR or RGB image to single-channel grayscale.

    Args:
        image: Input image, shape (H, W, 3) or (H, W).

    Returns:
        Grayscale image, shape (H, W), dtype uint8.
    """
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (1024, 1024),
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize an image to the specified target dimensions.

    Uses INTER_AREA for downscaling (anti-aliased) and INTER_LINEAR for
    upscaling by default.

    Args:
        image:         Input image, any shape.
        target_size:   (width, height) tuple.
        interpolation: OpenCV interpolation flag.

    Returns:
        Resized image with the specified dimensions.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Choose interpolation method based on scaling direction
    if h * w > target_h * target_w:
        interp = cv2.INTER_AREA  # Shrinking — anti-alias
    else:
        interp = interpolation

    return cv2.resize(image, (target_w, target_h), interpolation=interp)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization).

    CLAHE divides the image into tiles and equalizes the histogram of each
    tile independently, with contrast clipping to prevent noise
    amplification.  This is essential for scanned manuscripts with:
      - Uneven illumination
      - Faded ink in certain areas
      - Yellowed/browned paper

    Args:
        image:          Grayscale image, shape (H, W), dtype uint8.
        clip_limit:     Threshold for contrast limiting.
        tile_grid_size: Grid of tiles for local equalization.

    Returns:
        CLAHE-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid_size
    )
    return clahe.apply(image)


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """
    Apply adaptive thresholding to binarize the image.

    Uses Gaussian-weighted neighborhood mean, which handles uneven
    illumination better than global thresholding.

    Args:
        image:      Grayscale image, shape (H, W), dtype uint8.
        block_size: Size of the neighborhood block (must be odd).
        c:          Constant subtracted from the weighted mean.

    Returns:
        Binary image, shape (H, W), values in {0, 255}.
    """
    return cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )


def denoise_image(
    image: np.ndarray,
    strength: int = 10,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """
    Apply Non-Local Means denoising.

    This algorithm averages similar patches across the image, which
    preserves fine edge details (crucial for Devanagari strokes) while
    effectively removing Gaussian noise from scanning.

    Args:
        image:                Grayscale image, shape (H, W).
        strength:             Filter strength (higher = more denoising,
                              but may blur fine text).
        template_window_size: Size of the template patch.
        search_window_size:   Size of the search area.

    Returns:
        Denoised grayscale image.
    """
    return cv2.fastNlMeansDenoising(
        image,
        h=strength,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] float32 range.

    Args:
        image: Input image, dtype uint8.

    Returns:
        Normalized image, dtype float32, range [0, 1].
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (1024, 1024),
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    denoise_strength: int = 10,
    apply_threshold: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a single image.

    Pipeline:
        Grayscale → Resize → CLAHE → Denoise → [Threshold] → [Normalize]

    Args:
        image:                Raw input image (BGR or grayscale).
        target_size:          (width, height) to resize to.
        clahe_clip_limit:     CLAHE contrast clipping limit.
        clahe_tile_grid_size: CLAHE tile grid size.
        denoise_strength:     NLM denoising filter strength.
        apply_threshold:      If True, apply adaptive thresholding.
        normalize:            If True, normalize to [0, 1] float32.

    Returns:
        Preprocessed image as numpy array.
    """
    # Step 1: Grayscale
    gray = convert_to_grayscale(image)

    # Step 2: Resize
    resized = resize_image(gray, target_size=target_size)

    # Step 3: CLAHE
    enhanced = apply_clahe(
        resized,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_grid_size,
    )

    # Step 4: Denoise
    denoised = denoise_image(enhanced, strength=denoise_strength)

    # Step 5: Optional adaptive threshold
    if apply_threshold:
        denoised = adaptive_threshold(denoised)

    # Step 6: Normalize
    if normalize:
        denoised = normalize_image(denoised)

    return denoised


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    target_size: Tuple[int, int] = (1024, 1024),
    clahe_clip_limit: float = 2.0,
    denoise_strength: int = 10,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> int:
    """
    Batch-preprocess all images in a directory.

    Args:
        input_dir:        Directory containing raw manuscript scans.
        output_dir:       Directory to save preprocessed images.
        target_size:      Target resize dimensions (width, height).
        clahe_clip_limit: CLAHE clipping limit.
        denoise_strength: Denoising filter strength.
        extensions:       Supported image file extensions.

    Returns:
        Number of images processed.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    count = 0
    for img_path in tqdm(image_paths, desc="Preprocessing"):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue

        processed = preprocess_image(
            image,
            target_size=target_size,
            clahe_clip_limit=clahe_clip_limit,
            denoise_strength=denoise_strength,
            normalize=False,  # Save as uint8 for storage
        )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), processed)
        count += 1

    print(f"Preprocessed {count} images → {output_dir}")
    return count
