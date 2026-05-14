"""
auto_mask_generator.py — Automatic Text Mask Generation for Sanskrit Manuscripts
==================================================================================

Generates pseudo ground-truth binary masks from raw manuscript scans
using classical image processing techniques.  These masks serve as
training labels when manual annotation is unavailable.

Strategy (multi-stage, tuned for Devanagari):
    1. Grayscale conversion + CLAHE contrast enhancement
    2. Bilateral filter to preserve edges while smoothing background
    3. Adaptive thresholding (Gaussian) to binarize text vs background
    4. Morphological closing to reconnect shirorekha and broken strokes
    5. Morphological opening to remove small noise
    6. Connected-component filtering to remove tiny artifacts
    7. Horizontal structure detection to reinforce shirorekha lines
    8. Edge-margin cropping to exclude page borders / decorative edges
    9. Large-blob suppression to exclude illustrations and artwork

The resulting masks are NOT perfect — they are "pseudo" ground truth.
For best results, manually review and correct a subset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def _estimate_text_region(gray: np.ndarray) -> np.ndarray:
    """
    Estimate a rectangular text-body region, excluding page borders
    and decorative margins.

    Uses projection profiles: sum pixel intensities along rows/cols
    and crop to the region with high activity.

    Args:
        gray: Grayscale image, shape (H, W), dtype uint8.

    Returns:
        Boolean mask marking the estimated text body region.
    """
    h, w = gray.shape

    # Invert so text is bright
    inv = 255 - gray

    # Horizontal projection (sum along rows)
    h_proj = inv.sum(axis=1).astype(np.float64)
    h_proj = h_proj / h_proj.max() if h_proj.max() > 0 else h_proj

    # Vertical projection (sum along columns)
    v_proj = inv.sum(axis=0).astype(np.float64)
    v_proj = v_proj / v_proj.max() if v_proj.max() > 0 else v_proj

    # Find rows/cols with sufficient text activity
    h_thresh = 0.05
    v_thresh = 0.05

    active_rows = np.where(h_proj > h_thresh)[0]
    active_cols = np.where(v_proj > v_thresh)[0]

    if len(active_rows) == 0 or len(active_cols) == 0:
        return np.ones((h, w), dtype=bool)

    # Add margin
    margin_y = int(h * 0.02)
    margin_x = int(w * 0.02)

    r_start = max(0, active_rows[0] - margin_y)
    r_end = min(h, active_rows[-1] + margin_y)
    c_start = max(0, active_cols[0] - margin_x)
    c_end = min(w, active_cols[-1] + margin_x)

    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[r_start:r_end, c_start:c_end] = True

    return region_mask


def _suppress_large_blobs(
    mask: np.ndarray,
    max_area_ratio: float = 0.15,
    min_aspect_ratio: float = 0.3,
) -> np.ndarray:
    """
    Remove large connected components that are likely illustrations
    or decorative artwork rather than text.

    Text lines are typically wide and thin.  Large square-ish blobs
    are likely images or ornamental blocks.

    Args:
        mask:            Binary mask, dtype uint8, {0, 255}.
        max_area_ratio:  Max ratio of component area to total image area.
        min_aspect_ratio: Minimum width/height ratio (text lines > 2).

    Returns:
        Filtered mask.
    """
    h, w = mask.shape
    total_area = h * w

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    result = mask.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        area_ratio = area / total_area
        aspect = comp_w / max(comp_h, 1)

        # Large blob with roughly square aspect ratio → likely illustration
        if area_ratio > max_area_ratio and aspect < 3.0:
            result[labels == i] = 0

        # Very tall blob spanning most of the page → likely border/illustration
        if comp_h > h * 0.7 and comp_w > w * 0.5:
            result[labels == i] = 0

    return result


def _reinforce_shirorekha(
    mask: np.ndarray,
    binary: np.ndarray,
    kernel_width: int = 25,
) -> np.ndarray:
    """
    Detect and reinforce horizontal lines (shirorekha) that may have
    been fragmented during thresholding.

    Uses a horizontal morphological kernel to extract long horizontal
    structures, then merges them into the mask.

    Args:
        mask:         Current binary mask, dtype uint8, {0, 255}.
        binary:       Thresholded binary image (text=255).
        kernel_width: Width of horizontal kernel.

    Returns:
        Mask with reinforced shirorekha.
    """
    # Horizontal kernel to detect long horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Dilate slightly to connect to nearby text
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    horizontal_lines = cv2.dilate(horizontal_lines, dilate_kernel, iterations=1)

    # Merge with existing mask
    result = cv2.bitwise_or(mask, horizontal_lines)
    return result


def generate_mask(
    image: np.ndarray,
    clahe_clip: float = 3.0,
    adaptive_block: int = 15,
    adaptive_c: int = 8,
    close_kernel: int = 5,
    close_iter: int = 2,
    open_kernel: int = 3,
    open_iter: int = 1,
    min_component_area: int = 80,
    suppress_blobs: bool = True,
    reinforce_lines: bool = True,
    exclude_margins: bool = True,
    margin_pct: float = 0.03,
) -> np.ndarray:
    """
    Generate a binary text mask from a raw manuscript image.

    Pipeline:
        Grayscale → CLAHE → Bilateral → Adaptive Thresh → Close → Open
        → Component Filter → Shirorekha Reinforce → Blob Suppress
        → Margin Exclude

    Args:
        image:              Raw input image (BGR or grayscale).
        clahe_clip:         CLAHE clip limit.
        adaptive_block:     Adaptive threshold block size (must be odd).
        adaptive_c:         Constant subtracted from mean in threshold.
        close_kernel:       Closing kernel size.
        close_iter:         Closing iterations.
        open_kernel:        Opening kernel size.
        open_iter:          Opening iterations.
        min_component_area: Min connected component area to keep.
        suppress_blobs:     Remove large illustration-like blobs.
        reinforce_lines:    Reinforce horizontal shirorekha lines.
        exclude_margins:    Exclude page edge margins.
        margin_pct:         Margin percentage to exclude from edges.

    Returns:
        Binary mask, dtype uint8, values {0, 255}.
        255 = text, 0 = non-text/background.
    """
    # --- Step 1: Grayscale ---
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # --- Step 2: CLAHE enhancement ---
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # --- Step 3: Bilateral filter (edge-preserving smoothing) ---
    smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # --- Step 4: Adaptive thresholding ---
    # Text becomes WHITE (255), background becomes BLACK (0)
    binary = cv2.adaptiveThreshold(
        smoothed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert: text=255
        blockSize=adaptive_block,
        C=adaptive_c,
    )

    # --- Step 5: Morphological closing (reconnect broken strokes) ---
    kern_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
    )
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern_close, iterations=close_iter)

    # --- Step 6: Morphological opening (remove small noise) ---
    kern_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern_open, iterations=open_iter)

    # --- Step 7: Connected component filtering ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            filtered[labels == i] = 255
    mask = filtered

    # --- Step 8: Reinforce shirorekha ---
    if reinforce_lines:
        mask = _reinforce_shirorekha(mask, binary, kernel_width=max(w // 20, 15))

    # --- Step 9: Suppress large illustration blobs ---
    if suppress_blobs:
        mask = _suppress_large_blobs(mask, max_area_ratio=0.12)

    # --- Step 10: Exclude page margins ---
    if exclude_margins:
        margin_y = int(h * margin_pct)
        margin_x = int(w * margin_pct)
        mask[:margin_y, :] = 0
        mask[-margin_y:, :] = 0
        mask[:, :margin_x] = 0
        mask[:, -margin_x:] = 0

    return mask


def generate_masks_for_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    **kwargs,
) -> int:
    """
    Generate binary text masks for all images in a directory.

    Args:
        input_dir:  Directory containing raw manuscript images.
        output_dir: Directory to save generated masks.
        extensions: Supported file extensions.
        **kwargs:   Arguments forwarded to generate_mask().

    Returns:
        Number of masks generated.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        return 0

    count = 0
    text_ratios = []

    for img_path in tqdm(image_paths, desc="Generating masks"):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  Warning: Could not read {img_path}, skipping.")
            continue

        mask = generate_mask(image, **kwargs)

        # Save mask
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), mask)

        # Track statistics
        text_ratio = (mask > 0).sum() / mask.size
        text_ratios.append(text_ratio)
        count += 1

    # Print summary statistics
    if text_ratios:
        ratios = np.array(text_ratios)
        print(f"\n{'='*50}")
        print(f"Mask Generation Summary")
        print(f"{'='*50}")
        print(f"  Images processed:  {count}")
        print(f"  Text ratio (mean): {ratios.mean():.3f}")
        print(f"  Text ratio (min):  {ratios.min():.3f}")
        print(f"  Text ratio (max):  {ratios.max():.3f}")
        print(f"  Text ratio (std):  {ratios.std():.3f}")
        print(f"  Output directory:  {output_dir}")
        print(f"{'='*50}")

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text masks from manuscripts")
    parser.add_argument("--input", default="dataset/raw", help="Input image directory")
    parser.add_argument("--output", default="dataset/masks", help="Output mask directory")
    parser.add_argument("--clahe-clip", type=float, default=3.0)
    parser.add_argument("--adaptive-block", type=int, default=15)
    parser.add_argument("--adaptive-c", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=80)
    args = parser.parse_args()

    generate_masks_for_directory(
        input_dir=args.input,
        output_dir=args.output,
        clahe_clip=args.clahe_clip,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        min_component_area=args.min_area,
    )
