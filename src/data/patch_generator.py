"""
patch_generator.py — Overlapping Patch Extraction & Metadata Tracking
======================================================================

Divides large manuscript images into overlapping patches for CNN training
and inference. Tracks coordinates for later stitching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class PatchInfo:
    """Metadata for a single extracted patch."""
    patch_id: int
    source_image: str
    row_start: int
    col_start: int
    row_end: int
    col_end: int
    patch_height: int
    patch_width: int
    padded: bool = False


@dataclass
class ExtractionResult:
    """Result of patch extraction from one image."""
    source_image: str
    image_height: int
    image_width: int
    patch_size: int
    overlap: int
    patches: List[np.ndarray] = field(default_factory=list)
    metadata: List[PatchInfo] = field(default_factory=list)


class PatchExtractor:
    """
    Extracts overlapping patches from images with coordinate tracking.

    Args:
        patch_size: Side length of square patches (256 or 512).
        overlap:    Overlap in pixels between adjacent patches.
    """

    def __init__(self, patch_size: int = 256, overlap: int = 64) -> None:
        if overlap >= patch_size:
            raise ValueError(f"Overlap ({overlap}) must be < patch_size ({patch_size}).")
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

    def extract_patches(self, image: np.ndarray, source_name: str = "unknown") -> ExtractionResult:
        """
        Extract overlapping patches from a single image.
        Edge patches are zero-padded to maintain uniform dimensions.

        Args:
            image:       Grayscale image, shape (H, W) or (H, W, 1).
            source_name: Filename for metadata.

        Returns:
            ExtractionResult with patches and metadata.
        """
        if len(image.shape) == 3:
            image = image[:, :, 0]

        h, w = image.shape
        result = ExtractionResult(
            source_image=source_name, image_height=h, image_width=w,
            patch_size=self.patch_size, overlap=self.overlap,
        )

        patch_id = 0
        for row in range(0, h, self.stride):
            for col in range(0, w, self.stride):
                row_end = min(row + self.patch_size, h)
                col_end = min(col + self.patch_size, w)
                patch = image[row:row_end, col:col_end]

                padded = False
                if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                    padded_patch = np.zeros((self.patch_size, self.patch_size), dtype=image.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded_patch
                    padded = True

                info = PatchInfo(
                    patch_id=patch_id, source_image=source_name,
                    row_start=row, col_start=col, row_end=row_end, col_end=col_end,
                    patch_height=row_end - row, patch_width=col_end - col, padded=padded,
                )
                result.patches.append(patch)
                result.metadata.append(info)
                patch_id += 1

        return result

    def extract_patches_with_masks(
        self, image: np.ndarray, mask: np.ndarray, source_name: str = "unknown",
    ) -> Tuple[ExtractionResult, List[np.ndarray]]:
        """
        Extract patches from both image and mask simultaneously.

        Args:
            image:       Grayscale image, shape (H, W).
            mask:        Binary mask, shape (H, W).
            source_name: Filename identifier.

        Returns:
            Tuple of (ExtractionResult, list of mask patches).
        """
        image_result = self.extract_patches(image, source_name)
        mask_patches: List[np.ndarray] = []

        for info in image_result.metadata:
            r0, c0 = info.row_start, info.col_start
            r1, c1 = info.row_end, info.col_end
            m_patch = mask[r0:r1, c0:c1]

            if info.padded:
                padded_mask = np.zeros((self.patch_size, self.patch_size), dtype=mask.dtype)
                padded_mask[:m_patch.shape[0], :m_patch.shape[1]] = m_patch
                m_patch = padded_mask

            mask_patches.append(m_patch)

        return image_result, mask_patches


def save_patches(
    result: ExtractionResult, output_dir: str | Path,
    mask_patches: Optional[List[np.ndarray]] = None,
) -> Path:
    """Save extracted patches and metadata to disk."""
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source_image).stem

    for patch, info in zip(result.patches, result.metadata):
        fname = f"{stem}_patch_{info.patch_id:04d}.png"
        p_save = (patch * 255).astype(np.uint8) if patch.dtype in (np.float32, np.float64) else patch
        cv2.imwrite(str(img_dir / fname), p_save)

    if mask_patches is not None:
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for m_patch, info in zip(mask_patches, result.metadata):
            fname = f"{stem}_patch_{info.patch_id:04d}.png"
            m_save = (m_patch * 255).astype(np.uint8) if m_patch.max() <= 1 else m_patch.astype(np.uint8)
            cv2.imwrite(str(mask_dir / fname), m_save)

    meta_path = meta_dir / f"{stem}_metadata.json"
    meta_dict = {
        "source_image": result.source_image, "image_height": result.image_height,
        "image_width": result.image_width, "patch_size": result.patch_size,
        "overlap": result.overlap, "num_patches": len(result.metadata),
        "patches": [asdict(m) for m in result.metadata],
    }
    with open(meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)

    return meta_path


def generate_patches_for_directory(
    images_dir: str | Path, masks_dir: Optional[str | Path],
    output_dir: str | Path, patch_size: int = 256, overlap: int = 64,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> int:
    """Extract patches from all images (and optionally masks) in a directory."""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    extractor = PatchExtractor(patch_size=patch_size, overlap=overlap)
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)

    total = 0
    for img_path in tqdm(image_paths, desc="Extracting patches"):
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        mask_patches = None
        if masks_dir is not None:
            mask_path = Path(masks_dir) / img_path.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                result, mask_patches = extractor.extract_patches_with_masks(image, mask, img_path.name)
            else:
                result = extractor.extract_patches(image, img_path.name)
        else:
            result = extractor.extract_patches(image, img_path.name)

        save_patches(result, output_dir, mask_patches)
        total += len(result.patches)

    print(f"Generated {total} patches → {output_dir}")
    return total
