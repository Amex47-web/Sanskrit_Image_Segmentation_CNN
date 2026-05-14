"""
infer.py — Full Inference Pipeline
=====================================

Runs patch-based inference on full-resolution manuscript images:
    1. Preprocess input image
    2. Extract overlapping patches
    3. Run model prediction on each patch
    4. Stitch patches back together with weighted blending
    5. Apply thresholding
    6. Run postprocessing
    7. Generate outputs (mask, text extraction, bboxes, overlay)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..data.patch_generator import PatchExtractor
from ..data.preprocessing import preprocess_image
from ..models.mini_unet import MiniUNet
from .postprocess import full_postprocess, extract_text_region, get_bounding_boxes
from .stitch import PatchStitcher
from .threshold import apply_threshold


class Inferencer:
    """
    Full inference pipeline for Sanskrit text segmentation.

    Args:
        model:       Trained MiniUNet model.
        device:      Device for inference.
        patch_size:  Patch size for extraction.
        overlap:     Overlap between patches.
        batch_size:  Batch size for patch inference.
        use_amp:     Whether to use mixed precision.
    """

    def __init__(
        self,
        model: MiniUNet,
        device: torch.device,
        patch_size: int = 256,
        overlap: int = 64,
        batch_size: int = 16,
        use_amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.use_amp = use_amp and device.type == "cuda"

        self.extractor = PatchExtractor(
            patch_size=patch_size, overlap=overlap,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: dict,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "Inferencer":
        """
        Create an Inferencer from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            model_config:    Model configuration dict.
            device:          Device for inference.
            **kwargs:        Additional arguments for Inferencer.

        Returns:
            Configured Inferencer instance.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MiniUNet.from_config(model_config)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model=model, device=device, **kwargs)

    def predict_image(
        self,
        image: np.ndarray,
        threshold_config: Optional[dict] = None,
        postprocess_config: Optional[dict] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run full inference on a single image.

        Args:
            image:             Input image (BGR or grayscale).
            threshold_config:  Thresholding config dict.
            postprocess_config: Postprocessing config dict.

        Returns:
            Dict with keys:
                'probability_map': float32 (H, W), range [0, 1]
                'binary_mask':     uint8 (H, W), values {0, 255}
                'cleaned_mask':    uint8 (H, W), postprocessed
                'text_only':       uint8 (H, W), text-only extraction
                'bounding_boxes':  list of (x, y, w, h)
        """
        # Preprocess
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        normalized = gray.astype(np.float32) / 255.0

        # Extract patches
        result = self.extractor.extract_patches(normalized, "inference")

        # Run model on patches in batches
        predictions = self._predict_patches(result.patches)

        # Stitch patches back together
        stitcher = PatchStitcher(h, w, self.patch_size, self.overlap)
        for pred_patch, meta in zip(predictions, result.metadata):
            stitcher.add_patch(
                pred_patch, meta.row_start, meta.col_start,
                meta.patch_height, meta.patch_width,
            )

        prob_map = stitcher.get_result()

        # Threshold
        thresh_cfg = threshold_config.copy() if threshold_config else {}
        method = thresh_cfg.pop("method", "fixed")
        binary_mask = apply_threshold(
            prob_map,
            method=method,
            **thresh_cfg,
        )

        # Postprocess
        cleaned_mask = full_postprocess(binary_mask, postprocess_config)

        # Extract text
        text_only = extract_text_region(gray, cleaned_mask)

        # Bounding boxes
        bboxes = get_bounding_boxes(
            cleaned_mask,
            min_area=postprocess_config.get("min_component_area", 50) if postprocess_config else 50,
        )

        return {
            "probability_map": prob_map,
            "binary_mask": binary_mask,
            "cleaned_mask": cleaned_mask,
            "text_only": text_only,
            "bounding_boxes": bboxes,
        }

    def _predict_patches(self, patches: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run model prediction on a list of patches in batches.

        Args:
            patches: List of grayscale patches, shape (ps, ps), float32.

        Returns:
            List of prediction patches, same shape, float32 [0, 1].
        """
        predictions = []

        for i in range(0, len(patches), self.batch_size):
            batch_patches = patches[i : i + self.batch_size]

            # Stack into batch tensor: (B, 1, H, W)
            batch_tensor = torch.stack([
                torch.from_numpy(p).unsqueeze(0).float()
                for p in batch_patches
            ]).to(self.device)

            with torch.no_grad():
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        outputs = self.model(batch_tensor)
                else:
                    outputs = self.model(batch_tensor)

            # Convert back to numpy
            for j in range(outputs.shape[0]):
                pred = outputs[j, 0].cpu().numpy()
                predictions.append(pred)

        return predictions


def run_inference_on_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    inferencer: Inferencer,
    threshold_config: Optional[dict] = None,
    postprocess_config: Optional[dict] = None,
    output_modes: Optional[dict] = None,
) -> int:
    """
    Run inference on all images in a directory.

    Args:
        input_dir:          Directory containing input images.
        output_dir:         Directory to save results.
        inferencer:         Configured Inferencer instance.
        threshold_config:   Thresholding configuration.
        postprocess_config: Postprocessing configuration.
        output_modes:       Dict specifying which outputs to save.

    Returns:
        Number of images processed.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    modes = output_modes or {
        "binary_mask": True, "text_extraction": True,
        "bounding_boxes": True, "overlay_visualization": True,
    }

    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)

    # Create output subdirectories
    for subdir in ["masks", "text_only", "overlays", "probability_maps"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in tqdm(image_paths, desc="Inference"):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        results = inferencer.predict_image(image, threshold_config, postprocess_config)

        stem = img_path.stem

        if modes.get("binary_mask", True):
            cv2.imwrite(str(output_dir / "masks" / f"{stem}_mask.png"), results["cleaned_mask"])

        if modes.get("text_extraction", True):
            cv2.imwrite(str(output_dir / "text_only" / f"{stem}_text.png"), results["text_only"])

        if modes.get("overlay_visualization", True):
            overlay = _create_overlay(image, results["cleaned_mask"])
            cv2.imwrite(str(output_dir / "overlays" / f"{stem}_overlay.png"), overlay)

        count += 1

    print(f"Processed {count} images → {output_dir}")
    return count


def _create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Create a green overlay of the mask on the original image."""
    if len(image.shape) == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display = image.copy()

    overlay = display.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)
