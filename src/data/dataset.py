"""
dataset.py — PyTorch Dataset for Sanskrit Text Segmentation
=============================================================

Loads image-mask patch pairs from disk and applies augmentations.
Supports both pre-extracted patches and on-the-fly patch extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .augmentations import get_train_augmentations, get_val_augmentations


class SanskritDataset(Dataset):
    """
    PyTorch Dataset for Sanskrit text segmentation patches.

    Loads grayscale image patches and their corresponding binary masks
    from disk. Applies augmentation transforms synchronously to both.

    Args:
        images_dir:  Directory containing image patch PNG files.
        masks_dir:   Directory containing mask patch PNG files.
        transform:   Albumentations Compose pipeline.
        extensions:  Supported file extensions.
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: Optional[A.Compose] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Collect and sort image paths
        self.image_paths: List[Path] = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in extensions
        )

        # Validate mask availability
        self.valid_pairs: List[Tuple[Path, Path]] = []
        for img_path in self.image_paths:
            mask_path = self.masks_dir / img_path.name
            if mask_path.exists():
                self.valid_pairs.append((img_path, mask_path))

        if len(self.valid_pairs) == 0:
            raise ValueError(
                f"No valid image-mask pairs found.\n"
                f"Images dir: {self.images_dir}\n"
                f"Masks dir:  {self.masks_dir}"
            )

        print(f"Dataset: {len(self.valid_pairs)} image-mask pairs loaded.")

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a single image-mask pair.

        Returns:
            Dict with keys:
                'image': Tensor (1, H, W), float32, [0, 1]
                'mask':  Tensor (1, H, W), float32, {0, 1}
                'name':  str, filename stem
        """
        img_path, mask_path = self.valid_pairs[idx]

        # Load grayscale image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)  # Binarize at 127

        # Add channel dimension for Albumentations: (H, W) → (H, W, 1)
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"].float()
            mask_tensor = augmented["mask"].float()
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()

        # Ensure mask is (1, H, W)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.shape[0] != 1 and len(mask_tensor.shape) == 3:
            # Handle case where ToTensorV2 puts channel last
            if mask_tensor.shape[-1] == 1:
                mask_tensor = mask_tensor.permute(2, 0, 1)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": img_path.stem,
        }


def create_dataloaders(
    images_dir: str | Path,
    masks_dir: str | Path,
    batch_size: int = 8,
    val_split: float = 0.2,
    patch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    augmentation_config: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        images_dir:           Path to image patches.
        masks_dir:            Path to mask patches.
        batch_size:           Batch size for both loaders.
        val_split:            Fraction of data for validation.
        patch_size:           Patch size (for augmentation config).
        num_workers:          Number of data loading workers.
        pin_memory:           Pin memory for GPU transfer.
        augmentation_config:  Dict of augmentation hyperparameters.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Build augmentation pipelines
    aug_kwargs = augmentation_config or {}
    train_transform = get_train_augmentations(patch_size=patch_size, **aug_kwargs)
    val_transform = get_val_augmentations()

    # Create full dataset with training augmentation
    full_dataset = SanskritDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=train_transform,
    )

    # Split into train/val
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Override val subset transform — no augmentation during validation
    val_dataset_wrapper = _ValSubset(val_dataset, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset_wrapper,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"Train: {train_size} samples | Val: {val_size} samples")
    return train_loader, val_loader


class _ValSubset(Dataset):
    """Wrapper to apply validation transforms to a Subset."""

    def __init__(self, subset, transform: A.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> dict:
        item = self.subset[idx]
        # Re-apply with val transform by reloading from disk
        # Since the subset uses train transform, we access the underlying dataset
        real_idx = self.subset.indices[idx]
        dataset = self.subset.dataset
        img_path, mask_path = dataset.valid_pairs[real_idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"].float()
        mask_tensor = augmented["mask"].float()

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.shape[0] != 1 and len(mask_tensor.shape) == 3:
            if mask_tensor.shape[-1] == 1:
                mask_tensor = mask_tensor.permute(2, 0, 1)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": img_path.stem,
        }
