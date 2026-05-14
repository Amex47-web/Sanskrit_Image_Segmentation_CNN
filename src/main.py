"""
main.py — CLI Entry Point for Sanskrit Text Segmentation
==========================================================

Provides subcommands:
    preprocess  — Run image preprocessing pipeline
    patches     — Generate patches from preprocessed images
    train       — Train the Mini U-Net model
    infer       — Run inference on new images
    visualize   — Generate training visualizations
    summary     — Print model architecture summary

Usage:
    python -m src.main train --config configs/train.yaml
    python -m src.main infer --config configs/inference.yaml
    python -m src.main summary --config configs/model.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    """Merge multiple config dicts (later configs override earlier)."""
    merged = {}
    for cfg in configs:
        merged.update(cfg)
    return merged


# ─── Subcommands ──────────────────────────────────────────────────────


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run image preprocessing pipeline."""
    config = load_config(args.config)
    prep_cfg = config.get("preprocessing", {})
    paths_cfg = config.get("paths", {})

    from .data.preprocessing import preprocess_directory

    input_dir = args.input or paths_cfg.get("images_dir", "dataset/raw")
    output_dir = args.output or paths_cfg.get("images_dir", "dataset/processed")
    target = tuple(prep_cfg.get("target_size", [1024, 1024]))

    preprocess_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        target_size=target,
        clahe_clip_limit=prep_cfg.get("clahe", {}).get("clip_limit", 2.0),
        denoise_strength=prep_cfg.get("denoise", {}).get("strength", 10),
    )


def cmd_patches(args: argparse.Namespace) -> None:
    """Generate patches from preprocessed images."""
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    paths_cfg = config.get("paths", {})

    from .data.patch_generator import generate_patches_for_directory

    generate_patches_for_directory(
        images_dir=args.input or paths_cfg.get("images_dir", "dataset/images"),
        masks_dir=args.masks or paths_cfg.get("masks_dir", "dataset/masks") if not args.no_masks else None,
        output_dir=args.output or paths_cfg.get("patches_dir", "dataset/patches"),
        patch_size=data_cfg.get("patch_size", 256),
        overlap=data_cfg.get("overlap", 64),
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train the Mini U-Net model."""
    train_config = load_config(args.config)

    # Optionally merge model config
    model_config_path = args.model_config or "configs/model.yaml"
    model_config = load_config(model_config_path) if Path(model_config_path).exists() else {}

    paths_cfg = train_config.get("paths", {})
    data_cfg = train_config.get("data", {})

    from .data.dataset import create_dataloaders
    from .models.mini_unet import MiniUNet
    from .training.trainer import Trainer

    # Create dataloaders
    patches_dir = paths_cfg.get("patches_dir", "dataset/patches")
    images_dir = Path(patches_dir) / "images"
    masks_dir = Path(patches_dir) / "masks"

    train_loader, val_loader = create_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=train_config.get("training", {}).get("batch_size", 8),
        val_split=data_cfg.get("val_split", 0.2),
        patch_size=data_cfg.get("patch_size", 256),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    # Create model
    model = MiniUNet.from_config(model_config)

    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        project_root=".",
    )

    history = trainer.train()

    # Save training curves
    from .utils.visualization import plot_training_curves

    vis_dir = paths_cfg.get("vis_dir", "outputs/visualizations")
    plot_training_curves(history, save_path=f"{vis_dir}/training_curves.png")


def cmd_infer(args: argparse.Namespace) -> None:
    """Run inference on images."""
    config = load_config(args.config)
    paths_cfg = config.get("paths", {})

    # Load model config
    model_config_path = args.model_config or "configs/model.yaml"
    model_config = load_config(model_config_path) if Path(model_config_path).exists() else {}

    from .inference.infer import Inferencer, run_inference_on_directory
    from .training.trainer import get_device

    device = get_device(config.get("hardware", {}))
    patch_cfg = config.get("patch", {})

    inferencer = Inferencer.from_checkpoint(
        checkpoint_path=paths_cfg.get("checkpoint", "outputs/checkpoints/best_model.pth"),
        model_config=model_config,
        device=device,
        patch_size=patch_cfg.get("size", 256),
        overlap=patch_cfg.get("overlap", 64),
        batch_size=patch_cfg.get("batch_size", 16),
        use_amp=config.get("hardware", {}).get("mixed_precision", True),
    )

    run_inference_on_directory(
        input_dir=args.input or paths_cfg.get("input_dir", "dataset/raw"),
        output_dir=args.output or paths_cfg.get("output_dir", "outputs/predictions"),
        inferencer=inferencer,
        threshold_config=config.get("threshold", {}),
        postprocess_config=config.get("postprocessing", {}),
        output_modes=config.get("output", {}),
    )


def cmd_summary(args: argparse.Namespace) -> None:
    """Print model architecture summary."""
    config = load_config(args.config)

    from .models.mini_unet import MiniUNet, model_summary, estimate_flops

    model = MiniUNet.from_config(config)
    model_summary(model)
    print()
    estimate_flops(model)


# ─── CLI Parser ───────────────────────────────────────────────────────


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="sanskrit-seg",
        description="Sanskrit Text Segmentation — Mini U-Net Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- preprocess ---
    p_pre = subparsers.add_parser("preprocess", help="Preprocess raw images")
    p_pre.add_argument("--config", default="configs/train.yaml")
    p_pre.add_argument("--input", help="Input directory override")
    p_pre.add_argument("--output", help="Output directory override")

    # --- patches ---
    p_pat = subparsers.add_parser("patches", help="Generate patches")
    p_pat.add_argument("--config", default="configs/train.yaml")
    p_pat.add_argument("--input", help="Images directory override")
    p_pat.add_argument("--masks", help="Masks directory override")
    p_pat.add_argument("--output", help="Output directory override")
    p_pat.add_argument("--no-masks", action="store_true", help="Skip mask extraction")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--config", default="configs/train.yaml")
    p_train.add_argument("--model-config", default="configs/model.yaml")

    # --- infer ---
    p_infer = subparsers.add_parser("infer", help="Run inference")
    p_infer.add_argument("--config", default="configs/inference.yaml")
    p_infer.add_argument("--model-config", default="configs/model.yaml")
    p_infer.add_argument("--input", help="Input directory override")
    p_infer.add_argument("--output", help="Output directory override")

    # --- summary ---
    p_sum = subparsers.add_parser("summary", help="Print model summary")
    p_sum.add_argument("--config", default="configs/model.yaml")

    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "preprocess": cmd_preprocess,
        "patches": cmd_patches,
        "train": cmd_train,
        "infer": cmd_infer,
        "summary": cmd_summary,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
