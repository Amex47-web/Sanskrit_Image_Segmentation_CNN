"""
mini_unet.py — Complete Mini U-Net Architecture
=================================================

Assembles the full segmentation model:
    Encoder → Bottleneck → Decoder → 1×1 Output Head

Input:  (B, 1, 256, 256) grayscale patch
Output: (B, 1, 256, 256) probability mask (after sigmoid)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .encoder import Encoder
from .bottleneck import Bottleneck
from .decoder import Decoder


class MiniUNet(nn.Module):
    """
    Lightweight U-Net for Sanskrit text segmentation.

    Architecture:
        Encoder (3 stages) → Bottleneck (MultiScale + Dilated) → Decoder (3 stages) → 1×1 Conv → Sigmoid

    Args:
        in_channels:         Input channels (1 for grayscale).
        out_channels:        Output channels (1 for binary segmentation).
        encoder_channels:    Channel counts per encoder stage.
        bottleneck_channels: Bottleneck channel count.
        dropout_rate:        Dropout probability.
        use_residual:        Enable residual connections.
        use_multiscale:      Enable multi-scale bottleneck module.
        use_dilated:         Enable dilated convolution bottleneck module.
        dilated_rates:       Dilation rates for dilated conv module.
        use_attention:       Enable attention gates (research extension).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: List[int] = [32, 64, 128],
        bottleneck_channels: int = 256,
        dropout_rate: float = 0.3,
        use_residual: bool = True,
        use_multiscale: bool = True,
        use_dilated: bool = True,
        dilated_rates: List[int] = [2, 4],
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
        )

        self.bottleneck = Bottleneck(
            in_channels=encoder_channels[-1],
            bottleneck_channels=bottleneck_channels,
            use_multiscale=use_multiscale,
            use_dilated=use_dilated,
            dilated_rates=dilated_rates,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
        )

        self.decoder = Decoder(
            bottleneck_channels=bottleneck_channels,
            encoder_channels=encoder_channels,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_attention=use_attention,
        )

        # Final 1×1 convolution → sigmoid for binary probability mask
        self.output_head = nn.Sequential(
            nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            x: Input tensor, shape (B, 1, H, W).

        Returns:
            Probability mask, shape (B, 1, H, W), values in [0, 1].
        """
        # Encoder
        encoded, skip_features = self.encoder(x)

        # Bottleneck
        bottleneck_out = self.bottleneck(encoded)

        # Decoder
        decoded = self.decoder(bottleneck_out, skip_features)

        # Output head
        return self.output_head(decoded)

    @staticmethod
    def from_config(config: dict) -> "MiniUNet":
        """
        Create a MiniUNet instance from a configuration dictionary.

        Args:
            config: Model configuration dict (from model.yaml).

        Returns:
            Configured MiniUNet instance.
        """
        model_cfg = config.get("model", config)
        ext = model_cfg.get("extensions", {})

        return MiniUNet(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128]),
            bottleneck_channels=model_cfg.get("bottleneck_channels", 256),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            use_residual=model_cfg.get("use_residual", True),
            use_multiscale=model_cfg.get("multiscale", {}).get("enabled", True),
            use_dilated=model_cfg.get("dilated", {}).get("enabled", True),
            dilated_rates=model_cfg.get("dilated", {}).get("rates", [2, 4]),
            use_attention=ext.get("attention_gates", False),
        )


def model_summary(model: nn.Module, input_size: tuple = (1, 1, 256, 256)) -> Dict:
    """
    Print model summary: layer names, shapes, and parameter counts.

    Args:
        model:      PyTorch model.
        input_size: Example input shape (B, C, H, W).

    Returns:
        Dict with 'total_params', 'trainable_params', 'input_size', 'output_size'.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)

    # Forward pass to get output shape
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device("cpu")
    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():
        output = model(dummy)

    print(f"Input shape:      {list(input_size)}")
    print(f"Output shape:     {list(output.shape)}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Model size:       {total_params * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 60)

    # Print per-module parameter counts
    print(f"\n{'Module':<40} {'Params':>12}")
    print("-" * 54)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:<40} {params:>12,}")
    print("-" * 54)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_size": list(input_size),
        "output_size": list(output.shape),
    }


def estimate_flops(model: nn.Module, input_size: tuple = (1, 1, 256, 256)) -> int:
    """
    Rough FLOPs estimation for the model.

    Counts multiply-accumulate operations for Conv2d and Linear layers.
    This is an approximation — does not include BN, ReLU, etc.

    Args:
        model:      PyTorch model.
        input_size: Example input shape.

    Returns:
        Estimated FLOPs count.
    """
    flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal flops
        batch = input[0].shape[0]
        out_channels, in_channels_per_group = module.weight.shape[:2]
        kernel_h, kernel_w = module.weight.shape[2:]
        out_h, out_w = output.shape[2:]
        groups = module.groups

        # MACs = out_channels × in_channels_per_group × kernel_h × kernel_w × out_h × out_w
        flops += batch * out_channels * in_channels_per_group * kernel_h * kernel_w * out_h * out_w

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(conv_hook))

    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device("cpu")
    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    print(f"Estimated FLOPs: {flops:,}")
    print(f"Estimated GFLOPs: {flops / 1e9:.2f}")

    return flops
