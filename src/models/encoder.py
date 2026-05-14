"""
encoder.py — U-Net Encoder (Contracting Path)
================================================

Three-stage encoder that progressively downsamples the input while
extracting hierarchical features. Each stage consists of a ConvBlock
followed by MaxPool2d.

Feature hierarchy for Sanskrit manuscripts:
    Stage 1 (32 ch): Low-level edges, stroke boundaries, shirorekha
    Stage 2 (64 ch): Character-level patterns, glyph shapes
    Stage 3 (128 ch): Word/line-level structures, border patterns
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .blocks import ConvBlock


class Encoder(nn.Module):
    """
    Three-stage contracting encoder.

    Each stage: ConvBlock → MaxPool2d(2×2)

    Returns both the final encoded features and a list of skip
    connection features (before pooling) for the decoder.

    Args:
        in_channels:      Number of input channels (1 for grayscale).
        encoder_channels: List of channel counts per stage, e.g. [32, 64, 128].
        dropout_rate:     Dropout probability in ConvBlocks.
        use_residual:     Enable residual connections in ConvBlocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: List[int] = [32, 64, 128],
        dropout_rate: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for ch in encoder_channels:
            self.stages.append(
                ConvBlock(
                    in_channels=prev_channels,
                    out_channels=ch,
                    dropout_rate=dropout_rate,
                    use_residual=use_residual,
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor, shape (B, C_in, H, W).

        Returns:
            Tuple of:
                - Encoded features after final pooling, shape (B, C[-1], H/8, W/8)
                - List of skip features [stage1, stage2, stage3] (before pooling)
        """
        skip_features: List[torch.Tensor] = []

        for stage, pool in zip(self.stages, self.pools):
            x = stage(x)
            skip_features.append(x)  # Save BEFORE pooling
            x = pool(x)

        return x, skip_features
