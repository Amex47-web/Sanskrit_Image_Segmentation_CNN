"""
bottleneck.py — Bottleneck Module with Multi-Scale & Dilated Convolutions
==========================================================================

The bottleneck sits between the encoder and decoder. It processes the
most compressed representation and enriches it with:

1. Standard ConvBlock for deep feature extraction
2. MultiScaleBlock for capturing features at different scales
3. DilatedConvBlock for expanded receptive field

This combination is specifically designed for Sanskrit manuscripts where:
    - Thin strokes and thick borders coexist
    - Decorative patterns can span large spatial areas
    - Context is needed to distinguish ornaments from text
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .blocks import ConvBlock, MultiScaleBlock, DilatedConvBlock


class Bottleneck(nn.Module):
    """
    Bottleneck module: ConvBlock → MultiScale → DilatedConv.

    Args:
        in_channels:        Input channels (from last encoder stage after pooling).
        bottleneck_channels: Number of channels in the bottleneck.
        use_multiscale:     Enable multi-scale enhancement module.
        use_dilated:        Enable dilated convolution module.
        dilated_rates:      Dilation rates for the dilated conv module.
        dropout_rate:       Dropout probability.
        use_residual:       Enable residual connections.
    """

    def __init__(
        self,
        in_channels: int = 128,
        bottleneck_channels: int = 256,
        use_multiscale: bool = True,
        use_dilated: bool = True,
        dilated_rates: List[int] = [2, 4],
        dropout_rate: float = 0.3,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        # Core bottleneck convolution
        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
        )

        # Multi-scale enhancement
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale = MultiScaleBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
            )

        # Dilated convolutions
        self.use_dilated = use_dilated
        if use_dilated:
            self.dilated = DilatedConvBlock(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                rates=dilated_rates,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bottleneck.

        Args:
            x: Encoded features, shape (B, C_in, H, W).

        Returns:
            Enhanced features, shape (B, bottleneck_channels, H, W).
        """
        x = self.conv_block(x)

        if self.use_multiscale:
            x = self.multiscale(x)

        if self.use_dilated:
            x = self.dilated(x)

        return x
