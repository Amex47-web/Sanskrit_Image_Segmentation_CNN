"""
decoder.py — U-Net Decoder (Expanding Path)
=============================================

Three-stage decoder that progressively upsamples features and merges
them with encoder skip connections to recover spatial detail.

Each stage: TransposeConv(upsample) → Concat(skip) → ConvBlock
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, AttentionGate


class DecoderBlock(nn.Module):
    """
    Single decoder stage: upsample → concatenate skip → ConvBlock.

    Args:
        in_channels:   Input channels from previous decoder stage.
        skip_channels: Channels from the encoder skip connection.
        out_channels:  Output channels after this stage.
        dropout_rate:  Dropout probability.
        use_residual:  Enable residual connections in ConvBlock.
        use_attention: Enable attention gating on skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        use_residual: bool = True,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        # Transposed convolution for 2× upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2,
        )

        # Optional attention gate
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(
                gate_channels=in_channels,
                skip_channels=skip_channels,
            )

        # ConvBlock processes concatenated features
        self.conv_block = ConvBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    Features from previous decoder stage, shape (B, C_in, H, W).
            skip: Encoder skip features, shape (B, C_skip, 2H, 2W).

        Returns:
            Decoded features, shape (B, C_out, 2H, 2W).
        """
        x = self.upsample(x)

        # Handle spatial dimension mismatches (can occur with odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Optional attention gating
        if self.use_attention:
            skip = self.attention(gate=x, skip=skip)

        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)

        return self.conv_block(x)


class Decoder(nn.Module):
    """
    Three-stage expanding decoder with skip connections.

    Takes bottleneck features and skip connections from the encoder,
    progressively upsamples and refines the segmentation map.

    Args:
        bottleneck_channels: Channels from the bottleneck output.
        encoder_channels:    List of encoder channel counts [32, 64, 128].
        dropout_rate:        Dropout probability.
        use_residual:        Enable residual connections.
        use_attention:       Enable attention gates on skip connections.
    """

    def __init__(
        self,
        bottleneck_channels: int = 256,
        encoder_channels: List[int] = [32, 64, 128],
        dropout_rate: float = 0.0,
        use_residual: bool = True,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        # Decoder stages are built in reverse order of encoder
        # bottleneck(256) → skip3(128) → 128
        # 128 → skip2(64) → 64
        # 64 → skip1(32) → 32
        reversed_channels = list(reversed(encoder_channels))

        self.stages = nn.ModuleList()
        in_ch = bottleneck_channels

        for skip_ch in reversed_channels:
            self.stages.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=skip_ch,
                    dropout_rate=dropout_rate,
                    use_residual=use_residual,
                    use_attention=use_attention,
                )
            )
            in_ch = skip_ch

    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x:              Bottleneck features, shape (B, C_bn, H, W).
            skip_features:  List of encoder skip features [s1, s2, s3],
                           ordered from shallow to deep.

        Returns:
            Decoded features, shape (B, encoder_channels[0], H_in, W_in).
        """
        # Reverse skip features: process deepest first
        reversed_skips = list(reversed(skip_features))

        for stage, skip in zip(self.stages, reversed_skips):
            x = stage(x, skip)

        return x
