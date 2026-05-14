"""
blocks.py — Fundamental Building Blocks for the Mini U-Net
============================================================

Contains all reusable neural network blocks:
    - ConvBlock:        Double convolution with BN + ReLU + optional residual
    - MultiScaleBlock:  Parallel 1×1, 3×3, 5×5 branches (Inception-style)
    - DilatedConvBlock: Dilated convolutions for broader receptive field
    - AttentionGate:    Optional attention mechanism (research extension)

Design notes for Sanskrit manuscripts:
    - MultiScaleBlock captures thin strokes (1×1), medium words (3×3),
      and large decorative structures (5×5) simultaneously.
    - DilatedConvBlock with rates [2, 4] provides broader context to
      discriminate ornamental curves from text characters.
    - Residual connections help gradient flow through the network,
      important for the thin-stroke details in Devanagari script.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU.

    Optionally adds a residual (skip) connection from input to output.
    When use_residual=True and in_channels != out_channels, a 1×1 conv
    is used to match dimensions.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        dropout_rate: Dropout probability (0 = no dropout).
        use_residual: Whether to add a residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Residual projection if channel dimensions differ
        self.use_residual = use_residual
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.residual_bn = nn.BatchNorm2d(out_channels)
        else:
            self.residual_conv = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for Conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)

        if self.use_residual:
            if self.residual_conv is not None:
                residual = self.residual_bn(self.residual_conv(residual))
            out = out + residual

        return self.relu(out)


class MultiScaleBlock(nn.Module):
    """
    Multi-scale feature extraction with parallel convolution branches.

    Three parallel branches:
        - 1×1 conv: captures fine-grained features (thin Sanskrit strokes)
        - 3×3 conv: captures medium-scale features (word-level patterns)
        - 5×5 conv: captures large-scale features (decorative borders)

    Outputs are concatenated and projected back to the target channels.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels after projection.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Each branch outputs in_channels // 3 features
        branch_ch = in_channels // 3

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )

        # 1×1 projection to unify channel count
        self.projection = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        b5 = self.branch_5x5(x)

        concat = torch.cat([b1, b3, b5], dim=1)
        return self.projection(concat)


class DilatedConvBlock(nn.Module):
    """
    Dilated convolution module for expanded receptive field.

    Two parallel dilated convolutions (dilation=2, dilation=4) are
    concatenated and projected, providing broader spatial context
    without increasing parameter count or losing resolution.

    This is critical for:
        - Distinguishing ornamental curves from Devanagari strokes
        - Understanding spatial context around text regions

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        rates:        List of dilation rates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rates: List[int] = [2, 4],
    ) -> None:
        super().__init__()
        branch_ch = in_channels // len(rates)

        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            ))

        self.projection = nn.Sequential(
            nn.Conv2d(branch_ch * len(rates), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)
        return self.projection(concat)


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections (optional research extension).

    Learns to weight spatial regions of the skip connection features
    based on the gating signal from the decoder. This helps the model
    focus on text regions and suppress decorative backgrounds.

    Args:
        gate_channels: Channels from the decoder (gating signal).
        skip_channels: Channels from the encoder (skip connection).
        inter_channels: Internal channel count for attention computation.
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Decoder features (gating signal), shape (B, C_gate, H, W).
            skip: Encoder features (skip connection), shape (B, C_skip, H, W).

        Returns:
            Attention-weighted skip features, same shape as skip.
        """
        g = self.W_gate(gate)
        s = self.W_skip(skip)

        # Resize gate to match skip spatial dims if needed
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)

        attention = self.psi(self.relu(g + s))
        return skip * attention
