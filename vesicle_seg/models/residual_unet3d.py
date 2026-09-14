"""Tiny residual 3-D U-Net with dropout.

Original teaching model. Residual blocks + Dropout3d match the poster
description; channel counts are CI-sized (8→16→32), not a lab architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.drop = nn.Dropout3d(dropout)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class ResidualUNet3d(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 8, dropout: float = 0.10) -> None:
        super().__init__()
        self.enc1 = ResidualConv(in_ch, base, dropout)
        self.enc2 = ResidualConv(base, base * 2, dropout)
        self.enc3 = ResidualConv(base * 2, base * 4, dropout)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ResidualConv(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ResidualConv(base * 2, base, dropout)
        self.head = nn.Conv3d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self._match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self._match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

    @staticmethod
    def _match(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[2:] == skip.shape[2:]:
            return up
        return F.interpolate(up, size=skip.shape[2:], mode="trilinear", align_corners=False)
