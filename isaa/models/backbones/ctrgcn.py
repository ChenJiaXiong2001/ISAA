"""Reusable full CTR-GCN: adaptive channel topology and multi-scale TCN."""
from __future__ import annotations

import torch
from torch import nn

from isaa.models.original_ctrgcn import TCNGCNUnit
from .common import prepare_masked_input, downsample_mask, validate_stack


class OfficialCTRGCNFeatureExtractor(nn.Module):
    """CTR-GCN without its classifier; default is the standard ten blocks."""

    CHANNELS = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256)

    STRIDES = (1, 1, 1, 1, 2, 1, 1, 2, 1, 1)

    def __init__(self, adjacency: torch.Tensor, in_channels: int = 3,
                 channels: tuple[int, ...] | None = None,
                 strides: tuple[int, ...] | None = None, adaptive: bool = True):
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 3 or adjacency.shape[0] != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("CTR-GCN adjacency must have shape 3 x V x V")
        self.num_point = int(adjacency.shape[1])
        self.in_channels = int(in_channels)
        self.channels = tuple(self.CHANNELS if channels is None else channels)
        self.strides = tuple(self.STRIDES if strides is None else strides)
        validate_stack(self.channels, self.strides, temporal_branches=4)
        if self.in_channels not in (3, 6, 9) and self.in_channels < 8:
            raise ValueError("CTR-GCN input width must be 3/6/9 or at least 8")
        if any(c < 8 for c in self.channels[:-1]):
            raise ValueError("Intermediate CTR-GCN widths must be at least 8")
        self.adaptive = bool(adaptive)
        self.register_buffer("A", adjacency)
        self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_point)
        layers = []
        cin = self.in_channels
        for index, cout in enumerate(self.channels):
            layers.append(TCNGCNUnit(
                cin, cout, self.A, stride=self.strides[index],
                residual=index != 0, adaptive=self.adaptive,
            ))
            cin = cout
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if x.ndim != 5:
            raise ValueError("Expected B x C x T x V x 1")
        b, c, t, v, m = x.shape
        if (c, v, m) != (self.in_channels, self.num_point, 1):
            raise ValueError(f"Expected C,V,M=({self.in_channels},{self.num_point},1), got {(c,v,m)}")
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(b, v * c, t)
        x = self.data_bn(x)
        x = x.view(b, 1, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(b, c, t, v)
        for block in self.blocks:
            x = block(x)
        return x, int(x.shape[2])

    def forward_masked(self, x, mask):
        x, mask = prepare_masked_input(x, mask, self.in_channels, self.num_point)
        b, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, v * c, t)
        x = self.data_bn(x)
        x = x.reshape(b, v, c, t).permute(0, 2, 3, 1).contiguous()
        x = x.masked_fill(~mask, 0)
        for block, stride in zip(self.blocks, self.strides):
            x = block(x)
            mask = downsample_mask(mask, stride)
            x = x.masked_fill(~mask, 0)
        return x, mask


CTRGCNBackbone = OfficialCTRGCNFeatureExtractor
