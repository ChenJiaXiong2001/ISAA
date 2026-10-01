"""Reference ST-GCN feature extractor used by the official model variant.

This follows the structure of yysijie/st-gcn: a three-part spatial graph,
9-frame temporal convolutions, residual ST-GCN blocks, and learnable
edge-importance weights.  The classifier is intentionally omitted because
the fusion model consumes the per-node features.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _conv_init(module: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_out")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class ConvTemporalGraphical(nn.Module):
    """Official ST-GCN temporal convolution followed by graph propagation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 t_kernel_size: int = 9, t_stride: int = 1,
                 t_padding: int = 4, t_dilation: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )
        _conv_init(self.conv)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        n, kc, t, v = x.shape
        if kc % self.kernel_size:
            raise RuntimeError("ST-GCN graph convolution channel shape is invalid")
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, a)
        return x.contiguous(), a


class STGCNBlock(nn.Module):
    """Reference ST-GCN block with residual, temporal BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1,
                 residual: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1),
                      stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        nn.init.constant_(self.tcn[0].weight, 1)
        nn.init.constant_(self.tcn[0].bias, 0)
        nn.init.constant_(self.tcn[3].weight, 1e-6)
        nn.init.constant_(self.tcn[3].bias, 0)
        if isinstance(self.residual, nn.Sequential):
            nn.init.constant_(self.residual[1].weight, 1)
            nn.init.constant_(self.residual[1].bias, 0)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        res = self.residual(x)
        x, a = self.gcn(x, a)
        x = self.tcn(x) + res
        return x.relu(), a


class OfficialSTGCNFeatureExtractor(nn.Module):
    """Reference ST-GCN blocks without the final classifier."""

    CHANNELS = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256)
    STRIDES = (1, 1, 1, 1, 2, 1, 1, 2, 1, 1)

    def __init__(self, in_channels: int, adjacency: torch.Tensor,
                 num_person: int = 1, dropout: float = 0.0,
                 channels: tuple[int, ...] | None = None,
                 strides: tuple[int, ...] | None = None) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 3 or adjacency.shape[0] != 3:
            raise ValueError("ST-GCN adjacency must have shape 3 x V x V")
        self.num_person = int(num_person)
        self.num_point = int(adjacency.shape[1])
        self.channels = tuple(channels or self.CHANNELS)
        self.strides = tuple(strides or self.STRIDES)
        if len(self.channels) != len(self.strides) or not self.channels:
            raise ValueError("ST-GCN channels and strides must have equal non-zero lengths")
        self.register_buffer("A", adjacency)
        self.data_bn = nn.BatchNorm1d(self.num_person * in_channels * self.num_point)
        layers = []
        cin = in_channels
        for index, cout in enumerate(self.channels):
            layers.append(STGCNBlock(
                cin, cout, adjacency.shape[0], stride=self.strides[index],
                residual=index != 0, dropout=dropout,
            ))
            cin = cout
        self.st_gcn_networks = nn.ModuleList(layers)
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones_like(adjacency)) for _ in self.st_gcn_networks
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if x.ndim != 5:
            raise ValueError("Expected B x C x T x V x M")
        b, c, t, v, m = x.shape
        if (c, v, m) != (self.data_bn.num_features // self.num_point, self.num_point, self.num_person):
            raise ValueError(f"Expected C,V,M compatible with ({self.data_bn.num_features},{self.num_point},{self.num_person})")
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(b, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(b, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(b * m, c, t, v)
        a = self.A.to(device=x.device, dtype=x.dtype)
        for block, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = block(x, a * importance)
        target_t = int(x.shape[2])
        x = x.view(b, m, x.shape[1], target_t, v).mean(1)
        return x, target_t


__all__ = ["ConvTemporalGraphical", "STGCNBlock", "OfficialSTGCNFeatureExtractor"]
