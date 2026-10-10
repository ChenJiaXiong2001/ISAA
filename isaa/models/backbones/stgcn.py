"""Reusable full ST-GCN with three partitions and trainable edge weights."""
import torch
from torch import nn
from isaa.models.official_stgcn import OfficialSTGCNFeatureExtractor, ConvTemporalGraphical
from .common import prepare_masked_input, downsample_mask, validate_stack


class STGCNBackbone(OfficialSTGCNFeatureExtractor):
    def __init__(self, in_channels, adjacency, *, channels=None, strides=None, dropout=0.0):
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 3 or adjacency.shape[0] != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("ST-GCN adjacency must have shape 3 x V x V")
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        channels = tuple(self.CHANNELS if channels is None else channels)
        strides = tuple(self.STRIDES if strides is None else strides)
        validate_stack(channels, strides)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        super().__init__(in_channels, adjacency, channels=channels, strides=strides, dropout=dropout)
        self.in_channels = int(in_channels)
        # Reference ST-GCN: pointwise spatial projection, then 9-frame TCN.
        # Historical project variants keep their original implementation.
        cin = in_channels
        for block, cout in zip(self.st_gcn_networks, self.channels):
            block.gcn = ConvTemporalGraphical(cin, cout, 3, t_kernel_size=1, t_padding=0)
            nn.init.ones_(block.tcn[3].weight)
            cin = cout

    def forward_masked(self, x, mask):
        x, mask = prepare_masked_input(x, mask, self.in_channels, self.num_point)
        b, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 3, 1, 2).reshape(b, v * c, t))
        x = x.reshape(b, v, c, t).permute(0, 2, 3, 1).contiguous().masked_fill(~mask, 0)
        a = self.A.to(device=x.device, dtype=x.dtype)
        for block, importance, stride in zip(self.st_gcn_networks, self.edge_importance, self.strides):
            x, _ = block(x, a * importance)
            mask = downsample_mask(mask, stride)
            x = x.masked_fill(~mask, 0)
        return x, mask
