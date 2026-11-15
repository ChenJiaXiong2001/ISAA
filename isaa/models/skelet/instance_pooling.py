"""Official Instance Pooling, expressed without einops or Python tensor branches.

Adapted from YijieYang23/SkeleT-GCN (Apache-2.0); see NOTICE.md.
"""
import torch
from torch import nn


class InstancePoolingModule(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_point=65,
                 drop_path=0.1, pe=None):
        super().__init__()
        if pe not in {None, "V", "VC"} or not 0 <= drop_path <= 1:
            raise ValueError("Invalid Instance Pooling configuration")
        self.point_encoder = nn.Sequential(nn.Linear(in_channels, base_channels),
                                           nn.BatchNorm1d(base_channels), nn.ReLU())
        self.concat_pool_fc = nn.Sequential(nn.Linear(base_channels * 2, base_channels),
                                            nn.BatchNorm1d(base_channels))
        self.group_pool_fc = nn.Sequential(nn.Linear(base_channels * 2, base_channels),
                                           nn.BatchNorm1d(base_channels), nn.GELU())
        self.gelu = nn.GELU()
        self.pe, self.drop_path = pe, drop_path
        if pe is not None:
            self.pos_embedding = nn.Parameter(torch.randn(
                1, num_point, 1 if pe == "V" else base_channels))

    def forward(self, x):
        # Input N,I,V,C,T; intermediate N,T,V,I,C.
        n, i, v, c, t = x.shape
        y = self.point_encoder(x.permute(0, 4, 2, 1, 3).reshape(-1, c))
        y = y.reshape(n, t, v, i, -1)
        if self.pe is not None:
            y = y + self.pos_embedding[:, None, :, None, :]
        residual = y
        if self.training and self.drop_path:
            # Same one Bernoulli decision for the entire batch as upstream;
            # tensor multiplication keeps the neural path in the compiled graph.
            residual = residual * (torch.rand((), device=x.device) >= self.drop_path)
        global_person = y.amax(dim=3, keepdim=True).expand_as(y)
        y = torch.cat((y, global_person), dim=-1)
        y = self.concat_pool_fc(y.reshape(-1, y.shape[-1])).reshape_as(residual)
        y = self.gelu(y + residual)
        local = y.amax(dim=3)
        global_scene = y.amax(dim=(1, 2, 3), keepdim=True).squeeze(3).expand_as(local)
        y = torch.cat((global_scene, local), dim=-1)
        y = self.group_pool_fc(y.reshape(-1, y.shape[-1]))
        return y.reshape(n, t, v, -1).permute(0, 3, 1, 2).contiguous()
