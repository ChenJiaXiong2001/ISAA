"""Experimental RTMW body CTR-GCN + hand/face ST-GCN classifier.

This model is opt-in and leaves the existing model implementations untouched.
It consumes the full normalized RTMW-133 tensor and fuses pooled branch
features only immediately before classification.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from isaa.graph.adjacency import build_joint_spatial_partitions, build_joint_adjacency
from isaa.graph.regions import build_region_partition
from isaa.layouts.rtmw_133 import RTMW_32_NODE_INDICES
from isaa.models.rtmw_local_ctr import CTRGCNBlock, FaceTokenCompression, PointBatchNorm


class LocalSTBlock(nn.Module):
    def __init__(self, cin: int, cout: int, adjacency: torch.Tensor, stride: int = 1):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.spatial = nn.Conv2d(cin, cout, 1, bias=False)
        self.temporal = nn.Conv2d(cout, cout, (5, 1), stride=(stride, 1), padding=(2, 0), bias=False)
        self.norm = nn.BatchNorm2d(cout)
        self.stride = stride
        self.residual = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x, mask):
        y = torch.einsum("bctn,nm->bctm", self.spatial(x), self.adjacency)
        y = self.temporal(y.masked_fill(~mask, 0))
        if self.stride > 1:
            mask = F.max_pool2d(mask.float(), (self.stride, 1), (self.stride, 1), ceil_mode=True).bool()
        residual = self.residual(x)
        if residual.size(2) != y.size(2):
            residual = residual[:, :, :y.size(2)]
        return F.relu(self.norm(y) + residual).masked_fill(~mask, 0), mask


class BodyLocalFusion(nn.Module):
    ARCHITECTURE = "rtmw_ctr22_st48_handface_v1"
    BODY_INDICES = tuple(i for i in RTMW_32_NODE_INDICES
                         if i not in {95, 99, 103, 107, 111, 116, 120, 124, 128, 132})
    HAND_INDICES = tuple(range(91, 133))
    CHANNELS = (48, 48, 48, 48, 96, 96, 96, 192, 192, 192)
    LOCAL_CHANNELS = (24, 24, 48, 48)

    def __init__(self, num_classes: int = 120, body_channels=None, local_channels=None):
        super().__init__()
        body_channels = tuple(body_channels or self.CHANNELS)
        local_channels = tuple(local_channels or self.LOCAL_CHANNELS)
        self.register_buffer("body_indices", torch.tensor(self.BODY_INDICES, dtype=torch.long))
        partition = build_region_partition("rtmw_133", 133)
        full_graph = build_joint_spatial_partitions(133, partition, "rtmw_133", scope="full")
        ids = self.body_indices
        body_graph = full_graph.index_select(1, ids).index_select(2, ids)
        degree = body_graph.sum(-1, keepdim=True).clamp_min(1)
        body_graph = body_graph / degree
        self.body_input_norm = PointBatchNorm(3 * len(self.BODY_INDICES))
        self.body_blocks = nn.ModuleList()
        cin = 3
        for i, cout in enumerate(body_channels):
            stride = 2 if i in (4, 7) else 1
            self.body_blocks.append(CTRGCNBlock(cin, cout, body_graph, stride=stride, residual=i != 0))
            cin = cout

        compressor = FaceTokenCompression(full_graph, torch.tensor(partition.joint_to_region))
        self.register_buffer("face_original_to_token", compressor.original_to_token.clone())
        self.register_buffer("face_indices", compressor.face_indices.clone())
        self.register_buffer("face_members", compressor.face_members.clone())
        # Six face tokens are appended after 42 hand joints: 48 local nodes total.
        face_hand_graph = torch.eye(48)
        hand_graph = build_joint_adjacency(133, "rtmw_133")
        hidx = torch.tensor(self.HAND_INDICES)
        hand_sub = hand_graph.index_select(0, hidx).index_select(1, hidx)
        face_hand_graph[:42, :42] = hand_sub
        face_hand_graph[42:, 42:] = compressor.adjacency[0, 65:, 65:]
        face_hand_graph = face_hand_graph / face_hand_graph.sum(-1, keepdim=True).clamp_min(1)
        self.local_blocks = nn.ModuleList()
        cin = 3
        for i, cout in enumerate(local_channels):
            stride = 2 if i == 1 else 1
            self.local_blocks.append(LocalSTBlock(cin, cout, face_hand_graph, stride))
            cin = cout
        self.body_projection = nn.Sequential(nn.Linear(body_channels[-1], 192), nn.ReLU())
        self.local_projection = nn.Sequential(nn.Linear(local_channels[-1], 96), nn.ReLU())
        self.classifier = nn.Linear(288, num_classes)

    @staticmethod
    def _masked_pool(x, mask):
        w = mask.to(x.dtype)
        return (x * w).sum((2, 3)) / w.sum((2, 3)).clamp_min(1)

    def forward(self, x, valid_frame_mask=None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != 133:
            raise ValueError("Expected B x 3 x T x 133 x M RTMW input")
        b, c, t, _, m = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask &= valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(b*m, c, t, 133).masked_fill(~mask.permute(0,4,1,2,3).reshape(b*m,1,t,133), 0)
        mask = mask.permute(0,4,1,2,3).reshape(b*m,1,t,133)
        body = x.index_select(-1, self.body_indices)
        bm = mask.index_select(-1, self.body_indices)
        norm_mask = bm.expand(-1, 3, -1, -1).permute(0, 1, 3, 2).reshape(
            b*m, 3*len(self.BODY_INDICES), t, 1)
        body = self.body_input_norm(body.permute(0,1,3,2).reshape(
            b*m, 3*len(self.BODY_INDICES), t, 1), norm_mask)
        body = body.reshape(b*m,3,len(self.BODY_INDICES),t).permute(0,1,3,2)
        for block in self.body_blocks:
            body, bm = block(body, bm)
        hands = x.index_select(-1, torch.tensor(self.HAND_INDICES, device=x.device))
        hm = mask.index_select(-1, torch.tensor(self.HAND_INDICES, device=x.device))
        fidx = self.face_indices.flatten().to(x.device)
        fx = x.index_select(-1, fidx).reshape(b*m,3,t,6,12)
        fm = mask.index_select(-1, fidx).reshape(b*m,1,t,6,12) & self.face_members.to(x.device)[None,None,None]
        fc = fm.sum(-1)
        face = fm.to(x.dtype).mul(fx).sum(-1) / fc.clamp_min(1)
        local = torch.cat((hands, face), -1)
        lm = torch.cat((hm, fc > 0), -1)
        for block in self.local_blocks:
            local, lm = block(local, lm)
        bv = self._masked_pool(body, bm).reshape(b,m,-1).mean(1)
        lv = self._masked_pool(local, lm).reshape(b,m,-1).mean(1)
        return self.classifier(torch.cat((self.body_projection(bv), self.local_projection(lv)), dim=1))

