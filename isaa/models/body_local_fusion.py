"""Current RTMW body CTR-GCN + hand/face ST-GCN action classifier.

BodyLocalFusion is the project's default research baseline. It consumes the
full RTMW-133 tensor with raw x/y/score channels and fuses pooled body and local branch
features immediately before classification. Legacy 32-node ISAA models and
newer attention variants remain selectable for explicit comparisons.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.graph.adjacency import build_joint_spatial_partitions, build_joint_adjacency
from isaa.graph.regions import build_region_partition
from isaa.layouts.rtmw_133 import RTMW_32_NODE_INDICES
from isaa.models.original_ctrgcn import TCNGCNUnit, build_official_rtmw_adjacency
from isaa.models.official_stgcn import OfficialSTGCNFeatureExtractor
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
    LOCAL_STRIDES = (1,)
    LOCAL_COORDINATE_MODE = "raw"

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
            stride = 2 if i in self.LOCAL_STRIDES else 1
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
        local_source = x
        if self.LOCAL_COORDINATE_MODE == "torso_relative":
            torso_indices = torch.tensor((5, 6, 11, 12), device=x.device)
            torso_points = x[:, :2].index_select(-1, torso_indices)
            torso_mask = mask.to(x.dtype).index_select(-1, torso_indices)
            torso_center = (torso_points * torso_mask).sum(-1, keepdim=True) / torso_mask.sum(-1, keepdim=True).clamp_min(1)
            local_source = x.clone()
            local_source[:, :2] = local_source[:, :2] - torso_center

        hands = local_source.index_select(-1, torch.tensor(self.HAND_INDICES, device=x.device))
        hm = mask.index_select(-1, torch.tensor(self.HAND_INDICES, device=x.device))
        fidx = self.face_indices.flatten().to(x.device)
        fx = local_source.index_select(-1, fidx).reshape(b*m,3,t,6,12)
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


class BodyLocalRelativeFusion(BodyLocalFusion):
    """Body-local model using torso-relative xy for the hand/face branch only."""

    ARCHITECTURE = "rtmw_ctr22_st48_handface_torso_relative_v1"
    LOCAL_COORDINATE_MODE = "torso_relative"


class BodyLocalHandCTRRelativeFusion(BodyLocalRelativeFusion):
    """Torso-relative model with shared CTR-GCN blocks for each 21-joint hand."""

    ARCHITECTURE = "rtmw_ctr22_handctr21_facest48_torso_relative_v1"
    HAND_CHANNELS = (24, 24, 48, 48)

    def __init__(self, num_classes: int = 120):
        super().__init__(num_classes=num_classes)
        partition = build_region_partition("rtmw_133", 133)
        full_graph = build_joint_spatial_partitions(133, partition, "rtmw_133", scope="full")
        hand_ids = torch.tensor(self.HAND_INDICES[:21], dtype=torch.long)
        hand_graph = full_graph.index_select(1, hand_ids).index_select(2, hand_ids)
        hand_graph = hand_graph / hand_graph.sum(-1, keepdim=True).clamp_min(1)
        # One shared module is applied to the left and right hand separately.
        self.hand_ctr_blocks = nn.ModuleList()
        cin = 3
        for cout in self.HAND_CHANNELS:
            self.hand_ctr_blocks.append(CTRGCNBlock(cin, cout, hand_graph, stride=1, residual=cin != 3 or cout != 3))
            cin = cout
        face_graph = self.local_blocks[0].adjacency[42:, 42:]
        self.face_st_blocks = nn.ModuleList()
        cin = 3
        for cout in self.LOCAL_CHANNELS:
            self.face_st_blocks.append(LocalSTBlock(cin, cout, face_graph, stride=1))
            cin = cout
        # Hand and face branches are fused frame-by-frame.  Keep the face
        # representation at 48 channels while allowing wider hand stacks in
        # derived ablations.
        self.hand_to_local = (nn.Identity() if self.HAND_CHANNELS[-1] == self.LOCAL_CHANNELS[-1]
                              else nn.Conv2d(self.HAND_CHANNELS[-1], self.LOCAL_CHANNELS[-1], 1, bias=False))

    def _run_hand(self, hand, hand_mask):
        for block in self.hand_ctr_blocks:
            hand, hand_mask = block(hand, hand_mask)
        return hand, hand_mask

    @staticmethod
    def _pool_nodes_per_frame(x, mask):
        w = mask.to(x.dtype)
        count = w.sum(-1).squeeze(1)
        feat = (x * w).sum(-1).transpose(1, 2) / count.unsqueeze(-1).clamp_min(1)
        return feat, count > 0

    def forward(self, x, valid_frame_mask=None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != 133:
            raise ValueError("Expected B x 3 x T x 133 x M RTMW input")
        b, c, t, _, m = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask &= valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(b * m, c, t, 133)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b * m, 1, t, 133)
        x = x.masked_fill(~mask, 0)

        body = x.index_select(-1, self.body_indices)
        bm = mask.index_select(-1, self.body_indices)
        norm_mask = bm.expand(-1, 3, -1, -1).permute(0, 1, 3, 2).reshape(
            b * m, 3 * len(self.BODY_INDICES), t, 1)
        body = self.body_input_norm(body.permute(0, 1, 3, 2).reshape(
            b * m, 3 * len(self.BODY_INDICES), t, 1), norm_mask)
        body = body.reshape(b * m, 3, len(self.BODY_INDICES), t).permute(0, 1, 3, 2)
        for block in self.body_blocks:
            body, bm = block(body, bm)

        torso_indices = torch.tensor((5, 6, 11, 12), device=x.device)
        torso_points = x[:, :2].index_select(-1, torso_indices)
        torso_mask = mask.to(x.dtype).index_select(-1, torso_indices)
        torso_center = (torso_points * torso_mask).sum(-1, keepdim=True) / torso_mask.sum(-1, keepdim=True).clamp_min(1)
        local_source = x.clone()
        local_source[:, :2] = local_source[:, :2] - torso_center

        hand0 = local_source[..., 91:112]
        hand1 = local_source[..., 112:133]
        hm0 = mask[..., 91:112]
        hm1 = mask[..., 112:133]
        hand0, hm0 = self._run_hand(hand0, hm0)
        hand1, hm1 = self._run_hand(hand1, hm1)
        hand0 = self.hand_to_local(hand0)
        hand1 = self.hand_to_local(hand1)
        hand_t0, hand_valid0 = self._pool_nodes_per_frame(hand0, hm0)
        hand_t1, hand_valid1 = self._pool_nodes_per_frame(hand1, hm1)
        hand_t = (hand_t0 + hand_t1) * 0.5
        hand_valid = hand_valid0 | hand_valid1

        fidx = self.face_indices.flatten().to(x.device)
        fx = local_source.index_select(-1, fidx).reshape(b * m, 3, t, 6, 12)
        fm = mask.index_select(-1, fidx).reshape(b * m, 1, t, 6, 12)
        fm = fm & self.face_members.to(x.device)[None, None, None]
        fc = fm.sum(-1)
        face = fm.to(x.dtype).mul(fx).sum(-1) / fc.clamp_min(1)
        face_mask = fc > 0
        for block in self.face_st_blocks:
            face, face_mask = block(face, face_mask)
        face_t, face_valid = self._pool_nodes_per_frame(face, face_mask)

        body_t, body_valid = self._pool_nodes_per_frame(body, bm)
        target_t = body_t.shape[1]
        if hand_t.shape[1] != target_t:
            hand_t = F.interpolate(hand_t.transpose(1, 2), size=target_t, mode="linear", align_corners=False).transpose(1, 2)
            hand_valid = F.interpolate(hand_valid[:, None].float(), size=target_t, mode="nearest")[:, 0].bool()
        if face_t.shape[1] != target_t:
            face_t = F.interpolate(face_t.transpose(1, 2), size=target_t, mode="linear", align_corners=False).transpose(1, 2)
            face_valid = F.interpolate(face_valid[:, None].float(), size=target_t, mode="nearest")[:, 0].bool()
        bv = self.body_projection(body_t.reshape(b, m, target_t, -1).mean(1).mean(1))
        local_valid = hand_valid | face_valid
        local_t = (hand_t + face_t) * 0.5
        local_t = local_t * local_valid.unsqueeze(-1).to(local_t.dtype)
        # Keep the person aggregation consistent with the torso branch.  The
        # hand/face path is still flattened as ``B*M`` at this point; reducing
        # only the time axis would leave ``lv`` with a ``B*M`` batch dimension
        # and make the final concat fail whenever more than one person is
        # present.  Restore ``(B, M, T, C)`` before averaging people and time.
        lv = self.local_projection(
            local_t.reshape(b, m, target_t, -1).mean(1).mean(1)
        )
        return self.classifier(torch.cat((bv, lv), dim=1))


class BodyLocalHandCTRWideRelativeFusion(BodyLocalHandCTRRelativeFusion):
    """Torso-relative hand CTR-GCN width ablation.

    The two hands still share one CTR-GCN stack.  The wider 96-channel output
    is projected to the existing 48-channel local fusion width so the face
    branch and classifier contract remain unchanged.
    """

    ARCHITECTURE = "rtmw_ctr22_handctr21_wide_facest48_torso_relative_v1"
    HAND_CHANNELS = (32, 32, 64, 64, 96, 96)


class BodyLocalRelativeSplitFusion(BodyLocalRelativeFusion):
    """Relative-coordinate model with independent hand and face ST-GCNs."""

    ARCHITECTURE = "rtmw_ctr22_st42_hand_st6_face_torso_relative_v1"
    LOCAL_COORDINATE_MODE = "torso_relative_precomputed"

    def __init__(self, num_classes: int = 120):
        super().__init__(num_classes=num_classes)
        # Replace the joint 48-node graph with two independent ST-GCN stacks.
        # Keep the same layer widths and temporal strides as the best relative
        # model so this experiment isolates separate branch processing.
        self.local_blocks = nn.ModuleList()
        self.hand_blocks = nn.ModuleList()
        self.face_blocks = nn.ModuleList()

        hand_graph = build_joint_adjacency(133, "rtmw_133")
        hidx = torch.tensor(self.HAND_INDICES)
        hand_graph = hand_graph.index_select(0, hidx).index_select(1, hidx)
        hand_graph = hand_graph / hand_graph.sum(-1, keepdim=True).clamp_min(1)
        # The best relative model's face-token partition uses self-loops only.
        face_graph = torch.eye(6, dtype=hand_graph.dtype)

        for graph, blocks in ((hand_graph, self.hand_blocks), (face_graph, self.face_blocks)):
            cin = 3
            for i, cout in enumerate(self.LOCAL_CHANNELS):
                stride = 2 if i in self.LOCAL_STRIDES else 1
                blocks.append(LocalSTBlock(cin, cout, graph, stride))
                cin = cout

        # Preserve the original 96-dimensional local representation and 288-D
        # classifier input: hand contributes 64 dims, face contributes 32.
        self.hand_projection = nn.Sequential(nn.Linear(self.LOCAL_CHANNELS[-1], 64), nn.ReLU())
        self.face_projection = nn.Sequential(nn.Linear(self.LOCAL_CHANNELS[-1], 32), nn.ReLU())
        self.local_projection = nn.Identity()

    def forward(self, x, valid_frame_mask=None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] != 5 or x.shape[3] != 133:
            raise ValueError("Expected B x 5 x T x 133 x M RTMW input with precomputed relative xy")
        b, c, t, _, m = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x[:, :3]).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask &= valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(b * m, c, t, 133)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b * m, 1, t, 133)
        x = x.masked_fill(~mask, 0)

        body = x[:, :3].index_select(-1, self.body_indices)
        bm = mask.index_select(-1, self.body_indices)
        norm_mask = bm.expand(-1, 3, -1, -1).permute(0, 1, 3, 2).reshape(
            b * m, 3 * len(self.BODY_INDICES), t, 1)
        body = self.body_input_norm(body.permute(0, 1, 3, 2).reshape(
            b * m, 3 * len(self.BODY_INDICES), t, 1), norm_mask)
        body = body.reshape(b * m, 3, len(self.BODY_INDICES), t).permute(0, 1, 3, 2)
        for block in self.body_blocks:
            body, bm = block(body, bm)

        local_source = torch.cat((x[:, 3:5], x[:, 2:3]), dim=1)

        hand_indices = torch.tensor(self.HAND_INDICES, device=x.device)
        hands = local_source.index_select(-1, hand_indices)
        hm = mask.index_select(-1, hand_indices)
        for block in self.hand_blocks:
            hands, hm = block(hands, hm)

        fidx = self.face_indices.flatten().to(x.device)
        fx = local_source.index_select(-1, fidx).reshape(b * m, 3, t, 6, 12)
        fm = mask.index_select(-1, fidx).reshape(b * m, 1, t, 6, 12)
        fm = fm & self.face_members.to(x.device)[None, None, None]
        fc = fm.sum(-1)
        face = fm.to(x.dtype).mul(fx).sum(-1) / fc.clamp_min(1)
        fm = fc > 0
        for block in self.face_blocks:
            face, fm = block(face, fm)

        bv = self._masked_pool(body, bm).reshape(b, m, -1).mean(1)
        hv = self._masked_pool(hands, hm).reshape(b, m, -1).mean(1)
        fv = self._masked_pool(face, fm).reshape(b, m, -1).mean(1)
        local = torch.cat((self.hand_projection(hv), self.face_projection(fv)), dim=1)
        return self.classifier(torch.cat((self.body_projection(bv), local), dim=1))


class BodyLocalDropoutFusion(BodyLocalFusion):
    """Body-local baseline with dropout only at branch fusion projections."""

    ARCHITECTURE = "rtmw_ctr22_st48_handface_dropout_v1"
    DROPOUT = 0.2

    def __init__(self, num_classes: int = 120):
        super().__init__(num_classes=num_classes)
        self.body_projection = nn.Sequential(
            nn.Linear(self.CHANNELS[-1], 192), nn.ReLU(), nn.Dropout(self.DROPOUT)
        )
        self.local_projection = nn.Sequential(
            nn.Linear(self.LOCAL_CHANNELS[-1], 96), nn.ReLU(), nn.Dropout(self.DROPOUT)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.DROPOUT), nn.Linear(288, num_classes)
        )


class BodyLocalFullFusion(BodyLocalFusion):
    """Full-width 10-layer ST-GCN local branch for comparison with baseline."""

    ARCHITECTURE = "rtmw_ctr22_st48_handface_full"
    CHANNELS = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256)
    LOCAL_CHANNELS = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256)
    LOCAL_STRIDES = (4, 7)


class TorsoCenteredCrossBranchFusion(BodyLocalFusion):
    """Per-frame torso-query attention with independent hand/face gates."""

    ARCHITECTURE = "rtmw_torso_centered_hand_face_cross_attention_v1"
    FUSION_WIDTH = 96
    # Match the original ST-GCN paper: nine layers, with temporal
    # downsampling at layers 4 and 7 (zero-based indices 3 and 6).
    LOCAL_CHANNELS = (64, 64, 64, 128, 128, 128, 256, 256, 256)
    LOCAL_STRIDES = (3, 6)

    def __init__(self, num_classes: int = 120):
        super().__init__(num_classes=num_classes)
        self.body_projection = nn.Linear(self.CHANNELS[-1], self.FUSION_WIDTH)
        self.hand_projection = nn.Linear(self.LOCAL_CHANNELS[-1], self.FUSION_WIDTH)
        self.face_projection = nn.Linear(self.LOCAL_CHANNELS[-1], self.FUSION_WIDTH)
        self.torso_norm = nn.LayerNorm(self.FUSION_WIDTH)
        self.hand_norm = nn.LayerNorm(self.FUSION_WIDTH)
        self.face_norm = nn.LayerNorm(self.FUSION_WIDTH)
        self.torso_query_hand = nn.Linear(self.FUSION_WIDTH, self.FUSION_WIDTH, bias=False)
        self.hand_key = nn.Linear(self.FUSION_WIDTH, self.FUSION_WIDTH, bias=False)
        self.torso_query_face = nn.Linear(self.FUSION_WIDTH, self.FUSION_WIDTH, bias=False)
        self.face_key = nn.Linear(self.FUSION_WIDTH, self.FUSION_WIDTH, bias=False)
        self.hand_gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.face_gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.classifier = nn.Linear(self.FUSION_WIDTH, num_classes)

    @staticmethod
    def _pool_nodes_per_frame(x, mask):
        # x [B,C,T,V], mask [B,1,T,V] -> [B,T,C], valid [B,T]
        w = mask.to(x.dtype)
        count = w.sum(-1).squeeze(1)
        feat = (x * w).sum(-1).transpose(1, 2) / count.unsqueeze(-1).clamp_min(1)
        return feat, count > 0

    def forward(self, x, valid_frame_mask=None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != 133:
            raise ValueError("Expected B x 3 x T x 133 x M RTMW input")
        b, c, t, _, m = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask &= valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(b*m, c, t, 133)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b*m, 1, t, 133)
        x = x.masked_fill(~mask, 0)

        body = x.index_select(-1, self.body_indices)
        bm = mask.index_select(-1, self.body_indices)
        norm_mask = bm.expand(-1, 3, -1, -1).permute(0, 1, 3, 2).reshape(
            b*m, 3*len(self.BODY_INDICES), t, 1)
        body = self.body_input_norm(body.permute(0, 1, 3, 2).reshape(
            b*m, 3*len(self.BODY_INDICES), t, 1), norm_mask)
        body = body.reshape(b*m, 3, len(self.BODY_INDICES), t).permute(0, 1, 3, 2)
        for block in self.body_blocks:
            body, bm = block(body, bm)

        hands = x.index_select(-1, torch.tensor(self.HAND_INDICES, device=x.device))
        hm = mask.index_select(-1, torch.tensor(self.HAND_INDICES, device=x.device))
        fidx = self.face_indices.flatten().to(x.device)
        fx = x.index_select(-1, fidx).reshape(b*m, 3, t, 6, 12)
        fm = mask.index_select(-1, fidx).reshape(b*m, 1, t, 6, 12)
        fm = fm & self.face_members.to(x.device)[None, None, None]
        fc = fm.sum(-1)
        face = fm.to(x.dtype).mul(fx).sum(-1) / fc.clamp_min(1)
        local = torch.cat((hands, face), -1)
        lm = torch.cat((hm, fc > 0), -1)
        for block in self.local_blocks:
            local, lm = block(local, lm)

        body_t, body_valid = self._pool_nodes_per_frame(body, bm)
        hand_t, hand_valid = self._pool_nodes_per_frame(local[..., :42], lm[..., :42])
        face_t, face_valid = self._pool_nodes_per_frame(local[..., 42:], lm[..., 42:])
        target_t = body_t.shape[1]
        if hand_t.shape[1] != target_t:
            hand_t = F.interpolate(hand_t.transpose(1, 2), size=target_t, mode="linear",
                                   align_corners=False).transpose(1, 2)
            hand_valid = F.interpolate(hand_valid[:, None].float(), size=target_t,
                                       mode="nearest")[:, 0].bool()
        if face_t.shape[1] != target_t:
            face_t = F.interpolate(face_t.transpose(1, 2), size=target_t, mode="linear",
                                   align_corners=False).transpose(1, 2)
            face_valid = F.interpolate(face_valid[:, None].float(), size=target_t,
                                       mode="nearest")[:, 0].bool()
        torso = self.torso_norm(self.body_projection(body_t))
        hand = self.hand_norm(self.hand_projection(hand_t))
        face = self.face_norm(self.face_projection(face_t))
        hand_score = (self.torso_query_hand(torso) * self.hand_key(hand)).sum(-1) / self.FUSION_WIDTH**0.5
        face_score = (self.torso_query_face(torso) * self.face_key(face)).sum(-1) / self.FUSION_WIDTH**0.5
        alpha_hand = torch.sigmoid(hand_score + self.hand_gate_bias) * hand_valid.to(hand.dtype)
        alpha_face = torch.sigmoid(face_score + self.face_gate_bias) * face_valid.to(face.dtype)
        fused = torso + alpha_hand.unsqueeze(-1) * hand + alpha_face.unsqueeze(-1) * face
        valid = body_valid | hand_valid | face_valid
        fused = fused * valid.unsqueeze(-1).to(fused.dtype)
        fused = fused.reshape(b, m, target_t, self.FUSION_WIDTH).mean(1)
        valid = valid.reshape(b, m, target_t).any(1)
        pooled = (fused * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.classifier(pooled)


class OfficialCTRGCNFeatureExtractor(nn.Module):
    """Official CTR-GCN ten-block backbone without its classifier."""

    CHANNELS = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256)

    def __init__(self, adjacency: torch.Tensor, in_channels: int = 3,
                 channels: tuple[int, ...] | None = None):
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.shape[0] != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("CTR-GCN adjacency must have shape 3 x V x V")
        self.num_point = int(adjacency.shape[1])
        self.in_channels = int(in_channels)
        self.channels = tuple(channels or self.CHANNELS)
        if len(self.channels) != 10:
            raise ValueError("CTR-GCN feature extractor requires ten channel widths")
        self.register_buffer("A", adjacency)
        self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_point)
        layers = []
        cin = self.in_channels
        for index, cout in enumerate(self.channels):
            layers.append(TCNGCNUnit(
                cin, cout, self.A, stride=2 if index in (4, 7) else 1,
                residual=index != 0, adaptive=True,
            ))
            cin = cout
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        b, c, t, v, m = x.shape
        if (c, v, m) != (self.in_channels, self.num_point, 1):
            raise ValueError(f"Expected C,V,M=({self.in_channels},{self.num_point},1), got {(c,v,m)}")
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(b, v * c, t)
        x = self.data_bn(x)
        x = x.view(b, 1, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(b, c, t, v)
        for block in self.blocks:
            x = block(x)
        return x, int(x.shape[2])


class OfficialTorsoCenteredCrossBranchFusion(nn.Module):
    """Official CTR-GCN torso + official ST-GCN hand/face attention fusion."""

    ARCHITECTURE = "rtmw_official_ctr_body22_official_st_handface_torso_attention"
    BODY_INDICES = tuple(i for i in RTMW_32_NODE_INDICES
                         if i not in {95, 99, 103, 107, 111, 116, 120, 124, 128, 132})
    HAND_INDICES = tuple(range(91, 133))
    FUSION_WIDTH = 256
    BODY_CHANNELS = (48, 48, 48, 48, 96, 96, 96, 192, 192, 192)
    LOCAL_CHANNELS = (24, 24, 48, 48, 96, 96)
    LOCAL_STRIDES = (1, 2, 1, 1, 1, 1)

    def __init__(self, num_classes: int = 120):
        super().__init__()
        partition = build_region_partition("rtmw_133", 133)
        official_graph = build_official_rtmw_adjacency()
        body_ids = torch.tensor(self.BODY_INDICES, dtype=torch.long)
        body_graph = official_graph.index_select(1, body_ids).index_select(2, body_ids)
        # CTR-GCN's reference graph utility uses column-normalized directed
        # partitions (normalize_digraph), including after the RTMW body
        # subgraph is induced.
        body_graph = body_graph / body_graph.sum(1, keepdim=True).clamp_min(1)
        self.register_buffer("body_indices", body_ids)
        self.body_encoder = OfficialCTRGCNFeatureExtractor(body_graph, channels=self.BODY_CHANNELS)

        compressor = FaceTokenCompression(
            build_joint_spatial_partitions(133, partition, "rtmw_133", scope="full"),
            torch.tensor(partition.joint_to_region),
        )
        self.register_buffer("face_indices", compressor.face_indices.clone())
        self.register_buffer("face_members", compressor.face_members.clone())
        self.register_buffer("hand_indices", torch.tensor(self.HAND_INDICES, dtype=torch.long))
        full_graph = build_joint_spatial_partitions(133, partition, "rtmw_133", scope="full")
        local_graph = full_graph.index_select(1, self.hand_indices).index_select(2, self.hand_indices)
        face_graph = compressor.adjacency[:, 65:, 65:]
        local_graph = torch.zeros(3, 48, 48, dtype=local_graph.dtype)
        local_graph[:, :42, :42] = full_graph.index_select(1, self.hand_indices).index_select(2, self.hand_indices)
        local_graph[:, 42:, 42:] = face_graph
        # Match ST-GCN's ``normalize_digraph`` convention: normalize each
        # source column independently within every spatial partition.
        local_graph = local_graph / local_graph.sum(1, keepdim=True).clamp_min(1)
        self.local_encoder = OfficialSTGCNFeatureExtractor(
            3, local_graph, channels=self.LOCAL_CHANNELS, strides=self.LOCAL_STRIDES,
        )

        self.body_projection = nn.Linear(self.BODY_CHANNELS[-1], self.FUSION_WIDTH)
        self.hand_projection = nn.Linear(self.LOCAL_CHANNELS[-1], self.FUSION_WIDTH)
        self.face_projection = nn.Linear(self.LOCAL_CHANNELS[-1], self.FUSION_WIDTH)
        self.torso_norm = nn.LayerNorm(self.FUSION_WIDTH)
        self.hand_norm = nn.LayerNorm(self.FUSION_WIDTH)
        self.face_norm = nn.LayerNorm(self.FUSION_WIDTH)
        self.torso_query_hand = nn.Linear(self.FUSION_WIDTH, self.FUSION_WIDTH, bias=False)
        self.hand_key = nn.Linear(self.FUSION_WIDTH, self.FUSION_WIDTH, bias=False)
        self.torso_query_face = nn.Linear(self.FUSION_WIDTH, self.FUSION_WIDTH, bias=False)
        self.face_key = nn.Linear(self.FUSION_WIDTH, self.FUSION_WIDTH, bias=False)
        self.hand_gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.face_gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.classifier = nn.Linear(self.FUSION_WIDTH, num_classes)

    @staticmethod
    def _pool_nodes_per_frame(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = mask.to(x.dtype)
        count = w.sum(-1).squeeze(1)
        feat = (x * w).sum(-1).transpose(1, 2) / count.unsqueeze(-1).clamp_min(1)
        return feat, count > 0

    @staticmethod
    def _downsample_mask(mask: torch.Tensor, target_t: int) -> torch.Tensor:
        while mask.shape[2] > target_t:
            mask = F.max_pool2d(mask.float(), (2, 1), (2, 1), ceil_mode=True).bool()
        return mask[:, :, :target_t]

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != 133:
            raise ValueError("Expected B x 3 x T x 133 x M RTMW input")
        b, c, t, _, m = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask &= valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(b * m, c, t, 133)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b * m, 1, t, 133)
        x = x.masked_fill(~mask, 0)

        body = x.index_select(-1, self.body_indices).unsqueeze(-1)
        bm = mask.index_select(-1, self.body_indices)
        body, body_t_len = self.body_encoder(body)
        bm = self._downsample_mask(bm, body_t_len)

        hands = x.index_select(-1, self.hand_indices)
        hm = mask.index_select(-1, self.hand_indices)
        fidx = self.face_indices.flatten().to(x.device)
        fx = x.index_select(-1, fidx).reshape(b * m, 3, t, 6, 12)
        fm = mask.index_select(-1, fidx).reshape(b * m, 1, t, 6, 12)
        fm = fm & self.face_members.to(x.device)[None, None, None]
        fc = fm.sum(-1)
        face = fm.to(x.dtype).mul(fx).sum(-1) / fc.clamp_min(1)
        local = torch.cat((hands, face), -1).unsqueeze(-1)
        lm = torch.cat((hm, fc > 0), -1)
        local, local_t_len = self.local_encoder(local)
        lm = self._downsample_mask(lm, local_t_len)
        body_t, body_valid = self._pool_nodes_per_frame(body, bm)
        local_t, local_valid = self._pool_nodes_per_frame(local, lm)
        if local_t.shape[1] != body_t.shape[1]:
            local_t = F.interpolate(local_t.transpose(1, 2), size=body_t.shape[1], mode="linear", align_corners=False).transpose(1, 2)
            local_valid = F.interpolate(local_valid[:, None].float(), size=body_t.shape[1], mode="nearest")[:, 0].bool()
        hand_t, hand_valid = self._pool_nodes_per_frame(local[..., :42], lm[..., :42])
        face_t, face_valid = self._pool_nodes_per_frame(local[..., 42:], lm[..., 42:])
        target_t = body_t.shape[1]
        if hand_t.shape[1] != target_t:
            hand_t = F.interpolate(hand_t.transpose(1, 2), size=target_t, mode="linear", align_corners=False).transpose(1, 2)
            hand_valid = F.interpolate(hand_valid[:, None].float(), size=target_t, mode="nearest")[:, 0].bool()
        if face_t.shape[1] != target_t:
            face_t = F.interpolate(face_t.transpose(1, 2), size=target_t, mode="linear", align_corners=False).transpose(1, 2)
            face_valid = F.interpolate(face_valid[:, None].float(), size=target_t, mode="nearest")[:, 0].bool()
        torso = self.torso_norm(self.body_projection(body_t))
        hand = self.hand_norm(self.hand_projection(hand_t))
        face = self.face_norm(self.face_projection(face_t))
        hand_score = (self.torso_query_hand(torso) * self.hand_key(hand)).sum(-1) / math.sqrt(self.FUSION_WIDTH)
        face_score = (self.torso_query_face(torso) * self.face_key(face)).sum(-1) / math.sqrt(self.FUSION_WIDTH)
        alpha_hand = torch.sigmoid(hand_score + self.hand_gate_bias) * hand_valid.to(hand.dtype)
        alpha_face = torch.sigmoid(face_score + self.face_gate_bias) * face_valid.to(face.dtype)
        fused = torso + alpha_hand.unsqueeze(-1) * hand + alpha_face.unsqueeze(-1) * face
        valid = body_valid | hand_valid | face_valid
        fused = fused * valid.unsqueeze(-1).to(fused.dtype)
        fused = fused.reshape(b, m, target_t, self.FUSION_WIDTH).mean(1)
        valid = valid.reshape(b, m, target_t).any(1)
        pooled = (fused * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.classifier(pooled)

