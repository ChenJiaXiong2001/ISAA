"""Face-light full-body fusion with shared hand CTR-GCN wrist injection.

The model keeps five coarse head landmarks for head-orientation cues, removes
the dense face branch, encodes each 21-point hand with one shared CTR-GCN, and
injects the resulting per-frame descriptors into the matching body wrists
before the second half of the body CTR-GCN.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.graph.adjacency import build_joint_spatial_partitions, normalize_adjacency_partitions
from isaa.graph.regions import build_region_partition
from isaa.layouts.rtmw_133 import register as register_rtmw_layout
from isaa.models.rtmw_local_ctr import CTRGCNBlock, PointBatchNorm


HAND_CTR_WRIST_VARIANT = "hand-ctr-wrist-fusion"


class HandCTRWristFusion(nn.Module):
    """19-point body CTR-GCN with hand and coarse-head residual injection."""

    ARCHITECTURE = "rtmw_body19_handctr21_head5_wrist_injection_v1"
    BODY_INDICES = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22)
    HAND_INDICES = tuple(range(91, 133))
    HEAD_INDICES = (0, 1, 2, 3, 4)
    TORSO_INDICES = (5, 6, 11, 12)
    BODY_CHANNELS = (48, 48, 48, 48, 96, 96, 96, 192, 192, 192)
    HAND_CHANNELS = (32, 32, 64, 64, 96, 96)
    HAND_INPUT_CHANNELS = 6
    HEAD_OUTPUT_CHANNELS = 48
    INJECTION_BLOCK = 4
    LOCAL_COORDINATE_MODE = "torso_relative"

    def __init__(self, num_classes: int = 60):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        register_rtmw_layout()
        self.num_classes = int(num_classes)
        self.body_indices_tensor = torch.tensor(self.BODY_INDICES, dtype=torch.long)
        self.hand_indices_tensor = torch.tensor(self.HAND_INDICES, dtype=torch.long)
        self.head_indices_tensor = torch.tensor(self.HEAD_INDICES, dtype=torch.long)
        self.register_buffer("body_indices", self.body_indices_tensor)
        self.register_buffer("hand_indices", self.hand_indices_tensor)
        self.register_buffer("head_indices", self.head_indices_tensor)
        self.register_buffer("torso_indices", torch.tensor(self.TORSO_INDICES, dtype=torch.long))
        self.register_buffer("head_injection_index", torch.tensor(0, dtype=torch.long))
        self.register_buffer("wrist_positions", torch.tensor((5, 6), dtype=torch.long))
        wrist_map = torch.zeros(2, len(self.BODY_INDICES), dtype=torch.float32)
        wrist_map[0, 5] = 1.0
        wrist_map[1, 6] = 1.0
        head_map = torch.zeros(1, len(self.BODY_INDICES), dtype=torch.float32)
        head_map[0, 0] = 1.0
        self.register_buffer("wrist_map", wrist_map)
        self.register_buffer("head_map", head_map)

        partition = build_region_partition("rtmw_133", 133)
        full_graph = build_joint_spatial_partitions(133, partition, "rtmw_133", scope="full")
        body_graph = full_graph.index_select(1, self.body_indices).index_select(2, self.body_indices)
        # The 19-point induced graph otherwise leaves the nose isolated. Add
        # explicit coarse head-to-shoulder links to preserve head orientation.
        body_graph = body_graph.clone()
        body_graph[:, 0, 1] = 1.0
        body_graph[:, 1, 0] = 1.0
        body_graph[:, 0, 2] = 1.0
        body_graph[:, 2, 0] = 1.0
        body_graph = normalize_adjacency_partitions(body_graph)
        hand_graph = full_graph.index_select(1, self.hand_indices[:21]).index_select(2, self.hand_indices[:21])
        hand_graph = normalize_adjacency_partitions(hand_graph)
        self.register_buffer("body_graph", body_graph)
        self.register_buffer("hand_graph", hand_graph)

        self.body_input_norm = PointBatchNorm(3 * len(self.BODY_INDICES))
        self.body_blocks = nn.ModuleList()
        cin = 3
        for index, cout in enumerate(self.BODY_CHANNELS):
            self.body_blocks.append(CTRGCNBlock(
                cin, cout, body_graph, stride=2 if index in (4, 7) else 1,
                residual=index != 0,
            ))
            cin = cout

        self.hand_blocks = nn.ModuleList()
        cin = self.HAND_INPUT_CHANNELS
        for cout in self.HAND_CHANNELS:
            self.hand_blocks.append(CTRGCNBlock(
                cin, cout, hand_graph, stride=1, residual=cin != cout,
            ))
            cin = cout

        # Per-frame five-point head pose encoding: relative x/y plus score.
        self.head_encoder = nn.Sequential(
            nn.Linear(len(self.HEAD_INDICES) * 3, 32),
            nn.ReLU(),
            nn.Linear(32, self.HEAD_OUTPUT_CHANNELS),
        )
        self.hand_projection = nn.Sequential(
            nn.Linear(self.HAND_CHANNELS[-1], self.BODY_CHANNELS[3]),
            nn.LayerNorm(self.BODY_CHANNELS[3]),
        )
        self.head_projection = nn.Sequential(
            nn.Linear(self.HEAD_OUTPUT_CHANNELS, self.BODY_CHANNELS[3]),
            nn.LayerNorm(self.BODY_CHANNELS[3]),
        )
        self.injection_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.body_projection = nn.Sequential(
            nn.Linear(self.BODY_CHANNELS[-1], 192), nn.ReLU(),
        )
        self.classifier = nn.Linear(192, num_classes)

    @property
    def injection_scale(self) -> torch.Tensor:
        return self.injection_logit.sigmoid()

    @staticmethod
    def _masked_pool_nodes(x: torch.Tensor, mask: torch.Tensor):
        weights = mask.to(x.dtype)
        count = weights.sum(-1).squeeze(1)
        features = (x * weights).sum(-1).transpose(1, 2)
        features = features / count.unsqueeze(-1).clamp_min(1)
        return features, count > 0

    def _torso_geometry(self, raw: torch.Tensor, mask: torch.Tensor):
        # raw: BM x 3 x T x 133; mask: BM x 1 x T x 133
        torso_xy = raw[:, :2].index_select(-1, self.torso_indices)
        torso_valid = mask.index_select(-1, self.torso_indices).to(raw.dtype)
        shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
        shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
        hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
        scale = torch.linalg.vector_norm(shoulder - hip, dim=1)
        # ``shoulder_w``/``hip_w`` retain the singleton channel dimension
        # (BM x 1 x T).  Remove it before combining with the (BM x T)
        # torso scale so the reliability mask stays two-dimensional.
        reliable = (
            (shoulder_w.sum(-1) > 0)
            & (hip_w.sum(-1) > 0)
            & (scale[:, None, :] > 1e-3)
        ).squeeze(1)
        return (shoulder + hip) * 0.5, scale.clamp_min(1e-3), reliable

    def _hand_input(self, raw, mask, torso_center, torso_scale):
        left = raw[..., 91:112]
        right = raw[..., 112:133]
        left_mask = mask[..., 91:112]
        right_mask = mask[..., 112:133]
        local_xy = raw[:, :2] - torso_center[:, :, :, None]
        local_xy = local_xy / torso_scale[:, None, :, None].clamp_min(1e-3)
        local = torch.cat((local_xy, raw[:, 2:3]), dim=1)
        pair_valid = left_mask & right_mask
        pair_vector = right[:, :2] - left[:, :2]
        pair_distance = torch.linalg.vector_norm(pair_vector, dim=1) / torso_scale[:, :, None].clamp_min(1e-3)
        pair_distance = pair_distance * pair_valid[:, 0].to(raw.dtype)
        pair_direction = pair_vector / torch.linalg.vector_norm(pair_vector, dim=1, keepdim=True).clamp_min(1e-6)
        pair_direction = pair_direction * pair_valid.to(raw.dtype)
        distance = raw.new_zeros((raw.size(0), 1, raw.size(2), 133))
        direction = raw.new_zeros((raw.size(0), 2, raw.size(2), 133))
        distance[..., 91:112] = pair_distance[:, None]
        distance[..., 112:133] = pair_distance[:, None]
        direction[..., 91:112] = pair_direction
        direction[..., 112:133] = -pair_direction
        source = torch.cat((local, distance, direction), dim=1)
        return source[..., 91:112], source[..., 112:133], left_mask, right_mask

    def _head_features(self, raw, mask, torso_center, torso_scale, torso_reliable):
        head = raw.index_select(-1, self.head_indices)
        head_mask = mask.index_select(-1, self.head_indices)
        xy = (head[:, :2] - torso_center[:, :, :, None]) / torso_scale[:, None, :, None].clamp_min(1e-3)
        values = torch.cat((xy, head[:, 2:3]), dim=1).permute(0, 2, 3, 1)
        values = values.reshape(raw.size(0), raw.size(2), -1)
        encoded = self.head_encoder(values)
        head_quality = (head_mask[:, 0].to(raw.dtype) * head[:, 2].clamp(0, 1)).mean(-1)
        valid = head_mask[:, 0].any(-1) & torso_reliable
        return encoded, head_quality, valid

    def _run_body(self, body, body_mask):
        batch, _, time, nodes = body.shape
        norm_mask = body_mask.expand(-1, 3, -1, -1).permute(0, 1, 3, 2).reshape(batch, 3 * nodes, time, 1)
        body = self.body_input_norm(body.permute(0, 1, 3, 2).reshape(batch, 3 * nodes, time, 1), norm_mask)
        body = body.reshape(batch, 3, nodes, time).permute(0, 1, 3, 2)
        for index, block in enumerate(self.body_blocks):
            body, body_mask = block(body, body_mask)
            if index == self.INJECTION_BLOCK - 1:
                return body, body_mask, (body, body_mask)
        raise RuntimeError("Body backbone did not reach the configured injection block")

    def _run_hand(self, hand, hand_mask):
        for block in self.hand_blocks:
            hand, hand_mask = block(hand, hand_mask)
        return hand, hand_mask

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != 133:
            raise ValueError("Expected B x 3 x T x 133 x M raw RTMW input")
        batch, channels, time, _, people = x.shape
        raw = x
        mask = (raw[:, 2:3] > 0) & torch.isfinite(raw).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask = mask & valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(batch * people, channels, time, 133)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(batch * people, 1, time, 133)
        x = x.masked_fill(~mask, 0)
        torso_center, torso_scale, torso_reliable = self._torso_geometry(x, mask)

        body = x.index_select(-1, self.body_indices)
        body_mask = mask.index_select(-1, self.body_indices)
        _, _, (body_mid, body_mid_mask) = self._run_body(body, body_mask)

        hand_left, hand_right, left_mask, right_mask = self._hand_input(x, mask, torso_center, torso_scale)
        hand_left, left_mask = self._run_hand(hand_left, left_mask)
        hand_right, right_mask = self._run_hand(hand_right, right_mask)
        left_t, left_valid = self._masked_pool_nodes(hand_left, left_mask)
        right_t, right_valid = self._masked_pool_nodes(hand_right, right_mask)
        hand_quality_left = (left_mask[:, 0].to(x.dtype) * x[:, 2, :, 91:112].clamp(0, 1)).mean(-1)
        hand_quality_right = (right_mask[:, 0].to(x.dtype) * x[:, 2, :, 112:133].clamp(0, 1)).mean(-1)
        hand_valid_left = left_valid & torso_reliable
        hand_valid_right = right_valid & torso_reliable
        hand_left = (self.hand_projection(left_t) * hand_quality_left.unsqueeze(-1)
                     * hand_valid_left.unsqueeze(-1).to(x.dtype))
        hand_right = (self.hand_projection(right_t) * hand_quality_right.unsqueeze(-1)
                      * hand_valid_right.unsqueeze(-1).to(x.dtype))

        head_t, head_quality, head_valid = self._head_features(x, mask, torso_center, torso_scale, torso_reliable)
        head_t = self.head_projection(head_t) * head_quality.unsqueeze(-1) * head_valid.unsqueeze(-1).to(x.dtype)

        scale = self.injection_scale
        hand_residual = torch.stack((hand_left, hand_right), dim=-1).permute(0, 2, 1, 3) * scale
        head_residual = head_t.permute(0, 2, 1).unsqueeze(-1) * scale
        # BM x C x T x 2 multiplied by 2 x 19 one-hot wrist map, plus the
        # independent BM x C x T x 1 head descriptor at the nose.
        injected = torch.einsum("bctw,wn->bctn", hand_residual, self.wrist_map)
        injected = injected + torch.einsum("bctw,wn->bctn", head_residual, self.head_map)
        mid_mask = body_mid_mask | torch.stack((hand_valid_left, hand_valid_right), dim=-1)[:, None, :, :].any(-1, keepdim=True).expand_as(body_mid_mask)
        mid_mask = mid_mask | head_valid[:, None, :, None].expand_as(body_mid_mask)
        body_mid = body_mid + injected
        body_mid = body_mid.masked_fill(~mid_mask, 0)
        # Continue the body stack after block 4.
        for block in self.body_blocks[self.INJECTION_BLOCK:]:
            body_mid, mid_mask = block(body_mid, mid_mask)
        body_t, body_valid = self._masked_pool_nodes(body_mid, mid_mask)
        pooled = body_t.reshape(batch, people, body_t.size(1), body_t.size(2)).mean(1).mean(1)
        return self.classifier(self.body_projection(pooled))

