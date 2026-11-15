"""Original pooled wide-hand model with a temporal two-hand relation residual."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.best_no_pool import BestNoPoolFusion


BEST_HAND_RELATION_VARIANT = "best-hand-relation-fusion"


class BestHandRelationFusion(BestNoPoolFusion):
    """Preserve the original pooled classifier and add same-person hand relations."""

    ARCHITECTURE = "rtmw_ctr22_handctr21_wide_face6_pooled_hand_relation_v1"
    RELATION_WIDTH = 48

    def __init__(self, num_classes: int = 120, *, window_size: int = 100,
                 max_persons: int = 2):
        if window_size != 100:
            raise ValueError("Hand relation fusion requires the 100-frame baseline")
        super().__init__(num_classes=num_classes, window_size=window_size,
                         max_persons=max_persons)
        for name in ("body_readout", "hand_readout", "face_readout",
                     "body_person_projection", "local_slot_projection"):
            delattr(self, name)
        del self.readout_config
        width = self.RELATION_WIDTH
        # Mean, absolute difference, product, distance, direction, distance
        # change and pair confidence: 3*48 + 1 + 2 + 1 + 1 channels.
        self.pair_projection = nn.Conv2d(3 * width + 5, width, 1, bias=False)
        self.joint_bias = nn.Parameter(torch.zeros(21))
        self.body_query = nn.Linear(self.CHANNELS[-1], width, bias=False)
        self.pair_key = nn.Conv2d(width, width, 1, bias=False)
        self.pair_value = nn.Conv2d(width, width, 1, bias=False)
        self.short_temporal = nn.Conv1d(width, width, 3, padding=1,
                                         groups=width, bias=False)
        self.long_temporal = nn.Conv1d(width, width, 5, dilation=2, padding=4,
                                        groups=width, bias=False)
        self.temporal_mix = nn.Conv1d(2 * width, width, 1, bias=False)
        self.relation_norm = nn.LayerNorm(width)
        self.gate = nn.Sequential(
            nn.Linear(self.CHANNELS[-1] + 2 * width + 1, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -1.0)
        self.residual_scale_logit = nn.Parameter(torch.tensor(math.log(1 / 9)))
        self.relation_config = {
            "baseline": "body-local-hand-ctr-wide-relative",
            "paired_joints": 21,
            "relation_width": width,
            "joint_attention": "body_query_masked_softmax",
            "temporal_kernels": [3, 5],
            "temporal_dilations": [1, 2],
            "fusion": "original_hand_face_mean_plus_gated_relation_residual",
            "original_pooling": "masked_node_mean_fixed_hand_person_time_mean",
            "initial_residual_scale": 0.1,
        }

    @staticmethod
    def _pair_geometry(x: torch.Tensor, frame_mask: torch.Tensor | None):
        b, _, t, _, m = x.shape
        raw = x[:, :3].permute(0, 4, 1, 2, 3).reshape(b * m, 3, t, 133)
        valid = (raw[:, 2:3] > 0) & torch.isfinite(raw).all(1, keepdim=True)
        if frame_mask is not None:
            valid = valid & frame_mask[:, None, None, :, None].expand(b, m, 1, t, 133).reshape(b * m, 1, t, 133)
        raw = raw.masked_fill(~valid, 0)
        xy = raw[:, :2]
        torso_ids = torch.tensor((5, 6, 11, 12), device=x.device)
        torso_xy = xy.index_select(-1, torso_ids)
        torso_valid = valid.index_select(-1, torso_ids).to(x.dtype)
        shoulder = (torso_xy[..., :2] * torso_valid[..., :2]).sum(-1) / torso_valid[..., :2].sum(-1).clamp_min(1)
        hip = (torso_xy[..., 2:] * torso_valid[..., 2:]).sum(-1) / torso_valid[..., 2:].sum(-1).clamp_min(1)
        scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
        delta = xy[..., 112:133] - xy[..., 91:112]
        distance = torch.linalg.vector_norm(delta, dim=1) / scale[..., None]
        direction = delta / torch.linalg.vector_norm(delta, dim=1)[:, None].clamp_min(1e-6)
        pair_valid = valid[..., 91:112] & valid[..., 112:133]
        previous_valid = F.pad(pair_valid[:, 0, :-1], (0, 0, 1, 0))
        change = F.pad(distance[:, 1:] - distance[:, :-1], (0, 0, 1, 0))
        change = change * (pair_valid[:, 0] & previous_valid).to(change.dtype)
        confidence = torch.minimum(raw[:, 2, :, 91:112], raw[:, 2, :, 112:133])
        geometry = torch.cat((distance[:, None], direction, change[:, None],
                              confidence[:, None]), dim=1)
        return geometry.masked_fill(~pair_valid, 0), pair_valid

    def _relation(self, left: torch.Tensor, right: torch.Tensor,
                  pair_valid: torch.Tensor, geometry: torch.Tensor,
                  body_t: torch.Tensor, hand_t: torch.Tensor):
        width = self.RELATION_WIDTH
        pair = torch.cat(((left + right) * 0.5, (left - right).abs(),
                          left * right, geometry), dim=1)
        pair = F.gelu(self.pair_projection(pair))
        query = self.body_query(F.interpolate(body_t.transpose(1, 2),
                                                size=left.shape[2], mode="linear",
                                                align_corners=False).transpose(1, 2))
        scores = (query.transpose(1, 2)[..., None] * self.pair_key(pair)).sum(1) / math.sqrt(width)
        scores = scores + self.joint_bias[None, None]
        valid = pair_valid[:, 0]
        weights = torch.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        relation = (self.pair_value(pair) * weights[:, None]).sum(-1)
        frame_valid = valid.any(-1)
        relation = torch.cat((self.short_temporal(relation),
                              self.long_temporal(relation)), dim=1)
        relation = self.temporal_mix(F.gelu(relation))
        relation = self.relation_norm(relation.transpose(1, 2))
        relation = relation * frame_valid[..., None].to(relation.dtype)
        target_t = body_t.shape[1]
        valid_weight = frame_valid[:, None].to(relation.dtype)
        numerator = F.adaptive_avg_pool1d(relation.transpose(1, 2), target_t)
        denominator = F.adaptive_avg_pool1d(valid_weight, target_t)
        relation = (numerator / denominator.clamp_min(1e-6)).transpose(1, 2)
        quality = F.adaptive_avg_pool1d(valid.float().mean(-1)[:, None], target_t).transpose(1, 2)
        gate = torch.sigmoid(self.gate(torch.cat((body_t, hand_t, relation, quality), dim=-1)))
        relation = relation * (denominator.transpose(1, 2) > 0).to(relation.dtype)
        return relation * gate * torch.sigmoid(self.residual_scale_logit)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None,
                *, interaction_enabled: bool = True):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] not in (3, 8) or x.shape[3] != 133:
            raise ValueError("Expected B x 3/8 x T x 133 x M RTMW input")
        b, _, t, _, m = x.shape
        if t != self.window_size or not 1 <= m <= self.max_persons:
            raise ValueError("Input window or person count does not match the configured model")
        (body, body_mask), (left, left_mask), (right, right_mask), (face, face_mask) = self._encode_branches(x, valid_frame_mask)
        body_t, _ = self._pool_nodes_per_frame(body, body_mask)
        left_t, left_valid = self._pool_nodes_per_frame(left, left_mask)
        right_t, right_valid = self._pool_nodes_per_frame(right, right_mask)
        hand_t = (left_t + right_t) * 0.5
        hand_valid = left_valid | right_valid
        face_t, face_valid = self._pool_nodes_per_frame(face, face_mask)
        target_t = body_t.shape[1]
        if hand_t.shape[1] != target_t:
            hand_t = F.interpolate(hand_t.transpose(1, 2), size=target_t,
                                   mode="linear", align_corners=False).transpose(1, 2)
            hand_valid = F.interpolate(hand_valid[:, None].float(), size=target_t,
                                       mode="nearest")[:, 0].bool()
        if face_t.shape[1] != target_t:
            face_t = F.interpolate(face_t.transpose(1, 2), size=target_t,
                                   mode="linear", align_corners=False).transpose(1, 2)
            face_valid = F.interpolate(face_valid[:, None].float(), size=target_t,
                                       mode="nearest")[:, 0].bool()
        local_t = (hand_t + face_t) * 0.5
        if interaction_enabled:
            geometry, pair_valid = self._pair_geometry(x, valid_frame_mask)
            relation = self._relation(left, right, pair_valid & left_mask & right_mask,
                                      geometry, body_t, hand_t)
            local_t = local_t + relation
        local_t = local_t * (hand_valid | face_valid)[..., None].to(local_t.dtype)
        body_v = body_t.reshape(b, m, target_t, -1).mean(1).mean(1)
        local_v = local_t.reshape(b, m, target_t, -1).mean(1).mean(1)
        return self.classifier(torch.cat((self.body_projection(body_v),
                                          self.local_projection(local_v)), dim=1))
