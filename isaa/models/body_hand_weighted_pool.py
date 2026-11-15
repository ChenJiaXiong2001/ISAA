"""Pooled torso/hand model with learned, mask-aware hand joint weights."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.body_hand_no_pool import BodyHandNoPoolFusion


BODY_HAND_WEIGHTED_POOL_VARIANT = "body-hand-weighted-pool"


class BodyHandWeightedPoolFusion(BodyHandNoPoolFusion):
    """Keep the original person/time means and learn only hand node pooling."""

    ARCHITECTURE = "rtmw_body23_handctr21x2_wide_crosshanddistdir_weighted_hand_pool_v1"

    def __init__(self, num_classes: int = 120, *, window_size: int = 100,
                 max_persons: int = 2):
        super().__init__(num_classes=num_classes, window_size=window_size,
                         max_persons=max_persons)
        for name in ("body_readout", "hand_readout", "body_person_projection",
                     "local_slot_projection", "hand_pair_projection"):
            delattr(self, name)
        del self.readout_config
        self.hand_score = nn.Conv2d(self.LOCAL_CHANNELS[-1], 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.hand_score.weight)
        self.hand_joint_logits = nn.Parameter(torch.zeros(21))
        self.pooling_config = {
            "body_nodes": "masked_mean",
            "hand_nodes": "masked_learned_softmax_mean",
            "hand_weights": "feature_score_plus_joint_logit; initialized_uniform",
            "hands": "fixed_mean",
            "persons": "fixed_mean",
            "time": "fixed_mean",
            "face_points": [0, 1, 2, 3, 4],
            "face_detail_indices_excluded": list(range(23, 91)),
        }

    def _pool_hand(self, x: torch.Tensor, mask: torch.Tensor):
        valid = mask[:, 0]
        scores = self.hand_score(x).squeeze(1) + self.hand_joint_logits
        weights = torch.softmax(scores.masked_fill(~valid, -1e4), dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = (x * weights[:, None]).sum(-1).transpose(1, 2)
        return pooled, valid.any(-1)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] not in (3, 8) or x.shape[3] != 133:
            raise ValueError("Expected B x 3/8 x T x 133 x M RTMW input")
        b, _, t, _, m = x.shape
        if t != self.window_size or not 1 <= m <= self.max_persons:
            raise ValueError("Input window or person count does not match the configured model")

        (body, body_mask), (left, left_mask), (right, right_mask) = self._encode_branches(x, valid_frame_mask)
        body_t, _ = self._pool_nodes_per_frame(body, body_mask)
        left_t, left_valid = self._pool_hand(left, left_mask)
        right_t, right_valid = self._pool_hand(right, right_mask)
        hand_t = (left_t + right_t) * 0.5
        hand_valid = left_valid | right_valid
        target_t = body_t.shape[1]
        if hand_t.shape[1] != target_t:
            hand_t = F.interpolate(hand_t.transpose(1, 2), size=target_t,
                                   mode="linear", align_corners=False).transpose(1, 2)
            hand_valid = F.interpolate(hand_valid[:, None].float(), size=target_t,
                                       mode="nearest")[:, 0].bool()
        hand_t = hand_t * hand_valid[..., None].to(hand_t.dtype)
        body_v = body_t.reshape(b, m, target_t, -1).mean(1).mean(1)
        hand_v = hand_t.reshape(b, m, target_t, -1).mean(1).mean(1)
        return self.classifier(torch.cat((self.body_projection(body_v),
                                          self.local_projection(hand_v)), dim=1))
