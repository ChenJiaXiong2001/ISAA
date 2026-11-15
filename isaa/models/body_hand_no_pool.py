"""Two-branch torso/hand model with ordered, learned readouts.

The torso branch owns RTMW nodes 0..22, including only the five coarse face
points (nose, eyes and ears).  The hand branch encodes both 21-joint hands
with the existing six-channel torso-relative features.  Classification uses
learned ordered readouts instead of masked means over nodes, people or time.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from isaa.models.best_no_pool import BestNoPoolFusion, OrderedBranchReadout


BODY_HAND_NO_POOL_VARIANT = "body-hand-no-pool"


class BodyHandNoPoolFusion(BestNoPoolFusion):
    """Scratch-trained torso/hand-only variant of the best wide backbone."""

    ARCHITECTURE = "rtmw_body23_handctr21x2_wide_crosshanddistdir_ordered_no_pool_v1"
    BODY_INDICES = tuple(range(23))

    def __init__(self, num_classes: int = 120, *, window_size: int = 100,
                 max_persons: int = 2):
        super().__init__(num_classes=num_classes, window_size=window_size,
                         max_persons=max_persons)
        local_width = self.LOCAL_CHANNELS[-1]
        # There is one logical hand branch: left and right ordered readouts are
        # concatenated and mixed by a bias-free projection, preserving zeros for
        # a missing hand/person slot.
        self.hand_pair_projection = torch.nn.Linear(2 * local_width, local_width, bias=False)
        self.local_slot_projection = torch.nn.Linear(
            max_persons * local_width, local_width, bias=False
        )
        self.hand_readout = OrderedBranchReadout(local_width, 21, window_size)
        self.readout_config = {
            "window_size": window_size,
            "max_persons": max_persons,
            "body_frames": self.body_readout.frames,
            "local_frames": window_size,
            "mode": "ordered_flatten_linear",
            "branches": ["torso_body_0_22", "hands_left_right_combined"],
            "face_points": [0, 1, 2, 3, 4],
            "classification_pooling": "none",
        }
        # The parent builds the old face path for weight-compatible subclasses;
        # this variant has no face encoder/readout or six-token buffers.
        for name in ("face_st_blocks", "face_readout"):
            if hasattr(self, name):
                delattr(self, name)
        for name in ("face_original_to_token", "face_indices", "face_members"):
            self._buffers.pop(name, None)

    def _encode_branches(self, x, valid_frame_mask):
        b, c, t, _, m = x.shape
        raw = x[:, :3]
        mask = (raw[:, 2:3] > 0) & torch.isfinite(raw).all(1, keepdim=True)
        if valid_frame_mask is not None:
            if valid_frame_mask.shape != (b, t):
                raise ValueError("valid_frame_mask must have shape B x T")
            mask &= valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(b * m, c, t, 133)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b * m, 1, t, 133)
        x = x.masked_fill(~mask, 0)
        x = x.masked_fill(~torch.isfinite(x), 0)

        body, bm = self._run_body(
            x[:, :3].index_select(-1, self.body_indices),
            mask.index_select(-1, self.body_indices),
        )

        if c == 8:
            local_source = torch.cat((x[:, 3:5], x[:, 2:3]), dim=1)
            hand_features = torch.cat((x[:, 3:5], x[:, 2:3], x[:, 5:8]), dim=1)
        else:
            torso_ids = torch.tensor((5, 6, 11, 12), device=x.device)
            torso_xy = x[:, :2].index_select(-1, torso_ids)
            torso_valid = mask.index_select(-1, torso_ids).to(x.dtype)
            center = (torso_xy * torso_valid).sum(-1, keepdim=True) / torso_valid.sum(-1, keepdim=True).clamp_min(1)
            local_source = x[:, :3].clone()
            local_source[:, :2] = local_source[:, :2] - center
            shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
            shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
            hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
            scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
            pair_valid = mask[..., 91:112] & mask[..., 112:133]
            vector = local_source[:, :2, :, 112:133] - local_source[:, :2, :, 91:112]
            distance = torch.linalg.vector_norm(vector, dim=1)
            normalized = (distance / scale[..., None]) * pair_valid[:, 0].to(x.dtype)
            direction = vector / distance[:, None].clamp_min(1e-6) * pair_valid.to(x.dtype)
            distance_all = x.new_zeros((b * m, 1, t, 133))
            direction_all = x.new_zeros((b * m, 2, t, 133))
            distance_all[..., 91:112] = normalized[:, None]
            distance_all[..., 112:133] = normalized[:, None]
            direction_all[..., 91:112] = direction
            direction_all[..., 112:133] = -direction
            hand_features = torch.cat((local_source, distance_all, direction_all), dim=1)

        left, lm = self._run_hand(hand_features[..., 91:112], mask[..., 91:112])
        right, rm = self._run_hand(hand_features[..., 112:133], mask[..., 112:133])
        return (body, bm), (self.hand_to_local(left), lm), (self.hand_to_local(right), rm)

    def forward(self, x, valid_frame_mask=None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] not in (3, 8) or x.shape[3] != 133:
            raise ValueError("Expected B x 3/8 x T x 133 x M RTMW input")
        b, _, t, _, m = x.shape
        if t != self.window_size:
            raise ValueError(f"Expected {self.window_size} frames, received {t}; crop/pad in the data loader")
        if not 1 <= m <= self.max_persons:
            raise ValueError(f"Expected 1..{self.max_persons} person slots, received {m}")
        body, left, right = self._encode_branches(x, valid_frame_mask)
        body_v = self.body_readout(*body).reshape(b, m, -1)
        left_v = self.hand_readout(*left)
        right_v = self.hand_readout(*right)
        hand_v = self.hand_pair_projection(torch.cat((left_v, right_v), dim=-1)).reshape(b, m, -1)
        body_v = F.pad(body_v, (0, 0, 0, self.max_persons - m))
        hand_v = F.pad(hand_v, (0, 0, 0, self.max_persons - m))
        torso = F.relu(self.body_person_projection(body_v.flatten(1)))
        hands = F.relu(self.local_slot_projection(hand_v.flatten(1)))
        return self.classifier(torch.cat((self.body_projection(torso), self.local_projection(hands)), dim=1))

