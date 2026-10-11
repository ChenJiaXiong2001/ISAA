"""Best wide-relative backbone with ordered, learned classification readouts."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion


NO_POOL_VARIANT = "body-local-hand-ctr-wide-relative-no-pool"


class OrderedBranchReadout(nn.Module):
    """Project flattened joints, then flattened frames; never average them.

    Bias-free maps keep an entirely missing branch exactly zero. Joint and
    frame order determine separate columns of the learned weight matrices.
    """

    def __init__(self, channels: int, nodes: int, frames: int):
        super().__init__()
        self.channels, self.nodes, self.frames = channels, nodes, frames
        self.node_projection = nn.Linear(nodes * channels, channels, bias=False)
        self.time_projection = nn.Linear(frames * channels, channels, bias=False)

    def forward(self, x, mask):
        if x.shape[1:] != (self.channels, self.frames, self.nodes):
            raise ValueError("Branch shape does not match the configured ordered readout")
        x = x.masked_fill(~mask, 0)
        # N,C,T,V -> N,T,V*C: no node or time reduction before learning.
        frames = F.relu(self.node_projection(x.permute(0, 2, 3, 1).flatten(2)))
        frame_valid = mask[:, 0].any(-1)
        frames = frames.masked_fill(~frame_valid[..., None], 0)
        return F.relu(self.time_projection(frames.flatten(1)))


class BestNoPoolFusion(BodyLocalHandCTRWideRelativeFusion):
    """Keep the best feature encoders, replace all classification pooling.

    Face-region preprocessing, CTR relation statistics, mask propagation and
    temporal convolutions remain those of the best model. The learned head
    uses a fixed input window and ordered person slots, recorded in checkpoints.
    """

    ARCHITECTURE = "rtmw_ctr22_handctr21_wide_crosshanddistdir_face6_ordered_no_pool_v1"

    def __init__(self, num_classes: int = 120, *, window_size: int = 64, max_persons: int = 2):
        if min(num_classes, window_size, max_persons) < 1:
            raise ValueError("num_classes, window_size and max_persons must be positive")
        super().__init__(num_classes=num_classes)
        self.window_size, self.max_persons = window_size, max_persons
        # Inherited local_blocks belong to an unused historical combined path.
        del self.local_blocks
        body_frames = window_size
        for block in self.body_blocks:
            body_frames = (body_frames + block.stride - 1) // block.stride
        body_width, local_width = self.CHANNELS[-1], self.LOCAL_CHANNELS[-1]
        self.body_readout = OrderedBranchReadout(body_width, len(self.BODY_INDICES), body_frames)
        self.hand_readout = OrderedBranchReadout(local_width, 21, window_size)
        self.face_readout = OrderedBranchReadout(local_width, 6, window_size)
        self.body_person_projection = nn.Linear(max_persons * body_width, body_width, bias=False)
        self.local_slot_projection = nn.Linear(max_persons * 3 * local_width, local_width, bias=False)
        self.readout_config = {
            "window_size": window_size, "max_persons": max_persons,
            "body_frames": body_frames, "local_frames": window_size,
            "mode": "ordered_flatten_linear", "local_order": ["left_hand", "right_hand", "face"],
        }

    def _encode_branches(self, x, valid_frame_mask):
        """Same raw/cached features and active encoder paths as the best model."""
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
        # A corrupt auxiliary cache channel must not inject NaN into an encoder.
        x = x.masked_fill(~torch.isfinite(x), 0)
        body, bm = self._run_body(x[:, :3].index_select(-1, self.body_indices),
                                  mask.index_select(-1, self.body_indices))

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

        hand0, hm0 = self._run_hand(hand_features[..., 91:112], mask[..., 91:112])
        hand1, hm1 = self._run_hand(hand_features[..., 112:133], mask[..., 112:133])
        hand0, hand1 = self.hand_to_local(hand0), self.hand_to_local(hand1)
        fx = local_source.index_select(-1, self.face_indices.flatten()).reshape(b * m, 3, t, 6, 12)
        fm = mask.index_select(-1, self.face_indices.flatten()).reshape(b * m, 1, t, 6, 12)
        fm = fm & self.face_members[None, None, None]
        fc = fm.sum(-1)
        face = fm.to(x.dtype).mul(fx).sum(-1) / fc.clamp_min(1)
        face, face_mask = self._run_face(face, fc > 0)
        return (body, bm), (hand0, hm0), (hand1, hm1), (face, face_mask)

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
        body, left, right, face = self._encode_branches(x, valid_frame_mask)
        body_v = self.body_readout(*body).reshape(b, m, -1)
        local_v = torch.cat((self.hand_readout(*left), self.hand_readout(*right),
                             self.face_readout(*face)), dim=-1).reshape(b, m, -1)
        # Missing trailing person slots are zeros, rather than an averaging divisor.
        body_v = F.pad(body_v, (0, 0, 0, self.max_persons - m))
        local_v = F.pad(local_v, (0, 0, 0, self.max_persons - m))
        bv = F.relu(self.body_person_projection(body_v.flatten(1)))
        lv = F.relu(self.local_slot_projection(local_v.flatten(1)))
        return self.classifier(torch.cat((self.body_projection(bv), self.local_projection(lv)), dim=1))
