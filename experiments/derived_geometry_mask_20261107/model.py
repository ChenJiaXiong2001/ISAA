"""Scratch-only face-pair variants that guard anchor-derived geometry.

The production ``FacePairFusion`` path is untouched. These subclasses are
used for a frozen inference screen and, if promoted, a separate scratch run.
"""
from __future__ import annotations

import torch

from isaa.models.face_pair_fusion import FacePairFusion


class MaskAwareDerivedFacePairFusion(FacePairFusion):
    """Guard cross-hand geometry when torso anchors are incomplete.

    ``geometry_only`` keeps the existing torso-relative coordinates and masks
    only cross-hand distance/direction. ``center_scale`` additionally falls
    back to raw coordinates (center zero, scale one) for incomplete torso
    frames, avoiding a partial-anchor reference frame.
    """

    ARCHITECTURE = "rtmw_best_wide_relative_handface_bodyface_mask_aware_derived_v1"
    MASK_AWARE_MODE = "geometry_only"

    def __init__(self, num_classes: int = 60):
        super().__init__(num_classes=num_classes)
        self.model_config.update(
            variant="best-face-pair-mask-aware-derived",
            mask_aware_derived_mode=self.MASK_AWARE_MODE,
            scratch_only=True,
        )

    def forward(self, x, valid_frame_mask=None):
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] not in (3, 8) or x.shape[3] != 133:
            raise ValueError("Expected B x 3/8 x T x 133 x M RTMW input")
        batch, channels, time, _, people = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x[:, :3]).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask = mask & valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(batch * people, channels, time, 133)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(batch * people, 1, time, 133)
        x = x.masked_fill(~mask, 0)

        body, body_mask = self._run_body(
            x[:, :3].index_select(-1, self.body_indices), mask.index_select(-1, self.body_indices))
        torso_ids = torch.tensor((5, 6, 11, 12), device=x.device)
        if channels == 8:
            local_source = torch.cat((x[:, 3:5], x[:, 2:3]), dim=1)
            hand_features = torch.cat((x[:, 3:5], x[:, 2:3], x[:, 5:8]), dim=1)
        else:
            torso_xy = x[:, :2].index_select(-1, torso_ids)
            torso_valid = mask.index_select(-1, torso_ids).to(x.dtype)
            torso_complete = torso_valid.sum(-1) >= 4
            torso_center = (torso_xy * torso_valid).sum(-1, keepdim=True) / torso_valid.sum(-1, keepdim=True).clamp_min(1)
            if self.MASK_AWARE_MODE == "center_scale":
                torso_center = torch.where(torso_complete.unsqueeze(-1), torso_center, torch.zeros_like(torso_center))
            local_source = torch.cat((x[:, :2] - torso_center, x[:, 2:3]), dim=1)
            shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
            shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
            hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
            torso_scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
            if self.MASK_AWARE_MODE == "center_scale":
                torso_scale = torch.where(torso_complete[:, 0], torso_scale, torch.ones_like(torso_scale))
            pair_valid = mask[..., 91:112] & mask[..., 112:133]
            if self.MASK_AWARE_MODE in ("geometry_only", "center_scale"):
                pair_valid = pair_valid & torso_complete[..., None]
            pair_vector = local_source[:, :2, :, 112:133] - local_source[:, :2, :, 91:112]
            pair_distance = torch.linalg.vector_norm(pair_vector, dim=1)
            distance = (pair_distance / torso_scale[..., None]) * pair_valid[:, 0].to(x.dtype)
            direction = pair_vector / pair_distance[:, None].clamp_min(1e-6) * pair_valid.to(x.dtype)
            distance_all = x.new_zeros((batch * people, 1, time, 133))
            direction_all = x.new_zeros((batch * people, 2, time, 133))
            distance_all[..., 91:112] = distance[:, None]
            distance_all[..., 112:133] = distance[:, None]
            direction_all[..., 91:112] = direction
            direction_all[..., 112:133] = -direction
            hand_features = torch.cat((local_source, distance_all, direction_all), dim=1)

        if self.HAND_FACE_COORDINATE_MODE == "torso_relative_plus_raw_xy":
            hand_features = torch.cat((hand_features, x[:, :2]), dim=1)
        left, left_mask = self._run_hand(hand_features[..., 91:112], mask[..., 91:112])
        right, right_mask = self._run_hand(hand_features[..., 112:133], mask[..., 112:133])
        left_t, left_valid = self._pool_nodes_per_frame(self.hand_to_local(left), left_mask)
        right_t, right_valid = self._pool_nodes_per_frame(self.hand_to_local(right), right_mask)
        hand_t, hand_valid = (left_t + right_t) * 0.5, left_valid | right_valid
        face_source = local_source
        if self.HAND_FACE_COORDINATE_MODE == "torso_relative_plus_raw_xy":
            face_source = torch.cat((local_source, x[:, :2]), dim=1)
        face_input = face_source.index_select(-1, self.face_indices.flatten()).reshape(
            batch * people, face_source.shape[1], time, 6, 12)
        face_mask = mask.index_select(-1, self.face_indices.flatten()).reshape(
            batch * people, 1, time, 6, 12) & self.face_members[None, None, None]
        face_count = face_mask.sum(-1)
        face_input = (face_input * face_mask.to(x.dtype)).sum(-1) / face_count.clamp_min(1)
        face, face_mask = self._run_face(face_input, face_count > 0)
        body_t, body_valid = self._pool_nodes_per_frame(body, body_mask)
        face_t, face_valid = self._pool_nodes_per_frame(face, face_mask)
        target_time = body_t.shape[1]
        hand_t, hand_valid = self._align_time(hand_t, hand_valid, target_time)
        face_t, face_valid = self._align_time(face_t, face_valid, target_time)
        body_feature = self.body_projection(self._pool_people_time(body_t, body_valid, batch, people))
        hand_feature = self._pool_people_time(hand_t, hand_valid, batch, people)
        face_feature = self._pool_people_time(face_t, face_valid, batch, people)
        hf = self.hand_face_head(torch.cat((hand_feature, face_feature), dim=-1))
        bf = self.body_face_head(torch.cat((body_feature, face_feature), dim=-1))
        return {"logits": self.fuse_logits(hf, bf), "hand_face_logits": hf, "body_face_logits": bf}


class MaskAwareGeometryOnlyFusion(MaskAwareDerivedFacePairFusion):
    MASK_AWARE_MODE = "geometry_only"
    ARCHITECTURE = "rtmw_best_wide_relative_handface_bodyface_mask_aware_geometry_v1"


class MaskAwareCenterScaleFusion(MaskAwareDerivedFacePairFusion):
    MASK_AWARE_MODE = "center_scale"
    ARCHITECTURE = "rtmw_best_wide_relative_handface_bodyface_mask_aware_center_scale_v1"
