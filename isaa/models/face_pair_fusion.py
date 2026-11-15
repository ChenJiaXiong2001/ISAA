"""Independent hand/face and body/face classifiers with classwise routing.

The existing wide-relative body, shared hand and face encoders are retained.
Only this new variant changes the readout: each pair is classified separately,
then a training-internal holdout determines which pair supplies each class.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.body_local_fusion import LocalSTBlock


FACE_PAIR_VARIANT = "best-face-pair"
FACE_PAIR_RAW_XY_VARIANT = "best-face-pair-raw-xy"
FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT = "best-face-pair-hand-face-geometry"
FACE_PAIR_HAND_MOTION_VARIANT = "best-face-pair-hand-motion"
VARIANT = FACE_PAIR_VARIANT


class FacePairFusion(BodyLocalHandCTRWideRelativeFusion):
    """Share the baseline encoders, classify HF/BF pairs, and route by class."""

    ARCHITECTURE = "rtmw_best_wide_relative_handface_bodyface_classwise_v1"
    HAND_FACE_COORDINATE_MODE = "torso_relative_xy"

    def __init__(self, num_classes: int = 60):
        super().__init__(num_classes=num_classes)
        self.num_classes = num_classes
        self.model_config = {
            "variant": FACE_PAIR_VARIANT, "num_classes": num_classes,
            "hand_face_feature_width": 96, "body_face_feature_width": 240,
            "feature_fusion": "concatenation", "classwise_selection": "heldout_accuracy",
            "tie_policy": "equal_probability_average", "scratch_only": True,
        }
        # These modules belong to the baseline's old combined readout. Removing
        # them here affects this subclass only and avoids inactive parameters.
        del self.local_blocks
        del self.local_projection
        del self.classifier
        self.hand_face_head = nn.Linear(96, num_classes)
        self.body_face_head = nn.Linear(240, num_classes)
        self.register_buffer("hand_face_weight", torch.full((num_classes,), 0.5))
        self.register_buffer("hand_face_temperature", torch.tensor(1.0))
        self.register_buffer("body_face_temperature", torch.tensor(1.0))
        self.register_buffer("routing_class_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("routing_hand_face_correct", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("routing_body_face_correct", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def set_routing_counts(self, hand_face_correct, body_face_correct, class_counts):
        """Choose the more accurate head per class; ties/unseen classes average.

        All three count arrays must come from the same internal calibration
        split. They are saved as buffers with the inference routing table.
        """
        arrays = [torch.as_tensor(value, device=self.hand_face_weight.device)
                  for value in (hand_face_correct, body_face_correct, class_counts)]
        if any(value.shape != (self.num_classes,) for value in arrays):
            raise ValueError("Routing counts must each have num_classes entries")
        if any(not bool(torch.isfinite(value).all()) or bool((value < 0).any())
               or bool((value != value.floor()).any()) for value in arrays):
            raise ValueError("Routing counts must be finite nonnegative integers")
        hf, bf, total = arrays
        if bool((hf > total).any()) or bool((bf > total).any()):
            raise ValueError("Correct predictions cannot exceed class counts")
        weights = torch.where(hf > bf, 1.0, torch.where(hf < bf, 0.0, 0.5))
        self.hand_face_weight.copy_(weights)
        self.routing_class_counts.copy_(total)
        self.routing_hand_face_correct.copy_(hf)
        self.routing_body_face_correct.copy_(bf)

    @torch.no_grad()
    def set_temperatures(self, hand_face: float, body_face: float):
        temperatures = torch.tensor([hand_face, body_face], dtype=torch.float64)
        if not bool(torch.isfinite(temperatures).all()) or bool((temperatures <= 0).any()):
            raise ValueError("Classifier temperatures must be finite and positive")
        self.hand_face_temperature.fill_(hand_face)
        self.body_face_temperature.fill_(body_face)

    def fuse_logits(self, hand_face_logits, body_face_logits):
        """Select comparable per-class probabilities, then normalize globally.

        No ground-truth label is consumed here. Each possible class already
        has a fixed head choice in ``hand_face_weight``. Returning log
        probabilities permits the usual cross-entropy and argmax operations.
        """
        # Keep selected classes in log space: softmax followed by log loses
        # very unlikely classes to underflow and clamp then removes gradients.
        hf = F.log_softmax(hand_face_logits.float() / self.hand_face_temperature, dim=-1)
        bf = F.log_softmax(body_face_logits.float() / self.body_face_temperature, dim=-1)
        average = torch.logaddexp(hf, bf) - math.log(2.0)
        selected = torch.where(self.hand_face_weight == 1, hf,
                               torch.where(self.hand_face_weight == 0, bf, average))
        # Hard routing can discard both heads' high-probability classes. Shift
        # before the final reduction so a large negative common offset cannot
        # destroy normalization through float32 cancellation.
        selected = selected - selected.amax(dim=-1, keepdim=True)
        return selected - torch.logsumexp(selected, dim=-1, keepdim=True)

    @staticmethod
    def _pool_people_time(feature, valid, batch, people):
        time, width = feature.shape[1:]
        feature = feature.reshape(batch, people, time, width)
        weights = valid.reshape(batch, people, time, 1).to(feature.dtype)
        return (feature * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)

    @staticmethod
    def _align_time(feature, valid, target_time):
        if feature.shape[1] != target_time:
            feature = F.interpolate(feature.transpose(1, 2), size=target_time,
                                    mode="linear", align_corners=False).transpose(1, 2)
            valid = F.interpolate(valid[:, None].float(), size=target_time,
                                  mode="nearest")[:, 0].bool()
        return feature, valid

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
            torso_center = (torso_xy * torso_valid).sum(-1, keepdim=True) / torso_valid.sum(-1, keepdim=True).clamp_min(1)
            local_source = torch.cat((x[:, :2] - torso_center, x[:, 2:3]), dim=1)
            shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
            shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
            hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
            torso_scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
            pair_valid = mask[..., 91:112] & mask[..., 112:133]
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
        return {"logits": self.fuse_logits(hf, bf),
                "hand_face_logits": hf, "body_face_logits": bf}


class CompiledRawXYFaceBlock(LocalSTBlock):
    # This expanded input block is new neural computation and must participate
    # in Inductor. Subsequent inherited face blocks keep their existing boundary.
    forward = LocalSTBlock.forward.__wrapped__


class FacePairRawXYFusion(FacePairFusion):
    """Face-pair classifier retaining relative XY and appending source XY."""

    ARCHITECTURE = "rtmw_best_relplusrawxy_handface_bodyface_classwise_v1"
    HAND_FACE_COORDINATE_MODE = "torso_relative_plus_raw_xy"
    HAND_INPUT_CHANNELS = 8

    def __init__(self, num_classes: int = 60):
        super().__init__(num_classes=num_classes)
        first = self.face_st_blocks[0]
        self.face_st_blocks[0] = CompiledRawXYFaceBlock(5, self.LOCAL_CHANNELS[0], first.adjacency, stride=1)
        self.model_config.update(
            variant=FACE_PAIR_RAW_XY_VARIANT,
            hand_face_coordinate_mode=self.HAND_FACE_COORDINATE_MODE,
            hand_input_channels=["relative_x", "relative_y", "score", "cross_hand_distance",
                                 "cross_hand_direction_x", "cross_hand_direction_y", "raw_x", "raw_y"],
            face_input_channels=["relative_x", "relative_y", "score", "raw_x", "raw_y"],
        )


class FacePairHandFaceGeometryFusion(FacePairFusion):
    """Face-pair model with explicit hand-to-face geometry and hand motion.

    The body/face classifier and the gamma=1 routing contract are inherited
    unchanged.  Only the hand encoder receives six extra per-frame channels:
    face-to-hand direction (2), normalized distance (1), face validity (1),
    and first temporal differences of direction (2).
    """

    ARCHITECTURE = "rtmw_best_hand_face_geometry_temporal_bodyface_v1"
    HAND_FACE_COORDINATE_MODE = "torso_relative_xy_hand_face_geometry_temporal"
    HAND_INPUT_CHANNELS = 12

    def __init__(self, num_classes: int = 60):
        super().__init__(num_classes=num_classes)
        self.model_config.update(
            variant=FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT,
            hand_face_coordinate_mode=self.HAND_FACE_COORDINATE_MODE,
            hand_input_channels=[
                "relative_x", "relative_y", "score", "cross_hand_distance",
                "cross_hand_direction_x", "cross_hand_direction_y",
                "face_to_hand_direction_x", "face_to_hand_direction_y",
                "face_to_hand_distance", "face_valid",
                "face_to_hand_direction_delta_x", "face_to_hand_direction_delta_y",
            ],
            face_input_channels=["relative_x", "relative_y", "score"],
        )

    @staticmethod
    def _torso_scale(x, mask):
        torso_ids = torch.tensor((5, 6, 11, 12), device=x.device)
        torso_xy = x[:, :2].index_select(-1, torso_ids)
        torso_valid = mask.index_select(-1, torso_ids).to(x.dtype)
        shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
        shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
        hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
        return torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)

    @staticmethod
    def _face_token_centroids(local_source, mask, face_indices, face_members):
        fidx = face_indices.flatten().to(local_source.device)
        face_points = local_source.index_select(-1, fidx).reshape(
            local_source.shape[0], local_source.shape[1], local_source.shape[2], 6, 12)
        face_mask = mask.index_select(-1, fidx).reshape(
            mask.shape[0], 1, mask.shape[2], 6, 12)
        face_mask = face_mask & face_members.to(local_source.device)[None, None, None]
        weights = face_mask.to(local_source.dtype)
        count = weights.sum(-1)
        centroid = (face_points[:, :2] * weights).sum(-1) / count.clamp_min(1)
        return centroid, count > 0

    def _hand_face_geometry(self, x, mask, local_source):
        token_centroid, token_valid = self._face_token_centroids(
            local_source, mask, self.face_indices, self.face_members)
        hand_xy = local_source[:, :2, :, 91:133]
        relative = hand_xy.unsqueeze(-1) - token_centroid.unsqueeze(3)
        token_distance = torch.linalg.vector_norm(relative, dim=1)
        token_mask = token_valid[:, 0].unsqueeze(2)
        token_distance = torch.where(token_mask, token_distance,
                                     torch.full_like(token_distance, float("inf")))
        nearest = token_distance.argmin(-1)
        gather_index = nearest[:, None, :, :, None].expand(-1, 2, -1, -1, 1)
        relative = torch.gather(relative, -1, gather_index).squeeze(-1)
        raw_distance = torch.linalg.vector_norm(relative, dim=1).clamp_min(1e-6)
        direction = relative / raw_distance[:, None]
        distance = raw_distance / self._torso_scale(x, mask)[:, :, None]
        face_valid = token_valid.any(-1)
        valid = face_valid.unsqueeze(-1).expand(-1, 1, -1, 42).to(x.dtype)
        direction = direction * valid
        distance = distance * valid[:, 0]
        delta = torch.cat((torch.zeros_like(direction[:, :, :1]),
                           direction[:, :, 1:] - direction[:, :, :-1]), dim=2)
        transition = torch.cat((torch.zeros_like(valid[:, :, :1]),
                                valid[:, :, 1:] * valid[:, :, :-1]), dim=2)
        delta = delta * transition.to(x.dtype)
        return torch.cat((direction, distance[:, None], valid, delta), dim=1)

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
            base_hand_features = torch.cat((x[:, 3:5], x[:, 2:3], x[:, 5:8]), dim=1)
        else:
            torso_xy = x[:, :2].index_select(-1, torso_ids)
            torso_valid = mask.index_select(-1, torso_ids).to(x.dtype)
            torso_center = (torso_xy * torso_valid).sum(-1, keepdim=True) / torso_valid.sum(-1, keepdim=True).clamp_min(1)
            local_source = torch.cat((x[:, :2] - torso_center, x[:, 2:3]), dim=1)
            shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
            shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
            hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
            torso_scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
            pair_valid = mask[..., 91:112] & mask[..., 112:133]
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
            base_hand_features = torch.cat((local_source, distance_all, direction_all), dim=1)
        geometry = self._hand_face_geometry(x, mask, local_source)
        geometry_all = x.new_zeros((batch * people, geometry.shape[1], time, 133))
        geometry_all[..., 91:133] = geometry
        hand_features = torch.cat((base_hand_features, geometry_all), dim=1)

        left, left_mask = self._run_hand(hand_features[..., 91:112], mask[..., 91:112])
        right, right_mask = self._run_hand(hand_features[..., 112:133], mask[..., 112:133])
        left_t, left_valid = self._pool_nodes_per_frame(self.hand_to_local(left), left_mask)
        right_t, right_valid = self._pool_nodes_per_frame(self.hand_to_local(right), right_mask)
        hand_t, hand_valid = (left_t + right_t) * 0.5, left_valid | right_valid

        face_source = local_source
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
        return {"logits": self.fuse_logits(hf, bf),
                "hand_face_logits": hf, "body_face_logits": bf}


class FacePairHandMotionFusion(FacePairFusion):
    """Add hand-joint first differences to HF only; all other paths are baseline."""

    ARCHITECTURE = "rtmw_best_hand_motion_only_bodyface_v1"
    HAND_FACE_COORDINATE_MODE = "torso_relative_xy_hand_motion"
    HAND_INPUT_CHANNELS = 8

    def __init__(self, num_classes: int = 60):
        super().__init__(num_classes=num_classes)
        self.model_config.update(
            variant=FACE_PAIR_HAND_MOTION_VARIANT,
            hand_face_coordinate_mode=self.HAND_FACE_COORDINATE_MODE,
            hand_input_channels=["relative_x", "relative_y", "score", "cross_hand_distance",
                                 "cross_hand_direction_x", "cross_hand_direction_y",
                                 "delta_x", "delta_y"],
            face_input_channels=["relative_x", "relative_y", "score"],
        )

    @staticmethod
    def _hand_motion(local_source, mask):
        hand = local_source[..., 91:133]
        valid = mask[..., 91:133]
        delta = torch.cat((torch.zeros_like(hand[:, :2, :1]),
                           hand[:, :2, 1:] - hand[:, :2, :-1]), dim=2)
        transition = torch.cat((torch.zeros_like(valid[:, :, :1]),
                                valid[:, :, 1:] & valid[:, :, :-1]), dim=2)
        delta = delta * transition.to(delta.dtype)
        return delta

    def forward(self, x, valid_frame_mask=None):
        # Preserve the baseline path and only extend hand encoder features.
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
            base_hand_features = torch.cat((x[:, 3:5], x[:, 2:3], x[:, 5:8]), dim=1)
        else:
            torso_xy = x[:, :2].index_select(-1, torso_ids)
            torso_valid = mask.index_select(-1, torso_ids).to(x.dtype)
            torso_center = (torso_xy * torso_valid).sum(-1, keepdim=True) / torso_valid.sum(-1, keepdim=True).clamp_min(1)
            local_source = torch.cat((x[:, :2] - torso_center, x[:, 2:3]), dim=1)
            shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
            shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
            hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
            torso_scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
            pair_valid = mask[..., 91:112] & mask[..., 112:133]
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
            base_hand_features = torch.cat((local_source, distance_all, direction_all), dim=1)
        motion = self._hand_motion(local_source, mask)
        motion_all = x.new_zeros((batch * people, motion.shape[1], time, 133))
        motion_all[..., 91:133] = motion
        hand_features = torch.cat((base_hand_features, motion_all), dim=1)
        left, left_mask = self._run_hand(hand_features[..., 91:112], mask[..., 91:112])
        right, right_mask = self._run_hand(hand_features[..., 112:133], mask[..., 112:133])
        left_t, left_valid = self._pool_nodes_per_frame(self.hand_to_local(left), left_mask)
        right_t, right_valid = self._pool_nodes_per_frame(self.hand_to_local(right), right_mask)
        hand_t, hand_valid = (left_t + right_t) * 0.5, left_valid | right_valid
        face_input = local_source.index_select(-1, self.face_indices.flatten()).reshape(
            batch * people, local_source.shape[1], time, 6, 12)
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
        return {"logits": self.fuse_logits(hf, bf),
                "hand_face_logits": hf, "body_face_logits": bf}
