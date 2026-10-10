"""Body-first, class-dependent hard hand routing with a trained face fallback."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion


CLASS_HAND_VARIANT = "body-local-hand-ctr-wide-relative-class-routed"


class BodyLocalClassHandRoutedFusion(BodyLocalHandCTRWideRelativeFusion):
    ARCHITECTURE = "rtmw_ctr22_handctr21_wide_crosshanddistdir_facest48_class_hand_route_v1"
    AUXILIARY_PREFIXES = ("body_classifier.", "no_hand_classifier.", "no_hand_projection.")

    def __init__(self, num_classes=120, hand_threshold=0.5,
                 confidence_threshold=0.8, body_temperature=1.0):
        super().__init__(num_classes=num_classes)
        if not 0 <= hand_threshold <= 1 or not 0 <= confidence_threshold <= 1:
            raise ValueError("Routing thresholds must be in [0, 1]")
        if not math.isfinite(body_temperature) or body_temperature <= 0:
            raise ValueError("body_temperature must be finite and positive")
        self.body_classifier = nn.Linear(192, num_classes)
        self.no_hand_projection = copy.deepcopy(self.local_projection)
        self.no_hand_classifier = copy.deepcopy(self.classifier)
        self.register_buffer("hand_requirements", torch.ones(num_classes))
        self.register_buffer("requirements_ready", torch.tensor(False))
        self.hand_threshold = float(hand_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.body_temperature = float(body_temperature)
        self.backbone_frozen = False

    def initialize_from_baseline(self, checkpoint):
        """Import every baseline tensor; initialize only the additional heads."""
        if checkpoint.get("architecture") != BodyLocalHandCTRWideRelativeFusion.ARCHITECTURE:
            raise ValueError("Initialization requires the original wide-relative best checkpoint")
        incompatible = self.load_state_dict(checkpoint["model"], strict=False)
        expected = {key for key in self.state_dict()
                    if key.startswith(self.AUXILIARY_PREFIXES)
                    or key in {"hand_requirements", "requirements_ready"}}
        if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
            raise ValueError(f"Baseline checkpoint mismatch: {incompatible}")
        self.no_hand_projection.load_state_dict(self.local_projection.state_dict())
        self.no_hand_classifier.load_state_dict(self.classifier.state_dict())
        with torch.no_grad():
            self.body_classifier.weight.copy_(self.classifier.weight[:, :192])
            self.body_classifier.bias.copy_(self.classifier.bias)

    def freeze_backbone(self, frozen=True):
        self.backbone_frozen = bool(frozen)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not frozen or name.startswith(self.AUXILIARY_PREFIXES))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.backbone_frozen:
            # Freezing weights alone would still mutate the pretrained BN buffers.
            for name, module in self.named_children():
                if not any(prefix.startswith(name + ".") for prefix in self.AUXILIARY_PREFIXES):
                    module.eval()
        return self

    def set_hand_requirements(self, table):
        if int(table["num_classes"]) != self.hand_requirements.numel():
            raise ValueError("Hand requirement class count does not match the model")
        rows = table["classes"]
        ids = [int(row["class_index"]) for row in rows]
        if len(ids) != self.hand_requirements.numel() or set(ids) != set(range(len(ids))):
            raise ValueError("Every zero-based class must occur exactly once")
        values = self.hand_requirements.new_empty(self.hand_requirements.shape)
        for row in rows:
            score = float(row["need_score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Hand requirement scores must be finite and in [0, 1]")
            values[int(row["class_index"])] = score
        self.hand_requirements.copy_(values)
        self.requirements_ready.fill_(True)

    def decide_hands(self, body_logits, available):
        probabilities = (body_logits / self.body_temperature).softmax(-1)
        score = probabilities @ self.hand_requirements
        confidence = probabilities.max(-1).values
        if not bool(self.requirements_ready.item()):
            called = available.clone()
        else:
            called = ((score >= self.hand_threshold)
                      | (confidence < self.confidence_threshold)) & available
        return called, score, confidence

    def _body_features(self, x, mask, b, m, t):
        body = x[:, :3].index_select(-1, self.body_indices)
        bm = mask.index_select(-1, self.body_indices)
        norm_mask = bm.expand(-1, 3, -1, -1).permute(0, 1, 3, 2).reshape(
            b * m, 3 * len(self.BODY_INDICES), t, 1)
        body = self.body_input_norm(body.permute(0, 1, 3, 2).reshape(
            b * m, 3 * len(self.BODY_INDICES), t, 1), norm_mask)
        body = body.reshape(b * m, 3, len(self.BODY_INDICES), t).permute(0, 1, 3, 2)
        for block in self.body_blocks:
            body, bm = block(body, bm)
        body_t, _ = self._pool_nodes_per_frame(body, bm)
        target_t = body_t.shape[1]
        bv = self.body_projection(body_t.reshape(b, m, target_t, -1).mean(1).mean(1))
        return bv, target_t

    @staticmethod
    def _local_source(x, mask):
        if x.shape[1] == 8:
            return torch.cat((x[:, 3:5], x[:, 2:3]), dim=1)
        ids = torch.tensor((5, 6, 11, 12), device=x.device)
        points = x[:, :2].index_select(-1, ids)
        valid = mask.index_select(-1, ids).to(x.dtype)
        center = (points * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1)
        source = x[:, :3].clone()
        source[:, :2] -= center
        return source

    def _face_features(self, source, mask, target_t):
        bm, _, t, _ = source.shape
        ids = self.face_indices.flatten()
        fx = source.index_select(-1, ids).reshape(bm, 3, t, 6, 12)
        fm = mask.index_select(-1, ids).reshape(bm, 1, t, 6, 12)
        fm = fm & self.face_members[None, None, None]
        counts = fm.sum(-1)
        face = (fx * fm.to(fx.dtype)).sum(-1) / counts.clamp_min(1)
        face_mask = counts > 0
        for block in self.face_st_blocks:
            face, face_mask = block(face, face_mask)
        features, valid = self._pool_nodes_per_frame(face, face_mask)
        return self._resize(features, valid, target_t)

    @staticmethod
    def _resize(features, valid, target_t):
        if features.shape[1] != target_t:
            features = F.interpolate(features.transpose(1, 2), size=target_t,
                                     mode="linear", align_corners=False).transpose(1, 2)
            valid = F.interpolate(valid[:, None].float(), size=target_t,
                                  mode="nearest")[:, 0].bool()
        return features, valid

    def _hand_features(self, x, source, mask, target_t):
        # Only selected clips reach this function, including feature construction.
        if x.shape[1] == 8:
            features = torch.cat((x[:, 3:5], x[:, 2:3], x[:, 5:8]), dim=1)
        else:
            ids = torch.tensor((5, 6, 11, 12), device=x.device)
            xy = x[:, :2].index_select(-1, ids)
            valid = mask.index_select(-1, ids).to(x.dtype)
            sw, hw = valid[..., :2], valid[..., 2:]
            shoulder = (xy[..., :2] * sw).sum(-1) / sw.sum(-1).clamp_min(1)
            hip = (xy[..., 2:] * hw).sum(-1) / hw.sum(-1).clamp_min(1)
            scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
            pair_valid = mask[..., 91:112] & mask[..., 112:133]
            vector = source[:, :2, :, 112:133] - source[:, :2, :, 91:112]
            distance = torch.linalg.vector_norm(vector, dim=1)
            scaled = (distance / scale[..., None]) * pair_valid[:, 0].to(x.dtype)
            direction = vector / distance[:, None].clamp_min(1e-6) * pair_valid.to(x.dtype)
            all_distance = x.new_zeros((x.shape[0], 1, x.shape[2], 133))
            all_direction = x.new_zeros((x.shape[0], 2, x.shape[2], 133))
            all_distance[..., 91:112] = scaled[:, None]
            all_distance[..., 112:133] = scaled[:, None]
            all_direction[..., 91:112] = direction
            all_direction[..., 112:133] = -direction
            features = torch.cat((source, all_distance, all_direction), dim=1)
        h0, m0 = self._run_hand(features[..., 91:112], mask[..., 91:112])
        h1, m1 = self._run_hand(features[..., 112:133], mask[..., 112:133])
        h0, v0 = self._pool_nodes_per_frame(self.hand_to_local(h0), m0)
        h1, v1 = self._pool_nodes_per_frame(self.hand_to_local(h1), m1)
        return self._resize((h0 + h1) * 0.5, v0 | v1, target_t)

    @torch._dynamo.disable
    def forward(self, x, valid_frame_mask=None, *, hand_mode="adaptive",
                return_auxiliary=False, return_routing=False):
        if hand_mode not in {"adaptive", "all", "none"}:
            raise ValueError("hand_mode must be adaptive/all/none")
        if return_auxiliary and hand_mode != "all":
            raise ValueError("Auxiliary training requires hand_mode='all'")
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if x.ndim != 5 or x.shape[1] not in {3, 8} or x.shape[3] != 133:
            raise ValueError("Expected B x 3/8 x T x 133 x M input")
        b, c, t, _, m = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x[:, :3]).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask &= valid_frame_mask[:, None, :, None, None].bool()
        available = mask[:, :, :, 91:133].any(dim=(1, 2, 3, 4))
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b * m, 1, t, 133)
        x = x.permute(0, 4, 1, 2, 3).reshape(b * m, c, t, 133).masked_fill(~mask, 0)
        bv, target_t = self._body_features(x, mask, b, m, t)
        body_logits = self.body_classifier(bv)
        called, score, confidence = self.decide_hands(body_logits, available)
        if hand_mode == "all":
            called = torch.ones(b, device=x.device, dtype=torch.bool)
        elif hand_mode == "none":
            called = torch.zeros(b, device=x.device, dtype=torch.bool)
        # The decision is now final, before any hand CTR-GCN is executed.
        source = self._local_source(x, mask)
        face_t, face_valid = self._face_features(source, mask, target_t)
        face_only = 0.5 * face_t * face_valid.unsqueeze(-1).to(face_t.dtype)
        fv = self.no_hand_projection(face_only.reshape(b, m, target_t, -1).mean(1).mean(1))
        no_hand_logits = self.no_hand_classifier(torch.cat((bv, fv), dim=1))
        selected = called.nonzero(as_tuple=False).flatten()
        logits = no_hand_logits.clone()
        if selected.numel():
            person_ids = (selected[:, None] * m + torch.arange(m, device=x.device)).flatten()
            hands, hand_valid = self._hand_features(
                x.index_select(0, person_ids), source.index_select(0, person_ids),
                mask.index_select(0, person_ids), target_t)
            faces = face_t.index_select(0, person_ids)
            valid = hand_valid | face_valid.index_select(0, person_ids)
            local = 0.5 * (hands + faces) * valid.unsqueeze(-1).to(hands.dtype)
            lv = self.local_projection(local.reshape(len(selected), m, target_t, -1).mean(1).mean(1))
            full = self.classifier(torch.cat((bv.index_select(0, selected), lv), dim=1))
            logits = logits.index_copy(0, selected, full)
        if return_auxiliary or return_routing:
            return {"logits": logits, "body_logits": body_logits,
                    "no_hand_logits": no_hand_logits, "hand_called": called,
                    "hand_requirement": score, "body_confidence": confidence}
        return logits
