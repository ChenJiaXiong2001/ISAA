"""Radius-limited hand/body interactions with absolute torso context."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.body_local_full_gcn import BodyLocalFullGCNFusion


PROXIMITY_VARIANT = "hand-body-proximity"
REGION_NAMES = ("head_face", "torso", "left_arm", "right_arm",
                "left_leg", "right_leg", "left_foot", "right_foot")
REGION_NODES = (tuple(range(5)) + tuple(range(23, 91)), (5, 6, 11, 12),
                (7, 9), (8, 10), (13, 15), (14, 16), (17, 18, 19), (20, 21, 22))


class TorsoCoordinates(nn.Module):
    """Common reference; nearest valid torso frame fills missing references."""

    def __init__(self, confidence_threshold=0.0):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.register_buffer("torso_indices", torch.tensor((5, 6, 11, 12)))

    def forward(self, raw, frame_mask=None):
        if raw.ndim == 4:
            raw = raw.unsqueeze(-1)
        if raw.ndim != 5 or raw.shape[1] != 3 or raw.shape[3] != 133 or raw.shape[2] < 1:
            raise ValueError("Expected raw B x 3 x T x 133 x M input")
        b, _, t, _, m = raw.shape
        x = raw.permute(0, 4, 2, 3, 1).reshape(b*m, t, 133, 3).float()
        valid = torch.isfinite(x).all(-1) & (x[..., 2] > self.confidence_threshold)
        if frame_mask is not None:
            if frame_mask.shape != (b, t):
                raise ValueError("frame_mask must be B x T")
            valid &= frame_mask.to(device=x.device, dtype=torch.bool)[:, None, :, None].expand(b, m, t, 133).reshape(b*m, t, 133)
        x = torch.nan_to_num(x).masked_fill(~valid[..., None], 0)
        torso = x.index_select(2, self.torso_indices)
        tm = valid.index_select(2, self.torso_indices)
        w = torso[..., 2] * tm
        center = (torso[..., :2] * w[..., None]).sum(2) / w.sum(2).clamp_min(1e-6)[..., None]
        shoulder = (torso[:, :, :2, :2] * w[:, :, :2, None]).sum(2) / w[:, :, :2].sum(2).clamp_min(1e-6)[..., None]
        hip = (torso[:, :, 2:, :2] * w[:, :, 2:, None]).sum(2) / w[:, :, 2:].sum(2).clamp_min(1e-6)[..., None]
        length = torch.linalg.vector_norm(shoulder - hip, dim=-1)
        reference_valid = tm[:, :, :2].any(-1) & tm[:, :, 2:].any(-1) & (length > 1e-6)
        # Reference geometry is preprocessing, not a learned path.
        scale = length.masked_fill(~reference_valid, float("nan")).nanmedian(1).values
        scale = torch.nan_to_num(scale, nan=1.0).clamp_min(1e-6)
        times = torch.arange(t, device=x.device)
        nearest = (times[:, None] - times[None, :]).abs()[None].expand(b*m, -1, -1)
        nearest = nearest.masked_fill(~reference_valid[:, None, :], t+1).argmin(-1)
        center = center.gather(1, nearest[..., None].expand(-1, -1, 2))
        valid &= reference_valid.any(1)[:, None, None]
        q = ((x[..., :2] - center[:, :, None]) / scale[:, None, None, None]).masked_fill(~valid[..., None], 0)
        motion_valid = torch.zeros_like(valid)
        motion_valid[:, 1:] = valid[:, 1:] & valid[:, :-1]
        velocity = torch.zeros_like(q)
        velocity[:, 1:] = (q[:, 1:] - q[:, :-1]).masked_fill(~motion_valid[:, 1:, :, None], 0)
        center_motion = torch.zeros_like(center)
        center_motion[:, 1:] = (center[:, 1:] - center[:, :-1]) / scale[:, None, None]
        center_motion[:, 1:] *= (reference_valid[:, 1:] & reference_valid[:, :-1])[..., None]
        context = torch.cat((torso[..., :2].flatten(2), center_motion,
                             scale[:, None, None].expand(-1, t, 1), tm.float().mean(2, keepdim=True)), -1)
        context = context.masked_fill(~valid.any(2)[..., None], 0)
        return {"relative": q, "velocity": velocity, "valid": valid,
                "motion_valid": motion_valid, "score": x[..., 2].masked_fill(~valid, 0),
                "torso_context": context, "scale": scale, "center": center}


class SparseHandBodyInteraction(nn.Module):
    """Cheap 42x91 distance screening; encode only selected edge rows."""

    def __init__(self, width=64, radius_on=0.4, radius_off=0.48):
        super().__init__()
        self.width, self.radius_on, self.radius_off = width, radius_on, radius_off
        owners = torch.empty(91, dtype=torch.long)
        for r, nodes in enumerate(REGION_NODES):
            owners[list(nodes)] = r
        self.register_buffer("owners", owners)
        self.node_identity = nn.Embedding(133, width)
        self.region_identity = nn.Embedding(len(REGION_NAMES), 8)
        self.node_encoder = nn.Sequential(nn.Linear(6, width), nn.ReLU(), nn.Linear(width, width))
        self.torso_encoder = nn.Sequential(nn.Linear(12, width), nn.LayerNorm(width), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(2*width+19, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.edge_weight = nn.Linear(width, 1)
        self.gate = nn.Linear(2*width+8, 1)
        nn.init.constant_(self.gate.bias, -2.0)
        self.temporal = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(2)])

    @torch.compiler.disable
    def build_edges(self, q, valid):
        # Geometry is shared by both message directions; no all-body pair graph.
        distance = torch.cdist(q[:, :, 91:].float(), q[:, :, :91].float(), compute_mode="donot_use_mm_for_euclid_dist")
        pair_valid = valid[:, :, 91:, None] & valid[:, :, None, :91]
        pair_valid[:, :, 0, 9] = False
        pair_valid[:, :, 21, 10] = False
        state = torch.zeros_like(pair_valid[:, 0])
        frames = []
        for t in range(q.shape[1]):
            state = pair_valid[:, t] & ((distance[:, t] < self.radius_on) | (state & (distance[:, t] <= self.radius_off)))
            frames.append(state)
        return torch.stack(frames, 1).nonzero(as_tuple=False)

    @torch.compiler.disable
    def forward(self, geometry, return_details=False):
        q, valid = geometry["relative"], geometry["valid"]
        n, t, _, _ = q.shape
        regions = len(REGION_NAMES)
        node_input = torch.cat((q, geometry["velocity"], geometry["score"][..., None], geometry["motion_valid"][..., None].float()), -1)
        features = self.node_encoder(node_input) + self.node_identity.weight[None, None]
        features = features.masked_fill(~valid[..., None], 0)
        torso = self.torso_encoder(geometry["torso_context"])
        edges = self.build_edges(q.detach(), valid)
        ni, ti, hi, ji = edges.unbind(1)
        hand_joint = hi + 91
        ri = self.owners[ji]
        delta = q[ni, ti, ji] - q[ni, ti, hand_joint]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        motion = geometry["motion_valid"][ni, ti, ji] & geometry["motion_valid"][ni, ti, hand_joint]
        relative_velocity = geometry["velocity"][ni, ti, ji] - geometry["velocity"][ni, ti, hand_joint]
        relative_velocity = relative_velocity.masked_fill(~motion[:, None], 0)
        prev = (ti-1).clamp_min(0)
        previous_delta = q[ni, prev, ji] - q[ni, prev, hand_joint]
        distance_change = (distance - torch.linalg.vector_norm(previous_delta, dim=-1, keepdim=True)).masked_fill(~motion[:, None], 0)
        edge_input = torch.cat((features[ni, ti, hand_joint], features[ni, ti, ji], delta,
                                distance, delta / distance.clamp_min(1e-6), relative_velocity,
                                distance_change, geometry["score"][ni, ti, hand_joint, None],
                                geometry["score"][ni, ti, ji, None], motion[:, None].float(),
                                self.region_identity(ri)), -1)
        encoded = self.edge_encoder(edge_input)
        weights = self.edge_weight(encoded).sigmoid()
        # Normalize neighbors per hand node and region, then normalize nodes.
        group = ((((ni*t+ti)*2+hi//21)*regions+ri)*21+hi%21)
        group_count = n*t*2*regions*21
        sums = encoded.new_zeros(group_count, self.width).index_add(0, group, encoded*weights)
        totals = weights.new_zeros(group_count, 1).index_add(0, group, weights)
        node_messages = (sums/totals.clamp_min(1e-6)).reshape(n, t, 2, regions, 21, self.width)
        node_active = (totals > 0).reshape(n, t, 2, regions, 21, 1)
        region_active = node_active.any(4).squeeze(-1)
        messages = node_messages.sum(4)/node_active.sum(4).clamp_min(1)
        region_emb = self.region_identity.weight[None, None, None].expand(n, t, 2, -1, -1)
        ctx = torso[:, :, None, None].expand(-1, -1, 2, regions, -1)
        gates = self.gate(torch.cat((messages, ctx, region_emb), -1)).sigmoid().squeeze(-1)
        gates = gates * region_active
        frame_features = (messages*gates[..., None]).sum(3)/region_active.sum(3).clamp_min(1)[..., None]
        active = region_active.any(3)
        y = frame_features.permute(0, 2, 3, 1).reshape(n*2, self.width, t)
        mask = active.permute(0, 2, 1).reshape(n*2, 1, t)
        for layer in self.temporal:
            y = F.relu(layer(y)).masked_fill(~mask, 0)
        pooled = (y.sum(2)/mask.sum(2).clamp_min(1)).reshape(n, 2, self.width)
        if return_details:
            return pooled, {"edges": edges, "edge_distance": distance.squeeze(-1),
                            "edge_weight": weights.squeeze(-1), "region_gate": gates,
                            "active": active, "region_names": REGION_NAMES}
        return pooled


class HandBodyProximityFusion(BodyLocalFullGCNFusion):
    """Full relative-coordinate backbones plus sparse hand/body interaction."""

    ARCHITECTURE = "rtmw_relative_full_gcn_absolute_torso_sparse_hand_body_v1"
    LOCAL_COORDINATE_MODE = "torso_relative_scaled"

    def __init__(self, num_classes=120, *, backbone_config=None, interaction_config=None):
        super().__init__(num_classes, backbone_config=backbone_config, hand_input_channels=3)
        defaults = dict(width=64, radius_on=0.4, radius_off=0.48, confidence_threshold=0.0)
        supplied = dict(interaction_config or {})
        if set(supplied)-set(defaults):
            raise ValueError("Unknown interaction_config keys")
        defaults.update(supplied)
        width = defaults["width"]
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("interaction width must be a positive integer")
        on, off, confidence = (defaults[k] for k in ("radius_on", "radius_off", "confidence_threshold"))
        if not all(math.isfinite(v) for v in (on, off, confidence)) or not 0 < on <= off or not 0 <= confidence <= 1:
            raise ValueError("Require 0 < radius_on <= radius_off and confidence in [0,1]")
        self.interaction_config = defaults
        self.coordinates = TorsoCoordinates(confidence)
        self.interaction = SparseHandBodyInteraction(width, on, off)
        self.interaction_projection = nn.Linear(2*width, 288, bias=False)
        self.torso_projection = nn.Linear(width, 288, bias=False)

    @staticmethod
    def _pool_people_time(features, valid, b, m):
        features = features.reshape(b, m, features.shape[1], -1)
        weight = valid.reshape(b, m, valid.shape[1], 1).to(features.dtype)
        return (features*weight).sum((1, 2))/weight.sum((1, 2)).clamp_min(1)

    def forward(self, x, valid_frame_mask=None, return_interaction=False):
        geometry = self.coordinates(x, valid_frame_mask)
        b, m = x.shape[0], (x.shape[-1] if x.ndim == 5 else 1)
        source = torch.cat((geometry["relative"], geometry["score"][..., None]), -1).permute(0, 3, 1, 2).contiguous()
        mask = geometry["valid"][:, None]
        body, bm = self._run_body(source.index_select(-1, self.body_indices), mask.index_select(-1, self.body_indices))
        body_t, body_valid = self._pool_nodes_per_frame(body, bm)
        body_vector = self.body_projection(self._pool_people_time(body_t, body_valid, b, m))
        local_vectors = []
        for start in (91, 112):
            hand, hm = self._run_hand(source[..., start:start+21], mask[..., start:start+21])
            hand_t, hand_valid = self._pool_nodes_per_frame(self.hand_to_local(hand), hm)
            local_vectors.append(self._pool_people_time(hand_t, hand_valid, b, m))
        n, c, t, _ = source.shape
        fx = source.index_select(-1, self.face_indices.flatten()).reshape(n, c, t, 6, 12)
        fm = mask.index_select(-1, self.face_indices.flatten()).reshape(n, 1, t, 6, 12)
        fm = fm & self.face_members[None, None, None]
        face = (fx*fm).sum(-1)/fm.sum(-1).clamp_min(1)
        face, face_mask = self._run_face(face, fm.any(-1))
        face_t, face_valid = self._pool_nodes_per_frame(face, face_mask)
        local_vectors.append(self._pool_people_time(face_t, face_valid, b, m))
        branch_valid = torch.stack((geometry["valid"][..., 91:112].any(2).any(1),
                                    geometry["valid"][..., 112:].any(2).any(1),
                                    geometry["valid"][..., 23:91].any(2).any(1)), -1).reshape(b, m, 3).any(1)
        local = torch.stack(local_vectors, 1)
        local = (local*branch_valid[..., None]).sum(1)/branch_valid.sum(1).clamp_min(1)[:, None]
        base = torch.cat((body_vector, self.local_projection(local)), -1)
        result = self.interaction(geometry, return_interaction)
        interaction, details = result if return_interaction else (result, None)
        person_active = geometry["valid"].any(2).any(1).reshape(b, m)
        interaction = interaction.reshape(b, m, -1)
        interaction = (interaction*person_active[..., None]).sum(1)/person_active.sum(1).clamp_min(1)[:, None]
        torso_features = self.interaction.torso_encoder(geometry["torso_context"])
        torso_vector = self._pool_people_time(torso_features, geometry["valid"].any(2), b, m)
        fused = base + self.interaction_projection(interaction) + self.torso_projection(torso_vector)
        logits = self.classifier(fused)
        if return_interaction:
            details.update(relative_coordinates=geometry["relative"], node_valid=geometry["valid"],
                           scale=geometry["scale"], batch_size=b, people=m)
            return logits, details
        return logits
