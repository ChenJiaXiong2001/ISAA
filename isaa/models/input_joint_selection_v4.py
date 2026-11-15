"""Input-conditioned, per-sequence stochastic joint selection."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.input_joint_selection import OPTIONAL_BODY
from isaa.models.input_joint_selection import InputJointSelectionFusion


INPUT_SELECTION_V4_VARIANT = "best-input-joint-selection-v4-dynamic"


class ConditionalJointGates(nn.Module):
    """Predict separate gates for each input clip; people share its selection."""

    def __init__(self, *, temperature=.67, initial_logit=.5):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Gate temperature must be finite and positive")
        self.temperature = float(temperature)
        self.node_embedding = nn.Parameter(torch.empty(len(OPTIONAL_BODY) + 20 + 6, 8))
        nn.init.normal_(self.node_embedding, std=.02)
        self.router = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, initial_logit)
        self.register_buffer("open_mix", torch.ones(()))
        self.register_buffer("noise_scale", torch.zeros(()))
        self.register_buffer("hard_train", torch.zeros((), dtype=torch.bool))
        self.register_buffer("hard_eval", torch.zeros((), dtype=torch.bool))
        self.register_buffer("threshold", torch.tensor(.5))
        self.register_buffer("cost", torch.tensor([1.] * len(OPTIONAL_BODY) + [2.] * 20 + [1.] * 6))

    def begin_epoch(self, epoch, warmup_epochs, noise_ramp_epochs):
        progress = min(1., max(0., (epoch - warmup_epochs) / noise_ramp_epochs))
        self.open_mix.fill_(1. - progress)
        self.noise_scale.fill_(progress)
        self.hard_eval.zero_()
        self.hard_train.fill_(progress >= 1.)

    def forward(self, descriptors, global_context):
        context = global_context[:, None, :].expand_as(descriptors)
        embedding = self.node_embedding[None].expand(descriptors.shape[0], -1, -1)
        logits = self.router(torch.cat((descriptors, context, embedding), dim=-1)).squeeze(-1)
        deterministic = torch.sigmoid(logits / self.temperature)
        deterministic = torch.lerp(deterministic, torch.ones_like(deterministic), self.open_mix)
        hard = (deterministic > self.threshold).to(logits.dtype)
        if self.training:
            uniform = torch.rand_like(logits).clamp(1e-6, 1. - 1e-6)
            noise = torch.log(uniform) - torch.log1p(-uniform)
            relaxed = torch.sigmoid((logits + noise * self.noise_scale) / self.temperature)
            soft = relaxed
            straight_through = (soft > self.threshold).to(soft.dtype) - soft.detach() + soft
            gates = torch.where(self.hard_train, straight_through, soft)
            gates = torch.lerp(gates, torch.ones_like(gates), self.open_mix)
        else:
            gates = torch.where(self.hard_eval, hard, deterministic)
        # Probability that the full-noise binary forward opens a gate. Match
        # the deployed threshold, including nondefault thresholds.
        cutoff = self.threshold.clamp(1e-6, 1. - 1e-6)
        expected_open = torch.sigmoid(logits - self.temperature * torch.logit(cutoff))
        expected_count = (expected_open * self.cost).sum(-1)
        return gates, expected_count.mean(), dict(gate_values=deterministic, sampled_gate_values=gates,
            expected_optional_counts=expected_count, gate_logits=logits)


class InputJointSelectionV4Fusion(InputJointSelectionFusion):
    ARCHITECTURE = "rtmw_original_pool_conditional_stg_input_subset_v4"

    def __init__(self, num_classes=60, *, gate_temperature=.67):
        super().__init__(num_classes, stage="select")
        self.gates = ConditionalJointGates(temperature=gate_temperature)
        positions = {joint: i for i, joint in enumerate(self.selection["input_indices"])}
        candidates = [[positions[j], positions[j]] for j in OPTIONAL_BODY]
        candidates.extend([[positions[91 + j], positions[112 + j]] for j in range(1, 21)])
        # Parent has already mapped original RTMW ids into input positions.
        self.register_buffer("face_positions", self.face_indices.clone())
        self.register_buffer("candidate_positions", torch.tensor(candidates))
        self.register_buffer("routing_torso_positions", torch.tensor([positions[j] for j in (5, 6, 11, 12)]))

    def export_selection(self, threshold=.5):
        raise ValueError("Dynamic selection requires an input clip; no single fixed subset exists")

    @staticmethod
    def _describe(raw, mask):
        # Raw input [B,3,T,N,M]; missing observations contribute no coordinates.
        xy = raw[:, :2].permute(0, 4, 3, 2, 1)
        valid = mask[:, 0].permute(0, 3, 2, 1).to(raw.dtype)
        count = valid.sum(-1)
        mean = (xy * valid[..., None]).sum(-2) / count[..., None].clamp_min(1.)
        variance = ((xy - mean[..., None, :]).square() * valid[..., None]).sum(-2) / count[..., None].clamp_min(1.)
        if xy.shape[-2] > 1:
            pair_valid = valid[..., 1:] * valid[..., :-1]
            velocity = (xy[..., 1:, :] - xy[..., :-1, :]).abs() * pair_valid[..., None]
            velocity = velocity.sum(-2) / pair_valid.sum(-1)[..., None].clamp_min(1.)
        else:
            velocity = torch.zeros_like(mean)
        confidence = (raw[:, 2].permute(0, 3, 2, 1) * valid).sum(-1) / count.clamp_min(1.)
        visibility = valid.mean(-1)
        features = torch.cat((mean, (variance + 1e-8).sqrt(), velocity, confidence[..., None], visibility[..., None]), dim=-1)
        return features, (count > 0).to(raw.dtype)

    def route(self, raw, raw_mask):
        # Normalize routing geometry by visible torso references.
        torso = raw[:, :2].index_select(3, self.routing_torso_positions)
        tm = raw_mask.index_select(3, self.routing_torso_positions).to(raw.dtype)
        center = (torso * tm).sum(3, keepdim=True) / tm.sum(3, keepdim=True).clamp_min(1.)
        shoulders = (torso[:, :, :, :2] * tm[:, :, :, :2]).sum(3) / tm[:, :, :, :2].sum(3).clamp_min(1.)
        hips = (torso[:, :, :, 2:] * tm[:, :, :, 2:]).sum(3) / tm[:, :, :, 2:].sum(3).clamp_min(1.)
        scale = torch.linalg.vector_norm(shoulders - hips, dim=1).clamp_min(.05)[:, None, :, None, :]
        normalized = torch.cat((((raw[:, :2] - center) / scale).clamp(-20., 20.), raw[:, 2:3]), dim=1)
        normalized = normalized.masked_fill(~raw_mask, 0.)
        descriptors, visible = self._describe(normalized, raw_mask)
        descriptors = (descriptors * visible[..., None]).sum(1) / visible.sum(1).clamp_min(1.)[..., None]
        node_visible = visible.any(1).to(raw.dtype)
        left = descriptors.index_select(1, self.candidate_positions[:, 0])
        right = descriptors.index_select(1, self.candidate_positions[:, 1])
        lv = node_visible.index_select(1, self.candidate_positions[:, 0])
        rv = node_visible.index_select(1, self.candidate_positions[:, 1])
        candidates = (left * lv[..., None] + right * rv[..., None]) / (lv + rv).clamp_min(1.)[..., None]
        face_points = descriptors.index_select(1, self.face_positions.flatten()).reshape(raw.shape[0], 6, 12, 8)
        face_visible = node_visible.index_select(1, self.face_positions.flatten()).reshape(raw.shape[0], 6, 12)
        face_visible = face_visible * self.face_members[None].to(raw.dtype)
        face_descriptors = (face_points * face_visible[..., None]).sum(2) / face_visible.sum(2).clamp_min(1.)[..., None]
        candidates = torch.cat((candidates, face_descriptors), dim=1)
        context = (descriptors * node_visible[..., None]).sum(1) / node_visible.sum(1).clamp_min(1.)[:, None]
        return self.gates(candidates, context)

    @staticmethod
    def _pool_gated_nodes(x, observed, gates):
        # x already contains the differentiable gate. Do not mask it again
        # with (gate > 0): that would erase ST gradients for closed nodes.
        weights = observed.to(x.dtype)
        count = (weights * gates.detach()).sum(-1).squeeze(1)
        features = (x * weights).sum(-1).transpose(1, 2)
        return features / count.unsqueeze(-1).clamp_min(1.), observed.any(-1).squeeze(1)

    def forward(self, x, valid_frame_mask=None, *, return_selection=False, return_gate_values=False):
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != len(self.selection["input_indices"]):
            raise ValueError("Expected B x 3 x T x manifest_input_count x M raw input; no full-input fallback")
        b, _, t, n, m = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask = mask & valid_frame_mask[:, None, :, None, None].bool()
        raw = x.masked_fill(~mask, 0)
        gate_values, penalty, gate_aux = self.route(raw, mask)
        values = torch.cat((gate_values.new_ones((b, 1)), gate_values), dim=1)
        values = values[:, None].expand(b, m, -1).reshape(b * m, -1)
        bg = values[:, self.body_gate_ids][:, None, None]
        hg = values[:, self.hand_gate_ids][:, None, None]
        x = raw.permute(0, 4, 1, 2, 3).reshape(b * m, 3, t, n)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b * m, 1, t, n)

        # Keep detection validity separate from learned structural gates.
        body = x.index_select(-1, self.body_indices)
        bm = mask.index_select(-1, self.body_indices)
        body = body.masked_fill(~bm, 0)
        batch, _, time, nodes = body.shape
        norm_mask = bm.expand(-1, 3, -1, -1).permute(0, 1, 3, 2).reshape(batch, 3 * nodes, time, 1)
        body = self.body_input_norm(body.permute(0, 1, 3, 2).reshape(batch, 3 * nodes, time, 1), norm_mask)
        body = body.reshape(batch, 3, nodes, time).permute(0, 1, 3, 2) * bg
        for block in self.body_blocks:
            body, bm = block(body, bm)
        body = body * bg  # Preserve a direct gate gradient despite intervening BN.

        torso_xy = x[:, :2].index_select(-1, self.torso_indices)
        torso_valid = mask.index_select(-1, self.torso_indices).to(x.dtype)
        center = (torso_xy * torso_valid).sum(-1, keepdim=True) / torso_valid.sum(-1, keepdim=True).clamp_min(1)
        source = torch.cat((x[:, :2] - center, x[:, 2:3]), dim=1)
        shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
        shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
        hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
        scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
        left, right = source.index_select(-1, self.left_indices), source.index_select(-1, self.right_indices)
        lm, rm = mask.index_select(-1, self.left_indices), mask.index_select(-1, self.right_indices)
        pair_valid = lm & rm
        vector = right[:, :2] - left[:, :2]
        distance = torch.linalg.vector_norm(vector, dim=1)
        relative_distance = (distance / scale[..., None])[:, None] * pair_valid.to(x.dtype)
        direction = vector / distance[:, None].clamp_min(1e-6) * pair_valid.to(x.dtype)
        left = (torch.cat((left, relative_distance, direction), dim=1) * hg).masked_fill(~lm, 0)
        right = (torch.cat((right, relative_distance, -direction), dim=1) * hg).masked_fill(~rm, 0)
        left, lm = self._run_hand(left, lm)
        right, rm = self._run_hand(right, rm)
        left, right = self.hand_to_local(left) * hg, self.hand_to_local(right) * hg
        lt, lv = self._pool_gated_nodes(left, lm, hg)
        rt, rv = self._pool_gated_nodes(right, rm, hg)
        hand_t, hand_valid = (lt + rt) * .5, lv | rv

        fx = source.index_select(-1, self.face_indices.flatten()).reshape(b * m, 3, t, 6, 12)
        fm = mask.index_select(-1, self.face_indices.flatten()).reshape(b * m, 1, t, 6, 12)
        fm = fm & self.face_members[None, None, None]
        fc = fm.sum(-1)
        face = (fx * fm.to(x.dtype)).sum(-1) / fc.clamp_min(1)
        fg = values[:, len(OPTIONAL_BODY) + 20 + 1:][:, None, None, :]
        face, fm = self._run_face(face, fc > 0)
        # Gate each encoded token once. Pre- and post-gating together would
        # make all-closed face features exactly zero and erase reopening
        # gradients through ReLU/BN. Encoding stays dense in this experiment.
        face = face * fg
        face_t, face_valid = self._pool_gated_nodes(face, fm, fg)
        body_t, _ = self._pool_gated_nodes(body, bm, bg)
        target_t = body_t.shape[1]
        if hand_t.shape[1] != target_t:
            hand_t = F.interpolate(hand_t.transpose(1, 2), size=target_t, mode="linear", align_corners=False).transpose(1, 2)
            hand_valid = F.interpolate(hand_valid[:, None].float(), size=target_t, mode="nearest")[:, 0].bool()
        if face_t.shape[1] != target_t:
            face_t = F.interpolate(face_t.transpose(1, 2), size=target_t, mode="linear", align_corners=False).transpose(1, 2)
            face_valid = F.interpolate(face_valid[:, None].float(), size=target_t, mode="nearest")[:, 0].bool()
        bv = self.body_projection(body_t.reshape(b, m, target_t, -1).mean(1).mean(1))
        local_t = (hand_t + face_t) * .5 * (hand_valid | face_valid).unsqueeze(-1).to(x.dtype)
        local = self.local_projection(local_t.reshape(b, m, target_t, -1).mean(1).mean(1))
        logits = self.classifier(torch.cat((bv, local), dim=1))
        if return_gate_values:
            return logits, penalty, gate_aux
        return (logits, penalty) if return_selection else logits
