"""Global input-joint selection and a physically compact pooled baseline.

STG (Yamada et al., ICML 2020) motivates the clipped Gaussian gates. This
grouped skeleton adaptation is new; it is not a reproduction of that paper.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.rtmw_local_ctr import PointBatchNorm

INPUT_SELECTION_VARIANT = "best-input-joint-selection"
BODY = BodyLocalHandCTRWideRelativeFusion.BODY_INDICES
FACE = tuple(range(23, 91))
# Body face landmarks, torso geometry and wrists remain explicit dependencies.
PROTECTED_BODY = (0, 5, 6, 11, 12, 56, 91, 112)
OPTIONAL_BODY = tuple(j for j in BODY if j not in PROTECTED_BODY)


def selection_spec(body_indices=None, hand_local_indices=None):
    body = list(BODY if body_indices is None else body_indices)
    hand = list(range(21) if hand_local_indices is None else hand_local_indices)
    for values, allowed, name in ((body, BODY, "body"), (hand, range(21), "hand")):
        if not values or any(type(v) is not int for v in values):
            raise ValueError(f"{name} indices must be nonempty integer lists")
        if values != sorted(set(values)) or not set(values).issubset(allowed):
            raise ValueError(f"{name} indices must be sorted, unique and in the baseline layout")
    if not set(PROTECTED_BODY).issubset(body) or 0 not in hand:
        raise ValueError("Selection must retain torso references, body face landmarks and both wrists")
    inputs = sorted(set(body) | set(FACE) | {91 + j for j in hand} | {112 + j for j in hand})
    return dict(version=1, variant=INPUT_SELECTION_VARIANT, body_indices=body,
                hand_local_indices=hand, input_indices=inputs, face_indices=list(FACE),
                protected_body_indices=list(PROTECTED_BODY), paired_hands=True,
                raw_input_count=len(inputs), face_policy="all_68_raw_face_points_retained")


def validate_spec(spec):
    if spec.get("version") != 1 or spec.get("variant") != INPUT_SELECTION_VARIANT:
        raise ValueError("Unsupported input selection manifest")
    canonical = selection_spec(spec["body_indices"], spec["hand_local_indices"])
    for key, value in canonical.items():
        if spec.get(key) != value:
            raise ValueError(f"Selection manifest has inconsistent {key}")
    return canonical


class JointStochasticGates(nn.Module):
    """One dataset-global gate per optional body joint / mirrored hand pair."""

    def __init__(self, sigma=0.5, initial_mu=1.0):
        super().__init__()
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(initial_mu):
            raise ValueError("Gate sigma must be finite/positive and initial_mu finite")
        self.sigma = float(sigma)
        self.mu = nn.Parameter(torch.full((len(OPTIONAL_BODY) + 20,), float(initial_mu)))
        self.register_buffer("cost", torch.tensor([1.] * len(OPTIONAL_BODY) + [2.] * 20))
        self.register_buffer("noise_scale", torch.ones(()))

    def forward(self):
        noise = torch.randn_like(self.mu) * self.sigma * self.noise_scale if self.training else 0.
        return (self.mu + noise).clamp(0., 1.)

    def expected_optional_count(self):
        probability = .5 * (1. + torch.erf(self.mu / (self.sigma * math.sqrt(2.))))
        return (probability * self.cost).sum()

    def deterministic(self):
        return self.mu.detach().clamp(0., 1.)


class InputJointSelectionFusion(BodyLocalHandCTRWideRelativeFusion):
    """Same pooled encoders/readout, with explicit raw-input dependencies.

    `select`: dense candidate graph, learn global gates; no speed-up claimed.
    `compact`: only manifest inputs exist, graph and input BN physically sliced;
    all neural weights are newly randomly initialized, never copied from search.
    """

    ARCHITECTURE = "rtmw_original_pool_stg_input_subset_v1"

    def __init__(self, num_classes=60, *, stage="select", subset=None,
                 gate_sigma=.5, gate_initial_mu=1.):
        if stage not in {"select", "compact"}:
            raise ValueError("stage must be select or compact")
        if (stage == "compact") != (subset is not None):
            raise ValueError("Only compact stage requires a fixed selection manifest")
        super().__init__(num_classes)
        self.stage = stage
        self.selection = validate_spec(subset) if subset is not None else selection_spec()
        body_ids = self.selection["body_indices"]
        hand_ids = self.selection["hand_local_indices"]
        input_ids = self.selection["input_indices"]
        position = {j: i for i, j in enumerate(input_ids)}
        self.register_buffer("input_indices", torch.tensor(input_ids))
        self.body_indices = torch.tensor([position[j] for j in body_ids])
        self.register_buffer("left_indices", torch.tensor([position[91 + j] for j in hand_ids]))
        self.register_buffer("right_indices", torch.tensor([position[112 + j] for j in hand_ids]))
        self.register_buffer("torso_indices", torch.tensor([position[j] for j in (5, 6, 11, 12)]))
        self.face_indices = torch.tensor([[position[j] for j in row] for row in self.face_indices.tolist()])
        body_positions = torch.tensor([BODY.index(j) for j in body_ids])
        hand_positions = torch.tensor(hand_ids)
        for blocks, indices in ((self.body_blocks, body_positions), (self.hand_ctr_blocks, hand_positions)):
            for block in blocks:
                topology = block.gcn.static_topology.detach().index_select(1, indices).index_select(2, indices)
                block.gcn.static_topology = nn.Parameter(topology.clone())
        self.body_input_norm = PointBatchNorm(3 * len(body_ids))
        # Remove the unused inherited ST hand stack (not used by the baseline's CTR forward).
        del self.local_blocks
        if stage == "select":
            self.gates = JointStochasticGates(gate_sigma, gate_initial_mu)
        body_groups = {j: i + 1 for i, j in enumerate(OPTIONAL_BODY)}
        self.register_buffer("body_gate_ids", torch.tensor([body_groups.get(j, 0) for j in body_ids]))
        self.register_buffer("hand_gate_ids", torch.tensor([0 if j == 0 else len(OPTIONAL_BODY) + j for j in hand_ids]))

    def export_selection(self, threshold=.5):
        if not math.isfinite(threshold) or not 0 <= threshold < 1:
            raise ValueError("Export threshold must be finite and in [0, 1)")
        if self.stage == "compact":
            return dict(self.selection)
        keep = self.gates.deterministic().cpu().tolist()
        body = sorted(set(PROTECTED_BODY) | {j for j, z in zip(OPTIONAL_BODY, keep) if z > threshold})
        hand = [0] + [j for j in range(1, 21) if keep[len(OPTIONAL_BODY) + j - 1] > threshold]
        return selection_spec(body, hand)

    def forward(self, x, valid_frame_mask=None, *, return_selection=False):
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3] != len(self.selection["input_indices"]):
            raise ValueError("Expected B x 3 x T x manifest_input_count x M raw input; no full-input fallback")
        b, _, t, n, m = x.shape
        mask = (x[:, 2:3] > 0) & torch.isfinite(x).all(1, keepdim=True)
        if valid_frame_mask is not None:
            mask = mask & valid_frame_mask[:, None, :, None, None].bool()
        x = x.permute(0, 4, 1, 2, 3).reshape(b * m, 3, t, n)
        mask = mask.permute(0, 4, 1, 2, 3).reshape(b * m, 1, t, n)
        x = x.masked_fill(~mask, 0)
        values = (self.gates() if self.stage == "select" else x.new_ones(len(OPTIONAL_BODY) + 20))
        values = torch.cat((values.new_ones(1), values))
        bg = values[self.body_gate_ids][None, None, None]
        hg = values[self.hand_gate_ids][None, None, None]

        # Keep detection validity separate from learned structural gates.
        body = x.index_select(-1, self.body_indices)
        bm = mask.index_select(-1, self.body_indices) & (bg > 0)
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
        pair_valid = lm & rm & (hg > 0)
        vector = right[:, :2] - left[:, :2]
        distance = torch.linalg.vector_norm(vector, dim=1)
        relative_distance = (distance / scale[..., None])[:, None] * pair_valid.to(x.dtype)
        direction = vector / distance[:, None].clamp_min(1e-6) * pair_valid.to(x.dtype)
        lm, rm = lm & (hg > 0), rm & (hg > 0)
        left = (torch.cat((left, relative_distance, direction), dim=1) * hg).masked_fill(~lm, 0)
        right = (torch.cat((right, relative_distance, -direction), dim=1) * hg).masked_fill(~rm, 0)
        left, lm = self._run_hand(left, lm)
        right, rm = self._run_hand(right, rm)
        left, right = self.hand_to_local(left) * hg, self.hand_to_local(right) * hg
        lt, lv = self._pool_nodes_per_frame(left, lm)
        rt, rv = self._pool_nodes_per_frame(right, rm)
        hand_t, hand_valid = (lt + rt) * .5, lv | rv

        fx = source.index_select(-1, self.face_indices.flatten()).reshape(b * m, 3, t, 6, 12)
        fm = mask.index_select(-1, self.face_indices.flatten()).reshape(b * m, 1, t, 6, 12)
        fm = fm & self.face_members[None, None, None]
        fc = fm.sum(-1)
        face = (fx * fm.to(x.dtype)).sum(-1) / fc.clamp_min(1)
        face, fm = self._run_face(face, fc > 0)
        face_t, face_valid = self._pool_nodes_per_frame(face, fm)
        body_t, _ = self._pool_nodes_per_frame(body, bm)
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
        penalty = (self.gates.expected_optional_count() / self.gates.cost.sum()
                   if self.stage == "select" else logits.new_zeros(()))
        return (logits, penalty) if return_selection else logits
