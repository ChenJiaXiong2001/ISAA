"""Add action-candidate and proximity routing to the complete best architecture."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.hand_body_proximity import REGION_NAMES, REGION_NODES, TorsoCoordinates


ACTION_PROXIMITY_VARIANT = "best-action-hand-body"
# Public configuration uses NTU's ONE-based action numbers, not tensor indices.
DEFAULT_REGION_ACTIONS = {
    "head_face": [1, 2, 3, 4, 18, 19, 20, 21, 28, 32, 37, 38, 41, 44, 47, 48],
    "torso": [14, 15, 25, 40, 45, 46],
    "left_arm": [14, 15, 40], "right_arm": [14, 15, 40],
    "left_leg": [16, 17], "right_leg": [16, 17],
    "left_foot": [16, 17], "right_foot": [16, 17],
}


def validated_config(config, num_classes):
    defaults = dict(width=64, radius_on=0.4, radius_off=0.48, confidence_threshold=0.0,
                    candidate_threshold=0.15, temperature=1.0, exploration_floor=0.1,
                    baseline_epochs=5, ramp_epochs=10, baseline_loss_weight=0.5,
                    auxiliary_loss_weight=0.2, interaction_weight_decay=0.00001,
                    region_actions=DEFAULT_REGION_ACTIONS)
    supplied = dict(config or {})
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown action interaction settings: {sorted(unknown)}")
    defaults.update(supplied)
    for key in ("width", "baseline_epochs", "ramp_epochs"):
        value = defaults[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < (1 if key != "baseline_epochs" else 0):
            raise ValueError(f"Invalid {key}")
    for key in set(defaults) - {"width", "baseline_epochs", "ramp_epochs", "region_actions"}:
        value = defaults[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if not 0 < defaults["radius_on"] <= defaults["radius_off"]:
        raise ValueError("Require 0 < radius_on <= radius_off")
    for key in ("confidence_threshold", "exploration_floor"):
        if not 0 <= defaults[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if not 0 < defaults["candidate_threshold"] <= 1 or defaults["temperature"] <= 0:
        raise ValueError("Candidate threshold must be in (0,1]; temperature must be positive")
    if defaults["baseline_loss_weight"] <= 0 or defaults["auxiliary_loss_weight"] <= 0:
        raise ValueError("Both auxiliary classification weights must be positive")
    if defaults["interaction_weight_decay"] < 0:
        raise ValueError("interaction_weight_decay must be nonnegative")
    mapping = defaults["region_actions"]
    if not isinstance(mapping, dict) or set(mapping) - set(REGION_NAMES):
        raise ValueError("region_actions must map known region names to one-based action lists")
    normalized = {}
    for region in REGION_NAMES:
        actions = mapping.get(region, [])
        if not isinstance(actions, list) or any(isinstance(a, bool) or not isinstance(a, int) or a < 1 for a in actions):
            raise ValueError("Action IDs must be positive one-based integers")
        if "region_actions" in supplied and any(a > num_classes for a in actions):
            raise ValueError("Configured action ID exceeds num_classes")
        normalized[region] = sorted({a for a in actions if a <= num_classes})
    defaults["region_actions"] = normalized
    return defaults


class ConditionalHandBodyInteraction(nn.Module):
    """Screen candidate regions, then encode only valid nearby edge rows."""

    def __init__(self, width, radius_on, radius_off):
        super().__init__()
        self.width, self.radius_on, self.radius_off = width, radius_on, radius_off
        owners = torch.empty(91, dtype=torch.long)
        for r, nodes in enumerate(REGION_NODES):
            owners[list(nodes)] = r
            self.register_buffer(f"region_nodes_{r}", torch.tensor(nodes))
        self.register_buffer("owners", owners)
        self.node_identity = nn.Embedding(133, width)
        self.region_identity = nn.Embedding(8, 8)
        self.node_encoder = nn.Sequential(nn.Linear(6, width), nn.LeakyReLU(0.1), nn.Linear(width, width))
        self.edge_encoder = nn.Sequential(nn.Linear(2 * width + 19, width), nn.LayerNorm(width),
                                          nn.LeakyReLU(0.1), nn.Linear(width, width), nn.LeakyReLU(0.1))
        self.edge_weight = nn.Linear(width, 1)
        self.torso_encoder = nn.Sequential(nn.Linear(12, width), nn.LayerNorm(width), nn.LeakyReLU(0.1))
        self.gate = nn.Linear(2 * width + 8, 1)
        nn.init.zeros_(self.gate.bias)
        self.temporal = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.LeakyReLU(0.1))

    @torch.compiler.disable
    def build_edges(self, q, valid, enabled):
        rows = []
        # Unselected regions/persons never enter cdist or neural edge encoding.
        for r in range(len(REGION_NAMES)):
            people = enabled[:, r].nonzero(as_tuple=True)[0]
            if people.numel() == 0:
                continue
            nodes = getattr(self, f"region_nodes_{r}")
            hands = q[people, :, 91:]
            targets = q.index_select(0, people).index_select(2, nodes)
            distance = torch.cdist(hands.float(), targets.float(), compute_mode="donot_use_mm_for_euclid_dist")
            pair_valid = valid[people, :, 91:, None] & valid.index_select(0, people).index_select(2, nodes)[:, :, None]
            # Root hand points duplicate the ordinary body wrist points.
            pair_valid[:, :, 0] &= nodes != 9
            pair_valid[:, :, 21] &= nodes != 10
            state = torch.zeros_like(pair_valid[:, 0])
            frames = []
            for ti in range(q.shape[1]):
                state = pair_valid[:, ti] & ((distance[:, ti] < self.radius_on) |
                                            (state & (distance[:, ti] <= self.radius_off)))
                frames.append(state)
            pi, ti, hi, ji = torch.stack(frames, 1).nonzero(as_tuple=True)
            rows.append(torch.stack((people[pi], ti, hi, nodes[ji]), 1))
        return torch.cat(rows, 0) if rows else torch.empty((0, 4), dtype=torch.long, device=q.device)

    def forward(self, geometry, enabled, return_details=False):
        q, valid = geometry["relative"], geometry["valid"]
        n, t = q.shape[:2]
        edges = self.build_edges(q.detach(), valid, enabled)
        active = torch.zeros((n, t, 2, 8), dtype=torch.bool, device=q.device)
        if edges.numel() == 0:
            pooled = q.new_zeros(n, 2, 8, self.width)
            details = dict(edges=edges, edge_distance=q.new_empty(0), region_gate=q.new_zeros(n, t, 2, 8), active=active)
            return pooled, details
        ni, ti, hi, ji = edges.unbind(1)
        hj, ri = hi + 91, self.owners[ji]
        flat_hand, flat_target = (ni * t + ti) * 133 + hj, (ni * t + ti) * 133 + ji
        selected, inverse = torch.unique(torch.cat((flat_hand, flat_target)), return_inverse=True)
        node_input = torch.cat((q, geometry["velocity"], geometry["score"][..., None],
                                geometry["motion_valid"][..., None].float()), -1)
        features = self.node_encoder(node_input.reshape(-1, 6)[selected]) + self.node_identity(selected % 133)
        hand_features, target_features = features[inverse].split(edges.shape[0])
        delta = q[ni, ti, ji] - q[ni, ti, hj]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        motion = geometry["motion_valid"][ni, ti, ji] & geometry["motion_valid"][ni, ti, hj]
        velocity = (geometry["velocity"][ni, ti, ji] - geometry["velocity"][ni, ti, hj]).masked_fill(~motion[:, None], 0)
        previous = (ti - 1).clamp_min(0)
        old_delta = q[ni, previous, ji] - q[ni, previous, hj]
        change = (distance - torch.linalg.vector_norm(old_delta, dim=-1, keepdim=True)).masked_fill(~motion[:, None], 0)
        encoded = self.edge_encoder(torch.cat((hand_features, target_features, delta, distance,
                    delta / distance.clamp_min(1e-6), velocity, change,
                    geometry["score"][ni, ti, hj, None], geometry["score"][ni, ti, ji, None],
                    motion[:, None].float(), self.region_identity(ri)), -1))
        quality = (geometry["score"][ni, ti, hj] * geometry["score"][ni, ti, ji]).clamp(0, 1)[:, None].to(encoded.dtype)
        weights = self.edge_weight(encoded).sigmoid() * quality
        group = ((((ni * t + ti) * 2 + hi // 21) * 8 + ri) * 21 + hi % 21)
        count = n * t * 2 * 8 * 21
        sums = encoded.new_zeros(count, self.width).index_add(0, group, encoded * weights)
        totals = encoded.new_zeros(count, 1).index_add(0, group, weights)
        close = weights.new_zeros(count, 1).index_add(
            0, group, weights * torch.exp(-distance / self.radius_on).to(weights.dtype))
        node_messages = (sums / totals.clamp_min(1e-6)).reshape(n, t, 2, 8, 21, self.width)
        node_active = (totals > 0).reshape(n, t, 2, 8, 21, 1)
        active = node_active.any(4).squeeze(-1)
        messages = node_messages.sum(4) / node_active.sum(4).clamp_min(1)
        proximity = (close / totals.clamp_min(1e-6)).reshape(n, t, 2, 8, 21).sum(4) / node_active.sum(4).squeeze(-1).clamp_min(1)
        ctx = self.torso_encoder(geometry["torso_context"])[:, :, None, None].expand(-1, -1, 2, 8, -1)
        identity = self.region_identity.weight[None, None, None].expand(n, t, 2, -1, -1)
        gates = self.gate(torch.cat((messages, ctx, identity), -1)).sigmoid().squeeze(-1) * proximity * active
        y = (messages * gates[..., None]).permute(0, 2, 3, 4, 1).reshape(n * 16, self.width, t)
        mask = active.permute(0, 2, 3, 1).reshape(n * 16, 1, t)
        y = self.temporal(y).masked_fill(~mask, 0)
        pooled = (y.sum(2) / mask.sum(2).clamp_min(1)).reshape(n, 2, 8, self.width)
        details = dict(edges=edges, edge_distance=distance.squeeze(-1), region_gate=gates, active=active)
        return pooled, details


class BestActionHandBodyFusion(BodyLocalHandCTRWideRelativeFusion):
    """Original best forward + conditional hand/body logit correction; scratch only."""

    ARCHITECTURE = "rtmw_best_additive_action_candidate_hand_body_v1"
    ADDITIONAL_PREFIXES = ("coordinates.", "interaction.", "interaction_head.",
                           "class_regions", "interaction_strength")

    def __init__(self, num_classes=60, *, interaction_config=None):
        super().__init__(num_classes=num_classes)
        self.interaction_config = validated_config(interaction_config, num_classes)
        cfg = self.interaction_config
        matrix = torch.zeros(num_classes, 8)
        for r, name in enumerate(REGION_NAMES):
            for action in cfg["region_actions"][name]:
                matrix[action - 1, r] = 1
        self.register_buffer("class_regions", matrix)
        # Saved in checkpoints so reload uses the same inference ramp strength.
        self.register_buffer("interaction_strength", torch.tensor(1.0))
        self.coordinates = TorsoCoordinates(cfg["confidence_threshold"])
        self.interaction = ConditionalHandBodyInteraction(cfg["width"], cfg["radius_on"], cfg["radius_off"])
        self.interaction_head = nn.Linear(cfg["width"], num_classes, bias=False)
        nn.init.normal_(self.interaction_head.weight, std=0.01)

    def set_training_epoch(self, epoch):
        cfg = self.interaction_config
        strength = min(1.0, max(0.0, (epoch - cfg["baseline_epochs"]) / cfg["ramp_epochs"]))
        self.interaction_strength.fill_(strength)
        return strength

    def route_candidates(self, baseline_logits):
        probabilities = (baseline_logits.detach() / self.interaction_config["temperature"]).softmax(-1)
        mass = probabilities @ self.class_regions
        mapped = self.class_regions.any(0)[None]
        if self.training:
            floor = self.interaction_config["exploration_floor"] * (1 - self.interaction_strength)
            gates = (mass + floor).clamp(max=1) * mapped
            enabled = mapped.expand_as(mass)
        else:
            enabled = (mass >= self.interaction_config["candidate_threshold"]) & mapped
            gates = mass * enabled
        return mass, gates, enabled

    def _addition(self, x, frame_mask, baseline_logits):
        b, m = x.shape[0], x.shape[-1] if x.ndim == 5 else 1
        mass, candidate_gate, enabled = self.route_candidates(baseline_logits)
        enabled = enabled & (self.interaction_strength > 0)
        geometry = self.coordinates(x[:, :3], frame_mask)
        person_enabled = enabled[:, None].expand(-1, m, -1).reshape(b * m, 8)
        pooled, details = self.interaction(geometry, person_enabled)
        active = details["active"].any(1).reshape(b, m, 2, 8)
        region_active = active.any(1).any(1)
        pooled = pooled.reshape(b, m, 2, 8, -1)
        region_features = (pooled * active[..., None]).sum((1, 2)) / active.sum((1, 2)).clamp_min(1)[..., None]
        region_logits = self.interaction_head(region_features)
        applied_gate = candidate_gate * region_active
        allowed = self.class_regions.t()[None]
        denominator = self.class_regions.sum(1).clamp_min(1)
        correction = (region_logits * applied_gate[..., None] * allowed).sum(1) / denominator
        correction = correction * self.interaction_strength
        aux_features = region_features.sum(1) / region_active.sum(1).clamp_min(1)[:, None]
        auxiliary_logits = self.interaction_head(aux_features)
        details.update(baseline_logits=baseline_logits, correction=correction, auxiliary_logits=auxiliary_logits,
                       candidate_mass=mass, candidate_gate=candidate_gate, region_enabled=enabled,
                       region_active=region_active, applied_gate=applied_gate, region_names=REGION_NAMES,
                       batch_size=b, people=m)
        return correction, details

    def forward(self, x, valid_frame_mask=None, return_interaction=False, return_routing=False):
        baseline_logits = super().forward(x, valid_frame_mask)
        correction, details = self._addition(x, valid_frame_mask, baseline_logits)
        logits = baseline_logits + correction
        if return_routing:
            return {"logits": logits, **details}
        return (logits, details) if return_interaction else logits

    def classification_loss(self, logits, details, labels):
        """Labels supervise losses only; forward routing never consumes labels."""
        cfg = self.interaction_config
        final = F.cross_entropy(logits, labels)
        baseline = F.cross_entropy(details["baseline_logits"], labels)
        supervised = (self.class_regions[labels].bool() & details["region_active"]).any(1)
        auxiliary = (F.cross_entropy(details["auxiliary_logits"][supervised], labels[supervised])
                     if supervised.any() else logits.sum() * 0)
        strength = self.interaction_strength
        # Warmup trains the original path once, rather than double-counting CE.
        baseline_weight = cfg["baseline_loss_weight"] * (strength > 0)
        loss = final + baseline_weight * baseline + cfg["auxiliary_loss_weight"] * auxiliary
        return loss, dict(final_loss=final, baseline_loss=baseline, interaction_aux_loss=auxiliary)

    def optimizer_groups(self, baseline_weight_decay):
        original, additional = [], []
        for name, parameter in self.named_parameters():
            (additional if name.startswith(self.ADDITIONAL_PREFIXES) else original).append(parameter)
        return [{"params": original, "weight_decay": baseline_weight_decay},
                {"params": additional, "weight_decay": self.interaction_config["interaction_weight_decay"]}]
