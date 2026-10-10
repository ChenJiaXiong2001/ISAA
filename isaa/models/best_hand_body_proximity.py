"""Append sparse interaction features to the unchanged best wide-relative model."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.hand_body_proximity import TorsoCoordinates, SparseHandBodyInteraction


BEST_PROXIMITY_VARIANT = "best-hand-body-proximity"


class BestHandBodyProximityFusion(BodyLocalHandCTRWideRelativeFusion):
    """The complete baseline path plus an initially zero feature residual."""

    ARCHITECTURE = "rtmw_best_wide_relative_additive_hand_body_proximity_v1"
    ADDITIONAL_PREFIXES = ("coordinates.", "interaction.", "interaction_projection.", "torso_projection.")

    def __init__(self, num_classes=120, *, interaction_config=None):
        super().__init__(num_classes=num_classes)
        defaults = dict(width=64, radius_on=0.4, radius_off=0.48, confidence_threshold=0.0)
        supplied = dict(interaction_config or {})
        if set(supplied) - set(defaults):
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
        self.interaction_projection = nn.Linear(2 * width, 288, bias=False)
        self.torso_projection = nn.Linear(width, 288, bias=False)
        # Importing the best checkpoint starts with exactly its predictions.
        nn.init.zeros_(self.interaction_projection.weight)
        nn.init.zeros_(self.torso_projection.weight)

    def initialize_from_baseline(self, checkpoint):
        if checkpoint.get("architecture") != BodyLocalHandCTRWideRelativeFusion.ARCHITECTURE:
            raise ValueError("Initialization requires the original best wide-relative checkpoint")
        source = checkpoint["model"]
        target = self.state_dict()
        baseline_keys = {key for key in target if not key.startswith(self.ADDITIONAL_PREFIXES)}
        if set(source) != baseline_keys or any(source[key].shape != target[key].shape for key in baseline_keys):
            raise ValueError("Baseline checkpoint keys and tensor shapes must match exactly")
        self.load_state_dict(source, strict=False)
        nn.init.zeros_(self.interaction_projection.weight)
        nn.init.zeros_(self.torso_projection.weight)

    def forward(self, x, valid_frame_mask=None, return_interaction=False):
        # Execute the original six-channel hand inputs, encoders and fusion verbatim.
        baseline_logits = super().forward(x, valid_frame_mask)
        geometry = self.coordinates(x[:, :3], valid_frame_mask)
        b, m = x.shape[0], (x.shape[-1] if x.ndim == 5 else 1)
        result = self.interaction(geometry, return_interaction)
        interaction, details = result if return_interaction else (result, None)
        valid_frames = geometry["valid"].any(2)
        person_active = valid_frames.any(1).reshape(b, m)
        interaction = interaction.reshape(b, m, -1)
        interaction = (interaction * person_active[..., None]).sum(1) / person_active.sum(1).clamp_min(1)[:, None]
        torso = self.interaction.torso_encoder(geometry["torso_context"]).reshape(b, m, valid_frames.shape[1], -1)
        weight = valid_frames.reshape(b, m, -1, 1).to(torso.dtype)
        torso = (torso * weight).sum((1, 2)) / weight.sum((1, 2)).clamp_min(1)
        added_features = self.interaction_projection(interaction) + self.torso_projection(torso)
        # Linear classification of the feature sum preserves the original bias once.
        logits = baseline_logits + F.linear(added_features, self.classifier.weight)
        if return_interaction:
            details.update(relative_coordinates=geometry["relative"], node_valid=geometry["valid"],
                           scale=geometry["scale"], batch_size=b, people=m,
                           baseline_logits=baseline_logits, added_features=added_features)
            return logits, details
        return logits
