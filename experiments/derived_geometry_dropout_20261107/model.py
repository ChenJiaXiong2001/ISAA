"""Scratch-only training variant for derived hand geometry dropout."""
from __future__ import annotations

import torch

from isaa.models.face_pair_fusion import FacePairFusion


class DerivedGeometryDropoutFacePairFusion(FacePairFusion):
    """Drop only distance/direction channels during training.

    Coordinates and confidence remain intact. The dropout is disabled during
    evaluation, so the inference contract matches the formal FacePairFusion.
    """

    ARCHITECTURE = "rtmw_best_wide_relative_handface_bodyface_derived_geometry_dropout_v1"
    DERIVED_GEOMETRY_DROPOUT_P = 0.025

    def __init__(self, num_classes: int = 60):
        super().__init__(num_classes=num_classes)
        self.model_config.update(
            variant="best-face-pair-derived-geometry-dropout",
            derived_geometry_dropout={
                "probability": self.DERIVED_GEOMETRY_DROPOUT_P,
                "channels": ["cross_hand_distance", "cross_hand_direction_x", "cross_hand_direction_y"],
                "training_only": True,
            },
            scratch_only=True,
        )

    def _run_hand(self, hand, hand_mask):
        if self.training and self.DERIVED_GEOMETRY_DROPOUT_P > 0:
            keep = torch.rand(
                hand[:, 3:6].shape, device=hand.device, dtype=hand.dtype
            ) >= self.DERIVED_GEOMETRY_DROPOUT_P
            hand = hand.clone()
            hand[:, 3:6] = torch.where(keep, hand[:, 3:6], torch.zeros_like(hand[:, 3:6]))
        return super()._run_hand(hand, hand_mask)
