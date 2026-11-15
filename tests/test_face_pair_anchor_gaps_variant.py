"""Regression checks for anchor-gap model variant selection."""
import unittest

from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_VARIANT,
    FacePairFusion, FacePairHandFaceGeometryFusion,
)
from tools.evaluate_face_pair_anchor_gaps import model_type


class AnchorGapVariantTests(unittest.TestCase):
    def test_baseline_and_geometry_checkpoints_select_matching_models(self):
        self.assertIs(model_type(FACE_PAIR_VARIANT), FacePairFusion)
        self.assertIs(model_type(FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT), FacePairHandFaceGeometryFusion)
        with self.assertRaises(ValueError):
            model_type("unsupported-face-pair")


if __name__ == "__main__":
    unittest.main()
