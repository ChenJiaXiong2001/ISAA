"""Regression checks for precompression evaluator model selection."""
import unittest

from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT,
    FACE_PAIR_HAND_MOTION_VARIANT,
    FACE_PAIR_RAW_XY_VARIANT,
    FACE_PAIR_VARIANT,
    FacePairFusion,
    FacePairHandFaceGeometryFusion,
    FacePairHandMotionFusion,
    FacePairRawXYFusion,
)
from tools.evaluate_face_pair_precompression_repair import model_type


class PrecompressionVariantTests(unittest.TestCase):
    def test_checkpoint_variants_select_matching_models(self):
        self.assertIs(model_type(FACE_PAIR_VARIANT), FacePairFusion)
        self.assertIs(model_type(FACE_PAIR_RAW_XY_VARIANT), FacePairRawXYFusion)
        self.assertIs(model_type(FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT), FacePairHandFaceGeometryFusion)
        self.assertIs(model_type(FACE_PAIR_HAND_MOTION_VARIANT), FacePairHandMotionFusion)
        with self.assertRaises(ValueError):
            model_type("unsupported-face-pair")


if __name__ == "__main__":
    unittest.main()
