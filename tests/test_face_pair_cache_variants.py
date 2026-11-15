"""Ensure cached evaluation selects the checkpoint's actual coordinate model."""
import unittest

from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_HAND_MOTION_VARIANT,
    FACE_PAIR_VARIANT, FACE_PAIR_RAW_XY_VARIANT, FacePairFusion,
    FacePairHandFaceGeometryFusion, FacePairHandMotionFusion, FacePairRawXYFusion,
)
from tools.cache_face_pair_logits import model_type
from tools.check_face_pair_checkpoint import check_args


class CacheVariantTests(unittest.TestCase):
    def test_model_selection_preserves_coordinate_architectures(self):
        self.assertIs(model_type(FACE_PAIR_VARIANT), FacePairFusion)
        self.assertIs(model_type(FACE_PAIR_RAW_XY_VARIANT), FacePairRawXYFusion)
        self.assertIs(model_type(FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT), FacePairHandFaceGeometryFusion)
        self.assertIs(model_type(FACE_PAIR_HAND_MOTION_VARIANT), FacePairHandMotionFusion)
        with self.assertRaises(RuntimeError):
            model_type("unknown")

    def test_raw_xy_validation_keeps_compile_and_scratch_requirements(self):
        args = dict(model_variant=FACE_PAIR_RAW_XY_VARIANT, compile=True,
                    init_checkpoint=None, resume_checkpoint=None, feature_mode="raw",
                    npy_dir=None, window_size=100, temporal_compression="confidence_weighted",
                    lr_scheduler="warmup_cosine")
        check_args(args, "test", expected_variant=FACE_PAIR_RAW_XY_VARIANT)
        with self.assertRaises(RuntimeError):
            check_args(args, "test")
        for change in ({"compile": False}, {"init_checkpoint": "old.pt"},
                       {"resume_checkpoint": "old.pt"}):
            with self.assertRaises(RuntimeError):
                check_args({**args, **change}, "test", expected_variant=FACE_PAIR_RAW_XY_VARIANT)

    def test_geometry_validation_keeps_compile_and_scratch_requirements(self):
        args = dict(model_variant=FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, compile=True,
                    init_checkpoint=None, resume_checkpoint=None, feature_mode="raw",
                    npy_dir=None, window_size=100, temporal_compression="confidence_weighted",
                    lr_scheduler="warmup_cosine")
        check_args(args, "test", expected_variant=FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT)
        for change in ({"compile": False}, {"init_checkpoint": "old.pt"},
                       {"resume_checkpoint": "old.pt"}):
            with self.assertRaises(RuntimeError):
                check_args({**args, **change}, "test", expected_variant=FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT)


if __name__ == "__main__":
    unittest.main()
