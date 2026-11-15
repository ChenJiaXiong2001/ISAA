import unittest

import torch

from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_MOTION_VARIANT, FacePairHandMotionFusion,
)
from train_best_face_pair_hand_motion_100 import training_arguments


class FacePairHandMotionTests(unittest.TestCase):
    def test_motion_is_xy_only_and_invalid_transitions_are_zero(self):
        local = torch.zeros(1, 3, 4, 133)
        local[:, 0, 1, 91] = 2.0
        local[:, 1, 2, 91] = 3.0
        mask = torch.ones(1, 1, 4, 133, dtype=torch.bool)
        mask[:, :, 2, 91] = False
        motion = FacePairHandMotionFusion._hand_motion(local, mask)
        self.assertEqual(tuple(motion.shape), (1, 2, 4, 42))
        self.assertTrue(torch.equal(motion[:, :, 0], torch.zeros_like(motion[:, :, 0])))
        self.assertTrue(torch.equal(motion[:, :, 2], torch.zeros_like(motion[:, :, 2])))

    def test_model_and_entry_use_eight_hand_channels_and_compile(self):
        model = FacePairHandMotionFusion(4)
        self.assertEqual(model.model_config["variant"], FACE_PAIR_HAND_MOTION_VARIANT)
        self.assertEqual(model.HAND_INPUT_CHANNELS, 8)
        self.assertEqual(model.hand_ctr_blocks[0].gcn.branches[0].feature_proj.in_channels, 8)
        flags = training_arguments([])
        self.assertEqual(flags[flags.index("--model-variant") + 1], FACE_PAIR_HAND_MOTION_VARIANT)
        self.assertIn("--compile", flags)
        with self.assertRaisesRegex(ValueError, "torch.compile"):
            training_arguments(["--no-compile"])


if __name__ == "__main__":
    unittest.main()
