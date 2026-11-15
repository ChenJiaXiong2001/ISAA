import sys
import unittest
from unittest.mock import patch

import torch

from isaa.models.body_hand_no_pool import BODY_HAND_NO_POOL_VARIANT, BodyHandNoPoolFusion
from isaa.train import parse_args
from train_body_hand_no_pool import training_arguments


class BodyHandNoPoolTests(unittest.TestCase):
    @staticmethod
    def sample(channels=3, frames=5, persons=2):
        x = torch.randn(2, channels, frames, 133, persons) * 0.05
        x[:, 2] = 1
        return x

    def test_has_only_torso_and_hand_readout_branches(self):
        model = BodyHandNoPoolFusion(7, window_size=5, max_persons=2)
        self.assertEqual(model.BODY_INDICES, tuple(range(23)))
        self.assertEqual(model.readout_config["branches"], ["torso_body_0_22", "hands_left_right_combined"])
        self.assertEqual(model.readout_config["face_points"], [0, 1, 2, 3, 4])
        self.assertEqual(model.readout_config["classification_pooling"], "none")
        self.assertFalse(any(name.startswith("face") for name, _ in model.named_modules()))
        self.assertFalse(any("face_" in name for name, _ in model.state_dict().items()))

    def test_forward_backward_and_missing_people_are_finite(self):
        model = BodyHandNoPoolFusion(7, window_size=5, max_persons=2)
        x = self.sample()
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        for module in (model.body_readout, model.hand_readout,
                       model.hand_pair_projection, model.local_slot_projection,
                       model.classifier):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in module.parameters()))
        one = self.sample(persons=1)
        padded = torch.cat((one, torch.zeros_like(one)), dim=-1)
        with torch.no_grad():
            torch.testing.assert_close(model(one), model(padded))

    def test_preset_forces_100_frame_cosine_compile_and_scratch(self):
        flags = training_arguments([])
        with patch.object(sys, "argv", ["main.py"] + flags):
            args = parse_args()
        self.assertEqual(args.model_variant, BODY_HAND_NO_POOL_VARIANT)
        self.assertEqual(args.window_size, 100)
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.lr_scheduler, "warmup_cosine")
        self.assertEqual(args.temporal_compression, "confidence_weighted")
        self.assertTrue(args.compile)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume_checkpoint)
        self.assertEqual(args.save_dir, "outputs/body_hand_no_pool_100_ntu60_xsub")


if __name__ == "__main__":
    unittest.main()

