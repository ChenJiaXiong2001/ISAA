import io
import sys
import unittest
from unittest.mock import patch

import torch

from isaa.models.body_hand_weighted_pool import (
    BODY_HAND_WEIGHTED_POOL_VARIANT, BodyHandWeightedPoolFusion,
)
from isaa.train import parse_args
from train_body_hand_weighted_pool import training_arguments


class BodyHandWeightedPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    @staticmethod
    def sample(persons=2):
        x = torch.randn(2, 3, 5, 133, persons) * 0.05
        x[:, 2] = 1
        return x

    def test_hand_weights_start_as_masked_mean_and_learn(self):
        model = BodyHandWeightedPoolFusion(7, window_size=5)
        features = torch.arange(21.0).reshape(1, 1, 1, 21).expand(1, 48, 2, 21)
        mask = torch.ones(1, 1, 2, 21, dtype=torch.bool)
        mask[..., 20] = False
        pooled, valid = model._pool_hand(features, mask)
        torch.testing.assert_close(pooled, torch.full_like(pooled, 9.5))
        self.assertTrue(valid.all())
        with torch.no_grad():
            model.hand_joint_logits[0] = 8
        weighted, _ = model._pool_hand(features, mask)
        self.assertTrue((weighted < pooled).all())
        empty, empty_valid = model._pool_hand(features, torch.zeros_like(mask))
        self.assertFalse(empty_valid.any())
        torch.testing.assert_close(empty, torch.zeros_like(empty))

    def test_two_branches_forward_backward_and_missing_person(self):
        model = BodyHandWeightedPoolFusion(7, window_size=5)
        self.assertEqual(model.BODY_INDICES, tuple(range(23)))
        self.assertEqual(model.pooling_config["face_points"], [0, 1, 2, 3, 4])
        self.assertFalse(any(name.startswith("face") for name, _ in model.named_modules()))
        self.assertFalse(hasattr(model, "body_readout"))
        logits = model(self.sample())
        self.assertEqual(tuple(logits.shape), (2, 7))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        for parameter in (model.hand_score.weight, model.hand_joint_logits):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        model.eval()
        one = self.sample(persons=1)
        padded = torch.cat((one, torch.zeros_like(one)), dim=-1)
        corrupt = padded.clone()
        corrupt[:, :2, :, :, 1] = 1000
        with torch.no_grad():
            torch.testing.assert_close(model(padded), model(corrupt))

    def test_preset_and_generic_parser_require_compile_and_scratch(self):
        flags = training_arguments([])
        with patch.object(sys, "argv", ["main.py"] + flags):
            args = parse_args()
        self.assertEqual(args.model_variant, BODY_HAND_WEIGHTED_POOL_VARIANT)
        self.assertEqual((args.epochs, args.window_size), (100, 100))
        self.assertEqual((args.temporal_compression, args.lr_scheduler),
                         ("confidence_weighted", "warmup_cosine"))
        self.assertTrue(args.compile)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume_checkpoint)
        for forbidden in (["--no-compile"], ["--resume-checkpoint", "old.pt"]):
            with self.assertRaises(ValueError):
                training_arguments(forbidden)
        with patch.object(sys, "argv", ["main.py", "--model-variant",
                                         BODY_HAND_WEIGHTED_POOL_VARIANT, "--no-compile"]), \
                patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parse_args()


if __name__ == "__main__":
    unittest.main()
