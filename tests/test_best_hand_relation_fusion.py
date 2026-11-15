"""Focused regression checks for the pooled hand-relation experiment."""
import sys
import unittest
from unittest.mock import patch

import torch

from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.best_hand_relation_fusion import (
    BEST_HAND_RELATION_VARIANT, BestHandRelationFusion,
)
from isaa.train import parse_args
from train_best_hand_relation_100_cosine import training_arguments


class BestHandRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def sample():
        x = torch.rand(1, 3, 100, 133, 1)
        x[:, 2] = 1
        return x, torch.ones(1, 100, dtype=torch.bool)

    def test_disabled_residual_matches_original_pooling(self):
        torch.manual_seed(3)
        baseline = BodyLocalHandCTRWideRelativeFusion(60).eval()
        model = BestHandRelationFusion(60).eval()
        missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(any(name.startswith("pair_projection") for name in missing))
        self.assertTrue(all(name.startswith("local_blocks") for name in unexpected))
        x, mask = self.sample()
        with torch.no_grad():
            expected = baseline(x, mask)
            actual = model(x, mask, interaction_enabled=False)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_relation_receives_gradients_and_missing_hands_are_inert(self):
        torch.manual_seed(7)
        model = BestHandRelationFusion(60).eval()
        x, mask = self.sample()
        model(x, mask).square().mean().backward()
        self.assertIsNotNone(model.pair_projection.weight.grad)
        self.assertGreater(model.pair_projection.weight.grad.abs().sum().item(), 0)
        x[:, :, :, 91:133] = 0
        with torch.no_grad():
            full = model(x, mask)
            without = model(x, mask, interaction_enabled=False)
        torch.testing.assert_close(full, without, rtol=1e-5, atol=1e-6)

    def test_preset_and_generic_parser_require_compile_and_scratch(self):
        args = training_arguments([])
        self.assertEqual(args[args.index("--model-variant") + 1], BEST_HAND_RELATION_VARIANT)
        self.assertEqual(args[args.index("--epochs") + 1], "100")
        self.assertEqual(args[args.index("--lr-scheduler") + 1], "warmup_cosine")
        with patch.object(sys, "argv", ["train", *args]):
            parsed = parse_args()
        self.assertTrue(parsed.compile)
        self.assertEqual(parsed.window_size, 100)
        self.assertEqual(parsed.temporal_compression, "confidence_weighted")
        for override in (["--no-compile"], ["--init-checkpoint", "old.pt"],
                         ["--resume-checkpoint", "old.pt"]):
            with self.assertRaises((SystemExit, ValueError)):
                rejected = training_arguments(override)
                with patch.object(sys, "argv", ["train", *rejected]):
                    parse_args()
        with patch.object(sys, "argv", ["train", "--model-variant",
                                        BEST_HAND_RELATION_VARIANT, "--window-size", "100",
                                        "--temporal-compression", "confidence_weighted",
                                        "--lr-scheduler", "warmup_cosine", "--no-compile"]):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
