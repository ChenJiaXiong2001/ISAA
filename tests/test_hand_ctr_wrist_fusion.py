"""Regression tests for the face-light hand CTR-GCN wrist-injection model."""

import io
import sys
import unittest
from unittest.mock import patch

import torch

from isaa.models.hand_ctr_wrist_fusion import HAND_CTR_WRIST_VARIANT, HandCTRWristFusion
from isaa.train import parse_args
from train_hand_ctr_wrist_fusion import training_arguments


class HandCTRWristFusionTests(unittest.TestCase):
    @staticmethod
    def sample(batch=2, time=16, people=2):
        x = torch.randn(batch, 3, time, 133, people) * 0.05
        x[:, 2] = torch.rand(batch, time, 133, people)
        return x

    def test_fixed_training_preset(self):
        with patch.object(sys, "argv", ["train_hand_ctr_wrist_fusion.py"] + training_arguments([])):
            args = parse_args()
        self.assertEqual(args.model_variant, HAND_CTR_WRIST_VARIANT)
        self.assertEqual(args.window_size, 100)
        self.assertEqual(args.temporal_compression, "confidence_weighted")
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.lr_scheduler, "warmup_cosine")
        self.assertEqual(args.warmup_start_lr, 0.001)
        self.assertEqual(args.min_lr, 0.0001)
        self.assertTrue(args.compile)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume_checkpoint)

    def test_entry_rejects_protocol_overrides_and_no_compile(self):
        for flags in (
            ["--no-compile"],
            ["--window-size", "64"],
            ["--epochs", "65"],
            ["--lr", "0.01"],
            ["--lr-scheduler", "step"],
            ["--init-checkpoint", "old.pt"],
        ):
            with self.assertRaises(ValueError):
                training_arguments(flags)

    def test_forward_backward_and_face_independence(self):
        model = HandCTRWristFusion(num_classes=7)
        x = self.sample()
        frame_mask = torch.ones(x.size(0), x.size(2), dtype=torch.bool)
        logits = model(x, frame_mask)
        self.assertEqual(tuple(logits.shape), (x.size(0), 7))
        self.assertTrue(torch.isfinite(logits).all())
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
        for name in ("hand_blocks", "head_encoder", "hand_projection", "head_projection", "body_blocks", "classifier"):
            self.assertTrue(any(p.grad is not None for n, p in model.named_parameters() if n.startswith(name)), name)

        altered = x.clone()
        altered[:, :, :, 23:91] = torch.randn_like(altered[:, :, :, 23:91]) * 100
        with torch.no_grad():
            torch.testing.assert_close(model(x, frame_mask), model(altered, frame_mask), rtol=0, atol=0)

    def test_missing_hand_and_head_are_finite_and_zero_injection(self):
        model = HandCTRWristFusion(num_classes=7).eval()
        x = self.sample()
        empty = x.clone()
        empty[:, 2, :, 1:5] = 0
        empty[:, 2, :, 91:133] = 0
        empty[:, 0:2, :, 91:133] = float("nan")
        empty[:, 0:2, :, 1:5] = float("nan")
        with torch.no_grad():
            logits = model(empty, torch.ones(x.size(0), x.size(2), dtype=torch.bool))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
