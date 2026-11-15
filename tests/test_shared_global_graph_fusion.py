"""Regression tests for the regional sparse shared-global-graph variant."""
import io
import sys
import unittest
from unittest.mock import patch

import torch

from isaa.models.shared_global_graph_fusion import (
    SHARED_GLOBAL_GRAPH_VARIANT,
    SharedGlobalGraphFusion,
)
from isaa.train import parse_args
from train_best_shared_global_graph_100 import training_arguments


class SharedGlobalGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def sample(persons=2):
        x = torch.randn(1, 3, 100, 133, persons) * 0.05
        x[:, 2] = 1
        return x

    def test_single_sparse_graph_keeps_regional_nodes_and_cross_edges(self):
        model = SharedGlobalGraphFusion(7, window_size=100, max_persons=2)
        self.assertEqual(model.GLOBAL_NODE_COUNT, 70)
        self.assertEqual(model.global_graph_config["edge_count"], 210)
        self.assertEqual(model.global_graph.source.numel(), 210)
        self.assertFalse(hasattr(model, "global_adjacency"))
        self.assertEqual(model.region_slices["body"], (0, 22))
        self.assertEqual(model.region_slices["left_hand"], (22, 43))
        self.assertEqual(model.region_slices["right_hand"], (43, 64))
        self.assertEqual(model.region_slices["face"], (64, 70))

    def test_forward_backward_and_missing_person_mask(self):
        model = SharedGlobalGraphFusion(7, window_size=100, max_persons=2)
        x = self.sample()
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        for parameter in (
            model.global_scale,
            model.global_graph.mix[0].weight,
            model.body_to_global.weight,
            model.global_to_body.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

        one = self.sample(persons=1)
        padded = torch.cat((one, torch.zeros_like(one)), dim=-1)
        corrupt = padded.clone()
        corrupt[:, :2, :, :, 1] = 1000
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(padded), model(corrupt), rtol=1e-5, atol=1e-6)

    def test_preset_and_generic_parser_require_compile_and_scratch(self):
        flags = training_arguments([])
        with patch.object(sys, "argv", ["main.py"] + flags):
            args = parse_args()
        self.assertEqual(args.model_variant, SHARED_GLOBAL_GRAPH_VARIANT)
        self.assertEqual((args.window_size, args.epochs), (100, 100))
        self.assertEqual((args.temporal_compression, args.lr_scheduler),
                         ("confidence_weighted", "warmup_cosine"))
        self.assertTrue(args.compile)
        for forbidden in (["--no-compile"], ["--init-checkpoint", "old.pt"],
                           ["--resume-checkpoint", "old.pt"]):
            with self.assertRaises(ValueError):
                training_arguments(forbidden)
        with patch.object(sys, "argv", ["main.py", "--model-variant",
                                         SHARED_GLOBAL_GRAPH_VARIANT, "--no-compile"]), \
                patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parse_args()


if __name__ == "__main__":
    unittest.main()
