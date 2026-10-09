"""Argument-level contracts for the implemented single-change stages."""

import sys
import unittest
from unittest.mock import patch

from isaa.train import parse_args


class AblationStageArgumentTests(unittest.TestCase):
    def test_s00_forces_original_raw_25_node_contract(self):
        with patch.object(sys, "argv", ["main.py", "--ablation-stage", "s00_original25"]):
            args = parse_args()
        self.assertEqual(args.ablation_stage, "s00_original25")
        self.assertEqual(args.model_variant, "original")
        self.assertEqual(args.feature_mode, "raw")
        self.assertEqual(args.node_count, 25)
        self.assertFalse(args.main_only)

    def test_s01_is_the_same_contract_with_only_32_nodes(self):
        with patch.object(sys, "argv", ["main.py", "--ablation-stage", "s01_original32"]):
            args = parse_args()
        self.assertEqual(args.ablation_stage, "s01_original32")
        self.assertEqual((args.model_variant, args.feature_mode, args.node_count),
                         ("original", "raw", 32))
        self.assertFalse(args.main_only)

    def test_stage_rejects_conflicting_feature_or_node_settings(self):
        with patch.object(sys, "argv", ["main.py", "--ablation-stage", "s00_original25",
                                         "--feature-mode", "isaa"]):
            with self.assertRaises(SystemExit):
                parse_args()
        with patch.object(sys, "argv", ["main.py", "--ablation-stage", "s01_original32",
                                         "--node-count", "25"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_wide_relative_routed_variant_is_registered(self):
        with patch.object(sys, "argv", ["main.py", "--model-variant",
                                         "body-local-hand-ctr-wide-relative-routed"]):
            args = parse_args()
        self.assertEqual(args.model_variant,
                         "body-local-hand-ctr-wide-relative-routed")


if __name__ == "__main__":
    unittest.main()
