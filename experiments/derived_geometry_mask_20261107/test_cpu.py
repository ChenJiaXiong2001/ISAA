"""Minimal CPU shape and finite-output checks for the isolated variants."""
import unittest

import torch

from experiments.derived_geometry_mask_20261107.model import (
    MaskAwareCenterScaleFusion,
    MaskAwareGeometryOnlyFusion,
)


class MaskAwareGeometryTest(unittest.TestCase):
    def test_variants_produce_finite_logits(self):
        torch.manual_seed(7)
        features = torch.zeros(2, 3, 8, 133, 2)
        features[:, :2] = torch.rand(2, 2, 8, 133, 2)
        features[:, 2] = 1.0
        frame_mask = torch.ones(2, 8, dtype=torch.bool)
        for cls in (MaskAwareGeometryOnlyFusion, MaskAwareCenterScaleFusion):
            output = cls(60)(features, frame_mask)
            self.assertEqual(tuple(output["logits"].shape), (2, 60))
            self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_incomplete_torso_does_not_create_nonfinite_geometry(self):
        features = torch.zeros(1, 3, 6, 133, 1)
        features[:, :2] = 0.5
        features[:, 2] = 1.0
        features[:, 2, 2, (5, 6)] = 0.0
        output = MaskAwareGeometryOnlyFusion(60)(features, torch.ones(1, 6, dtype=torch.bool))
        self.assertTrue(torch.isfinite(output["logits"]).all())


if __name__ == "__main__":
    unittest.main()
