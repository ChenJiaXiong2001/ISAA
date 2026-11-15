import unittest

import torch

from inference import arithmetic_geometric_fusion


class GeometricInferenceTest(unittest.TestCase):
    def test_normalized_and_finite(self):
        torch.manual_seed(3)
        hf = torch.randn(4, 60)
        bf = torch.randn(4, 60)
        weights = torch.linspace(0.0, 1.0, 60)
        result = arithmetic_geometric_fusion(hf, bf, weights, 0.5)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.allclose(result.exp().sum(-1), torch.ones(4), atol=1e-6))

    def test_endpoints_match_pool(self):
        hf = torch.randn(2, 5)
        bf = torch.randn(2, 5)
        weights = torch.full((5,), 0.5)
        for strength in (0.0, 1.0):
            result = arithmetic_geometric_fusion(hf, bf, weights, strength)
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(torch.allclose(result.exp().sum(-1), torch.ones(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
