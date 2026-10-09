"""Smoke tests for the stage/quality-gated wide hand CTR model."""

import unittest

try:
    import torch
    from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeRoutedFusion
except ImportError:  # pragma: no cover - environments without PyTorch
    torch = None
    BodyLocalHandCTRWideRelativeRoutedFusion = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WideRelativeRoutedModelTests(unittest.TestCase):
    def test_forward_and_routing_outputs(self):
        torch.manual_seed(1)
        model = BodyLocalHandCTRWideRelativeRoutedFusion(num_classes=7)
        model.eval()
        x = torch.zeros(2, 8, 16, 133, 1)
        x[:, 0:2] = torch.randn(2, 2, 16, 133, 1) * 0.01
        x[:, 2] = 1.0
        x[:, 5] = 0.25
        x[:, 6] = 1.0
        with torch.no_grad():
            logits = model(x)
            result = model(x, return_routing=True)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(tuple(result["logits"].shape), (2, 7))
        self.assertEqual(result["hand_gate"].shape[0], 2)
        self.assertEqual(result["face_gate"].shape[0], 2)
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
