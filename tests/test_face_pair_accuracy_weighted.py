"""Regression checks for frozen calibration-accuracy soft fusion."""
import unittest

import torch
from torch import nn

from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_RAW_XY_VARIANT, FACE_PAIR_VARIANT,
    FacePairFusion, FacePairHandFaceGeometryFusion, FacePairRawXYFusion,
)
from tools.evaluate_face_pair_accuracy_weighted import SoftAccuracyReadout, accuracy_weights, model_type


class FixedHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("hand_face_temperature", torch.tensor(1.0))
        self.register_buffer("body_face_temperature", torch.tensor(1.0))

    def forward(self, features, frame_mask=None):
        return {"hand_face_logits": torch.log(torch.tensor([[0.8, 0.2]])),
                "body_face_logits": torch.log(torch.tensor([[0.1, 0.9]]))}


class AccuracyWeightedTests(unittest.TestCase):
    def test_model_variants(self):
        self.assertIs(model_type(FACE_PAIR_VARIANT), FacePairFusion)
        self.assertIs(model_type(FACE_PAIR_RAW_XY_VARIANT), FacePairRawXYFusion)
        self.assertIs(model_type(FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT), FacePairHandFaceGeometryFusion)
        with self.assertRaises(RuntimeError):
            model_type("unknown")

    def test_calibration_weights_and_probability_fusion(self):
        hf_acc, bf_acc, hf_weight, bf_weight = accuracy_weights(
            torch.tensor([3, 0, 0]), torch.tensor([1, 0, 2]), torch.tensor([4, 0, 2]))
        torch.testing.assert_close(hf_acc, torch.tensor([0.75, 0., 0.]))
        torch.testing.assert_close(bf_acc, torch.tensor([0.25, 0., 1.]))
        torch.testing.assert_close(hf_weight, torch.tensor([0.75, 0.5, 0.]))
        torch.testing.assert_close(bf_weight, 1 - hf_weight)
        with self.assertRaises(RuntimeError):
            accuracy_weights(torch.tensor([5]), torch.tensor([0]), torch.tensor([4]))

        readout = SoftAccuracyReadout(FixedHeads(), torch.tensor([0.75, 0.25]))
        actual = readout(None)["soft_logits"].exp()
        unnormalized = torch.tensor([[0.75 * 0.8 + 0.25 * 0.1,
                                      0.25 * 0.2 + 0.75 * 0.9]])
        torch.testing.assert_close(actual, unnormalized / unnormalized.sum(-1, keepdim=True))
        self.assertFalse(torch.allclose(actual, torch.tensor([[0.8, 0.2]])))
        self.assertFalse(torch.allclose(actual, torch.tensor([[0.1, 0.9]])))


if __name__ == "__main__":
    unittest.main()
