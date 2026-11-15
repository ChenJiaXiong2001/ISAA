"""Numerical checks for same-batch geometric-consensus inference."""
import unittest

import torch
from torch.nn import functional as F

from evaluate_cuda import fuse
from tools.evaluate_face_pair_corruption import _soft_logits


class FusionNumerics(unittest.TestCase):
    def test_matches_reference_probability_formula(self):
        hf = torch.tensor([[1., -1., 0.5], [-0.5, 1.5, 0.]])
        bf = torch.tensor([[0., 1., -0.5], [1., -0.5, 0.5]])
        w = torch.tensor([0., 0.4, 1.])
        base, candidate = fuse(hf, bf, w)
        h, b = hf.softmax(-1), bf.softmax(-1)
        a = w * h + (1 - w) * b
        a = a / a.sum(-1, keepdim=True)
        g = torch.exp(w * h.log() + (1 - w) * b.log())
        g = g / g.sum(-1, keepdim=True)
        self.assertTrue(torch.allclose(base.exp(), a, atol=1e-7))
        self.assertTrue(torch.allclose(candidate.exp(), 0.5 * (a + g), atol=1e-7))
        old = _soft_logits({'hand_face_logits': hf, 'body_face_logits': bf}, w)
        self.assertTrue(torch.allclose(base, old, atol=1e-6))

    def test_extreme_finite_logits_and_exact_zero_one_weights(self):
        hf = torch.tensor([[10000., -10000., 0.], [-10000., 10000., 0.]])
        bf = -hf
        w = torch.tensor([0., 1., 0.5])
        base, candidate = fuse(hf, bf, w)
        self.assertTrue(torch.isfinite(base).all())
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertTrue(torch.allclose(base.exp().sum(-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(candidate.exp().sum(-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.isfinite(F.nll_loss(candidate, torch.tensor([0, 1]))))


if __name__ == '__main__':
    unittest.main()
