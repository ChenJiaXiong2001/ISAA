"""Numerical regression tests for the classwise HF/BF probability readout."""
import unittest

import torch
from torch.nn import functional as F

from isaa.face_pair_training import pair_loss
from isaa.models.face_pair_fusion import FacePairFusion


class FacePairNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_log_domain_matches_normal_probability_fusion_and_gradients(self):
        torch.manual_seed(27)
        model = FacePairFusion(6)
        model.set_routing_counts([3, 1, 2, 0, 4, 1], [1, 3, 2, 0, 2, 4], [4] * 6)
        model.set_temperatures(.8, 1.7)
        hf = (torch.randn(7, 6) * 2).requires_grad_()
        bf = (torch.randn(7, 6) * 2).requires_grad_()
        labels = torch.arange(7) % 6

        actual = model.fuse_logits(hf, bf)
        weights = model.hand_face_weight.double()
        probability = (weights * (hf.double() / model.hand_face_temperature.double()).softmax(-1)
                       + (1 - weights) * (bf.double() / model.body_face_temperature.double()).softmax(-1))
        reference = (probability / probability.sum(-1, keepdim=True)).log()
        torch.testing.assert_close(actual, reference.float(), atol=1e-6, rtol=1e-6)
        actual_grad = torch.autograd.grad(F.cross_entropy(actual, labels), (hf, bf), retain_graph=True)
        reference_grad = torch.autograd.grad(F.cross_entropy(reference, labels), (hf, bf))
        for found, expected in zip(actual_grad, reference_grad):
            torch.testing.assert_close(found, expected, atol=1e-6, rtol=1e-5)

    def test_hard_routing_retains_extreme_classes_and_finite_gradients(self):
        model = FacePairFusion(2)
        model.set_routing_counts([1, 0], [0, 1], [1, 1])
        # Each class chooses the head assigning it vanishing probability. A
        # probability-domain implementation underflows both to zero here.
        for magnitude in (100., 10000., 1e20):
            with self.subTest(magnitude=magnitude):
                hf = torch.tensor([[-magnitude, magnitude]], requires_grad=True)
                bf = torch.tensor([[magnitude, -magnitude]], requires_grad=True)
                fused = model.fuse_logits(hf, bf)
                self.assertTrue(torch.isfinite(fused).all())
                torch.testing.assert_close(fused.exp(), torch.full((1, 2), .5), atol=1e-6, rtol=1e-6)
                loss = F.cross_entropy(fused, torch.tensor([0]))
                loss.backward()
                for head in (hf, bf):
                    self.assertTrue(torch.isfinite(head.grad).all())
                    self.assertGreater(float(head.grad.abs().sum()), 0.)

        model = FacePairFusion(4)
        model.set_routing_counts([1, 0, 1, 0], [0, 1, 1, 0], [1] * 4)
        hf = torch.tensor([[10000., -10000., 3000., -5000.]], requires_grad=True)
        bf = torch.tensor([[-10000., 10000., -3000., 5000.]], requires_grad=True)
        fused = model.fuse_logits(hf, bf)
        self.assertTrue(torch.isfinite(fused).all())
        torch.testing.assert_close(fused.exp().sum(-1), torch.ones(1))
        F.cross_entropy(fused, torch.tensor([2])).backward()
        for head in (hf, bf):
            self.assertTrue(torch.isfinite(head.grad).all())

    def test_extreme_unselected_head_keeps_auxiliary_supervision_gradient(self):
        model = FacePairFusion(2)
        labels = torch.tensor([1, 0])
        for hand_selected in (True, False):
            with self.subTest(hand_selected=hand_selected):
                model.set_routing_counts([1 if hand_selected else 0] * 2,
                                         [0 if hand_selected else 1] * 2, [1, 1])
                hf = torch.tensor([[10000., -10000.], [-10000., 10000.]], requires_grad=True)
                bf = hf.detach().clone().requires_grad_()
                fused = model.fuse_logits(hf, bf)
                fusion_grad = torch.autograd.grad(F.cross_entropy(fused, labels), (hf, bf), retain_graph=True)
                torch.testing.assert_close(fusion_grad[1 if hand_selected else 0], torch.zeros_like(hf))
                loss = pair_loss({"logits": fused, "hand_face_logits": hf, "body_face_logits": bf},
                                 labels, head_weight=.5, fusion_weight=1.)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for head in (hf, bf):
                    self.assertTrue(torch.isfinite(head.grad).all())
                    self.assertGreater(float(head.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
