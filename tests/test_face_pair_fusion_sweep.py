"""Focused numerical and calibration tests for face-pair fusion sweeps."""
import math
import unittest

import isaa.train  # Match the evaluation entry points' CUDA library setup.
import torch

from tools.face_pair_fusion_sweep import (
    CONFIDENCE_MARGIN,
    CONFIDENCE_MAXPROB,
    CONFIDENCE_NORMALIZED_ENTROPY,
    CONFIDENCE_UNIFORM,
    LogitFusion,
    class_weights_from_counts,
    fit_temperature,
    temperature_nll,
    validate_strategy_tensors,
)


class FacePairFusionSweepTests(unittest.TestCase):
    @staticmethod
    def _inputs(device="cpu"):
        hf = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.25]], device=device)
        bf = torch.tensor([[-1.0, 1.0, 2.0], [2.5, -0.25, 0.0]], device=device)
        return hf, bf

    def test_accuracy_power_weights_cover_gamma_zero_one_hard_and_zero_counts(self):
        hf = torch.tensor([8, 0, 4, 0, 3])
        bf = torch.tensor([2, 7, 4, 0, 0])
        total = torch.tensor([10, 7, 8, 0, 3])
        weights = class_weights_from_counts(hf, bf, total, torch.tensor([0.0, 1.0, float("inf")]))
        torch.testing.assert_close(weights[0], torch.full((5,), 0.5))
        torch.testing.assert_close(weights[1], torch.tensor([.8, 0., .5, .5, 1.]))
        torch.testing.assert_close(weights[2], torch.tensor([1., 0., .5, .5, 1.]))
        self.assertTrue(torch.isfinite(weights).all())

    def test_dynamic_probability_matches_independent_arithmetic_reference(self):
        hf, bf = self._inputs()
        weights = torch.tensor([[.8, .2, .5]])
        temperatures = torch.tensor([[1.5, .75]])
        beta = torch.tensor([2.0])
        mode = torch.tensor([CONFIDENCE_MAXPROB], dtype=torch.long)
        validate_strategy_tensors(weights, temperatures, beta, mode)
        actual = LogitFusion()(hf, bf, weights, temperatures, beta, mode).exp()[0]

        hp = (hf / 1.5).softmax(-1)
        bp = (bf / .75).softmax(-1)
        hrel = hp.max(-1).values
        brel = bp.max(-1).values
        hcontrib = weights[0][None] * hrel[:, None].pow(2)
        bcontrib = (1 - weights[0])[None] * brel[:, None].pow(2)
        expected = (hcontrib * hp + bcontrib * bp) / (hcontrib + bcontrib)
        expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.isfinite(actual).all())

    def test_uniform_mode_and_zero_beta_reduce_to_static_class_prior(self):
        hf, bf = self._inputs()
        weights = torch.tensor([[.8, .2, .5], [.1, .9, .3]])
        temperatures = torch.tensor([[1.5, .75], [.6, 2.0]])
        static = LogitFusion()(hf, bf, weights, temperatures,
                               torch.tensor([0.0, 0.0]),
                               torch.tensor([CONFIDENCE_UNIFORM, CONFIDENCE_UNIFORM]))
        uniform_mode = LogitFusion()(hf, bf, weights, temperatures,
                                     torch.tensor([4.0, 2.0]),
                                     torch.tensor([CONFIDENCE_UNIFORM, CONFIDENCE_UNIFORM]))
        dynamic_mode = LogitFusion()(hf, bf, weights, temperatures,
                                     torch.tensor([4.0, 2.0]),
                                     torch.tensor([CONFIDENCE_MAXPROB, CONFIDENCE_MARGIN]))
        beta_zero = LogitFusion()(hf, bf, weights, temperatures,
                                  torch.tensor([0.0, 0.0]),
                                  torch.tensor([CONFIDENCE_MAXPROB, CONFIDENCE_NORMALIZED_ENTROPY]))
        torch.testing.assert_close(static, beta_zero, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(static, uniform_mode, atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(static, dynamic_mode))

    def test_hard_endpoint_priors_are_invariant_to_sample_reliability(self):
        hf, bf = self._inputs()
        weights = torch.tensor([[1.0, 0.0, 1.0]])
        temperatures = torch.tensor([[1.2, .8]])
        static = LogitFusion()(hf, bf, weights, temperatures, torch.tensor([0.0]),
                               torch.tensor([CONFIDENCE_UNIFORM]))
        dynamic = LogitFusion()(hf, bf, weights, temperatures, torch.tensor([8.0]),
                                torch.tensor([CONFIDENCE_NORMALIZED_ENTROPY]))
        torch.testing.assert_close(static, dynamic, atol=1e-6, rtol=1e-6)

    def test_hard_routing_ties_still_allow_dynamic_head_weighting(self):
        hf = torch.tensor([[4., 0.]])
        bf = torch.tensor([[0., 1.]])
        weights = class_weights_from_counts([2, 2], [2, 2], [4, 4], [float("inf")])
        temperatures = torch.ones(1, 2)
        static = LogitFusion()(hf, bf, weights, temperatures, torch.tensor([0.]),
                               torch.tensor([CONFIDENCE_UNIFORM]))
        dynamic = LogitFusion()(hf, bf, weights, temperatures, torch.tensor([2.]),
                                torch.tensor([CONFIDENCE_MAXPROB]))
        self.assertFalse(torch.allclose(static, dynamic))
        self.assertGreater(float(dynamic.exp()[0, 0, 0]), float(static.exp()[0, 0, 0]))

    def test_uniform_reliabilities_stay_finite_and_normalized(self):
        hf = torch.zeros(3, 4)
        bf = torch.zeros(3, 4)
        weights = torch.tensor([[.5, .2, .8, .5]] * 3)
        temperatures = torch.ones(3, 2)
        beta = torch.tensor([.5, 1.0, 2.0])
        for mode in (CONFIDENCE_MAXPROB, CONFIDENCE_MARGIN, CONFIDENCE_NORMALIZED_ENTROPY):
            output = LogitFusion()(hf, bf, weights, temperatures, beta,
                                   torch.tensor([mode] * 3))
            probabilities = output.exp()
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(3, 3), atol=1e-6, rtol=1e-6)

    def test_swapping_heads_and_priors_preserves_fused_distribution(self):
        hf, bf = self._inputs()
        weights = torch.tensor([[.8, .2, .5], [.3, .7, .1]])
        temperatures = torch.tensor([[1.5, .75], [.6, 2.0]])
        beta = torch.tensor([2.0, 1.0])
        modes = torch.tensor([CONFIDENCE_MAXPROB, CONFIDENCE_MARGIN])
        fusion = LogitFusion()
        original = fusion(hf, bf, weights, temperatures, beta, modes)
        swapped = fusion(bf, hf, 1 - weights, temperatures[:, [1, 0]], beta, modes)
        torch.testing.assert_close(original, swapped, atol=2e-6, rtol=2e-6)

    def test_temperature_fit_improves_or_matches_nll_without_changing_head_argmax(self):
        logits = torch.tensor([[10.0, -10.0], [10.0, -10.0], [-10.0, 10.0], [10.0, -10.0]])
        labels = torch.tensor([0, 1, 1, 0])
        before = temperature_nll(logits, labels, 1.0)
        fitted = fit_temperature(logits, labels)
        after = temperature_nll(logits, labels, fitted)
        self.assertTrue(math.isfinite(fitted) and fitted > 0)
        self.assertLessEqual(after, before + 1e-10)
        self.assertTrue(torch.equal(logits.argmax(-1), (logits / fitted).argmax(-1)))
        self.assertTrue(math.isfinite(temperature_nll(logits, labels, .1)))
        self.assertTrue(math.isfinite(temperature_nll(logits, labels, 10.0)))

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "compile"),
                         "CUDA and torch.compile are required for this integration check")
    def test_cuda_inductor_fullgraph_matches_eager(self):
        device = torch.device("cuda")
        hf, bf = self._inputs(device)
        weights = torch.tensor([[.8, .2, .5], [.3, .7, .1]], device=device)
        temperatures = torch.tensor([[1.5, .75], [.6, 2.0]], device=device)
        beta = torch.tensor([2.0, 1.0], device=device)
        modes = torch.tensor([CONFIDENCE_MAXPROB, CONFIDENCE_NORMALIZED_ENTROPY], device=device)
        validate_strategy_tensors(weights, temperatures, beta, modes)
        model = LogitFusion().to(device)
        eager = model(hf, bf, weights, temperatures, beta, modes)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        actual = compiled(hf, bf, weights, temperatures, beta, modes)
        torch.testing.assert_close(actual, eager, atol=3e-4, rtol=3e-4)
        hp = (hf.double()[None] / temperatures[:, 0, None, None].double()).softmax(-1)
        bp = (bf.double()[None] / temperatures[:, 1, None, None].double()).softmax(-1)
        hrel = torch.stack((hp[0].max(-1).values,
                            1 + (hp[1] * hp[1].log()).sum(-1) / math.log(hp.shape[-1])))
        brel = torch.stack((bp[0].max(-1).values,
                            1 + (bp[1] * bp[1].log()).sum(-1) / math.log(bp.shape[-1])))
        hcontrib = weights.double()[:, None] * hrel[:, :, None].pow(beta[:, None, None])
        bcontrib = (1 - weights.double())[:, None] * brel[:, :, None].pow(beta[:, None, None])
        direct = (hcontrib * hp + bcontrib * bp) / (hcontrib + bcontrib)
        direct = direct / direct.sum(-1, keepdim=True)
        torch.testing.assert_close(actual.exp().double(), direct, atol=3e-4, rtol=3e-4)
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
