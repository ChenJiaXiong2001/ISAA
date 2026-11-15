"""Tests for the fixed normalized average-rank fusion candidate."""
import unittest

import torch

from tools.analyze_face_pair_rank_fusion_subject_cv import normalized_average_rank_scores


class RankFusionTests(unittest.TestCase):
    def test_descending_average_rank_and_normalization(self):
        logits = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
        scores, ties = normalized_average_rank_scores(logits)
        torch.testing.assert_close(scores, torch.tensor([[4., 2., 3., 1.]]) / 10)
        self.assertEqual(ties, 0)

    def test_exact_ties_get_average_rank(self):
        logits = torch.tensor([[2.0, 2.0, 0.0]])
        scores, ties = normalized_average_rank_scores(logits)
        torch.testing.assert_close(scores, torch.tensor([[2.5, 2.5, 1.]]) / 6)
        self.assertEqual(ties, 1)

    def test_rejects_nonfinite_logits(self):
        with self.assertRaises(ValueError):
            normalized_average_rank_scores(torch.tensor([[float("nan"), 0.0]]))


if __name__ == "__main__":
    unittest.main()
