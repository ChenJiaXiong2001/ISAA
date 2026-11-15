"""Check the fixed synthetic gap protocol and bounded anchor interpolation."""
import unittest

import torch

from tools.evaluate_face_pair_anchor_gaps import contiguous_hidden, interpolate_short_anchor_gaps
from tools.evaluate_face_pair_anchor_interpolation import interpolate_isolated_anchors


class AnchorGapInterpolationTests(unittest.TestCase):
    def test_gap_starts_expand_forward(self):
        draws = torch.ones(1, 8, 1, 1)
        draws[:, 2] = 0
        actual = contiguous_hidden(draws, torch.ones(1, 8, dtype=torch.bool), 3, .5)
        self.assertEqual(actual.flatten().tolist(), [False, False, True, True, True, False, False, False])

    def test_bounded_interpolation_and_invalid_frame_barrier(self):
        features = torch.zeros(1, 3, 10, 133, 1)
        features[:, 0] = torch.arange(10).view(1, 10, 1, 1)
        features[:, 1] = 2 * features[:, 0]
        features[:, 2] = 1
        features[:, :, 2, 5] = 0
        features[:, :, 4:6, 6] = 0
        features[:, :, 0, 11] = 0
        features[:, :, 6:9, 12] = 0
        mask = torch.ones(1, 10, dtype=torch.bool)
        old, count = interpolate_isolated_anchors(features, mask)
        single, single_count = interpolate_short_anchor_gaps(features, mask, 1)
        torch.testing.assert_close(single, old)
        self.assertEqual((count, single_count), (1, 1))
        two, two_count = interpolate_short_anchor_gaps(features, mask, 2)
        self.assertEqual(two_count, 3)
        self.assertEqual((float(two[0, 0, 4, 6, 0]), float(two[0, 0, 5, 6, 0])), (4., 5.))
        five, five_count = interpolate_short_anchor_gaps(features, mask, 5)
        self.assertEqual(five_count, 6)
        self.assertEqual(float(five[0, 2, 0, 11, 0]), 0.)
        mask[:, 5] = False
        barrier, _ = interpolate_short_anchor_gaps(features, mask, 5)
        self.assertEqual(float(barrier[0, 2, 4, 6, 0]), 0.)
        self.assertEqual(float(barrier[0, 2, 5, 6, 0]), 0.)


if __name__ == "__main__":
    unittest.main()
