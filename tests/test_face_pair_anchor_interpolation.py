"""Checks for the isolated-frame body-anchor evaluation rule."""
import unittest

import torch

from tools.evaluate_face_pair_anchor_interpolation import interpolate_isolated_anchors


class AnchorInterpolationTests(unittest.TestCase):
    def test_fills_only_single_frame_anchor_gap_without_mutating_input(self):
        raw = torch.zeros(1, 3, 5, 133, 1)
        raw[0, 0, 1, 5, 0], raw[0, 0, 3, 5, 0] = 2.0, 6.0
        raw[0, 2, 1, 5, 0], raw[0, 2, 3, 5, 0] = 0.8, 0.4
        original = raw.clone()
        repaired, count = interpolate_isolated_anchors(raw, torch.ones(1, 5, dtype=torch.bool))
        self.assertEqual(count, 1)
        self.assertEqual(float(repaired[0, 0, 2, 5, 0]), 4.0)
        self.assertAlmostEqual(float(repaired[0, 2, 2, 5, 0]), 0.6, places=6)
        torch.testing.assert_close(raw, original)
        self.assertTrue(torch.equal(repaired[:, :, :, 6:, :], raw[:, :, :, 6:, :]))

    def test_does_not_fill_contiguous_gaps_or_invalid_frames(self):
        raw = torch.zeros(1, 3, 6, 133, 1)
        raw[:, 2, 1, 5] = 1.0
        raw[:, 2, 4, 5] = 1.0
        repaired, count = interpolate_isolated_anchors(raw, torch.ones(1, 6, dtype=torch.bool))
        self.assertEqual(count, 0)
        torch.testing.assert_close(repaired, raw)

        raw[:, 2, 3, 5] = 1.0
        frame_mask = torch.ones(1, 6, dtype=torch.bool)
        frame_mask[:, 2] = False
        repaired, count = interpolate_isolated_anchors(raw, frame_mask)
        self.assertEqual(count, 0)
        torch.testing.assert_close(repaired, raw)


if __name__ == "__main__":
    unittest.main()
