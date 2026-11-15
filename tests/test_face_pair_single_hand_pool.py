"""Boundary and frozen-model checks for the isolated single-hand experiment."""
import unittest

import torch

from isaa.models.face_pair_fusion import FacePairFusion
from tools.evaluate_face_pair_single_hand_pool import hand_visibility_at_pool, occlude_raw_hand
from tools.face_pair_single_hand_experiment import ValidHandMeanFacePair, valid_hand_mean


class SingleHandPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_masked_mean_cases(self):
        left = torch.tensor([[[2., 4.], [2., 4.], [2., 4.], [2., 4.]]])
        right = torch.tensor([[[6., 8.], [6., 8.], [6., 8.], [6., 8.]]])
        result, valid = valid_hand_mean(left, torch.tensor([[True, True, False, False]]),
                                       right, torch.tensor([[True, False, True, False]]))
        torch.testing.assert_close(result, torch.tensor([[[4., 6.], [2., 4.], [6., 8.], [0., 0.]]]))
        self.assertEqual(valid.tolist(), [[True, True, True, False]])

    def test_raw_occlusion_is_deterministic_scoped_and_preserves_source(self):
        raw = torch.ones(3, 30, 133, 2)
        first = occlude_raw_hand(raw, "sample_A", "left", 5)
        torch.testing.assert_close(first, occlude_raw_hand(raw, "sample_A", "left", 5))
        self.assertEqual(int((first[2, :, 91:112, 0] == 0).sum()), 5 * 21)
        self.assertTrue(bool((first[:, :, :91] == 1).all()))
        self.assertTrue(bool((first[:, :, 112:] == 1).all()))
        self.assertTrue(bool((first[:, :, :, 1] == 1).all()))
        self.assertTrue(bool((raw == 1).all()))
        all_right = occlude_raw_hand(raw, "sample_A", "right", 0)
        self.assertTrue(bool((all_right[:, :, 112:133, 0] == 0).all()))
        self.assertTrue(bool((all_right[:, :, 91:112, 0] == 1).all()))

    def test_pool_visibility_preserves_stride_one_mask_and_excludes_empty_people(self):
        x = torch.zeros(1, 3, 4, 133, 2)
        x[:, 2, :, 5, 0] = 1
        x[:, 2, (0, 1), 91, 0] = 1
        x[:, 2, (0, 2), 112, 0] = 1
        counts = hand_visibility_at_pool(x, torch.ones(1, 4, dtype=torch.bool), [1, 1, 1, 1])
        self.assertEqual({key: value.tolist() for key, value in counts.items()},
                         {"both": [1], "left_only": [1], "right_only": [1], "neither": [1]})

    def test_pool_visibility_stride_two_uses_same_mask_binning(self):
        x = torch.zeros(1, 3, 4, 133, 1)
        x[:, 2, :, 5, 0] = 1
        x[:, 2, 0, 91, 0] = 1
        x[:, 2, 1, 112, 0] = 1
        counts = hand_visibility_at_pool(x, torch.ones(1, 4, dtype=torch.bool), [2])
        self.assertEqual({key: value.tolist() for key, value in counts.items()},
                         {"both": [1], "left_only": [0], "right_only": [0], "neither": [1]})

    def test_model_matches_baseline_when_both_hands_valid(self):
        torch.manual_seed(19)
        base = FacePairFusion(4).eval()
        candidate = ValidHandMeanFacePair(4).eval()
        candidate.load_state_dict(base.state_dict(), strict=True)
        x = torch.rand(1, 3, 8, 133, 1)
        x[:, 2] = 1
        with torch.inference_mode():
            original, changed = base(x), candidate(x)
        for key in original:
            torch.testing.assert_close(original[key], changed[key], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original["hand_face_logits"], changed["original_hand_face_logits"],
                                   atol=1e-6, rtol=1e-6)

    def test_eight_channel_input_matches_when_both_hands_valid(self):
        torch.manual_seed(21)
        base = FacePairFusion(4).eval()
        candidate = ValidHandMeanFacePair(4).eval()
        candidate.load_state_dict(base.state_dict(), strict=True)
        x = torch.rand(1, 8, 8, 133, 1)
        x[:, 2] = 1
        with torch.inference_mode():
            original, changed = base(x), candidate(x)
        for key in original:
            torch.testing.assert_close(original[key], changed[key], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original["hand_face_logits"], changed["original_hand_face_logits"],
                                   atol=1e-6, rtol=1e-6)

    def test_model_single_hand_keeps_bf_and_is_finite(self):
        torch.manual_seed(20)
        base = FacePairFusion(4).eval()
        candidate = ValidHandMeanFacePair(4).eval()
        candidate.load_state_dict(base.state_dict(), strict=True)
        x = torch.rand(1, 3, 8, 133, 1)
        x[:, 2] = 1
        x[:, :, :, 91:112] = 0
        with torch.inference_mode():
            original, changed = base(x), candidate(x)
        torch.testing.assert_close(original["body_face_logits"], changed["body_face_logits"], atol=0, rtol=0)
        torch.testing.assert_close(original["hand_face_logits"], changed["original_hand_face_logits"], atol=0, rtol=0)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in changed.values()))
        self.assertGreater(float((original["hand_face_logits"] - changed["hand_face_logits"]).abs().max()), 0)


if __name__ == "__main__":
    unittest.main()
