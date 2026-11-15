"""CPU checks for deterministic precompression masks and paired statistics."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from precompression_mask import PairedPrecompression
from summarize import summarize


class FakeDataset:
    members = ["S001C001P001R001A001", "S001C001P002R001A001"]
    labels = [1, 2]
    window_size = 4
    deterministic_temporal_crop = "center"

    def _read_raw_sample(self, index):
        raw = torch.ones(3, 8, 133, 2)
        raw[0] *= index + 1
        return raw, torch.ones(8, dtype=torch.bool)


class PrecompressionTests(unittest.TestCase):
    def test_determinism_clean_and_anchor_only(self):
        dataset = FakeDataset()
        clean, same, label, mask, hidden, length = PairedPrecompression(dataset, dataset.members, 0, 1729)[0]
        self.assertTrue(torch.equal(clean, same))
        self.assertEqual((label, hidden, length), (1, 0, 8))
        c1, drop1, _, mask1, _, _ = PairedPrecompression(dataset, dataset.members, 0.05, 1729)[0]
        c2, drop2, _, mask2, _, _ = PairedPrecompression(dataset, dataset.members, 0.05, 1729)[0]
        self.assertTrue(all(torch.equal(a, b) for a, b in ((clean, c1), (c1, c2), (drop1, drop2), (mask, mask1), (mask1, mask2))))
        _, all_dropped, _, _, hidden, _ = PairedPrecompression(dataset, dataset.members, 1, 1729)[0]
        self.assertEqual(hidden, 8 * 4 * 2)
        self.assertEqual(int(all_dropped[:, :, [5, 6, 11, 12], :].count_nonzero()), 0)
        self.assertTrue(torch.equal(all_dropped[:, :, 0, :], clean[:, :, 0, :]))

    def test_three_condition_subject_statistics(self):
        rows = []
        for condition in ("clean", "anchor_pre_0.05", "anchor_pre_0.1"):
            for index, name in enumerate(FakeDataset.members):
                rows.append({"condition": condition, "sample": name, "true_class_index": "0",
                             "control_correct": "0", "candidate_correct": str(index)})
        result = summarize(rows, {FakeDataset.members[0]}, "all")
        self.assertEqual((result["samples"], result["subjects"]), (2, 2))
        self.assertEqual(result["conditions"]["clean"]["candidate_minus_control"]["difference_pp"], 50.)
        rows[2]["true_class_index"] = "1"
        with self.assertRaisesRegex(ValueError, "Labels differ"):
            summarize(rows, {FakeDataset.members[0]}, "all")


if __name__ == "__main__":
    unittest.main()
