import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_precompression_anchor_trial import compare


class ComparePrecompressionAnchorTrialTests(unittest.TestCase):
    def test_paired_accuracy_and_robustness_differences(self):
        names = [
            "S001C001P001R001A001_rgb.npz",
            "S001C002P001R001A002_rgb.npz",
            "S001C001P002R001A001_rgb.npz",
            "S001C002P002R001A002_rgb.npz",
        ]
        members_sha = hashlib.sha256("\n".join(names).encode()).hexdigest()

        def write_eval(path: Path, clean, corrupt):
            path.mkdir()
            summary = {
                "samples": len(names),
                "members_sha256": members_sha,
                "checkpoint_sha256": "checkpoint",
                "input_sha256": {"clean": "same-clean", "corrupted": "same-mask"},
            }
            (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with (path / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "sample", "label", "clean_unmodified_prediction", "pre_dropout_unmodified_prediction"
                ])
                writer.writeheader()
                for name, clean_pred, corrupt_pred in zip(names, clean, corrupt):
                    writer.writerow({"sample": name, "label": 0,
                                     "clean_unmodified_prediction": clean_pred,
                                     "pre_dropout_unmodified_prediction": corrupt_pred})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_clean = [0, 0, 1, 0]
            control_clean = [0, 1, 0, 0]
            candidate_corrupt = [0, 1, 0, 0]
            control_corrupt = [1, 1, 0, 1]
            paths = {}
            for probability in ("p05", "p10"):
                candidate = root / f"candidate_{probability}"
                control = root / f"control_{probability}"
                write_eval(candidate, candidate_clean, candidate_corrupt)
                write_eval(control, control_clean, control_corrupt)
                paths[probability] = (candidate, control)

            result = compare(
                {p: paths[p][0] for p in paths},
                {p: paths[p][1] for p in paths},
                repeats=100,
                seed=1,
            )

        self.assertEqual(result["conditions"]["clean"]["candidate_minus_control_pp"], 0.0)
        self.assertEqual(result["conditions"]["p05"]["candidate_minus_control_corrupted_pp"], 50.0)
        self.assertEqual(result["conditions"]["p05"]["robustness_gain_difference_in_degradation_pp"], 50.0)


if __name__ == "__main__":
    unittest.main()
