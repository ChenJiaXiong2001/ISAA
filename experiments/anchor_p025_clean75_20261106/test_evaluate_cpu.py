"""Check candidate-specific comparison guards."""
import importlib.util
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "hand_motion_paired_p025_20ep_20261106"))
from test_evaluate_cpu import args, manifest

spec = importlib.util.spec_from_file_location("candidate_evaluate", Path(__file__).with_name("evaluate.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProtocolTests(unittest.TestCase):
    def test_both_controls_and_protocol_mismatch(self):
        candidate = {"args": args(True), "epoch": 20}
        candidate["args"]["precompression_anchor_clean_sample_probability"] = .75
        controls = [{"args": args(flag), "epoch": 20} for flag in (False, True)]
        m = manifest()
        for control in controls:
            self.assertEqual(module.validate_checkpoints(control, candidate, [m, m]),
                             ["validation1", "validation2"])
        candidate["args"]["weight_decay"] = .001
        with self.assertRaisesRegex(ValueError, "weight_decay"):
            module.validate_checkpoints(controls[0], candidate, [m, m])
        candidate["args"]["weight_decay"] = .0004
        candidate["args"]["precompression_anchor_clean_sample_probability"] = .5
        with self.assertRaisesRegex(ValueError, "clean75"):
            module.validate_checkpoints(controls[0], candidate, [m, m])

    def test_logits_capture_follows_paired_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "results"
            first = {"hand_face_logits": torch.tensor([[1., 2.], [3., 4.]]),
                     "body_face_logits": torch.tensor([[5., 6.], [7., 8.]])}
            second = {key: value + 10 for key, value in first.items()}

            def fake_main():
                output.mkdir()
                for item in (first, second):
                    module.paired._soft_logits(item, torch.tensor([.5, .5]))
                with (output / "paired_samples.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["sample"])
                    writer.writeheader()
                    writer.writerows([{"sample": "a"}, {"sample": "b"}])

            original = module.paired._soft_logits
            with patch.object(module.paired, "main", fake_main), patch.object(sys, "argv", ["evaluate", "--condition-set", "standard", "--output-dir", str(output)]):
                try:
                    module.main()
                finally:
                    module.paired._soft_logits = original
            with np.load(output / "paired_branch_logits.npz") as saved:
                np.testing.assert_array_equal(saved["control"][:, 0], first["hand_face_logits"].numpy())
                np.testing.assert_array_equal(saved["candidate"][:, 1], second["body_face_logits"].numpy())


if __name__ == "__main__":
    unittest.main()
