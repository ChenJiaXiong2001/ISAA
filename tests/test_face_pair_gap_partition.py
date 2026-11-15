"""Check calibration and validation partition selection without a GPU."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools import evaluate_face_pair_anchor_gaps as evaluator


class FakeModel(torch.nn.Module):
    ARCHITECTURE = "fake"

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, features, frames):
        logits = torch.zeros(features.shape[0], self.num_classes)
        logits[:, 0] = 1
        return {"hand_face_logits": logits, "body_face_logits": logits}


class GapPartitionTests(unittest.TestCase):
    def run_partition(self, partition):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkpoint = base / "model.pt"
            checkpoint.write_bytes(b"fake checkpoint")
            members = {"calibration": ["train_member"], "validation": ["val_member"]}
            manifest = {key: {"members": value, "sha256": hashlib.sha256("\n".join(value).encode()).hexdigest()}
                        for key, value in members.items()}
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = base / "output"
            args = ["test", "--checkpoint", str(checkpoint), "--manifest", str(manifest_path),
                    "--archive", str(base / "data.zip"), "--output-dir", str(output),
                    "--device", "cpu", "--num-workers", "0", "--compare-max-gap"]
            if partition is not None:
                args.extend(["--partition", partition])
            observed_splits = []

            class FakeDataset:
                def __init__(self, archive, *, split, **kwargs):
                    observed_splits.append(split)
                    self.members = members["calibration" if split == "train" else "validation"]

                def __len__(self):
                    return 1

            fake_checkpoint = {"architecture": "fake", "model": {}, "args": {
                "model_variant": evaluator.FACE_PAIR_VARIANT, "num_classes": 2, "split": "xsub60",
                "window_size": 5, "max_persons": 1, "temporal_compression": "none"}}
            features = torch.ones(1, 3, 5, 133, 1)
            batch = [(features, torch.tensor([0]), torch.ones(1, 5, dtype=torch.bool))]
            with patch.object(sys, "argv", args), patch.object(evaluator.torch, "load", return_value=fake_checkpoint), \
                 patch.object(evaluator, "FacePairFusion", FakeModel), \
                 patch.object(FakeModel, "load_state_dict"), \
                 patch.object(evaluator, "RTMWZipDataset", FakeDataset), \
                 patch.object(evaluator, "DataLoader", return_value=batch), \
                 patch.object(evaluator, "_soft_weights", return_value=torch.ones(2)), \
                 patch.object(evaluator, "_soft_logits", side_effect=lambda out, weights: out["hand_face_logits"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluator.main()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            predictions = (output / "predictions.csv").read_text(encoding="utf-8")
            return observed_splits, summary, predictions

    def test_default_calibration_and_explicit_validation(self):
        for partition, dataset_split, summary_partition, member in (
                (None, "train", "training_subject_calibration", "train_member"),
                ("validation", "val", "validation", "val_member")):
            with self.subTest(partition=partition):
                observed, summary, predictions = self.run_partition(partition)
                self.assertEqual(observed, [dataset_split])
                self.assertEqual(summary["partition"], summary_partition)
                self.assertEqual(summary["dataset_split"], dataset_split)
                self.assertIn(member, predictions)
                self.assertIn("gap5_max5_soft", predictions)


if __name__ == "__main__":
    unittest.main()
