"""Schedule boundary and durable-record tests without a PyTorch dependency."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from isaa.utils.experiment import RunRecords, create_run_directory, ctrgcn_learning_rate


class ExperimentTests(unittest.TestCase):
    def test_official_schedule_displayed_epoch_boundaries(self):
        expected = {1: .02, 2: .04, 3: .06, 4: .08, 5: .1, 6: .1,
                    35: .1, 36: .01, 55: .01, 56: .001, 65: .001}
        for epoch, lr in expected.items():
            self.assertAlmostEqual(ctrgcn_learning_rate(epoch - 1, .1, 5, [35, 55], .1), lr)
        self.assertEqual(ctrgcn_learning_rate(0, .1, 0, [35, 55], .1), .1)

    def test_runs_are_isolated_and_completed_batches_survive_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "main.py").write_text("# source snapshot\n")
            first, second = create_run_directory(root), create_run_directory(root)
            self.assertNotEqual(first, second)
            records = RunRecords(first, {"epochs": 65}, root)
            try:
                records.batch({"epoch": 1, "phase": "train", "step": 1, "loss": 1.2})
                # Read before close: completed batch records must already be visible.
                row = json.loads((first / "batches.jsonl").read_text())
                self.assertEqual(row["loss"], 1.2)
                records.epoch({"epoch": 1, "lr": .02, "best_epoch": 1, "best_val_acc": .5})
                records.batch({"epoch": 2, "phase": "train", "step": 1, "loss": 1.1})
                records.finish("interrupted", "KeyboardInterrupt()")
            finally:
                records.close()
            status = json.loads((first / "status.json").read_text())
            self.assertEqual(status["status"], "interrupted")
            self.assertEqual(status["last_completed_epoch"], 1)
            self.assertEqual(status["last_batch"]["epoch"], 2)
            self.assertEqual(status["best_val_acc"], .5)
            with (first / "epochs.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            with ZipFile(first / "source.zip") as archive:
                self.assertEqual(archive.read("main.py"), (root / "main.py").read_bytes())
            self.assertEqual(list(second.iterdir()), [])
            self.assertFalse(list(first.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
