"""Check the pooled baseline preset, complete temporal coverage and CUDA training."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.train import parse_args
from train_best_weighted_100 import training_arguments


class BestWeighted100Tests(unittest.TestCase):
    def parse(self, flags):
        with patch.object(sys, "argv", ["main.py"] + flags):
            return parse_args()

    def test_preset_matches_original_best_schedule(self):
        args = self.parse(training_arguments([]))
        self.assertEqual(args.model_variant, "body-local-hand-ctr-wide-relative")
        self.assertEqual((args.window_size, args.temporal_compression), (100, "confidence_weighted"))
        self.assertEqual((args.feature_mode, args.node_count, args.max_persons), ("raw", 133, 2))
        self.assertEqual((args.lr, args.epochs, args.batch_size, args.test_batch_size), (.1, 65, 32, 32))
        self.assertEqual(args.save_dir, "outputs/best_weighted_100_ntu60_xsub")
        self.assertTrue(args.compile)
        self.assertIsNone(args.init_checkpoint)
        self.assertIsNone(args.resume_checkpoint)
        self.assertIsNone(args.gcn_config)

    def test_compile_required_in_both_entries(self):
        base = ["--model-variant", "body-local-hand-ctr-wide-relative",
                "--temporal-compression", "confidence_weighted", "--window-size", "100"]
        self.assertTrue(self.parse(base).compile)
        for flags in (["--no-compile"], ["--compile", "--no-compile"], ["--no-compile", "--compile"]):
            with self.assertRaisesRegex(ValueError, "requires torch.compile"):
                training_arguments(flags)
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.parse(base + flags)

    def test_protocol_and_scratch_initialization_cannot_be_replaced(self):
        for flag in ("--init-checkpoint", "--resume-checkpoint", "--model-variant", "--npy-dir"):
            with self.assertRaises(ValueError):
                training_arguments([flag, "unused"])
        for flags in (["--window-size", "64"], ["--temporal-compression", "none"], ["--feature-mode", "isaa"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                training_arguments(flags)

    def test_old_entries_keep_their_time_protocol_and_defaults(self):
        from train_best_no_pool import training_arguments as old_arguments
        from train_full_gcn import training_arguments as full_arguments
        old = self.parse(old_arguments([]))
        full = self.parse(full_arguments([]))
        pooled = self.parse(["--model-variant", "body-local-hand-ctr-wide-relative"])
        for args in (old, full, pooled):
            self.assertEqual((args.window_size, args.temporal_compression), (64, "none"))
        self.assertEqual(old.lr, .025)
        self.assertEqual(pooled.lr, .1)

    def test_uneven_bins_cover_every_frame_and_weight_each_joint(self):
        raw = torch.zeros(3, 237, 2, 2)
        raw[:2] = torch.arange(237).view(1, 237, 1, 1)
        raw[2] = torch.linspace(.1, 1, 237).view(237, 1, 1)
        raw[2, :, 1, 1] = 0
        mask = torch.arange(237) % 5 == 0
        out, out_mask = _confidence_weighted_temporal_compress(raw, mask, 100)
        for i in range(100):
            start, end = i * 237 // 100, (i + 1) * 237 // 100
            scores = raw[2, start:end]
            expected = (raw[:2, start:end] * scores).sum(1) / scores.sum(0).clamp_min(1e-6)
            torch.testing.assert_close(out[:2, i], expected, rtol=1e-4, atol=1e-3)
            torch.testing.assert_close(out[2, i], scores.mean(0), rtol=1e-4, atol=1e-5)
            self.assertEqual(out_mask[i].item(), mask[start:end].any().item())
        self.assertEqual(out[:2, :, 1, 1].count_nonzero(), 0)

    @staticmethod
    def make_archive(path):
        rng = np.random.default_rng(1)
        with zipfile.ZipFile(path, "w") as archive:
            for subject in (1, 3):
                for action, frames in ((1, 137), (2, 37)):
                    payload = io.BytesIO()
                    points = rng.normal(.3, .02, (frames, 2, 133, 2)).astype(np.float32)
                    np.savez(payload, keypoints=points, scores=np.ones(points.shape[:-1], dtype=np.float32))
                    archive.writestr(f"S001C001P{subject:03}R001A{action:03}_rgb.npz", payload.getvalue())

    def test_short_clip_keeps_frames_and_pad_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "samples.zip"
            self.make_archive(archive)
            options = dict(archive_path=archive, split="train", split_protocol="xsub60",
                           window_size=100, num_joints=133, num_classes=2, layout="rtmw_133",
                           feature_mode="raw", temporal_compression="confidence_weighted")
            dataset = RTMWZipDataset(**options)
            try:
                short_index = next(i for i, label in enumerate(dataset.labels) if label == 1)
                raw, _ = dataset._read_raw_sample(short_index)
                x, _, mask = dataset[short_index]
                torch.testing.assert_close(x[:, :37], raw, rtol=0, atol=0)
                self.assertEqual(mask.sum().item(), 37)
                self.assertEqual(x[:, 37:].count_nonzero(), 0)
            finally:
                if dataset._archive is not None:
                    dataset._archive.close()

    @unittest.skipUnless(os.environ.get("ISAA_TEST_DEVICE") == "cuda", "Real compiled training runs on the server with ISAA_TEST_DEVICE=cuda")
    def test_cuda_compiled_training_config_checkpoint_and_evaluation(self):
        self.assertTrue(torch.cuda.is_available())
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            archive = directory / "samples.zip"
            self.make_archive(archive)
            command = [sys.executable, str(root / "train_best_weighted_100.py"), "--archive", str(archive),
                       "--num-classes", "2", "--epochs", "1", "--batch-size", "2", "--test-batch-size", "2",
                       "--num-workers", "0", "--device", "cuda", "--save-dir", str(directory / "runs")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1200)
            self.assertEqual(result.returncode, 0, result.stdout[-5000:] + result.stderr[-5000:])
            run = next((directory / "runs").iterdir())
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["compile"]["enabled"])
            self.assertTrue(config["compile"]["required"])
            self.assertFalse(config["compile"]["suppress_errors"])
            self.assertFalse(config["preprocessing"]["augmentation"]["random_temporal_crop"])
            self.assertEqual(config["initialization"], "random_from_scratch")
            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["architecture"], BodyLocalHandCTRWideRelativeFusion.ARCHITECTURE)
            self.assertEqual(checkpoint["args"]["window_size"], 100)
            self.assertTrue(checkpoint["optimizer"]["state"])
            self.assertTrue(all(torch.isfinite(t).all() for t in checkpoint["model"].values()))
            evaluation = subprocess.run([sys.executable, str(root / "tools/evaluate_best.py"),
                "--checkpoint", str(run / "best.pt"), "--device", "cpu", "--num-workers", "0",
                "--output-dir", str(directory / "eval")], cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout[-3000:] + evaluation.stderr[-3000:])
            summary = json.loads((directory / "eval/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["samples"], 2)
            self.assertTrue(summary["same_top1_correct_count"])


if __name__ == "__main__":
    unittest.main()
