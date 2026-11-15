import unittest
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, RTMW_133_FINGER_GROUPS
from isaa.layouts.rtmw_133 import RTMW_133_FACE_GROUPS
from train_best_weighted_100_robustness import training_arguments


class BestWeighted100RobustnessTests(unittest.TestCase):
    def test_inherits_baseline_and_adds_only_joint_dropout(self):
        args = training_arguments([])
        self.assertEqual(args[args.index("--model-variant") + 1], "body-local-hand-ctr-wide-relative")
        self.assertEqual(args[args.index("--window-size") + 1], "100")
        self.assertEqual(args[args.index("--temporal-compression") + 1], "confidence_weighted")
        self.assertEqual(args[args.index("--lr-scheduler") + 1], "warmup_cosine")
        self.assertEqual(args[args.index("--epochs") + 1], "150")
        self.assertEqual(args[args.index("--joint-dropout-probability") + 1], "0.20")
        self.assertEqual(args[args.index("--grouped-joint-dropout-probability") + 1], "0.20")
        self.assertEqual(args[args.index("--save-dir") + 1], "outputs/best_weighted_100_robustness_ntu60_xsub")
        self.assertNotIn("--init-checkpoint", args)
        self.assertNotIn("--no-compile", args)

    def test_compile_cannot_be_disabled_or_model_changed(self):
        for overrides in (["--no-compile"], ["--model-variant", "original"]):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                training_arguments(overrides)

    def test_dropout_can_be_set_explicitly(self):
        args = training_arguments(["--joint-dropout-probability", "0.2"])
        self.assertEqual(args[args.index("--joint-dropout-probability") + 1], "0.2")

    def test_grouped_dropout_can_be_set_explicitly(self):
        args = training_arguments(["--grouped-joint-dropout-probability", "0.25"])
        self.assertEqual(args[args.index("--grouped-joint-dropout-probability") + 1], "0.25")

    def test_finger_and_face_group_dropout_masks_complete_groups(self):
        self.assertEqual(len(RTMW_133_FINGER_GROUPS), 10)
        self.assertTrue(all(len(group) == 4 for group in RTMW_133_FINGER_GROUPS))
        self.assertEqual(len(RTMW_133_FACE_GROUPS), 6)
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "samples.zip"
            payload = io.BytesIO()
            points = np.ones((4, 1, 133, 2), dtype=np.float32)
            scores = np.ones(points.shape[:-1], dtype=np.float32)
            np.savez(payload, keypoints=points, scores=scores)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("S001C001P001R001A001_rgb.npz", payload.getvalue())
            dataset = RTMWZipDataset(
                archive_path, split="train", split_protocol="xsub60", window_size=4,
                num_joints=133, num_classes=1, layout="rtmw_133", feature_mode="raw",
                temporal_compression="confidence_weighted", augment=True,
                augmentation_config={"grouped_joint_dropout_probability": 1.0},
            )
            try:
                data, _, _ = dataset[0]
                hidden = set(index for group in RTMW_133_FINGER_GROUPS + RTMW_133_FACE_GROUPS
                             for index in group)
                for index in hidden:
                    self.assertEqual(data[2, :, index, :].count_nonzero().item(), 0)
                self.assertGreater(data[2, :, 91, :].count_nonzero().item(), 0)
                self.assertGreater(data[2, :, 0, :].count_nonzero().item(), 0)
            finally:
                if dataset._archive is not None:
                    dataset._archive.close()

    def test_point_dropout_is_limited_to_body_and_preserves_wrists(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "samples.zip"
            payload = io.BytesIO()
            points = np.ones((4, 1, 133, 2), dtype=np.float32)
            scores = np.ones(points.shape[:-1], dtype=np.float32)
            np.savez(payload, keypoints=points, scores=scores)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("S001C001P001R001A001_rgb.npz", payload.getvalue())
            dataset = RTMWZipDataset(
                archive_path, split="train", split_protocol="xsub60", window_size=4,
                num_joints=133, num_classes=1, layout="rtmw_133", feature_mode="raw",
                temporal_compression="confidence_weighted", augment=True,
                augmentation_config={"joint_dropout_probability": 1.0},
            )
            try:
                data, _, _ = dataset[0]
                self.assertEqual(data[2, :, :23, :].count_nonzero().item(), 0)
                for index in (91, 112, 92, 23):
                    self.assertGreater(data[2, :, index, :].count_nonzero().item(), 0)
            finally:
                if dataset._archive is not None:
                    dataset._archive.close()


if __name__ == "__main__":
    unittest.main()
