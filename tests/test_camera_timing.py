"""Capture identity and missing-result accounting for the camera demo."""
from argparse import Namespace
import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

from tools.camera_timing import CameraLog, FrameStatistics, create_pose_worker


class CameraTimingTests(unittest.TestCase):
    def test_unfinished_capture_frames_are_not_counted_as_skipped(self):
        stats = FrameStatistics(0)
        for i in range(10):
            stats.captured(i / 10)
        self.assertEqual(stats.snapshot(1)["skipped_frames"], 0)
        self.assertEqual(stats.snapshot(1)["awaiting_result_frames"], 10)
        stats.processed(0, 0, 0.2)
        stats.processed(4, 0.4, 0.8)
        row = stats.snapshot(1)
        self.assertEqual(row["processed_frames"], 2)
        self.assertEqual(row["skipped_frames"], 3)
        self.assertAlmostEqual(row["skipped_ratio"], 0.6)
        self.assertEqual(row["awaiting_result_frames"], 5)
        self.assertAlmostEqual(row["result_gap_ms"], 400)
        self.assertEqual(row["read_fps"], 10)
        self.assertEqual(row["pose_fps"], 2)
        with self.assertRaises(ValueError):
            stats.processed(4, 0.4, 0.8)
        self.assertEqual(stats.snapshot(3)["read_fps"], 0)
        self.assertEqual(stats.snapshot(3)["pose_fps"], 0)

    def test_session_csv_records_throttled_and_final_counters(self):
        stats = FrameStatistics(0)
        with TemporaryDirectory() as directory:
            log = CameraLog(directory)
            try:
                log.write(stats.snapshot(0.1))
                log.write(stats.snapshot(0.2))
                stats.captured(0.3)
                stats.submitted_frames = 1
                stats.processed(0, 0.3, 0.5)
                log.write(stats.snapshot(0.5), force=True)
            finally:
                log.close()
            with log.path.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["processed_frames"], "1")
            self.assertEqual(rows[-1]["read_frames"], "1")
            self.assertIn("placeholder_frames", rows[-1])

    def test_worker_tags_consumed_image_even_when_pending_image_is_replaced(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nturgbd_skeleton_extraction"))
        from ntu_rtmw.camera import PoseWorker
        entered, release = Event(), Event()
        seen = []

        def infer(worker, frame, pipeline, started):
            value = int(frame[0, 0, 0])
            seen.append(value)
            if value == 0:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test inference timed out")
            return {"pixel": value}

        args = Namespace(det_backend="yolo", temporal_min_frames=1,
                         temporal_max_jump=150, temporal_min_keypoints=5,
                         kpt_thr=0.1, filter_output_to_bbox=False)
        with patch.object(PoseWorker, "_build_yolo_pipeline", return_value=object()), \
             patch.object(PoseWorker, "_infer_yolo", new=infer):
            worker = create_pose_worker(args)
            worker.display_filter.apply_latest = lambda result: result
            worker.start()
            try:
                worker.submit(np.zeros((1, 1, 3), dtype=np.uint8), 0, 10.0)
                self.assertTrue(entered.wait(5))
                worker.submit(np.ones((1, 1, 3), dtype=np.uint8), 1, 10.1)
                worker.submit(np.full((1, 1, 3), 4, dtype=np.uint8), 4, 10.4)
                release.set()
                with worker.condition:
                    finished = worker.condition.wait_for(
                        lambda: worker.latest is not None and worker.latest["frame_id"] == 4,
                        timeout=5)
                self.assertTrue(finished)
                result, _, error = worker.snapshot()
                self.assertIsNone(error)
                self.assertEqual(seen, [0, 4])
                self.assertEqual(result["pixel"], 4)
                self.assertEqual(result["frame_id"], 4)
                self.assertEqual(result["captured_at"], 10.4)
                self.assertIn("completed_at", result)
            finally:
                release.set()
                worker.stop()


if __name__ == "__main__":
    unittest.main()
