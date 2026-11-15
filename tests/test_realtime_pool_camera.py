"""Presence gating must prevent closed-set actions on empty camera input."""
from collections import deque
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from tools import realtime_pool_camera as demo


def pose(confidence=0.95, body_score=0.9):
    points = np.full((2, 133, 2), np.nan, dtype=np.float32)
    points[0, :, 0] = np.linspace(150, 300, 133)
    points[0, :, 1] = np.linspace(100, 350, 133)
    scores = np.zeros((2, 133), dtype=np.float32)
    scores[0] = body_score
    return dict(keypoints=points, scores=scores,
                bboxes=np.array([[100, 50, 400, 450], [np.nan] * 4], dtype=np.float32),
                bbox_scores=np.array([confidence, 0], dtype=np.float32), count=1, elapsed=0.1)


def empty_pose():
    result = pose()
    result["keypoints"][:] = np.nan
    result["scores"][:] = 0
    result["bboxes"][:] = np.nan
    result["bbox_scores"][:] = 0
    # Upstream count can still describe unfiltered detector boxes.
    return result


class CameraPresenceTests(unittest.TestCase):
    def test_background_and_weak_detections_are_rejected(self):
        for observation in (empty_pose(), pose(confidence=0.4), pose(body_score=0.2)):
            with self.subTest(confidence=observation["bbox_scores"].tolist()):
                filtered = demo.filter_people(observation, (480, 640, 3))
                self.assertEqual(filtered["count"], 0)
                self.assertFalse(filtered["scores"].any())
                self.assertTrue(np.isnan(filtered["keypoints"]).all())

    def test_face_and_hand_points_do_not_confirm_a_body(self):
        observation = pose()
        observation["scores"][0, :17] = 0
        self.assertEqual(demo.filter_people(observation, (480, 640, 3))["count"], 0)

    def test_rejected_person_is_removed_without_changing_original(self):
        observation = pose()
        observation["keypoints"][1] = observation["keypoints"][0]
        observation["scores"][1] = 0.1
        observation["bboxes"][1] = observation["bboxes"][0]
        observation["bbox_scores"][1] = 0.9
        filtered = demo.filter_people(observation, (480, 640, 3))
        self.assertEqual(filtered["count"], 1)
        self.assertFalse(filtered["scores"][1].any())
        self.assertTrue(observation["scores"][1].any())

    def test_confirm_then_leave_clears_prediction_and_history(self):
        state = demo.RecognitionWindow(confirm_frames=3)
        person = demo.filter_people(pose(), (480, 640, 3))
        self.assertFalse(state.observe(person))
        self.assertFalse(state.observe(person))
        self.assertTrue(state.observe(person))
        state.prediction, state.confidence = 9, 0.99
        state.top5 = [(9, 0.99)]
        self.assertFalse(state.observe(demo.filter_people(empty_pose(), (480, 640, 3))))
        self.assertEqual(len(state.frames), 0)
        self.assertIsNone(state.prediction)
        self.assertEqual(state.top5, [])
        self.assertFalse(state.present)
        self.assertFalse(state.observe(person))
        self.assertFalse(state.observe(person))
        self.assertTrue(state.observe(person))
        self.assertEqual(len(state.frames), 1)

    def test_zero_score_frames_are_invalid_even_with_finite_coordinates(self):
        x, mask = demo.make_input(deque([np.zeros((2, 133, 3), dtype=np.float32)]),
                                  window=100, max_persons=2)
        self.assertFalse(mask.any())
        self.assertFalse(x.any())

    def test_skipped_capture_ids_preserve_empty_positions_and_mask(self):
        state = demo.RecognitionWindow(confirm_frames=1)
        person = demo.filter_people(pose(), (480, 640, 3))
        state.observe(person, 10)
        state.observe(person, 14)
        self.assertEqual(state.last_gap_count, 3)
        self.assertEqual(len(state.frames), 2)
        self.assertEqual(state.valid_count, 2)
        x, mask = demo.make_input(state.frames, window=100, max_persons=2)
        self.assertEqual(mask[0, :2].tolist(), [True, True])
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(torch.isfinite(x).all())

    def test_skipped_capture_ids_do_not_consume_action_window_slots(self):
        state = demo.RecognitionWindow(window=100, confirm_frames=1)
        person = demo.filter_people(pose(), (480, 640, 3))
        for frame_id in range(100):
            state.observe(person, frame_id * 10)
        self.assertEqual(state.last_gap_count, 9)
        self.assertEqual(len(state.frames), 100)
        self.assertEqual(state.valid_count, 100)
        self.assertTrue(state.complete)

    def test_large_gap_keeps_latest_100_capture_positions(self):
        state = demo.RecognitionWindow(confirm_frames=1)
        person = demo.filter_people(pose(), (480, 640, 3))
        state.observe(person, 1)
        state.observe(person, 10000)
        self.assertEqual(len(state.frames), 2)
        self.assertEqual(state.valid_count, 2)
        x, mask = demo.make_input(state.frames, window=100, max_persons=2)
        self.assertTrue(mask[0, :2].all())
        self.assertFalse(x[:, :, :2].eq(0).all())

    def test_absence_does_not_bridge_two_person_sessions(self):
        state = demo.RecognitionWindow(confirm_frames=1)
        person = demo.filter_people(pose(), (480, 640, 3))
        state.observe(person, 10)
        state.observe(demo.filter_people(empty_pose(), (480, 640, 3)), 15)
        state.observe(person, 30)
        self.assertEqual(len(state.frames), 1)
        self.assertEqual(state.last_gap_count, 0)

    def run_camera(self, observations, confidences=None, distributions=None):
        sys.path.insert(0, str(demo.ROOT.parent / "nturgbd_skeleton_extraction"))
        import cv2
        from ntu_rtmw import camera
        worker = Mock()
        worker.start.return_value = worker
        worker.snapshot.side_effect = [(item, False, None) for item in observations]
        cap = Mock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap.read.side_effect = [(True, frame.copy()) for _ in observations] + [(False, None)]
        if distributions is not None:
            model = Mock(side_effect=[Mock(softmax=Mock(return_value=distribution))
                                      for distribution in distributions])
        elif confidences is None:
            logits = torch.zeros(1, 60)
            logits[0, 0] = 12
            model = Mock(return_value=logits)
        else:
            outputs = []
            for confidence in confidences:
                probabilities = torch.full((1, 60), (1 - confidence) / 59)
                probabilities[0, 0] = confidence
                outputs.append(Mock(softmax=Mock(return_value=probabilities)))
            model = Mock(side_effect=outputs)
        labels = []
        args = demo.parser().parse_args(["--device", "cpu", "--classify-after", "2"])
        with TemporaryDirectory() as log_dir, \
             patch.object(demo, "load_model", return_value=(model, {"args": {"window_size": 100, "max_persons": 2}})), \
             patch.object(demo, "create_pose_worker", return_value=worker), \
             patch.object(demo, "draw_label", side_effect=lambda frame, text, *rest: labels.append(text)), \
             patch.object(camera, "draw_skeleton"), \
             patch.object(cv2, "VideoCapture", return_value=cap), \
             patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", return_value=-1), \
             patch.object(cv2, "destroyAllWindows"):
            args.log_dir = log_dir
            demo.run(args)
        cap.release.assert_called_once()
        worker.stop.assert_called_once()
        return model, labels

    def test_empty_camera_never_calls_classifier(self):
        model, labels = self.run_camera([empty_pose() for _ in range(25)])
        model.assert_not_called()
        self.assertTrue(all("未检测到有效人物" in label for label in labels))

    def test_leave_removes_label_and_reentry_requires_new_history(self):
        observations = [pose() for _ in range(4)] + [empty_pose()] + [pose() for _ in range(4)]
        model, labels = self.run_camera(observations)
        self.assertEqual(model.call_count, 2)
        self.assertTrue(labels[3].startswith("Top 5"))
        self.assertIn("未检测到有效人物", labels[4])
        self.assertTrue(all("Top 5" not in label for label in labels[4:8]))
        self.assertTrue(labels[8].startswith("Top 5"))

    def test_repeated_snapshot_is_not_counted_as_more_people_observations(self):
        observation = pose()
        model, labels = self.run_camera([observation] * 25)
        model.assert_not_called()
        self.assertTrue(all("正在确认人物" in label for label in labels))

    def test_top5_is_shown_above_at_and_below_old_confidence_threshold(self):
        model, labels = self.run_camera([pose() for _ in range(7)],
                                       confidences=[0.500001, 0.499999, 0.5, 0.1])
        self.assertEqual(model.call_count, 4)
        for label in labels[3:]:
            self.assertTrue(label.startswith("Top 5"))
            self.assertIn("1. A001  喝水", label)
            self.assertEqual(len(label.splitlines()), 6)
            self.assertNotIn("等待明确动作", label)
        self.assertIn("10.0%", labels[6])

    def test_top5_is_sorted_and_keeps_original_60_class_probabilities(self):
        probabilities = torch.full((1, 60), 0.03 / 55)
        indices = [9, 28, 21, 7, 8]
        scores = [0.4, 0.3, 0.15, 0.08, 0.04]
        for index, score in zip(indices, scores):
            probabilities[0, index] = score
        model, labels = self.run_camera([pose() for _ in range(4)],
                                       distributions=[probabilities])
        model.assert_called_once()
        rows = labels[-1].splitlines()[1:]
        self.assertEqual(len(rows), 5)
        for rank, (row, index, score) in enumerate(zip(rows, indices, scores), start=1):
            self.assertEqual(row, f"{rank}. A{index + 1:03d}  {demo.NTU60_NAMES[index]}  {score:.1%}")

    def test_placeholders_do_not_meet_minimum_valid_pose_count(self):
        observations = [pose() for _ in range(5)]
        for observation, source_id in zip(observations, [0, 1, 2, 3, 150]):
            observation.update(frame_id=source_id, captured_at=source_id / 30)
        # Camera keeps reading while the worker retains the same last result.
        reads = observations[:4] + [observations[3]] * 146 + observations[4:]
        model, labels = self.run_camera(reads)
        self.assertEqual(model.call_count, 1)
        self.assertTrue(labels[3].startswith("Top 5"))
        self.assertIn("有效帧 1/2", labels[-1])
        self.assertNotIn("A001", labels[-1])


if __name__ == "__main__":
    unittest.main()
