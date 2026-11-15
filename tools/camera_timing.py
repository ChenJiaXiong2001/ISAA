"""Capture IDs, processing coverage, and lightweight camera-session CSV logs."""
import csv
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
import time


def create_pose_worker(args):
    """Preserve the capture ID of the exact image consumed by the pose thread."""
    from ntu_rtmw.camera import PoseWorker
    from ntu_rtmw.extract import build_inferencer, require_numpy

    class TaggedPoseWorker(PoseWorker):
        def submit(self, frame, frame_id, captured_at):
            with self.condition:
                self.pending_frame = (frame.copy(), frame_id, captured_at)
                self.condition.notify_all()

        def _run(self):
            require_numpy()
            try:
                pipeline = (self._build_yolo_pipeline() if self.args.det_backend == "yolo"
                            else build_inferencer(self.args))
            except Exception as exc:
                with self.condition:
                    self.error = RuntimeError(f"Pose worker initialization failed: {exc}\n{traceback.format_exc()}")
                return
            while True:
                with self.condition:
                    while self.pending_frame is None and not self.stop_requested:
                        self.condition.wait()
                    if self.stop_requested:
                        return
                    frame, frame_id, captured_at = self.pending_frame
                    self.pending_frame = None
                    self.busy = True
                started = time.perf_counter()
                try:
                    if self.args.det_backend == "yolo":
                        result = self._infer_yolo(frame, pipeline, started)
                    else:
                        prediction = next(iter(pipeline(frame, show=False, return_vis=False,
                                                        draw_bbox=False, kpt_thr=self.args.kpt_thr)))
                        result = self._result_to_arrays(prediction, time.perf_counter() - started)
                    if self.args.filter_output_to_bbox:
                        from ntu_rtmw.camera import filter_pose_output_to_bboxes
                        result = filter_pose_output_to_bboxes(result, self.args.output_bbox_margin)
                    result = self.display_filter.apply_latest(result)
                    result.update(frame_id=frame_id, captured_at=captured_at,
                                  completed_at=time.perf_counter())
                    with self.condition:
                        self.latest = result
                        self.error = None
                        self.condition.notify_all()
                except Exception as exc:
                    with self.condition:
                        self.error = RuntimeError(f"{exc}\n{traceback.format_exc()}")
                finally:
                    with self.condition:
                        self.busy = False

    return TaggedPoseWorker(args)


class FrameStatistics:
    """Count missing results only through the most recent completed capture ID.

    Frames newer than that ID are pending, not yet counted as finalized misses.
    No inference about frames lost inside the camera/driver is made here.
    """

    def __init__(self, started):
        self.started = started
        self.read_frames = 0
        self.submitted_frames = 0
        self.processed_frames = 0
        self.last_processed_id = -1
        self.placeholder_frames = 0
        self.last_result_at = None
        self.result_gap_ms = 0.0
        self.capture_times = deque()
        self.result_times = deque()

    def captured(self, now):
        self.read_frames += 1
        self.capture_times.append(now)

    def processed(self, frame_id, captured_at, completed_at):
        if frame_id <= self.last_processed_id:
            raise ValueError("Pose capture IDs must increase")
        self.processed_frames += 1
        self.last_processed_id = frame_id
        if self.last_result_at is not None:
            self.result_gap_ms = (captured_at - self.last_result_at) * 1000
        self.last_result_at = captured_at
        self.result_times.append(completed_at)

    def snapshot(self, now):
        for events in (self.capture_times, self.result_times):
            while events and events[0] < now - 1:
                events.popleft()
        elapsed = max(now - self.started, 1e-6)
        duration = min(1.0, elapsed)
        finalized = self.last_processed_id + 1
        skipped = max(0, finalized - self.processed_frames)
        return dict(elapsed_s=elapsed, read_frames=self.read_frames,
                    submitted_frames=self.submitted_frames, processed_frames=self.processed_frames,
                    read_fps=len(self.capture_times) / duration,
                    pose_fps=len(self.result_times) / duration,
                    skipped_frames=skipped, skipped_fps=skipped / elapsed,
                    skipped_ratio=skipped / finalized if finalized else 0.0,
                    awaiting_result_frames=max(0, self.read_frames - finalized),
                    placeholder_frames=self.placeholder_frames, result_gap_ms=self.result_gap_ms)


class CameraLog:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".csv")
        self.file = self.path.open("x", encoding="utf-8-sig", newline="")
        self.writer = None
        self.last_written_at = -1.0

    def write(self, row, *, force=False):
        if not force and row["elapsed_s"] - self.last_written_at < 1:
            return
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()
        self.last_written_at = row["elapsed_s"]

    def close(self):
        self.file.close()
