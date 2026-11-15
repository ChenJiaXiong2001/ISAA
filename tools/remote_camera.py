"""Persistent SSH transport for remote YOLO/RTMW camera inference.

The camera and display stay local. Images are JPEG-compressed locally and sent
over one long-lived SSH stdio channel; the remote process returns the pose
arrays with the original capture id. The trusted SSH channel carries pickle
frames, so this transport must only be used with the configured server.
"""
from __future__ import annotations

import base64
from collections import deque
import pickle
import shlex
import struct
import subprocess
import threading
import time

import cv2


REMOTE_PROGRAM = r'''import argparse
import os
import pickle
import struct
import sys
import time

# Reserve a duplicate of stdout for packets, then route Python AND native
# library stdout to stderr before importing any model/runtime dependencies.
protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

import cv2
import numpy as np


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_packet(stream):
    header = read_exact(stream, 4)
    if header is None:
        return None
    size = struct.unpack("!I", header)[0]
    if size > 32 * 1024 * 1024:
        raise ValueError("incoming packet is too large")
    payload = read_exact(stream, size)
    if payload is None:
        return None
    return pickle.loads(payload)


def write_packet(stream, value):
    payload = pickle.dumps(value, protocol=5)
    stream.write(struct.pack("!I", len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-project", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo-model", required=True)
    parser.add_argument("--yolo-conf", type=float, default=.6)
    parser.add_argument("--yolo-iou", type=float, default=.5)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--kpt-thr", type=float, default=.1)
    parser.add_argument("--max-persons", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.)
    args = parser.parse_args()

    sys.path.insert(0, args.pose_project)
    from ntu_rtmw.compat import patch_runtime
    patch_runtime()
    from ntu_rtmw.camera import PoseWorker
    from ntu_rtmw.constants import RTMDET_CONFIG, RTMDET_WEIGHTS_PATH, RTMW_CONFIG, RTMW_WEIGHTS_PATH

    pose_args = argparse.Namespace(
        pose2d=RTMW_CONFIG, pose2d_weights=str(RTMW_WEIGHTS_PATH),
        det_model=RTMDET_CONFIG, det_weights=str(RTMDET_WEIGHTS_PATH),
        device=args.device, det_backend="yolo", yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou, yolo_imgsz=args.yolo_imgsz,
        pose_input="crop", crop_margin=0., max_persons=args.max_persons,
        bbox_thr=.3, kpt_thr=args.kpt_thr, tracking_distance=120.,
        temporal_min_frames=2, temporal_max_jump=150., temporal_min_keypoints=5,
        filter_output_to_bbox=True, output_bbox_margin=0.,
    )
    worker = PoseWorker(pose_args).start()
    previous = None
    try:
        while True:
            request = read_packet(sys.stdin.buffer)
            if request is None:
                break
            frame = cv2.imdecode(np.frombuffer(request["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("remote process could not decode JPEG frame")
            worker.submit(frame)
            deadline = time.monotonic() + args.timeout
            while True:
                latest, busy, error = worker.snapshot()
                if error is not None:
                    raise RuntimeError(str(error))
                if latest is not None and latest is not previous:
                    previous = latest
                    result = dict(latest)
                    result["frame_id"] = request["frame_id"]
                    result["captured_at"] = request["captured_at"]
                    # Do not pickle NumPy arrays across the SSH boundary. The
                    # server and Windows client may use different NumPy major
                    # versions, whose internal module paths are not stable.
                    for name in ("keypoints", "scores", "bboxes", "bbox_scores"):
                        value = result.get(name)
                        if hasattr(value, "tolist"):
                            result[name] = value.tolist()
                    write_packet(protocol_output, result)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("remote pose inference timed out")
                time.sleep(.001)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
'''


def _packet(value: object) -> bytes:
    payload = pickle.dumps(value, protocol=5)
    return struct.pack("!I", len(payload)) + payload


def _read_packet(stream):
    def read_exact(size):
        chunks = []
        while size:
            chunk = stream.read(size)
            if not chunk:
                if not chunks:
                    return None
                raise ConnectionError("SSH remote stream ended in a partial packet")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    header = read_exact(4)
    if not header:
        return None
    if len(header) != 4:
        raise ConnectionError("SSH remote stream ended in packet header")
    size = struct.unpack("!I", header)[0]
    if size > 32 * 1024 * 1024:
        raise ConnectionError("SSH remote packet is too large")
    payload = read_exact(size)
    if payload is None:
        raise ConnectionError("SSH remote stream ended in packet payload")
    return pickle.loads(payload)


def _terminate_process(process):
    """Bound shutdown even when a remote dependency or SSH peer stalls."""
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


class RemotePoseWorker:
    """PoseWorker-compatible client backed by a persistent SSH process."""

    def __init__(self, config):
        self.config = config
        self.condition = threading.Condition()
        self.pending_frame = None
        self.latest = None
        self.error = None
        self.busy = False
        self.stop_requested = False
        self.process = None
        self.stderr_tail = deque(maxlen=32)
        self.stderr_thread = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame, frame_id, captured_at):
        with self.condition:
            self.pending_frame = (frame.copy(), frame_id, captured_at)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return self.latest, self.busy, self.error

    def stop(self):
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
        process = self.process
        if process is not None:
            _terminate_process(process)
        if self.thread.is_alive():
            self.thread.join(timeout=3)

    def _command(self):
        encoded = base64.b64encode(REMOTE_PROGRAM.encode("utf-8")).decode("ascii")
        c = self.config
        remote = [
            c.remote_python, "-u", "-c",
            f"import base64;exec(base64.b64decode('{encoded}'))",
            "--pose-project", c.remote_pose_project,
            "--device", c.remote_device,
            "--yolo-model", c.remote_yolo_model,
            "--yolo-conf", str(c.yolo_conf), "--yolo-iou", str(c.yolo_iou),
            "--yolo-imgsz", str(c.yolo_imgsz), "--kpt-thr", str(c.kpt_thr),
            "--max-persons", str(c.max_persons), "--timeout", str(c.remote_timeout),
        ]
        remote_command = " ".join(shlex.quote(str(part)) for part in remote)
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
                   "-p", str(c.ssh_port)]
        if c.ssh_key:
            command += ["-i", str(c.ssh_key)]
        command += [f"{c.ssh_user}@{c.ssh_host}", remote_command]
        return command

    def _drain_stderr(self):
        # Draining concurrently prevents model logs from filling the stderr
        # pipe and deadlocking the binary stdout request/response channel.
        while True:
            chunk = self.process.stderr.read1(4096)
            if not chunk:
                return
            self.stderr_tail.append(chunk)

    def _process_error(self, prefix):
        _terminate_process(self.process)
        self.stderr_thread.join(timeout=1)
        detail = b"".join(self.stderr_tail).decode("utf-8", "replace").strip()[-1800:]
        return ConnectionError(f"{prefix} (exit code {self.process.returncode})"
                               + (f": {detail}" if detail else ""))

    def _run(self):
        try:
            process = subprocess.Popen(self._command(), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.process = process
            self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self.stderr_thread.start()
            while True:
                with self.condition:
                    while self.pending_frame is None and not self.stop_requested:
                        self.condition.wait()
                    if self.stop_requested:
                        return
                    frame, frame_id, captured_at = self.pending_frame
                    self.pending_frame = None
                    self.busy = True
                ok, encoded = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
                if not ok:
                    raise RuntimeError("local camera frame JPEG encoding failed")
                expired = threading.Event()

                def expire():
                    expired.set()
                    _terminate_process(process)

                # Includes JPEG write, network transfer, initialization and
                # response read; remote-only timers cannot cover network hangs.
                timer = threading.Timer(self.config.remote_timeout + 15, expire)
                timer.daemon = True
                timer.start()
                try:
                    process.stdin.write(_packet({"frame_id": frame_id,
                                                 "captured_at": captured_at,
                                                 "jpeg": encoded.tobytes()}))
                    process.stdin.flush()
                    result = _read_packet(process.stdout)
                    if result is None:
                        raise ConnectionError("SSH remote process exited before sending a result")
                    if result.get("frame_id") != frame_id or result.get("captured_at") != captured_at:
                        raise ConnectionError("SSH remote result capture identity mismatch")
                except Exception as exc:
                    reason = "SSH连接或推理超时" if expired.is_set() else "SSH远端推理失败"
                    raise self._process_error(f"{reason}: {exc}") from exc
                finally:
                    timer.cancel()
                    timer.join(timeout=2)
                # perf_counter origins differ between Windows and Linux. Only
                # the local receipt time may enter local FPS/staleness metrics.
                result["completed_at"] = time.perf_counter()
                with self.condition:
                    self.latest = result
                    self.error = None
                    self.busy = False
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.busy = False
                self.condition.notify_all()
        finally:
            if self.process is not None:
                _terminate_process(self.process)
                if self.stderr_thread is not None:
                    self.stderr_thread.join(timeout=1)
                for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                    try:
                        stream.close()
                    except OSError:
                        pass


def create_remote_pose_worker(config):
    return RemotePoseWorker(config)
