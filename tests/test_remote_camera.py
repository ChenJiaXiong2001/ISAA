"""Exercise the SSH payload locally, including startup and protocol failures."""
from argparse import Namespace
import io
import shlex
import struct
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from tools import remote_camera as remote


# Actual subprocesses execute the exact remote payload and argparse flags,
# without requiring network access or GPU/model weights. The stub deliberately
# emits model-loader logs to both Python stdout and native file descriptor 1.
STUB_RUNTIME = '''
import sys, types, os
import numpy as np
for name in ('ntu_rtmw', 'ntu_rtmw.compat', 'ntu_rtmw.camera', 'ntu_rtmw.constants'):
    sys.modules[name] = types.ModuleType(name)
sys.modules['ntu_rtmw.compat'].patch_runtime = lambda: None
constants = sys.modules['ntu_rtmw.constants']
for name in ('RTMDET_CONFIG', 'RTMDET_WEIGHTS_PATH', 'RTMW_CONFIG', 'RTMW_WEIGHTS_PATH'):
    setattr(constants, name, 'test-weight')
class PoseWorker:
    def __init__(self, args): self.latest = None
    def start(self): return self
    def stop(self): pass
    def submit(self, frame):
        print('loading checkpoint: test-weight', flush=True)
        os.write(1, b'native model log\\n')
        self.latest = {'pixel': int(frame[0, 0, 0]), 'elapsed': .01,
                       'keypoints': np.zeros((2, 133, 2), dtype=np.float32),
                       'scores': np.ones((2, 133), dtype=np.float32),
                       'bboxes': np.zeros((2, 4), dtype=np.float32),
                       'bbox_scores': np.ones(2, dtype=np.float32)}
    def snapshot(self): return self.latest, False, None
sys.modules['ntu_rtmw.camera'].PoseWorker = PoseWorker
'''


def config():
    return Namespace(remote_python="python3", remote_pose_project="/test project/pose",
                     remote_device="cuda:0", remote_yolo_model="/test/yolov8n.pt",
                     remote_timeout=2, jpeg_quality=80, yolo_conf=.6, yolo_iou=.5,
                     yolo_imgsz=640, kpt_thr=.1, max_persons=2, ssh_port=22,
                     ssh_host="test.invalid", ssh_user="test", ssh_key=None)


class RemoteCameraTests(unittest.TestCase):
    def test_packet_reader_handles_fragmentation_and_truncation(self):
        class Fragmented(io.BytesIO):
            def read(self, size):
                return super().read(min(size, 2))

        value = {"frame_id": 123, "payload": "pose"}
        encoded = remote._packet(value)
        self.assertEqual(remote._read_packet(Fragmented(encoded)), value)
        self.assertIsNone(remote._read_packet(io.BytesIO()))
        for payload in (encoded[:2], encoded[:-1], encoded[:4]):
            with self.assertRaises(ConnectionError):
                remote._read_packet(Fragmented(payload))
        with self.assertRaises(ConnectionError):
            remote._read_packet(io.BytesIO(struct.pack("!I", 100_000_000)))

    def test_command_has_no_argparse_separator_and_quotes_paths(self):
        command = remote.RemotePoseWorker(config())._command()
        args = shlex.split(command[-1])
        self.assertNotIn("--", args[4:])
        self.assertEqual(args[4], "--pose-project")
        self.assertEqual(args[5], "/test project/pose")
        self.assertIn("BatchMode=yes", command)

    def _wait(self, worker, frame_id):
        with worker.condition:
            self.assertTrue(worker.condition.wait_for(lambda: worker.error is not None or
                (worker.latest is not None and worker.latest["frame_id"] == frame_id), timeout=12))
        return worker.snapshot()

    def test_exact_remote_program_roundtrip_is_not_corrupted_by_model_logs(self):
        real_popen = subprocess.Popen

        def launch(command, **kwargs):
            arguments = shlex.split(command[-1])
            # Execute the same base64-wrapped program that SSH receives.
            return real_popen([sys.executable, "-u", "-c", STUB_RUNTIME + arguments[3],
                               *arguments[4:]], **kwargs)

        worker = remote.RemotePoseWorker(config())
        with patch.object(remote.subprocess, "Popen", side_effect=launch):
            worker.start()
            try:
                before = time.perf_counter()
                worker.submit(np.full((8, 8, 3), 40, dtype=np.uint8), 10, 123.4)
                result, _, error = self._wait(worker, 10)
                self.assertIsNone(error, str(error))
                self.assertEqual(result["frame_id"], 10)
                self.assertEqual(result["captured_at"], 123.4)
                self.assertEqual(result["pixel"], 40)
                # The on-wire payload must not depend on NumPy's private
                # pickle module names (NumPy 1.x on Windows, 2.x on Linux).
                for name in ("keypoints", "scores", "bboxes", "bbox_scores"):
                    self.assertIsInstance(result[name], list)
                self.assertEqual(len(result["keypoints"][0]), 133)
                self.assertEqual(result["bbox_scores"], [1., 1.])
                self.assertGreaterEqual(result["completed_at"], before)
                self.assertLessEqual(result["completed_at"], time.perf_counter())
                worker.submit(np.full((8, 8, 3), 80, dtype=np.uint8), 20, 123.7)
                result, _, error = self._wait(worker, 20)
                self.assertIsNone(error, str(error))
                self.assertEqual(result["pixel"], 80)
            finally:
                worker.stop()
        self.assertIsNotNone(worker.process.poll())
        logs = b"".join(worker.stderr_tail)
        self.assertIn(b'loading checkpoint', logs)
        self.assertIn(b'native model log', logs)

    def test_early_process_exit_surfaces_stderr_and_is_reaped(self):
        real_popen = subprocess.Popen

        def launch(command, **kwargs):
            return real_popen([sys.executable, "-c",
                "import sys;sys.stderr.write('ModuleNotFoundError: mmpose\\n');sys.exit(2)"], **kwargs)

        worker = remote.RemotePoseWorker(config())
        with patch.object(remote.subprocess, "Popen", side_effect=launch):
            worker.start()
            try:
                worker.submit(np.zeros((8, 8, 3), dtype=np.uint8), 0, 1.)
                _, _, error = self._wait(worker, 0)
                self.assertIsNotNone(error)
                self.assertIn("ModuleNotFoundError: mmpose", str(error))
                self.assertIn("exit code 2", str(error))
            finally:
                worker.stop()
        self.assertIsNotNone(worker.process.poll())


if __name__ == "__main__":
    unittest.main()
