"""Independent pose extraction for the CTR-GCN comparison branch.

The public CTR-GCN checkpoint expects NTU RGB+D 25-joint xyz skeletons.
An RGB webcam does not expose Kinect's native 25-joint depth skeleton.  The
CTR-GCN branch therefore owns a separate pose model and never consumes the
RTMW-133 output.  The preferred extractor is MMPose ``human3d``; a lightweight
Ultralytics pose extractor is also provided for remote hosts where MMPose's
optional detector extensions are unavailable.  Both extract a fresh pose from
the image, then emit the NTU-25 semantic layout required by the checkpoint.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import deque

import numpy as np


def coco17_to_ntu25(points: np.ndarray, scores: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Convert a freshly extracted COCO-17 pose to an NTU-25 xyz sample.

    The source is an independent COCO pose model, never the RTMW result.  RGB
    has no metric depth channel, so coordinates are root-centred and scaled by
    the detected person size; a small image-depth proxy keeps the third
    channel non-degenerate for the official 3D CTR-GCN input.
    """
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (17, 2):
        raise ValueError(f"expected (17,2) COCO pose, got {p.shape}")
    c = np.ones((17,), dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
    if c.shape != (17,):
        raise ValueError(f"expected (17,) COCO scores, got {c.shape}")
    valid = np.isfinite(p).all(axis=1) & np.isfinite(c) & (c > 0)
    if int(valid.sum()) < 5:
        return np.zeros((25, 3), dtype=np.float32), False

    def mid(a: int, b: int) -> np.ndarray:
        available = [p[i] for i in (a, b) if valid[i]]
        return np.mean(available, axis=0) if available else np.zeros(2, dtype=np.float32)

    hip = mid(11, 12)
    shoulder = mid(5, 6)
    if not valid[11] and not valid[12]:
        hip = np.nanmean(p[valid], axis=0)
    if not valid[5] and not valid[6]:
        shoulder = hip.copy()
    neck = shoulder
    head = p[0] if valid[0] else mid(1, 2)
    extent = np.ptp(p[valid], axis=0)
    scale = float(max(float(extent.max()), 1.0))
    xyz17 = np.zeros((17, 3), dtype=np.float32)
    xyz17[:, 0] = (p[:, 0] - hip[0]) / scale
    xyz17[:, 1] = (hip[1] - p[:, 1]) / scale
    xyz17[:, 2] = 0.15 * xyz17[:, 1]
    xyz17[~valid] = 0

    out = np.zeros((25, 3), dtype=np.float32)
    out[0] = 0.0
    out[1] = 0.5 * (out[0] + xyz17[5] + xyz17[6]) / 2.0
    out[2] = (xyz17[5] + xyz17[6]) / 2.0 if valid[5] or valid[6] else 0.0
    out[3] = xyz17[0] if valid[0] else (xyz17[1] + xyz17[2]) / 2.0
    out[4], out[5], out[6] = xyz17[5], xyz17[7], xyz17[9]
    out[8], out[9], out[10] = xyz17[6], xyz17[8], xyz17[10]
    out[12], out[13], out[14] = xyz17[11], xyz17[13], xyz17[15]
    out[16], out[17], out[18] = xyz17[12], xyz17[14], xyz17[16]
    out[20] = (xyz17[5] + xyz17[6]) / 2.0
    out[7] = out[6] + (out[6] - out[5]) * 0.35
    out[11] = out[10] + (out[10] - out[9]) * 0.35
    out[15] = out[14] + (out[14] - out[13]) * 0.35
    out[19] = out[18] + (out[18] - out[17]) * 0.35
    out[21] = out[6] + (out[6] - out[5]) * 0.55
    out[22] = out[6] + (out[6] - out[5]) * 0.25
    out[23] = out[10] + (out[10] - out[9]) * 0.55
    out[24] = out[10] + (out[10] - out[9]) * 0.25
    return out, True


class UltralyticsPose25Extractor:
    """Independent COCO-17 pose extractor for the remote CTR-GCN branch."""

    def __init__(self, model_path: str, device="cuda:0", conf=0.25, imgsz=640,
                 max_persons=2):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = None if str(device) == "cpu" else device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.max_persons = int(max_persons)

    def __call__(self, frame, frame_id=None, captured_at=None):
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                     device=self.device, verbose=False)
        if not results or results[0].keypoints is None:
            return {"keypoints3d": np.zeros((self.max_persons, 25, 3), dtype=np.float32),
                    "count": 0, "frame_id": frame_id, "captured_at": captured_at}
        data = results[0].keypoints.data.detach().cpu().numpy().astype(np.float32)
        if data.ndim != 3 or data.shape[1:] != (17, 3):
            raise RuntimeError(f"independent YOLO pose output shape is {data.shape}, expected Nx17x3")
        order = np.argsort(-data[..., 2].mean(axis=1))[:self.max_persons]
        out = np.zeros((self.max_persons, 25, 3), dtype=np.float32)
        count = 0
        for slot, index in enumerate(order):
            pose, ok = coco17_to_ntu25(data[index, :, :2], data[index, :, 2])
            if ok:
                out[slot] = pose
                count += 1
        return {"keypoints3d": out, "count": count, "frame_id": frame_id,
                "captured_at": captured_at}


def human3d17_to_ntu25(points: np.ndarray) -> np.ndarray:
    """Map Human3.6M 17 xyz joints to an NTU-like 25-joint xyz array.

    Missing NTU hand/finger and foot-tip joints are synthesized from the
    corresponding wrist/ankle direction.  This is an explicit RGB-only
    adaptation, not Kinect ground-truth skeleton recovery.
    """
    p = np.asarray(points, dtype=np.float32)
    if p.shape != (17, 3):
        raise ValueError(f"expected (17,3) Human3D pose, got {p.shape}")
    # Human3.6M: pelvis, RH,RK,RA,LH,LK,LA,spine,thorax,neck,head,
    #            LS,LE,LW,RS,RE,RW.
    out = np.zeros((25, 3), dtype=np.float32)
    out[0] = p[0]                         # spine base / pelvis
    out[1] = p[7]                         # spine mid
    out[2] = p[9]                         # neck
    out[3] = p[10]                        # head
    out[4:7] = p[[11, 12, 13]]            # left shoulder/elbow/wrist
    out[8:11] = p[[14, 15, 16]]           # right shoulder/elbow/wrist
    out[12:15] = p[[4, 5, 6]]             # left hip/knee/ankle
    out[16:19] = p[[1, 2, 3]]             # right hip/knee/ankle
    out[20] = p[8]                         # spine shoulder / thorax
    # NTU has hand and foot tips.  Extend the last observed bone direction.
    out[7] = out[6] + (out[6] - out[5]) * 0.35
    out[11] = out[10] + (out[10] - out[9]) * 0.35
    out[15] = out[14] + (out[14] - out[13]) * 0.35
    out[19] = out[18] + (out[18] - out[17]) * 0.35
    out[21] = out[6] + (out[6] - out[5]) * 0.55
    out[22] = out[6] + (out[6] - out[5]) * 0.25
    out[23] = out[10] + (out[10] - out[9]) * 0.55
    out[24] = out[10] + (out[10] - out[9]) * 0.25
    return out


class MMPoseHuman3DWorker:
    """Background webcam worker dedicated to the CTR-GCN branch."""

    def __init__(self, device="cpu", max_persons=2, det_model=None, det_weights=None):
        self.device = device
        self.max_persons = max_persons
        self.det_model = det_model
        self.det_weights = det_weights
        self._jobs = queue.Queue(maxsize=1)
        self._latest = None
        self._error = None
        self._busy = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ctrgcn25-mmpose3d", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)

    def submit(self, frame, frame_id, captured_at):
        try:
            self._jobs.put_nowait((frame.copy(), frame_id, captured_at))
            return True
        except queue.Full:
            return False

    def snapshot(self):
        return self._latest, self._busy, self._error

    def _run(self):
        try:
            from mmpose.apis import MMPoseInferencer
            from ntu_rtmw.camera import MODEL_INIT_LOCK
            with MODEL_INIT_LOCK:
                from mmengine.registry import init_default_scope
                init_default_scope('mmpose')
                # MMPose's inferencer builds the optional detector while its
                # pose-lifter scope is active. Register the detector head in
                # the MMPose registry for mixed-version installations.
                import mmdet.models  # noqa: F401
                from mmdet.registry import MODELS as DET_MODELS
                from mmpose.registry import MODELS as POSE_MODELS
                det_head = DET_MODELS.get('RTMDetSepBNHead')
                if det_head is not None and POSE_MODELS.get('RTMDetSepBNHead') is None:
                    POSE_MODELS.register_module(module=det_head, force=True)
                inferencer = MMPoseInferencer(
                    pose3d="human3d", device=self.device, show_progress=False,
                    det_model=self.det_model, det_weights=self.det_weights)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._error = RuntimeError("CTR-GCN 独立 MMPose 3D 提取器初始化失败: " + str(exc))
            return
        while not self._stop.is_set():
            job = self._jobs.get()
            if job is None:
                break
            frame, frame_id, captured_at = job
            self._busy = True
            try:
                result = next(inferencer(frame, return_datasamples=True))
                predictions = result.get("predictions", [])
                if not predictions:
                    raise RuntimeError("MMPose 3D 未检测到人物")
                inst = predictions[0].pred_instances
                poses = np.asarray(inst.keypoints, dtype=np.float32)
                if poses.ndim != 3 or poses.shape[1:] != (17, 3):
                    raise RuntimeError(f"MMPose 3D 输出形状异常: {poses.shape}")
                # Human3D returns up to two tracked persons. Preserve the two
                # highest entries and pad missing slots with zeros.
                out = np.zeros((self.max_persons, 25, 3), dtype=np.float32)
                for slot, pose in enumerate(poses[: self.max_persons]):
                    out[slot] = human3d17_to_ntu25(pose)
                self._latest = {
                    "keypoints3d": out,
                    "count": min(len(poses), self.max_persons),
                    "frame_id": frame_id,
                    "captured_at": captured_at,
                    "completed_at": time.perf_counter(),
                }
                self._error = None
            except Exception as exc:
                self._error = RuntimeError(f"{exc}\n{traceback.format_exc()}")
            finally:
                self._busy = False


class CTRGCN25Window:
    def __init__(self, window=100, max_persons=2, max_gap=8):
        self.frames = deque(maxlen=window)
        self._valid_flags = deque(maxlen=window)
        self.max_persons = max_persons
        self.max_gap = int(max_gap)
        self.last_frame_id = None
        self._last_valid = None
        self._misses = 0
        self.valid_frames = 0

    def clear(self):
        self.frames.clear()
        self._valid_flags.clear()
        self.last_frame_id = None
        self._last_valid = None
        self._misses = 0
        self.valid_frames = 0

    def _append(self, value, valid):
        if len(self._valid_flags) == self._valid_flags.maxlen:
            self.valid_frames -= int(self._valid_flags[0])
        self.frames.append(value)
        self._valid_flags.append(bool(valid))
        self.valid_frames += int(bool(valid))

    @property
    def complete(self):
        return len(self.frames) == self.frames.maxlen

    def observe(self, result):
        """Append one timestamp, using an empty NTU25 skeleton for short gaps.

        A detector miss must not throw away the preceding 99 timestamps.  The
        remote camera stream can lose a pose frame while JPEG transport and
        GPU inference continue normally.  We therefore insert a zero skeleton
        for a short gap and reset only after the person has been absent for a
        sustained interval.  This keeps the CTR-GCN input at exactly 100
        timestamps while preventing a stale person from surviving forever.
        """
        valid = result is not None and int(result.get("count", 0)) > 0
        if valid:
            value = np.asarray(result["keypoints3d"], dtype=np.float32)
            expected = (self.max_persons, 25, 3)
            if value.shape != expected:
                raise ValueError(f"expected keypoints3d shape {expected}, got {value.shape}")
            self._last_valid = value.copy()
            self._misses = 0
            self._append(value, True)
            self.last_frame_id = result.get("frame_id")
            return True

        if self._last_valid is None:
            self.clear()
            return False
        self._misses += 1
        if self._misses > self.max_gap:
            self.clear()
            return False
        self._append(np.zeros_like(self._last_valid), False)
        if result is not None:
            self.last_frame_id = result.get("frame_id")
        return False


def make_input(frames, window=64, max_persons=2):
    import torch
    import torch.nn.functional as F
    values = np.stack(tuple(frames), axis=0).astype(np.float32, copy=False)
    if values.ndim != 4 or values.shape[1:] != (max_persons, 25, 3):
        raise ValueError(f"expected T x M x 25 x 3, got {values.shape}")
    x = torch.from_numpy(values.transpose(3, 0, 2, 1).copy()).unsqueeze(0)
    if x.shape[2] != window:
        x = F.interpolate(x.permute(0, 1, 3, 4, 2).reshape(1, 3 * max_persons * 25, -1),
                          size=window, mode="linear", align_corners=False)
        x = x.reshape(1, 3, max_persons, 25, window).permute(0, 1, 4, 3, 2).contiguous()
    return x
