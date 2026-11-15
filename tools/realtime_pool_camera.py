"""Live RTMW skeleton extraction and original pooled 100-frame recognition.

This is an inference/demo entry point.  It uses the already trained original
pooled model; it does not train, fine-tune, or load a different architecture.
Press ``q`` or Escape to stop the camera window.
"""
from __future__ import annotations

import argparse
import sys
import time
import textwrap
from argparse import Namespace
from collections import deque
from pathlib import Path
from functools import lru_cache

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.camera_timing import CameraLog, FrameStatistics, create_pose_worker
from tools.remote_camera import create_remote_pose_worker

NTU60_NAMES = [
    "喝水", "吃饭或吃零食", "刷牙", "梳头", "掉落物品", "捡起物品", "扔东西",
    "坐下", "站起", "拍手", "阅读", "写字", "撕纸", "穿外套", "脱外套",
    "穿鞋", "脱鞋", "戴眼镜", "摘眼镜", "戴帽子", "摘帽子", "挥手",
    "踢腿或踢脚", "踢东西", "手伸进口袋", "单脚跳", "跳起", "打电话",
    "玩手机或平板", "使用键盘", "指向某物", "自拍", "看时间", "搓手",
    "点头或鞠躬", "摇头", "擦脸", "敬礼", "合十", "交叉双臂",
    "打喷嚏或咳嗽", "踉跄", "跌倒", "头痛", "胸痛", "背痛", "颈部疼痛",
    "恶心或呕吐", "使用扇子", "拳击或拍打他人", "踢他人", "推他人",
    "拍他人后背", "指向他人", "拥抱", "给他人物品", "触碰他人口袋",
    "握手", "一起走近", "一起走开",
]


def filter_people(latest, frame_shape, *, person_threshold=0.6,
                  body_threshold=0.35, min_body_points=6):
    """Reject weak detector/pose matches before drawing or action inference."""
    if latest is None:
        return None
    filtered = dict(latest)
    for name in ("keypoints", "scores", "bboxes", "bbox_scores"):
        filtered[name] = np.array(latest[name], copy=True)
    height, width = frame_shape[:2]
    accepted = 0
    for slot, (points, scores, bbox, confidence) in enumerate(zip(
        filtered["keypoints"], filtered["scores"], filtered["bboxes"], filtered["bbox_scores"]
    )):
        bbox_valid = np.isfinite(bbox).all() and np.isfinite(confidence) and confidence >= person_threshold
        if bbox_valid:
            x1, y1, x2, y2 = bbox
            bbox_valid = min(x2, width) > max(x1, 0) and min(y2, height) > max(y1, 0)
        valid_body = np.zeros(17, dtype=bool)
        if bbox_valid:
            body = points[:17]
            valid_body = (np.isfinite(body).all(-1) & np.isfinite(scores[:17])
                          & (scores[:17] >= body_threshold)
                          & (body[:, 0] >= max(0, x1)) & (body[:, 0] <= min(width, x2))
                          & (body[:, 1] >= max(0, y1)) & (body[:, 1] <= min(height, y2)))
        # Face/hand hallucinations alone must never count as a person.
        if valid_body.sum() >= min_body_points and valid_body[[5, 6, 11, 12]].sum() >= 2:
            accepted += 1
        else:
            filtered["keypoints"][slot] = np.nan
            filtered["scores"][slot] = 0
            filtered["bboxes"][slot] = np.nan
            filtered["bbox_scores"][slot] = 0
    filtered["count"] = accepted
    return filtered


class RecognitionWindow:
    """Require consecutive valid observations; forget actions on any absence."""

    def __init__(self, window=100, confirm_frames=3):
        self.frames = deque(maxlen=window)
        self.confirm_frames = confirm_frames
        self.clear()

    def clear(self):
        self.frames.clear()
        self.streak = 0
        self.present = False
        self.prediction = None
        self.confidence = 0.0
        self.top5 = []
        self.last_frame_id = None
        self.last_gap_count = 0

    @property
    def valid_count(self):
        return sum(bool(np.any(np.isfinite(frame[..., :2]).all(-1) & (frame[..., 2] > 0)))
                   for frame in self.frames)

    @property
    def complete(self):
        """The rolling window is exactly 100 real pose observations."""
        return len(self.frames) == self.frames.maxlen and self.valid_count == self.frames.maxlen

    def observe(self, people, frame_id=None):
        self.last_gap_count = 0
        if people is None or people["count"] == 0:
            self.clear()
            return False
        self.streak += 1
        self.present = self.streak >= self.confirm_frames
        if not self.present:
            return False
        if frame_id is not None and self.last_frame_id is not None:
            if frame_id <= self.last_frame_id:
                raise ValueError("Pose capture IDs must increase")
            self.last_gap_count = frame_id - self.last_frame_id - 1
            # Capture gaps are telemetry only.  They must not consume slots in
            # the action window: the model requires 100 completed pose results,
            # not 100 camera timestamps padded with empty skeletons.
        self.frames.append(np.concatenate((people["keypoints"], people["scores"][..., None]), axis=-1))
        self.last_frame_id = frame_id
        return True


def _repair_short_anchor_gaps(values: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Repair bounded shoulder/hip anchor gaps before temporal compression.

    This is the frozen max-5 precompression rule from the latest face-pair
    robustness evaluation.  It only interpolates an anchor when valid samples
    exist on both sides, never extrapolates, and leaves all other joints alone.
    """
    repaired = np.array(values, copy=True)
    anchors = (5, 6, 11, 12)
    frames, persons = repaired.shape[:2]
    for person in range(persons):
        for joint in anchors:
            valid = np.isfinite(repaired[:, person, joint, :2]).all(axis=-1) & (repaired[:, person, joint, 2] > 0)
            for index in range(frames):
                if valid[index]:
                    continue
                left = index - 1
                while left >= 0 and not valid[left]:
                    left -= 1
                right = index + 1
                while right < frames and not valid[right]:
                    right += 1
                if left < 0 or right >= frames or right - left - 1 > max_gap:
                    continue
                alpha = (index - left) / float(right - left)
                repaired[index, person, joint] = (
                    repaired[left, person, joint] * (1.0 - alpha)
                    + repaired[right, person, joint] * alpha
                )
                valid[index] = True
    return repaired


def make_input(frames: deque[np.ndarray], *, window: int, max_persons: int,
               repair_anchor_gaps: bool = False):
    """Convert M x 133 x (x,y,score) frames to B x 3 x T x 133 x M."""
    import torch

    if not frames:
        raise ValueError("No skeleton frames available")
    from isaa.data.rtmw_zip_dataset import (
        _normalize_scores, _confidence_weighted_temporal_compress,
    )
    from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask

    values = np.stack(tuple(frames), axis=0).astype(np.float32, copy=False)
    if values.ndim != 4 or values.shape[2:] != (133, 3):
        raise ValueError("Expected T x M x 133 x (x,y,score)")
    if repair_anchor_gaps:
        values = _repair_short_anchor_gaps(values, max_gap=5)
    finite = np.isfinite(values[..., :2]).all(axis=-1)
    points = np.where(finite[..., None], values[..., :2], 0.0)
    scores = np.where(finite, _normalize_scores(values[..., 2], "auto"), 0.0)
    raw = torch.from_numpy(np.concatenate((points, scores[..., None]), axis=-1).transpose(3, 0, 2, 1).copy())
    if raw.shape[-1] < max_persons:
        raw = torch.cat((raw, raw.new_zeros(3, raw.shape[1], 133, max_persons - raw.shape[-1])), dim=-1)
    raw = raw[..., :max_persons]
    valid = torch.from_numpy((finite & (scores > 0)).any(axis=(1, 2)))
    raw, valid = _confidence_weighted_temporal_compress(raw, valid, window)
    raw, valid = _temporal_crop_or_pad_sample_with_mask(raw, valid, window, random_start=False)
    raw[:2] *= raw[2:3] > 0
    raw = raw.masked_fill(~valid[None, :, None, None], 0)
    return raw.contiguous().unsqueeze(0), valid.unsqueeze(0)


def load_model(checkpoint_path: Path, device):
    # Importing isaa.train first preserves the project's CUDA library bootstrap.
    from isaa.train import collate_rtmw  # noqa: F401
    import torch
    from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    if args.get("model_variant") not in (None, "body-local-hand-ctr-wide-relative"):
        raise ValueError("This demo requires the original pooled body-local-hand-ctr-wide-relative checkpoint")
    model = BodyLocalHandCTRWideRelativeFusion(num_classes=int(args.get("num_classes", 60)))
    if checkpoint.get("architecture") != model.ARCHITECTURE:
        raise ValueError(
            f"Checkpoint architecture {checkpoint.get('architecture')!r} does not match {model.ARCHITECTURE!r}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


@lru_cache(maxsize=8)
def label_font(size=22):
    from PIL import ImageFont
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        return ImageFont.truetype(str(font), size)
    return ImageFont.load_default()


def draw_label(frame, text: str, color, cv2) -> None:
    from PIL import Image, ImageDraw
    lines = max(1, len(text.splitlines()))
    # Leave the footer visible and fit all five action rows on smaller frames.
    available = max(1, frame.shape[0] - 72)
    size = min(22, max(10, (available - 20) // lines - 8))
    bottom = min(frame.shape[0] - 56, 8 + max(68, lines * (size + 8) + 20))
    right = min(frame.shape[1] - 8, 760)
    cv2.rectangle(frame, (8, 8), (right, bottom), color, -1)
    banner = Image.fromarray(cv2.cvtColor(frame[8:bottom, 8:right], cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(banner).multiline_text((10, 8), text, font=label_font(size),
                                        fill="white", spacing=6)
    frame[8:bottom, 8:right] = cv2.cvtColor(np.asarray(banner), cv2.COLOR_RGB2BGR)


def run(args) -> None:
    import cv2
    import torch

    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    if args.classify_after != 100:
        raise ValueError("--classify-after is fixed at 100: complete 100 real frames are required")
    if not np.isfinite(args.classify_interval) or args.classify_interval <= 0:
        raise ValueError("--classify-interval must be finite and positive")
    if not 0 < args.person_threshold <= 1 or not 0 < args.body_threshold <= 1:
        raise ValueError("Person and body thresholds must be in (0, 1]")
    if not 1 <= args.min_body_points <= 17 or args.presence_frames < 1:
        raise ValueError("Body points must be in [1, 17] and presence frames must be positive")
    if not np.isfinite(args.remote_timeout) or args.remote_timeout <= 0:
        raise ValueError("--remote-timeout must be finite and positive")
    if not 1 <= args.remote_jpeg_quality <= 100:
        raise ValueError("--remote-jpeg-quality must be between 1 and 100")
    torch.set_num_threads(args.cpu_threads)

    pose_root = Path(args.pose_project).resolve()
    if not pose_root.is_dir():
        raise FileNotFoundError(f"Pose extraction project not found: {pose_root}")
    yolo_model = Path(args.yolo_model or pose_root / "models" / "yolov8n.pt").resolve()
    if args.det_backend == "yolo" and not yolo_model.is_file():
        raise FileNotFoundError(f"Local YOLO detector weights not found: {yolo_model}")
    sys.path.insert(0, str(pose_root))
    from ntu_rtmw.compat import patch_runtime
    patch_runtime()
    from ntu_rtmw.extract import require_numpy
    require_numpy()
    from ntu_rtmw.camera import draw_skeleton
    from ntu_rtmw.constants import (
        RTMDET_CONFIG, RTMDET_WEIGHTS_PATH, RTMW_CONFIG, RTMW_WEIGHTS_PATH,
    )

    device = torch.device(args.device)
    model, checkpoint = load_model(Path(args.checkpoint).resolve(), device)
    model_args = checkpoint.get("args", {})
    window = int(model_args.get("window_size", 100))
    max_persons = int(model_args.get("max_persons", 2))
    if window != 100:
        raise ValueError(f"The selected checkpoint has window_size={window}; this demo requires 100 frames")
    worker_args = Namespace(
        pose2d=RTMW_CONFIG, pose2d_weights=str(args.pose_weights or RTMW_WEIGHTS_PATH),
        det_model=RTMDET_CONFIG, det_weights=str(RTMDET_WEIGHTS_PATH),
        device=args.pose_device or args.device,
        det_backend=args.det_backend, yolo_model=str(yolo_model),
        yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou, yolo_imgsz=args.yolo_imgsz,
        pose_input=args.pose_input, crop_margin=args.crop_margin, max_persons=max_persons,
        bbox_thr=args.bbox_thr, kpt_thr=args.kpt_thr, tracking_distance=args.tracking_distance,
        temporal_min_frames=2, temporal_max_jump=150.0, temporal_min_keypoints=5,
        filter_output_to_bbox=True, output_bbox_margin=0.0,
    )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open local camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.ssh_host:
        worker = create_remote_pose_worker(Namespace(
            ssh_host=args.ssh_host, ssh_user=args.ssh_user, ssh_port=args.ssh_port,
            ssh_key=args.ssh_key, remote_python=args.remote_python,
            remote_pose_project=args.remote_pose_project,
            remote_device=args.remote_device,
            remote_yolo_model=args.remote_yolo_model,
            remote_timeout=args.remote_timeout, jpeg_quality=args.remote_jpeg_quality,
            yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou, yolo_imgsz=args.yolo_imgsz,
            kpt_thr=args.kpt_thr, max_persons=max_persons,
        ))
    else:
        worker = create_pose_worker(worker_args)
    state = RecognitionWindow(window, args.presence_frames)
    last_pose = None
    last_pose_at = 0.0
    last_error = None
    people = None
    frame_idx = 0
    last_classified_at = 0.0
    started = time.perf_counter()
    statistics = FrameStatistics(started)
    log = CameraLog(args.log_dir)
    print(f"Camera frame log: {log.path}", flush=True)
    try:
        worker.start()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            captured_at = time.perf_counter()
            statistics.captured(captured_at)
            if args.flip:
                frame = cv2.flip(frame, 1)
            latest, busy, error = worker.snapshot()
            if not busy and frame_idx % max(1, args.infer_every) == 0:
                worker.submit(frame, frame_idx, captured_at)
                statistics.submitted_frames += 1
            if latest is not None and latest is not last_pose:
                last_pose = latest
                last_pose_at = time.perf_counter()
                source_id = latest.get("frame_id", frame_idx)
                statistics.processed(source_id, latest.get("captured_at", captured_at),
                                     latest.get("completed_at", last_pose_at))
                people = filter_people(latest, frame.shape, person_threshold=args.person_threshold,
                                       body_threshold=args.body_threshold, min_body_points=args.min_body_points)
                appended = state.observe(people, source_id)
                statistics.placeholder_frames += state.last_gap_count
                if appended and state.valid_count < args.classify_after:
                    state.prediction, state.confidence = None, 0.0
                    state.top5 = []
                now = time.perf_counter()
                if appended and state.complete and now - last_classified_at >= args.classify_interval:
                    x, mask = make_input(state.frames, window=window, max_persons=max_persons)
                    x, mask = x.to(device), mask.to(device)
                    with torch.inference_mode():
                        probability = model(x, mask).softmax(-1)[0]
                    values, indices = probability.topk(5)
                    state.top5 = list(zip(indices.tolist(), values.tolist()))
                    state.prediction, state.confidence = state.top5[0]
                    last_classified_at = now
            # Do not leave an old skeleton/action visible after inference fails or stalls.
            stale_after = max(3.0, 2 * float(latest.get("elapsed", 0)) + 1) if latest else 3.0
            if error is not None or (last_pose is not None and time.perf_counter() - last_pose_at > stale_after):
                state.clear()
                people = None
            display = frame.copy()
            if people is not None and state.present:
                draw_skeleton(display, people["keypoints"], people["scores"], people["bboxes"],
                              people["bbox_scores"], args.kpt_thr, cv2)
            if error is not None:
                if error is not last_error:
                    print(f"Camera inference failed:\n{error}", file=sys.stderr, flush=True)
                    last_error = error
                detail = " ".join(str(error).split())
                summary = "\n".join(textwrap.wrap(detail[-180:], width=44))
                draw_label(display, f"SSH/RTMW error（完整原因见控制台）\n{summary}", (30, 30, 190), cv2)
            elif latest is None:
                draw_label(display, "正在初始化人物检测和骨架模型…", (180, 100, 20), cv2)
            elif not state.present:
                text = "正在确认人物，请保持入镜" if state.streak else "未检测到有效人物，等待入镜"
                draw_label(display, text, (100, 100, 100), cv2)
            elif state.prediction is None:
                draw_label(display, f"已检测到人物 | 有效帧 {state.valid_count}/{args.classify_after} | 窗口 {len(state.frames)}/{window}", (180, 100, 20), cv2)
            else:
                lines = [f"Top 5 | 有效{state.valid_count}/{len(state.frames)}帧"]
                lines.extend(f"{rank}. A{index + 1:03d}  {NTU60_NAMES[index]}  {confidence:.1%}"
                             for rank, (index, confidence) in enumerate(state.top5, start=1))
                draw_label(display, "\n".join(lines), (20, 120, 40), cv2)
            metrics = statistics.snapshot(time.perf_counter())
            log.write(metrics)
            cv2.putText(display, f"Read {metrics['read_fps']:.1f} FPS | Pose {metrics['pose_fps']:.1f} FPS | "
                        f"Skipped {metrics['skipped_ratio']:.0%} | Gap {metrics['result_gap_ms']:.0f} ms",
                        (12, display.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display, "q/Esc quit", (12, display.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("ISAA original pooled 100-frame camera", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            frame_idx += 1
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        try:
            log.write(statistics.snapshot(time.perf_counter()), force=True)
        finally:
            log.close()
    print(f"Camera session finished after {time.perf_counter() - started:.1f}s")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runtime/pool_100/best.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--pose-device", default=None,
                   help="RTMW device; defaults to --device")
    p.add_argument("--ssh-host", default=None,
                   help="Use a persistent SSH channel for remote YOLO/RTMW inference")
    p.add_argument("--ssh-user", default="xiong")
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--ssh-key", default=None,
                   help="SSH private key; SSH config is used when omitted")
    p.add_argument("--remote-python", default="python3")
    p.add_argument("--remote-pose-project", default="/home/xiong/Desktop/nturgbd_skeleton_extraction")
    p.add_argument("--remote-device", default="cuda:0",
                   help="Device used by the remote RTMW/YOLO process")
    p.add_argument("--remote-yolo-model",
                   default="/home/xiong/Desktop/nturgbd_skeleton_extraction/models/yolov8n.pt")
    p.add_argument("--remote-timeout", type=float, default=30.0)
    p.add_argument("--remote-jpeg-quality", type=int, default=80)
    p.add_argument("--pose-project", default=str(ROOT.parent / "nturgbd_skeleton_extraction"))
    p.add_argument("--pose-weights", type=Path)
    p.add_argument("--yolo-model", type=Path,
                   help="Local detector weights; defaults to <pose-project>/models/yolov8n.pt")
    p.add_argument("--det-backend", choices=("yolo", "mmdet"), default="yolo")
    p.add_argument("--pose-input", choices=("crop", "frame"), default="crop")
    p.add_argument("--crop-margin", type=float, default=0.0)
    p.add_argument("--yolo-conf", type=float, default=0.6)
    p.add_argument("--yolo-iou", type=float, default=0.5)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--bbox-thr", type=float, default=0.3)
    p.add_argument("--kpt-thr", type=float, default=0.1)
    p.add_argument("--tracking-distance", type=float, default=120.0)
    p.add_argument("--person-threshold", type=float, default=0.6,
                   help="Minimum person box confidence before action recognition")
    p.add_argument("--body-threshold", type=float, default=0.35)
    p.add_argument("--min-body-points", type=int, default=6,
                   help="Confident body joints required, including at least two torso joints")
    p.add_argument("--presence-frames", type=int, default=3,
                   help="Consecutive valid pose results needed to confirm a person")
    p.add_argument("--infer-every", type=int, default=1)
    p.add_argument("--classify-after", type=int, default=100,
                   help="Fixed at 100: action recognition waits for a complete 100-real-frame window")
    p.add_argument("--classify-interval", type=float, default=1.0,
                   help="Analyze the latest complete 100-frame window at most once per second")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--log-dir", type=Path, default=ROOT / "outputs/camera_logs",
                   help="Directory for per-session FPS, skipped-frame and placeholder CSV logs")
    p.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
