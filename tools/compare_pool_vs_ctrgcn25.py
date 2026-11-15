"""Compare the pooled model and official CTR-GCN as two separate outputs.

The pooled branch uses RTMW-133.  CTR-GCN uses the independent MMPose 3D
extractor in :mod:`tools.ctrgcn25_pose`; their probabilities are deliberately
not fused because the input skeletons and training domains differ.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.realtime_pool_camera import (  # noqa: E402
    NTU60_NAMES,
    RecognitionWindow,
    draw_label,
    filter_people,
    label_font,
    make_input,
    parser as base_parser,
)
from tools.camera_timing import CameraLog, FrameStatistics, create_pose_worker  # noqa: E402
from tools.remote_camera import create_remote_pose_worker  # noqa: E402
from tools.ctrgcn25_pose import CTRGCN25Window, MMPoseHuman3DWorker, make_input as make_ctrgcn_input  # noqa: E402


def load_25_model(checkpoint_path: Path, device):
    import torch
    from isaa.models.original_ctrgcn import OriginalCTRGCN, build_official_ntu_adjacency

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "data_bn.weight" not in checkpoint:
        raise ValueError("Expected an official raw CTR-GCN state_dict checkpoint")
    # The public checkpoint is trained with the original NTU 25-joint graph.
    graph = build_official_ntu_adjacency()
    model = OriginalCTRGCN(num_classes=60, num_point=25, num_person=2, graph=graph)
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval(), {"source": "github_official", "window_size": 64}, None


def top5(model, x, mask=None):
    import torch
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(x, mask) if mask is not None else model(x)
        probability = logits.softmax(-1)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    values, indices = probability.topk(5)
    return list(zip(indices.tolist(), values.tolist())), probability, elapsed_ms


def format_top5(title: str, entries, elapsed_ms: float, window: str) -> str:
    lines = [f"{title}  ({window}, {elapsed_ms:.0f} ms)"]
    lines.extend(f"{rank}. A{index + 1:03d} {NTU60_NAMES[index]} {confidence:.1%}"
                 for rank, (index, confidence) in enumerate(entries, start=1))
    return "\n".join(lines)


def run(args) -> None:
    import cv2
    import torch

    args.infer_every = max(1, int(args.infer_every or 1))
    args.presence_frames = max(1, int(args.presence_frames or 1))
    args.classify_interval = max(0.0, float(args.classify_interval or 0.0))
    if args.classify_after != 100:
        raise ValueError("--classify-after is fixed at 100 for this comparison")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
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
    from ntu_rtmw.constants import RTMDET_CONFIG, RTMDET_WEIGHTS_PATH, RTMW_CONFIG, RTMW_WEIGHTS_PATH

    device = torch.device(args.device)
    from tools.realtime_pool_camera import load_model as load_pool_model
    pool_model, pool_checkpoint = load_pool_model(Path(args.checkpoint).resolve(), device)
    pool_args = pool_checkpoint.get("args", {})
    if int(pool_args.get("window_size", 100)) != 100:
        raise ValueError("The pooled checkpoint must be trained for 100 frames")
    ctr_model, ctr_checkpoint, _ = load_25_model(Path(args.ctrgcn25_checkpoint).resolve(), device)
    ctr_frames = int(ctr_checkpoint.get("args", {}).get("window_size", 64))
    max_persons = max(int(pool_args.get("max_persons", 2)), 2)
    worker_args = argparse.Namespace(
        pose2d=RTMW_CONFIG, pose2d_weights=str(args.pose_weights or RTMW_WEIGHTS_PATH),
        det_model=RTMDET_CONFIG, det_weights=str(RTMDET_WEIGHTS_PATH),
        device=args.pose_device or args.device, det_backend=args.det_backend,
        yolo_model=str(yolo_model), yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou,
        yolo_imgsz=args.yolo_imgsz, pose_input=args.pose_input, crop_margin=args.crop_margin,
        max_persons=max_persons, bbox_thr=args.bbox_thr, kpt_thr=args.kpt_thr,
        tracking_distance=args.tracking_distance, temporal_min_frames=2,
        temporal_max_jump=150.0, temporal_min_keypoints=5, filter_output_to_bbox=True,
        output_bbox_margin=0.0,
    )
    cap = cv2.VideoCapture(int(args.camera), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open local camera index {args.camera}; try --camera 1 or close other camera apps")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.ssh_host:
        worker = create_remote_pose_worker(argparse.Namespace(
            ssh_host=args.ssh_host, ssh_user=args.ssh_user, ssh_port=args.ssh_port,
            ssh_key=args.ssh_key, remote_python=args.remote_python,
            remote_pose_project=args.remote_pose_project, remote_device=args.remote_device,
            remote_yolo_model=args.remote_yolo_model, remote_timeout=args.remote_timeout,
            jpeg_quality=args.remote_jpeg_quality, yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou, yolo_imgsz=args.yolo_imgsz, kpt_thr=args.kpt_thr,
            max_persons=max_persons,
        ))
    else:
        worker = create_pose_worker(worker_args)
    ctr_worker = MMPoseHuman3DWorker(
        device=args.pose_device or args.device, max_persons=2,
        det_model=RTMDET_CONFIG,
        det_weights=str(RTMDET_WEIGHTS_PATH),
    )
    ctr_state = CTRGCN25Window(window=100, max_persons=2)
    state = RecognitionWindow(100, args.presence_frames)
    last_pose = None
    people = None
    last_classified_at = 0.0
    last_error = None
    frame_idx = 0
    infer_every = max(1, int(args.infer_every or 1))
    started = time.perf_counter()
    statistics = FrameStatistics(started)
    log = CameraLog(args.log_dir)
    print(f"Camera comparison log: {log.path}")
    try:
        worker.start()
        ctr_worker.start()
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Camera frame read failed at frame {frame_idx}; "
                    "the Windows camera driver returned no frame. "
                    "Close other camera applications or try --camera 1."
                )
            captured_at = time.perf_counter()
            statistics.captured(captured_at)
            if args.flip:
                frame = cv2.flip(frame, 1)
            latest, busy, error = worker.snapshot()
            if not busy and frame_idx % infer_every == 0:
                worker.submit(frame, frame_idx, captured_at)
                statistics.submitted_frames += 1
            ctr_latest, ctr_busy, ctr_error = ctr_worker.snapshot()
            if not ctr_busy and frame_idx % infer_every == 0:
                ctr_worker.submit(frame, frame_idx, captured_at)
            if ctr_latest is not None and (ctr_state.last_frame_id != ctr_latest.get("frame_id")):
                ctr_state.observe(ctr_latest)
            if latest is not None and latest is not last_pose:
                last_pose = latest
                source_id = latest.get("frame_id", frame_idx)
                statistics.processed(source_id, latest.get("captured_at", captured_at), latest.get("completed_at", time.perf_counter()))
                people = filter_people(latest, frame.shape, person_threshold=args.person_threshold,
                                       body_threshold=args.body_threshold, min_body_points=args.min_body_points)
                appended = state.observe(people, source_id)
                if appended and state.complete and time.perf_counter() - last_classified_at >= args.classify_interval:
                    x100, mask100 = make_input(state.frames, window=100, max_persons=2)
                    x100, mask100 = x100.to(device), mask100.to(device)
                    pool_top, pool_probability, pool_ms = top5(pool_model, x100, mask100)
                    ctr_top = None
                    ctr_ms = 0.0
                    if ctr_state.complete:
                        x25 = make_ctrgcn_input(ctr_state.frames, window=ctr_frames, max_persons=2).to(device)
                        ctr_top, ctr_probability, ctr_ms = top5(ctr_model, x25)
                    state.top5 = pool_top
                    state.prediction, state.confidence = pool_top[0]
                    state.compare_text = format_top5("池化100帧", pool_top, pool_ms, "100帧/133点")
                    if ctr_top is not None:
                        state.compare_text += ("\n\n" + format_top5(
                            "CTR-GCN25（独立MMPose 3D→NTU25）", ctr_top, ctr_ms,
                            f"{ctr_frames}帧/25点"))
                        print(
                            f"[compare] pool={NTU60_NAMES[pool_top[0][0]]} {pool_top[0][1]:.2%}; "
                            f"ctrgcn25={NTU60_NAMES[ctr_top[0][0]]} {ctr_top[0][1]:.2%}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[compare] pool={NTU60_NAMES[pool_top[0][0]]} {pool_top[0][1]:.2%}; "
                            f"ctrgcn25=collecting {len(ctr_state.frames)}/{ctr_state.frames.maxlen}",
                            flush=True,
                        )
                    last_classified_at = time.perf_counter()
            if error is not None:
                if error is not last_error:
                    print(f"Camera inference failed: {error!r}", file=sys.stderr)
                    last_error = error
                state.clear()
                state.compare_text = None
                people = None
            if ctr_error is not None:
                if ctr_error is not last_error:
                    print(f"CTR-GCN 独立 3D 骨架提取失败: {ctr_error}", file=sys.stderr)
                    last_error = ctr_error
                ctr_state.clear()
            if people is not None and state.present:
                draw_skeleton(frame, people["keypoints"], people["scores"], people["bboxes"], people["bbox_scores"], args.kpt_thr, cv2)
            if getattr(state, "compare_text", None) and state.complete:
                draw_label(frame, state.compare_text, (20, 100, 40), cv2)
            elif people is None or not state.present:
                draw_label(frame, "未检测到有效人物，等待入镜", (100, 100, 100), cv2)
            else:
                draw_label(frame, f"已检测到人物 | 有效帧 {state.valid_count}/100", (180, 100, 20), cv2)
            metrics = statistics.snapshot(time.perf_counter())
            log.write(metrics)
            cv2.putText(frame, f"Read {metrics['read_fps']:.1f} FPS | Pose {metrics['pose_fps']:.1f} FPS | Skipped {metrics['skipped_ratio']:.0%}",
                        (12, frame.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "q/Esc quit", (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("ISAA pooled vs CTR-GCN 25-point", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            frame_idx += 1
    finally:
        worker.stop()
        ctr_worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        log.write(statistics.snapshot(time.perf_counter()), force=True)
        log.close()
    print(f"Camera comparison finished after {time.perf_counter() - started:.1f}s")


def parser() -> argparse.ArgumentParser:
    p = base_parser()
    p.description = __doc__
    p.add_argument("--ctrgcn25-checkpoint", type=Path,
                   default=ROOT / "runtime/ctrgcn25/github_weights/NTU60_Xsub/CTRGCN_joint_89.9/runs-60-37560.pt")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
