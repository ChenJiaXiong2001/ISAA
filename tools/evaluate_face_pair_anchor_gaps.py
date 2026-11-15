"""Stress-test fixed bounded anchor-gap repair on a fixed model partition."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA wheel paths before importing torch.
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_VARIANT,
    FacePairFusion, FacePairHandFaceGeometryFusion,
)
try:
    from isaa.models.face_pair_fusion import FACE_PAIR_HAND_MOTION_VARIANT, FacePairHandMotionFusion
except ImportError:  # Older experiment snapshots do not contain this variant.
    FACE_PAIR_HAND_MOTION_VARIANT = "best-face-pair-hand-motion"
    FacePairHandMotionFusion = None
try:
    from isaa.models.face_pair_fusion import FACE_PAIR_RAW_XY_VARIANT, FacePairRawXYFusion
except ImportError:  # Older experiment snapshots only contain the baseline model.
    FACE_PAIR_RAW_XY_VARIANT = "best-face-pair-raw-xy"
    FacePairRawXYFusion = None
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_anchor_interpolation import ANCHORS, interpolate_isolated_anchors
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights


def contiguous_hidden(draws, valid_frames, length, start_probability):
    """Expand independent starts forward; truncate at the sequence boundary."""
    starts = (draws < start_probability) & valid_frames[:, :, None, None]
    batch, time, joints, people = starts.shape
    starts = starts.permute(0, 2, 3, 1).reshape(-1, 1, time).float()
    expanded = F.max_pool1d(F.pad(starts, (length - 1, 0)), kernel_size=length, stride=1)
    return expanded.reshape(batch, joints, people, time).permute(0, 3, 1, 2).bool()


def interpolate_short_anchor_gaps(features, frame_mask, max_gap):
    """Interpolate bounded anchor gaps without crossing invalid frames."""
    anchors = features[:, :, :, list(ANCHORS), :]
    batch, _, time, joints, people = anchors.shape
    position = torch.arange(time, device=features.device).view(1, time, 1, 1).expand(batch, time, joints, people)
    frames = frame_mask.bool()
    valid = (anchors[:, 2] > 0) & frames[:, :, None, None]
    previous = torch.where(valid, position, -1).cummax(dim=1).values
    after = torch.where(valid, position, time).flip(1).cummin(dim=1).values.flip(1)
    gap = after - previous - 1
    invalid_prefix = F.pad((~frames).long().cumsum(dim=1), (1, 0))
    invalid_prefix = invalid_prefix[:, :, None, None].expand(batch, time + 1, joints, people)
    lo = previous.clamp(0, time - 1)
    hi = after.clamp(0, time - 1)
    invalid_between = invalid_prefix.gather(1, hi + 1) - invalid_prefix.gather(1, lo)
    fill = (~valid & frames[:, :, None, None] & (previous >= 0) & (after < time)
            & (gap <= max_gap) & (invalid_between == 0))
    gather_shape = (-1, 3, -1, -1, -1)
    left = anchors.gather(2, lo[:, None].expand(*gather_shape))
    right = anchors.gather(2, hi[:, None].expand(*gather_shape))
    alpha = ((position - previous).float() / (after - previous).clamp_min(1).float())[:, None]
    replacement = left + alpha * (right - left)
    result = features.clone()
    result[:, :, :, list(ANCHORS), :] = torch.where(fill[:, None], replacement, anchors)
    return result, int(fill.sum())


def model_type(variant):
    models = {FACE_PAIR_VARIANT: FacePairFusion,
              FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT: FacePairHandFaceGeometryFusion}
    if FacePairHandMotionFusion is not None:
        models[FACE_PAIR_HAND_MOTION_VARIANT] = FacePairHandMotionFusion
    if FacePairRawXYFusion is not None:
        models[FACE_PAIR_RAW_XY_VARIANT] = FacePairRawXYFusion
    selected = models.get(variant)
    if selected is None:
        raise ValueError(f"Expected face-pair checkpoint, got model_variant={variant!r}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("calibration", "validation"), default="calibration",
                        help="Manifest partition to evaluate; calibration remains the default")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare-max-gap", action="store_true",
                        help="Also compare fixed maximum-2 and maximum-5-frame repairs")
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure synchronized model and repair latency on the first calibration batch")
    opts = parser.parse_args()
    if opts.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {opts.output_dir}")
    if opts.batch_size <= 0 or opts.num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    if opts.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    checkpoint = torch.load(opts.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model_class = model_type(args["model_variant"])
    model = model_class(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(opts.device).eval()
    weights = _soft_weights(model).detach()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)

    dataset_split = "train" if opts.partition == "calibration" else "val"
    dataset = RTMWZipDataset(opts.archive, split=dataset_split, split_protocol=args["split"],
        window_size=args["window_size"], num_joints=133, num_classes=args["num_classes"],
        layout="rtmw_133", max_persons=args["max_persons"], feature_mode="raw",
        augment=False, temporal_compression=args["temporal_compression"])
    manifest = json.loads(opts.manifest.read_text(encoding="utf-8"))
    members = manifest[opts.partition]["members"]
    digest = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if digest != manifest[opts.partition]["sha256"]:
        raise RuntimeError(f"{opts.partition} manifest digest mismatch")
    index_by_name = {member: i for i, member in enumerate(dataset.members)}
    if len(index_by_name) != len(dataset.members) or len(set(members)) != len(members):
        raise RuntimeError("Duplicate dataset or manifest members")
    missing = [member for member in members if member not in index_by_name]
    if missing:
        raise RuntimeError(f"{opts.partition} members missing from dataset: {missing[:3]}")
    indices = [index_by_name[name] for name in members]
    loader_args = dict(batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                       collate_fn=partial(collate_rtmw, feature_mode="raw"))
    if opts.num_workers:
        loader_args.update(prefetch_factor=2, multiprocessing_context="spawn")
    loader = DataLoader(Subset(dataset, indices), **loader_args)

    # Fixed before observing results: approximately 5% marginal anchor loss.
    conditions = (("clean", 1, 0.0), ("independent05", 1, 0.05),
                  ("gap2", 2, 0.025), ("gap5", 5, 0.01), ("gap10", 10, 0.005))
    methods = ("unmodified", "repaired", "max2", "max5") if opts.compare_max_gap else ("unmodified", "repaired")
    counts = {f"{name}_{method}": {head: 0 for head in ("HF", "BF", "soft")}
              for name, _, _ in conditions for method in methods}
    hidden_counts = dict.fromkeys((name for name, _, _ in conditions), 0)
    synthetic_filled = {name: dict.fromkeys(methods, 0) for name, _, _ in conditions}
    original_filled = {name: dict.fromkeys(methods, 0) for name, _, _ in conditions}
    anchor_valid_total = 0
    rng = torch.Generator().manual_seed(opts.mask_seed)
    rows = []
    benchmark_input = benchmark_frames = None
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (features, labels, frames) in enumerate(loader, 1):
            features, labels, frames = features.to(opts.device), labels.to(opts.device), frames.to(opts.device)
            draws = torch.rand((features.shape[0], features.shape[2], len(ANCHORS), features.shape[4]),
                               generator=rng).to(opts.device)
            valid = (features[:, 2, :, list(ANCHORS), :] > 0) & frames[:, :, None, None].bool()
            anchor_valid_total += int(valid.sum())
            batch_rows = [{"sample": members[len(rows) + i], "label": int(y)}
                          for i, y in enumerate(labels.cpu().tolist())]
            for name, length, start_probability in conditions:
                hidden = contiguous_hidden(draws, frames.bool(), length, start_probability) & valid
                corrupted = features.clone()
                anchors = corrupted[:, :3, :, list(ANCHORS), :]
                anchors = anchors.masked_fill(hidden[:, None], 0)
                corrupted[:, :3, :, list(ANCHORS), :] = anchors
                hidden_counts[name] += int(hidden.sum())
                if opts.benchmark and name == "independent05" and benchmark_input is None:
                    benchmark_input, benchmark_frames = corrupted.clone(), frames.clone()
                for method in methods:
                    if method == "unmodified":
                        current, total_filled = corrupted, 0
                    elif method == "repaired":
                        current, total_filled = interpolate_isolated_anchors(corrupted, frames)
                    else:
                        current, total_filled = interpolate_short_anchor_gaps(corrupted, frames,
                                                                               2 if method == "max2" else 5)
                    restored = hidden & (current[:, 2, :, list(ANCHORS), :] > 0)
                    synthetic_filled[name][method] += int(restored.sum())
                    original_filled[name][method] += total_filled - int(restored.sum())
                    output = model(current, frames)
                    for head, scores in (("HF", output["hand_face_logits"]),
                                         ("BF", output["body_face_logits"]),
                                         ("soft", _soft_logits(output, weights))):
                        if not bool(torch.isfinite(scores).all()):
                            raise RuntimeError(f"Nonfinite {name} {method} {head} logits")
                        prediction = scores.argmax(-1)
                        counts[f"{name}_{method}"][head] += int((prediction == labels).sum())
                        for row, value in zip(batch_rows, prediction.cpu().tolist()):
                            row[f"{name}_{method}_{head}"] = value
            rows.extend(batch_rows)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if len(rows) != len(members):
        raise RuntimeError("Incomplete evaluation")
    timing = {}
    if opts.benchmark:
        def synchronize():
            if opts.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(opts.device))

        def timed(current, frames, method, include_model):
            if method == "unmodified":
                repaired = current
            elif method == "repaired":
                repaired, _ = interpolate_isolated_anchors(current, frames)
            else:
                repaired, _ = interpolate_short_anchor_gaps(current, frames, 2 if method == "max2" else 5)
            if include_model:
                _soft_logits(model(repaired, frames), weights)

        print("benchmark_started", flush=True)
        with torch.inference_mode():
            for size in (1, min(opts.batch_size, len(benchmark_input))):
                current, frames = benchmark_input[:size], benchmark_frames[:size]
                group = timing.setdefault(str(size), {})
                for method in methods:
                    group[method] = {}
                    for include_model in (False, True):
                        if method == "unmodified" and not include_model:
                            continue
                        for _ in range(5):
                            timed(current, frames, method, include_model)
                        synchronize()
                        durations = []
                        for _ in range(20):
                            start = time.perf_counter()
                            timed(current, frames, method, include_model)
                            synchronize()
                            durations.append(1000 * (time.perf_counter() - start))
                        durations.sort()
                        group[method]["repair_and_model" if include_model else "repair_only"] = {
                            "median_ms_per_batch": (durations[9] + durations[10]) / 2,
                            "p95_ms_per_batch": durations[18], "repeats": len(durations)}
    opts.output_dir.mkdir(parents=True, exist_ok=False)
    with (opts.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"checkpoint": str(opts.checkpoint),
        "checkpoint_sha256": hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest(),
        "partition": "training_subject_calibration" if opts.partition == "calibration" else "validation",
        "dataset_split": dataset_split, "members_sha256": digest,
        "samples": len(rows),
        "weight_source": ("same checkpoint's training-subject calibration counts; this is not an independent test"
                          if opts.partition == "calibration" else
                          "same checkpoint's training-subject calibration counts; validation is independent of that calibration set"),
        "mask_seed": opts.mask_seed, "batch_size": opts.batch_size,
        "mask_scheme": "shared per-anchor/frame/person starts, expanded forward for fixed spans; starts restricted to valid frames",
        "interpolation": "bounded gap linear interpolation of xy and score; valid endpoints and no invalid frames; no extrapolation",
        "valid_anchor_point_frames": anchor_valid_total,
        "conditions": {name: {"duration_frames": length, "start_probability": probability,
            "hidden_valid_point_frames": hidden_counts[name],
            "hidden_fraction_of_valid_anchor_point_frames": hidden_counts[name] / anchor_valid_total,
            "synthetic_hidden_point_frames_filled": synthetic_filled[name],
            "other_preexisting_missing_point_frames_filled": original_filled[name],
            "metrics": {method: {head: {"correct": count, "top1_percent": 100 * count / len(rows)}
                                 for head, count in counts[f"{name}_{method}"].items()}
                        for method in methods}}
            for name, length, probability in conditions},
        "inference": "frozen eager model, HF/BF and gamma=1 from checkpoint calibration counts",
        "seconds": time.perf_counter() - started}
    if opts.benchmark:
        result["timing"] = {"device": opts.device, "synchronized": True,
            "scope": "repair plus frozen eager model and gamma=1 readout; already-loaded compressed 100-frame inputs; excludes detector, archive loading and any look-ahead delay",
            "warmups": 5, "results_by_batch_size": timing}
    (opts.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
