"""Evaluate low-confidence anchor repair on a frozen face-pair checkpoint.

This is a standalone companion to the existing anchor-gap evaluator. It first
turns nonzero anchor scores at or below one or more fixed thresholds into
missing anchors, then optionally applies the existing bounded max-gap repair.
The original evaluator and all training entry points are unchanged.
"""
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
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_anchor_gaps import ANCHORS, contiguous_hidden
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights


def apply_score_threshold(features: torch.Tensor, threshold: float) -> tuple[torch.Tensor, int]:
    """Mark low-confidence nonzero anchor points missing without changing others."""
    result = features.clone()
    scores = result[:, 2, :, list(ANCHORS), :]
    low = (scores > 0) & (scores <= threshold)
    result[:, :3, :, list(ANCHORS), :] = result[:, :3, :, list(ANCHORS), :].masked_fill(
        low[:, None], 0
    )
    return result, int(low.sum())


def interpolate_short_anchor_gaps(
    features: torch.Tensor, frame_mask: torch.Tensor, max_gap: int
) -> tuple[torch.Tensor, int]:
    """Fill bounded anchor gaps without crossing invalid sequence frames.

    This copy keeps the new evaluator compatible with older experiment
    snapshots whose gap evaluator predates the max-gap helper.
    """
    anchors = features[:, :, :, list(ANCHORS), :]
    batch, _, time, joints, people = anchors.shape
    position = torch.arange(time, device=features.device).view(1, time, 1, 1)
    position = position.expand(batch, time, joints, people)
    frames = frame_mask.bool()
    valid = (anchors[:, 2] > 0) & frames[:, :, None, None]
    previous = torch.where(valid, position, -1).cummax(dim=1).values
    following = torch.where(valid, position, time).flip(1).cummin(dim=1).values.flip(1)
    gap = following - previous - 1
    invalid_prefix = F.pad((~frames).long().cumsum(dim=1), (1, 0))
    invalid_prefix = invalid_prefix[:, :, None, None].expand(batch, time + 1, joints, people)
    lo = previous.clamp(0, time - 1)
    hi = following.clamp(0, time - 1)
    invalid_between = invalid_prefix.gather(1, hi + 1) - invalid_prefix.gather(1, lo)
    fill = (
        ~valid
        & frames[:, :, None, None]
        & (previous >= 0)
        & (following < time)
        & (gap <= max_gap)
        & (invalid_between == 0)
    )
    gather_shape = (-1, 3, -1, -1, -1)
    left = anchors.gather(2, lo[:, None].expand(*gather_shape))
    right = anchors.gather(2, hi[:, None].expand(*gather_shape))
    alpha = ((position - previous).float() / (following - previous).clamp_min(1).float())[:, None]
    replacement = left + alpha * (right - left)
    result = features.clone()
    result[:, :, :, list(ANCHORS), :] = torch.where(fill[:, None], replacement, anchors)
    return result, int(fill.sum())


def _threshold_name(value: float) -> str:
    return f"threshold_{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("calibration", "validation"), default="validation")
    parser.add_argument("--max-samples", type=int, default=4096,
                        help="Use a deterministic prefix; zero evaluates the whole partition")
    parser.add_argument("--score-threshold", type=float, action="append", default=None,
                        help="Convert positive anchor scores <= threshold to missing; repeatable")
    parser.add_argument("--max-gap", type=int, default=5,
                        help="Bound for threshold-plus-interpolation repair")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    opts = parser.parse_args()
    if opts.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {opts.output_dir}")
    if opts.max_samples < 0 or opts.max_gap < 1 or opts.batch_size <= 0 or opts.num_workers < 0:
        raise ValueError("Invalid max-samples, max-gap, batch-size or num-workers")
    # The calibration score audit places almost all positive anchor scores above
    # 0.9, so lower defaults would be an effectively empty intervention.
    thresholds = [0.95, 0.98] if opts.score_threshold is None else list(opts.score_threshold)
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("score thresholds must be finite values in (0, 1)")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("duplicate score thresholds")
    if opts.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    checkpoint = torch.load(opts.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args["model_variant"] != FACE_PAIR_VARIANT:
        raise ValueError("Expected best-face-pair checkpoint")
    model = FacePairFusion(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(opts.device).eval()
    weights = _soft_weights(model).detach()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)

    manifest = json.loads(opts.manifest.read_text(encoding="utf-8"))
    full_members = manifest[opts.partition]["members"]
    full_digest = hashlib.sha256("\n".join(full_members).encode()).hexdigest()
    if full_digest != manifest[opts.partition]["sha256"]:
        raise RuntimeError(f"{opts.partition} manifest digest mismatch")
    members = full_members if opts.max_samples == 0 else full_members[:opts.max_samples]
    if not members:
        raise ValueError("empty partition")
    used_digest = hashlib.sha256("\n".join(members).encode()).hexdigest()

    dataset_split = "train" if opts.partition == "calibration" else "val"
    dataset = RTMWZipDataset(
        opts.archive, split=dataset_split, split_protocol=args["split"],
        window_size=args["window_size"], num_joints=133, num_classes=args["num_classes"],
        layout="rtmw_133", max_persons=args["max_persons"], feature_mode="raw",
        augment=False, temporal_compression=args["temporal_compression"])
    index_by_name = {member: i for i, member in enumerate(dataset.members)}
    missing = [member for member in members if member not in index_by_name]
    if missing:
        raise RuntimeError(f"Manifest members missing from dataset: {missing[:3]}")
    indices = [index_by_name[name] for name in members]
    loader_args = dict(batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                       collate_fn=partial(collate_rtmw, feature_mode="raw"))
    if opts.num_workers:
        loader_args.update(prefetch_factor=2, multiprocessing_context="spawn")
    loader = DataLoader(Subset(dataset, indices), **loader_args)

    conditions = (("clean", 1, 0.0), ("independent05", 1, 0.05),
                  ("gap2", 2, 0.025), ("gap5", 5, 0.01), ("gap10", 10, 0.005))
    methods = ["unmodified", f"max{opts.max_gap}"]
    for threshold in thresholds:
        name = _threshold_name(threshold)
        methods.extend((name, f"{name}_max{opts.max_gap}"))
    heads = ("HF", "BF", "soft")
    counts = {f"{condition}_{method}": dict.fromkeys(heads, 0)
              for condition, _, _ in conditions for method in methods}
    hidden_counts = dict.fromkeys((condition for condition, _, _ in conditions), 0)
    thresholded_counts = {
        f"{condition}_{name}": 0
        for condition, _, _ in conditions
        for name in methods
        if name.startswith("threshold_") and not name.endswith(f"_max{opts.max_gap}")
    }
    synthetic_filled = {f"{condition}_{method}": 0 for condition, _, _ in conditions for method in methods}
    threshold_filled = {f"{condition}_{method}": 0 for condition, _, _ in conditions for method in methods}
    rng = torch.Generator().manual_seed(opts.mask_seed)
    rows: list[dict[str, int]] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for step, (features, labels, frames) in enumerate(loader, 1):
            features, labels, frames = features.to(opts.device), labels.to(opts.device), frames.to(opts.device)
            draws = torch.rand((features.shape[0], features.shape[2], len(ANCHORS), features.shape[4]),
                               generator=rng).to(opts.device)
            valid = (features[:, 2, :, list(ANCHORS), :] > 0) & frames[:, :, None, None].bool()
            batch_rows = [{"sample": members[len(rows) + i], "label": int(value)}
                          for i, value in enumerate(labels.cpu().tolist())]
            for condition, length, start_probability in conditions:
                hidden = contiguous_hidden(draws, frames.bool(), length, start_probability) & valid
                corrupted = features.clone()
                corrupted[:, :3, :, list(ANCHORS), :] = corrupted[:, :3, :, list(ANCHORS), :].masked_fill(
                    hidden[:, None], 0)
                hidden_counts[condition] += int(hidden.sum())
                base_inputs = {"unmodified": corrupted,
                               f"max{opts.max_gap}": interpolate_short_anchor_gaps(
                                   corrupted, frames, opts.max_gap)[0]}
                low_masks = {}
                for threshold in thresholds:
                    name = _threshold_name(threshold)
                    scores = corrupted[:, 2, :, list(ANCHORS), :]
                    low_masks[name] = (scores > 0) & (scores <= threshold)
                    thresholded, threshold_count = apply_score_threshold(corrupted, threshold)
                    thresholded_counts[f"{condition}_{name}"] += threshold_count
                    repaired, _ = interpolate_short_anchor_gaps(thresholded, frames, opts.max_gap)
                    base_inputs[name] = thresholded
                    base_inputs[f"{name}_max{opts.max_gap}"] = repaired
                for method in methods:
                    current = base_inputs[method]
                    repaired_mask = current[:, 2, :, list(ANCHORS), :] > 0
                    synthetic_filled[f"{condition}_{method}"] += int((hidden & repaired_mask).sum())
                    if method.startswith("threshold_"):
                        before_name = method.removesuffix(f"_max{opts.max_gap}")
                        thresholded_mask = low_masks[before_name]
                        threshold_filled[f"{condition}_{method}"] += int(
                            (thresholded_mask & repaired_mask).sum())
                    output = model(current, frames)
                    for head_name in ("hand_face_logits", "body_face_logits", "logits"):
                        if not bool(torch.isfinite(output[head_name]).all()):
                            raise RuntimeError(f"Nonfinite {condition} {method} {head_name}")
                    predictions = {
                        "HF": output["hand_face_logits"].argmax(-1),
                        "BF": output["body_face_logits"].argmax(-1),
                        "soft": _soft_logits(output, weights).argmax(-1),
                    }
                    for head, prediction in predictions.items():
                        counts[f"{condition}_{method}"][head] += int((prediction == labels).sum())
                        for row, value in zip(batch_rows, prediction.cpu().tolist()):
                            row[f"{condition}_{method}_{head}"] = int(value)
            rows.extend(batch_rows)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if len(rows) != len(members):
        raise RuntimeError("Incomplete evaluation")

    opts.output_dir.mkdir(parents=True, exist_ok=False)
    with (opts.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(opts.checkpoint),
        "checkpoint_sha256": hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest(),
        "partition": opts.partition,
        "samples": len(rows),
        "members_sha256": used_digest,
        "full_manifest_members_sha256": full_digest,
        "prefix_evaluation": opts.max_samples != 0,
        "mask_seed": opts.mask_seed,
        "conditions": {},
        "thresholds": thresholds,
        "max_gap": opts.max_gap,
        "score_rule": "positive anchor score <= threshold is zeroed before bounded interpolation",
        "inference": "frozen eager FacePairFusion; gamma=1 weights from checkpoint calibration counts",
        "seconds": time.perf_counter() - started,
    }
    for condition, length, start_probability in conditions:
        condition_result = {
            "duration_frames": length,
            "start_probability": start_probability,
            "hidden_valid_point_frames": hidden_counts[condition],
            "methods": {},
        }
        for method in methods:
            key = f"{condition}_{method}"
            condition_result["methods"][method] = {
                head: {"correct": counts[key][head], "top1_percent": 100 * counts[key][head] / len(rows)}
                for head in heads
            }
            condition_result["methods"][method]["synthetic_hidden_point_frames_filled"] = synthetic_filled[key]
            if method.startswith("threshold_"):
                threshold_name = method.removesuffix(f"_max{opts.max_gap}")
                condition_result["methods"][method]["thresholded_point_frames"] = thresholded_counts[
                    f"{condition}_{threshold_name}"
                ]
                condition_result["methods"][method]["thresholded_point_frames_filled"] = threshold_filled[key]
        summary["conditions"][condition] = condition_result
    (opts.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
