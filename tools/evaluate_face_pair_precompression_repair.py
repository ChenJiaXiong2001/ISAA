"""Paired frozen-model screen of anchor loss before temporal compression."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA libraries before importing torch.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress
from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
try:
    from isaa.models.face_pair_fusion import (
        FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT,
        FacePairHandFaceGeometryFusion,
    )
except ImportError:  # Older frozen experiment snapshots only contain baseline.
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT = None
    FacePairHandFaceGeometryFusion = None
try:
    from isaa.models.face_pair_fusion import FACE_PAIR_HAND_MOTION_VARIANT, FacePairHandMotionFusion
except ImportError:
    FACE_PAIR_HAND_MOTION_VARIANT = None
    FacePairHandMotionFusion = None
try:
    from isaa.models.face_pair_fusion import FACE_PAIR_RAW_XY_VARIANT, FacePairRawXYFusion
except ImportError:
    FACE_PAIR_RAW_XY_VARIANT = None
    FacePairRawXYFusion = None
from tools.evaluate_face_pair_anchor_confidence_repair import apply_score_threshold, interpolate_short_anchor_gaps
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights

ANCHORS = (5, 6, 11, 12)

FACE_PAIR_MODELS = {FACE_PAIR_VARIANT: FacePairFusion}
if FACE_PAIR_RAW_XY_VARIANT is not None:
    FACE_PAIR_MODELS[FACE_PAIR_RAW_XY_VARIANT] = FacePairRawXYFusion
if FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT is not None:
    FACE_PAIR_MODELS[FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT] = FacePairHandFaceGeometryFusion
if FACE_PAIR_HAND_MOTION_VARIANT is not None:
    FACE_PAIR_MODELS[FACE_PAIR_HAND_MOTION_VARIANT] = FacePairHandMotionFusion


def model_type(variant):
    """Select the exact frozen model architecture recorded in a checkpoint."""
    selected = FACE_PAIR_MODELS.get(variant)
    if selected is None:
        raise ValueError(f"Unsupported face-pair variant: {variant!r}")
    return selected


def select_members(full_members, members_file, max_samples):
    if len(set(full_members)) != len(full_members):
        raise ValueError("Duplicate manifest members")
    if members_file is None:
        members = full_members if max_samples == 0 else full_members[:max_samples]
        source = {"kind": "manifest" if max_samples == 0 else "manifest_prefix"}
    else:
        if max_samples:
            raise ValueError("--max-samples cannot be combined with --members-file")
        payload = members_file.read_bytes()
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
        if not reader.fieldnames or reader.fieldnames.count("sample") != 1:
            raise ValueError("Members CSV must have exactly one sample column")
        members = [row["sample"] for row in reader]
        if any(not isinstance(name, str) or not name for name in members):
            raise ValueError("Members CSV contains a blank sample")
        if len(set(members)) != len(members):
            raise ValueError("Members CSV contains duplicate samples")
        extra = set(members) - set(full_members)
        if extra:
            raise ValueError(f"Members outside selected manifest: {sorted(extra)[:3]}")
        source = {"kind": "members_file", "path": str(members_file.resolve()),
                  "sha256": hashlib.sha256(payload).hexdigest(), "order": "members file row order"}
    if not members:
        raise ValueError("Empty evaluation partition")
    return members, source


def member_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class PairedPrecompression(Dataset):
    def __init__(self, dataset, members, probability, seed):
        self.dataset, self.members = dataset, members
        self.probability, self.seed = probability, seed
        index = {name: i for i, name in enumerate(dataset.members)}
        if len(index) != len(dataset.members) or len(set(members)) != len(members):
            raise ValueError("Duplicate members")
        self.indices = [index[name] for name in members]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        raw, frame = self.dataset._read_raw_sample(self.indices[i])
        rng = torch.Generator().manual_seed(member_seed(self.members[i], self.seed))
        anchor = raw[:, :, list(ANCHORS), :]
        hidden = (torch.rand(anchor[2].shape, generator=rng) < self.probability)
        hidden &= (anchor[2] > 0) & frame[:, None, None]
        corrupt = raw.clone()
        corrupt[:, :, list(ANCHORS), :] = anchor.masked_fill(hidden[None], 0)
        outputs = []
        for value in (raw, corrupt):
            value, mask = _confidence_weighted_temporal_compress(value, frame, self.dataset.window_size)
            value, mask = _temporal_crop_or_pad_sample_with_mask(
                value, mask, self.dataset.window_size, random_start=False,
                deterministic_crop=self.dataset.deterministic_temporal_crop)
            value[:2] *= value[2:3] > 0
            value = value.masked_fill(~mask[None, :, None, None], 0)
            outputs.append(value.contiguous())
        return outputs[0], outputs[1], int(self.dataset.labels[self.indices[i]]), mask, int(hidden.sum()), raw.shape[1]


def tensor_hash(digest, *values):
    for value in values:
        digest.update(value.contiguous().cpu().numpy().tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--partition", choices=("calibration", "validation"), default="calibration")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--members-file", type=Path, help="CSV with unique sample names from the selected manifest, in evaluation order")
    p.add_argument("--probability", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--threshold", type=float, action="append")
    p.add_argument("--max5-only", action="store_true", help="Evaluate unmodified and max5 only")
    p.add_argument(
        "--include-head-predictions",
        action="store_true",
        help="Also save HF/BF argmax columns for read-only fusion-gate analysis",
    )
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    o = p.parse_args()
    if o.max5_only and (o.threshold is not None or o.max_gap != 5):
        raise ValueError("--max5-only requires --max-gap 5 and no --threshold")
    thresholds = [] if o.max5_only else (o.threshold if o.threshold is not None else [0.95, 0.98])
    if o.output_dir.exists():
        raise FileExistsError(o.output_dir)
    if not (0 <= o.probability <= 1) or o.max_samples < 0 or o.max_gap < 1 or o.batch_size < 1 or o.num_workers < 0:
        raise ValueError("Invalid evaluation parameters")
    if any(not 0 < t < 1 for t in thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be unique and in (0, 1)")
    if o.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1 if o.device == "cpu" else 4)
    checkpoint = torch.load(o.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args["temporal_compression"] != "confidence_weighted":
        raise ValueError("Expected confidence-weighted temporal compression")
    selected_model = model_type(args["model_variant"])
    model = selected_model(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(o.device).eval()
    weights = _soft_weights(model).to(o.device)
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)
    manifest = json.loads(o.manifest.read_text(encoding="utf-8"))[o.partition]
    full_members = manifest["members"]
    if hashlib.sha256("\n".join(full_members).encode()).hexdigest() != manifest["sha256"]:
        raise ValueError("Manifest digest mismatch")
    members, member_source = select_members(full_members, o.members_file, o.max_samples)
    dataset = RTMWZipDataset(o.archive, split="train" if o.partition == "calibration" else "val",
        split_protocol=args["split"], window_size=args["window_size"], num_joints=133,
        num_classes=args["num_classes"], layout="rtmw_133", max_persons=args["max_persons"],
        score_normalization=args.get("score_normalization", "auto"), feature_mode="raw", augment=False,
        temporal_compression="confidence_weighted", deterministic_temporal_crop=args.get("deterministic_temporal_crop", "center"))
    paired = PairedPrecompression(dataset, members, o.probability, o.seed)
    loader = DataLoader(paired, batch_size=o.batch_size, shuffle=False, num_workers=o.num_workers)
    methods = ["unmodified", f"max{o.max_gap}"] + [f"threshold_{t:g}_max{o.max_gap}" for t in thresholds]
    hashes = {name: hashlib.sha256() for name in ("clean", "corrupted", "frame_mask")}
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (clean, corrupted, labels, frames, hidden, original_frames) in enumerate(loader, 1):
            tensor_hash(hashes["clean"], clean)
            tensor_hash(hashes["corrupted"], corrupted)
            tensor_hash(hashes["frame_mask"], frames)
            clean, corrupted, labels, frames = (x.to(o.device) for x in (clean, corrupted, labels, frames))
            batch = [{"sample": members[len(rows) + i], "label": int(y), "hidden_original_points": int(h),
                      "original_frames": int(n)} for i, (y, h, n) in enumerate(zip(labels.cpu(), hidden, original_frames))]
            for condition, original in (("clean", clean), ("pre_dropout", corrupted)):
                variants = {"unmodified": original, f"max{o.max_gap}": interpolate_short_anchor_gaps(original, frames, o.max_gap)[0]}
                for threshold in thresholds:
                    thresholded, _ = apply_score_threshold(original, threshold)
                    variants[f"threshold_{threshold:g}_max{o.max_gap}"] = interpolate_short_anchor_gaps(thresholded, frames, o.max_gap)[0]
                for method in methods:
                    output = model(variants[method], frames)
                    logits = _soft_logits(output, weights)
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError(f"Nonfinite logits: {condition}/{method}")
                    for row, pred in zip(batch, logits.argmax(-1).cpu().tolist()):
                        row[f"{condition}_{method}_prediction"] = pred
                    if o.include_head_predictions:
                        for head_name, head_key in (("HF", "hand_face_logits"), ("BF", "body_face_logits")):
                            head_logits = output[head_key]
                            if not bool(torch.isfinite(head_logits).all()):
                                raise RuntimeError(f"Nonfinite {head_name} logits: {condition}/{method}")
                            for row, pred in zip(batch, head_logits.argmax(-1).cpu().tolist()):
                                row[f"{condition}_{method}_{head_name}_prediction"] = pred
            rows.extend(batch)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    metrics = {}
    for condition in ("clean", "pre_dropout"):
        metrics[condition] = {}
        base = np.array([int(row[f"{condition}_unmodified_prediction"] == row["label"]) for row in rows])
        for method in methods:
            hits = np.array([int(row[f"{condition}_{method}_prediction"] == row["label"]) for row in rows])
            metrics[condition][method] = {"correct": int(hits.sum()), "top1_percent": float(hits.mean() * 100),
                "difference_pp_vs_unmodified": float((hits - base).mean() * 100),
                "corrected": int(((hits == 1) & (base == 0)).sum()), "new_errors": int(((hits == 0) & (base == 1)).sum())}
    o.output_dir.mkdir(parents=True, exist_ok=False)
    with (o.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"checkpoint": str(o.checkpoint), "checkpoint_sha256": hashlib.sha256(o.checkpoint.read_bytes()).hexdigest(),
        "partition": o.partition, "samples": len(rows), "members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
        "member_source": member_source,
        "manifest_sha256": manifest["sha256"], "input_sha256": {k: v.hexdigest() for k, v in hashes.items()},
        "probability": o.probability, "seed": o.seed, "thresholds": thresholds, "max_gap": o.max_gap,
        "original_hidden_points": sum(row["hidden_original_points"] for row in rows),
        "compressed_samples": sum(row["original_frames"] > args["window_size"] for row in rows),
        "mask_scheme": "independent valid anchor/frame/person dropout before compression; member-specific SHA256 seed",
        "fusion": "frozen checkpoint training-subject gamma=1 calibration counts", "device": o.device,
        "metrics": metrics, "seconds": time.perf_counter()-started,
        "caveat": "synthetic original-frame dropout; calibration partition also supplied checkpoint fusion weights"}
    (o.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
