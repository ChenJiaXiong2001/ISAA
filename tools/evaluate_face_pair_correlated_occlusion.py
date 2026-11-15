"""Evaluate fixed same-side body/hand occlusion on original frames."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA libraries before importing torch.
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress
from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask
from isaa.face_pair_training import compile_model
from tools.evaluate_face_pair_anchor_gaps import interpolate_short_anchor_gaps
from tools.evaluate_face_pair_corruption import _soft_weights
from tools.evaluate_face_pair_precompression_repair import model_type, select_members, tensor_hash

REGIONS = {"left": (5, 7, 9, 11, *range(91, 112)),
           "right": (6, 8, 10, 12, *range(112, 133))}
LENGTH = 5
SALT = 1729
CHECKPOINT_SHA256 = "aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7"


def choose(member, tag, size):
    if size < 1:
        raise ValueError("Cannot choose from an empty set")
    value = hashlib.sha256(f"{SALT}|{member}|{tag}".encode()).digest()
    return int.from_bytes(value[:8], "big") % size


def corrupt(raw, frames, member):
    if raw.ndim != 4 or raw.shape[0] != 3 or raw.shape[2] != 133 or frames.shape != (raw.shape[1],):
        raise ValueError("Expected raw 3 x time x 133 x people and frame mask")
    body = (raw[2, :, :23, :] > 0) & frames[:, None, None]
    counts = body.sum((0, 1))
    person = int(counts.argmax())
    side = ("left", "right")[choose(member, "side", 2)]
    info = {"person": person, "side": side, "eligible": False, "start": -1,
            "hidden_points": 0, "eligible_windows": 0}
    if not int(counts[person]) or len(frames) < LENGTH:
        return raw.clone(), info
    region = list(REGIONS[side])
    present = (raw[2, :, :23, person] > 0).any(1) & frames
    region_present = (raw[2, :, region, person] > 0).any(1)
    eligible = present & region_present
    starts = eligible.unfold(0, LENGTH, 1).all(1).nonzero().flatten().tolist()
    info["eligible_windows"] = len(starts)
    if not starts:
        return raw.clone(), info
    start = starts[choose(member, "start", len(starts))]
    result = raw.clone()
    points = result[:, start:start + LENGTH, region, person]
    valid = points[2] > 0
    points = points.masked_fill(valid[None], 0)
    result[:, start:start + LENGTH, region, person] = points
    info.update(eligible=True, start=start, hidden_points=int(valid.sum()))
    return result, info


def self_test():
    raw = torch.ones(3, 8, 133, 2)
    raw[:, :, :, 1] = 0
    raw[:, 2, 5, 0] = 0
    frames = torch.ones(8, dtype=torch.bool)
    before = raw.clone()
    a, info = corrupt(raw, frames, "S001C001P001R001A001")
    b, info2 = corrupt(raw, frames, "S001C001P001R001A001")
    assert torch.equal(a, b) and info == info2 and info["eligible"] and info["person"] == 0
    expected = raw.clone()
    region = list(REGIONS[info["side"]])
    points = expected[:, info["start"]:info["start"] + LENGTH, region, 0]
    valid = points[2] > 0
    expected[:, info["start"]:info["start"] + LENGTH, region, 0] = points.masked_fill(valid[None], 0)
    assert torch.equal(a, expected) and torch.equal(raw, before)
    assert info["hidden_points"] == int(valid.sum()) and torch.equal(a[..., 1], raw[..., 1])
    short, short_info = corrupt(raw[:, :4], frames[:4], "short")
    assert torch.equal(short, raw[:, :4]) and not short_info["eligible"]
    missing = raw.clone()
    for region in REGIONS.values():
        missing[:, :, list(region), :] = 0
    unchanged, missing_info = corrupt(missing, frames, "absent")
    assert torch.equal(unchanged, missing) and not missing_info["eligible"]
    frames[3] = False
    _, barrier = corrupt(raw, frames, "barrier")
    assert not barrier["eligible"]
    print("CPU self-test passed: deterministic side/window, original missingness, other person, invalid frames, short input, no mutation")


class Paired(Dataset):
    def __init__(self, dataset, members):
        self.dataset, self.members = dataset, members
        indices = {name: index for index, name in enumerate(dataset.members)}
        if len(indices) != len(dataset.members):
            raise ValueError("Duplicate members")
        self.indices = [indices[name] for name in members]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        raw, frames = self.dataset._read_raw_sample(self.indices[index])
        damaged, info = corrupt(raw, frames, self.members[index])
        inputs = []
        for value in (raw, damaged):
            value, mask = _confidence_weighted_temporal_compress(value, frames, self.dataset.window_size)
            value, mask = _temporal_crop_or_pad_sample_with_mask(value, mask, self.dataset.window_size,
                random_start=False, deterministic_crop=self.dataset.deterministic_temporal_crop)
            value[:2] *= value[2:3] > 0
            inputs.append(value.masked_fill(~mask[None, :, None, None], 0).contiguous())
        return inputs[0], inputs[1], int(self.dataset.labels[self.indices[index]]), mask, info


def scores(output, weights):
    hand = F.log_softmax(output["hand_face_logits"].float(), -1)
    body = F.log_softmax(output["body_face_logits"].float(), -1)
    arithmetic = torch.logaddexp(weights.log() + hand, torch.log1p(-weights) + body)
    arithmetic -= torch.logsumexp(arithmetic, -1, keepdim=True)
    geometric = weights * hand + (1 - weights) * body
    geometric -= torch.logsumexp(geometric, -1, keepdim=True)
    return arithmetic, torch.logaddexp(arithmetic, geometric) - math.log(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--partition", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    options = parser.parse_args()
    if options.self_test:
        self_test()
        return
    if not all((options.checkpoint, options.manifest, options.archive, options.output_dir)):
        parser.error("checkpoint, manifest, archive and output-dir are required")
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if options.max_samples < 0 or options.batch_size < 1 or options.num_workers < 0:
        raise ValueError("Invalid sample, batch, or worker count")
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1 if options.device == "cpu" else 4)
    checkpoint_sha256 = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("Expected the frozen epoch-83 checkpoint")
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args["temporal_compression"] != "confidence_weighted":
        raise ValueError("Expected confidence-weighted compression")
    cls = model_type(args["model_variant"])
    model = cls(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    if float(model.hand_face_temperature) != 1 or float(model.body_face_temperature) != 1:
        raise ValueError("Expected unit head temperatures")
    model = model.to(options.device).eval()
    weights = _soft_weights(model).to(options.device)
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)
    compiled = compile_model(model, args["compile_mode"]) if options.device == "cuda" else model
    section = json.loads(options.manifest.read_text(encoding="utf-8"))[options.partition]
    full = section["members"]
    if hashlib.sha256("\n".join(full).encode()).hexdigest() != section["sha256"]:
        raise ValueError("Manifest digest mismatch")
    members, source = select_members(full, None, options.max_samples)
    dataset = RTMWZipDataset(options.archive, split="train" if options.partition == "calibration" else "val",
        split_protocol=args["split"], window_size=args["window_size"], num_joints=133,
        num_classes=args["num_classes"], layout="rtmw_133", max_persons=args["max_persons"],
        score_normalization=args.get("score_normalization", "auto"), feature_mode="raw", augment=False,
        temporal_compression="confidence_weighted", deterministic_temporal_crop=args.get("deterministic_temporal_crop", "center"))
    loader = DataLoader(Paired(dataset, members), batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers)
    hashes = {name: hashlib.sha256() for name in ("clean", "corrupted", "frame_mask")}
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (clean, damaged, labels, mask, info) in enumerate(loader, 1):
            for name, value in (("clean", clean), ("corrupted", damaged), ("frame_mask", mask)):
                tensor_hash(hashes[name], value)
            batch = [{"sample": members[len(rows) + index], "label": int(label),
                      **{key: value[index].item() if isinstance(value, torch.Tensor) else value[index]
                         for key, value in info.items()}} for index, label in enumerate(labels)]
            clean, damaged, mask = clean.to(options.device), damaged.to(options.device), mask.to(options.device)
            for condition, original in (("clean", clean), ("occlusion", damaged)):
                repaired, _ = interpolate_short_anchor_gaps(original, mask, 5)
                for method, value in (("original", original), ("max5", repaired)):
                    arithmetic, mixed = scores(compiled(value, mask), weights)
                    if not torch.isfinite(arithmetic).all() or not torch.isfinite(mixed).all():
                        raise RuntimeError("Nonfinite fused scores")
                    for name, prediction in (("arithmetic", arithmetic.argmax(-1)), ("geometric_mix", mixed.argmax(-1))):
                        for row, value in zip(batch, prediction.cpu().tolist()):
                            row[f"{condition}_{method}_{name}"] = value
            rows.extend(batch)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    metrics = {}
    for condition in ("clean", "occlusion"):
        metrics[condition] = {}
        base = np.asarray([row[f"{condition}_original_arithmetic"] == row["label"] for row in rows])
        for method in ("original_arithmetic", "original_geometric_mix", "max5_arithmetic", "max5_geometric_mix"):
            hits = np.asarray([row[f"{condition}_{method}"] == row["label"] for row in rows])
            metrics[condition][method] = {"correct": int(hits.sum()), "top1_percent": float(100 * hits.mean()),
                "difference_pp": float(100 * (hits.astype(int) - base.astype(int)).mean()),
                "corrected": int((hits & ~base).sum()), "new_errors": int((base & ~hits).sum())}
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"checkpoint_sha256": checkpoint_sha256,
        "partition": options.partition, "samples": len(rows), "member_source": source,
        "members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
        "input_sha256": {name: value.hexdigest() for name, value in hashes.items()},
        "eligible_samples": sum(row["eligible"] for row in rows), "hidden_points": sum(row["hidden_points"] for row in rows),
        "regions": REGIONS, "original_frame_length": LENGTH, "salt": SALT, "metrics": metrics,
        "compile": "Inductor; existing face LocalSTBlock eager boundary" if options.device == "cuda" else "CPU eager smoke only",
        "seconds": time.perf_counter() - started,
        "caveat": "Synthetic related body/hand loss, not a measured detector failure rate; partitions reused."}
    (options.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
