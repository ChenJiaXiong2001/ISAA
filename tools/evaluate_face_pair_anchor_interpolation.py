"""Compare single-frame anchor interpolation on a frozen model."""
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
import isaa.train  # Configure the CUDA wheel paths before importing torch.
import torch
from torch.utils.data import DataLoader, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights


ANCHORS = (5, 6, 11, 12)


def interpolate_isolated_anchors(features, frame_mask):
    """Fill only a missing anchor surrounded by two valid adjacent frames."""
    anchors = features[:, :, :, list(ANCHORS), :]
    valid = anchors[:, 2] > 0
    frames = frame_mask[:, :, None, None].bool()
    fill = (~valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
            & frames[:, 1:-1] & frames[:, :-2] & frames[:, 2:])
    replacement = (anchors[:, :, :-2] + anchors[:, :, 2:]) * 0.5
    repaired = anchors.clone()
    repaired[:, :, 1:-1] = torch.where(fill[:, None], replacement, anchors[:, :, 1:-1])
    result = features.clone()
    result[:, :, :, list(ANCHORS), :] = repaired
    return result, int(fill.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(f"Output already exists: {options.output_dir}")
    if options.batch_size <= 0 or options.num_workers < 0:
        raise ValueError("Invalid loader configuration")
    if options.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args["model_variant"] != FACE_PAIR_VARIANT:
        raise ValueError("Expected best-face-pair checkpoint")
    model = FacePairFusion(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(options.device).eval()
    weights = _soft_weights(model).detach()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)

    dataset_split = "train" if options.partition == "calibration" else "val"
    dataset = RTMWZipDataset(options.archive, split=dataset_split, split_protocol=args["split"],
        window_size=args["window_size"], num_joints=133, num_classes=args["num_classes"],
        layout="rtmw_133", max_persons=args["max_persons"], feature_mode="raw",
        augment=False, temporal_compression=args["temporal_compression"])
    manifest = json.loads(options.manifest.read_text(encoding="utf-8"))
    members = manifest[options.partition]["members"]
    if hashlib.sha256("\n".join(members).encode()).hexdigest() != manifest[options.partition]["sha256"]:
        raise RuntimeError("Partition manifest digest mismatch")
    index_by_name = {member: i for i, member in enumerate(dataset.members)}
    if len(index_by_name) != len(dataset.members) or len(set(members)) != len(members):
        raise RuntimeError("Duplicate members")
    indices = [index_by_name[name] for name in members]
    loader_args = dict(batch_size=options.batch_size, shuffle=False,
        num_workers=options.num_workers, collate_fn=partial(collate_rtmw, feature_mode="raw"))
    if options.num_workers:
        loader_args.update(prefetch_factor=2, multiprocessing_context="spawn")
    loader = DataLoader(Subset(dataset, indices), **loader_args)
    generator = torch.Generator().manual_seed(options.mask_seed)
    counts = {f"{condition}_{method}": 0 for condition in ("clean", "point05", "point10")
              for method in ("unmodified", "repaired")}
    repairs = dict.fromkeys(("clean", "point05", "point10"), 0)
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (features, labels, frames) in enumerate(loader, 1):
            features, labels, frames = (features.to(options.device), labels.to(options.device),
                                        frames.to(options.device))
            draws = torch.rand((features.shape[0], features.shape[2], 23, features.shape[4]),
                               generator=generator).to(options.device)
            batch_rows = [{"sample": members[len(rows) + i], "label": int(value)}
                          for i, value in enumerate(labels.cpu().tolist())]
            for condition, probability in (("clean", 0.0), ("point05", 0.05), ("point10", 0.10)):
                corrupted = features.clone()
                hidden = ((draws < probability) & (features[:, 2, :, :23, :] > 0))
                corrupted[:, :3, :, :23, :] = corrupted[:, :3, :, :23, :].masked_fill(hidden[:, None], 0)
                repaired, fill_count = interpolate_isolated_anchors(corrupted, frames)
                repairs[condition] += fill_count
                for method, current in (("unmodified", corrupted), ("repaired", repaired)):
                    output = model(current, frames)
                    scores = _soft_logits(output, weights)
                    if not bool(torch.isfinite(scores).all()):
                        raise RuntimeError(f"Nonfinite predictions: {condition} {method}")
                    predicted = scores.argmax(-1)
                    counts[f"{condition}_{method}"] += int((predicted == labels).sum())
                    for row, value in zip(batch_rows, predicted.cpu().tolist()):
                        row[f"{condition}_{method}"] = value
            rows.extend(batch_rows)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if len(rows) != len(members):
        raise RuntimeError("Incomplete evaluation")
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"checkpoint": str(options.checkpoint),
        "checkpoint_sha256": hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
        "samples": len(rows), "split": options.partition,
        "members_sha256": manifest[options.partition]["sha256"],
        "mask_seed": options.mask_seed, "batch_size": options.batch_size,
        "mask_scheme": "same random draws reused for body 5% and 10%",
        "interpolation": "one missing anchor frame between two valid adjacent frames; xy-score mean; no endpoints",
        "anchors": ANCHORS, "repair_counts": repairs,
        "metrics": {name: {"correct": value, "top1_percent": 100.0 * value / len(rows)}
                    for name, value in counts.items()},
        "inference": "frozen eager model, gamma=1 from frozen checkpoint calibration counts",
        "seconds": time.perf_counter() - started}
    (options.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
