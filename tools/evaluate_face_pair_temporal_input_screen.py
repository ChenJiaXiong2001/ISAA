"""Screen deterministic temporal input construction on a frozen face-pair model."""
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
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights


SCREEN_MODES = (
    ("confidence_weighted", "confidence_weighted", "center"),
    ("none_start", "none", "start"),
    ("none_center", "none", "center"),
    ("none_end", "none", "end"),
)


def mode_specs():
    """Return the fixed screen modes in their recorded order."""
    return SCREEN_MODES


def _digest_update(digest, members, features, labels, frame_mask):
    """Include each member name and its exact collated input in order."""
    for index, member in enumerate(members):
        digest.update(member.encode("utf-8"))
        digest.update(b"\0")
        for value in (features[index], labels[index], frame_mask[index]):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))


def _member_digest(members):
    return hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_counts(predictions, labels):
    labels = labels.detach().cpu()
    return {
        name: {"correct": int((prediction == labels).sum()),
               "top1_percent": float(100 * (prediction == labels).float().mean())}
        for name, prediction in predictions.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Evaluate only the first N members of the fixed validation manifest")
    parser.add_argument("--prefix-samples", type=int, default=4096,
                        help="Fixed validation-manifest prefix reported as the short result")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0 or args.prefix_samples <= 0 or args.max_samples < 0:
        raise ValueError("batch-size/num-workers/prefix-samples are invalid")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing an eager CPU fallback")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    if saved_args["model_variant"] != FACE_PAIR_VARIANT:
        raise ValueError("Expected best-face-pair checkpoint")
    model = FacePairFusion(saved_args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(args.device).eval()
    weights = _soft_weights(model).detach()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_members = manifest["validation"]["members"]
    expected_member_digest = manifest["validation"]["sha256"]
    if _member_digest(manifest_members) != expected_member_digest:
        raise RuntimeError("Validation manifest digest mismatch")
    members = manifest_members
    if args.max_samples:
        if args.max_samples > len(manifest_members):
            raise ValueError("max-samples exceeds validation manifest")
        members = manifest_members[:args.max_samples]
    if args.prefix_samples > len(members):
        raise ValueError("prefix-samples exceeds validation manifest")
    archive_digest = _file_sha256(args.archive)

    all_results = {}
    all_prediction_rows = []
    started = time.perf_counter()
    for mode_name, compression, crop in mode_specs():
        dataset = RTMWZipDataset(
            args.archive, split="val", split_protocol=saved_args["split"],
            window_size=saved_args["window_size"], num_joints=133,
            num_classes=saved_args["num_classes"], layout="rtmw_133",
            max_persons=saved_args["max_persons"], feature_mode="raw", augment=False,
            max_samples=len(members) if args.max_samples else 0,
            deterministic_temporal_crop=crop, temporal_compression=compression,
        )
        if list(dataset.members) != members:
            raise RuntimeError(f"Dataset member order differs for {mode_name}")
        indices = list(range(len(dataset)))
        loader_options = dict(batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=partial(collate_rtmw, feature_mode="raw"))
        if args.num_workers:
            loader_options.update(prefetch_factor=2, multiprocessing_context="spawn")
        loader = DataLoader(Subset(dataset, indices), **loader_options)
        digests = {"full": hashlib.sha256(), "prefix": hashlib.sha256()}
        counts = {scope: {name: 0 for name in ("HF", "BF", "soft")}
                  for scope in ("full", "prefix")}
        rows = 0
        mode_started = time.perf_counter()
        with torch.inference_mode():
            for step, (features, labels, frames) in enumerate(loader, 1):
                features = features.to(args.device)
                labels = labels.to(args.device)
                frames = frames.to(args.device)
                output = model(features, frames)
                scores = {
                    "HF": output["hand_face_logits"],
                    "BF": output["body_face_logits"],
                    "soft": _soft_logits(output, weights),
                }
                if not all(bool(torch.isfinite(value).all()) for value in scores.values()):
                    raise RuntimeError(f"Nonfinite logits in {mode_name} batch {step}")
                predictions = {name: value.argmax(-1).cpu() for name, value in scores.items()}
                batch_members = members[rows:rows + len(labels)]
                _digest_update(digests["full"], batch_members, features, labels, frames)
                prefix_count = max(0, min(args.prefix_samples - rows, len(labels)))
                if prefix_count:
                    _digest_update(digests["prefix"], batch_members[:prefix_count],
                                   features[:prefix_count], labels[:prefix_count], frames[:prefix_count])
                for scope, offset, size in (("full", 0, len(labels)), ("prefix", 0, prefix_count)):
                    if size:
                        scope_predictions = {name: value[offset:offset + size]
                                             for name, value in predictions.items()}
                        metric = _metric_counts(scope_predictions, labels[:size].cpu())
                        for name in counts[scope]:
                            counts[scope][name] += metric[name]["correct"]
                for index, member in enumerate(batch_members):
                    row = {"sample": member, "label": int(labels[index].cpu())}
                    row.update({f"{name}_prediction": int(value[index])
                                for name, value in predictions.items()})
                    all_prediction_rows.append({"mode": mode_name, **row})
                rows += len(labels)
                if step == 1 or step % 100 == 0 or step == len(loader):
                    print(f"mode={mode_name} batch={step}/{len(loader)} samples={rows} "
                          f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
        if rows != len(members):
            raise RuntimeError(f"Incomplete {mode_name}: {rows} != {len(members)}")
        all_results[mode_name] = {
            "temporal_compression": compression,
            "deterministic_temporal_crop": crop,
            "input_digest_scope": "member name + exact float/bool input tensors in manifest order",
            "input_sha256_full": digests["full"].hexdigest(),
            "input_sha256_prefix": digests["prefix"].hexdigest(),
            "metrics": {
                "full": {name: {"correct": value, "top1_percent": 100 * value / len(members)}
                          for name, value in counts["full"].items()},
                "prefix": {name: {"correct": value, "top1_percent": 100 * value / args.prefix_samples}
                           for name, value in counts["prefix"].items()},
            },
            "seconds": time.perf_counter() - mode_started,
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ["mode", "sample", "label", "HF_prediction", "BF_prediction", "soft_prediction"]
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_prediction_rows)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "archive_sha256": archive_digest,
        "manifest_validation_sha256": expected_member_digest,
        "evaluated_members_sha256": _member_digest(members),
        "partition": "validation",
        "samples_manifest": len(manifest_members),
        "samples_full": len(members),
        "samples_prefix": args.prefix_samples,
        "prefix_definition": "first N members in the fixed validation manifest",
        "gamma": 1.0,
        "weight_source": "frozen checkpoint calibration counters; no validation recalibration",
        "checkpoint_hand_face_weights_sha256": hashlib.sha256(
            model.hand_face_weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "modes": all_results,
        "seconds": time.perf_counter() - started,
        "inference": "frozen eager model; raw x/y/score input; no augmentation; no synthetic corruption",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
