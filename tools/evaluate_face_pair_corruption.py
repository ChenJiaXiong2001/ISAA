"""Evaluate face-pair robustness under input joint/group corruption.

The checkpoint is kept frozen.  Each condition uses the same validation
members and the checkpoint's training-subject accuracy-proportional HF/BF
weights (gamma=1); only RTMW input corruption is changed.  ``point`` is the
existing body-joint dropout path and ``group`` is the existing complete
finger/face-token dropout path in :class:`RTMWZipDataset`.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Import the project entry module before torch so the configured CUDA wheel
# library paths are installed on the verified server environment.
import isaa.train  # noqa: E402

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_HAND_MOTION_VARIANT,
    FACE_PAIR_VARIANT, FacePairFusion, FacePairHandFaceGeometryFusion,
    FacePairHandMotionFusion,
)

try:
    from isaa.models.face_pair_fusion import FACE_PAIR_RAW_XY_VARIANT, FacePairRawXYFusion
except ImportError:  # Older remote experiment snapshots only contain best-face-pair.
    FACE_PAIR_RAW_XY_VARIANT = "best-face-pair-raw-xy"
    FacePairRawXYFusion = None
from isaa.train import collate_rtmw
from isaa.utils.seed import seed_everything
from tools.check_face_pair_checkpoint import check_args, check_compile


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _soft_weights(model: FacePairFusion) -> torch.Tensor:
    """Return gamma=1 weights from the frozen calibration counters."""
    counts = model.routing_class_counts.float()
    hf = model.routing_hand_face_correct.float()
    bf = model.routing_body_face_correct.float()
    hf_acc = torch.where(counts > 0, hf / counts.clamp_min(1), torch.zeros_like(counts))
    bf_acc = torch.where(counts > 0, bf / counts.clamp_min(1), torch.zeros_like(counts))
    total = hf_acc + bf_acc
    return torch.where(total > 0, hf_acc / total, torch.full_like(total, 0.5))


def _soft_logits(output: dict[str, torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    hf = F.log_softmax(output["hand_face_logits"].float(), dim=-1)
    bf = F.log_softmax(output["body_face_logits"].float(), dim=-1)
    w = weights.to(device=hf.device, dtype=hf.dtype).view(1, -1)
    # logaddexp handles zero per-class weights without a special branch.
    fused = torch.logaddexp(torch.log(w) + hf, torch.log1p(-w) + bf)
    fused = fused - fused.amax(dim=-1, keepdim=True)
    return fused - torch.logsumexp(fused, dim=-1, keepdim=True)


def _condition_specs(options: argparse.Namespace) -> List[Tuple[str, Dict[str, float], int]]:
    specs: List[Tuple[str, Dict[str, float], int]] = [("clean", {}, options.seed)]
    for kind, values in (("point", options.point_levels), ("group", options.group_levels)):
        for index, value in enumerate(values):
            level = float(value)
            if not 0.0 < level <= 1.0:
                raise ValueError(f"{kind} dropout level must be in (0, 1], got {level}")
            config_key = "joint_dropout_probability" if kind == "point" else "grouped_joint_dropout_probability"
            specs.append((f"{kind}_{level:g}", {config_key: level}, options.seed + index + 1 + (100 if kind == "group" else 0)))
    return specs


def _run_condition(model, weights, args, archive: Path, name: str, corruption: dict, seed: int):
    seed_everything(seed)
    dataset = RTMWZipDataset(
        archive,
        split="val",
        split_protocol=args.split,
        window_size=args.window_size,
        num_joints=133,
        num_classes=args.num_classes,
        layout="rtmw_133",
        max_persons=args.max_persons,
        score_normalization=args.score_normalization,
        feature_mode="raw",
        augment=bool(corruption),
        augmentation_config={"random_temporal_crop": False, **corruption},
        deterministic_temporal_crop=args.deterministic_temporal_crop,
        temporal_compression=args.temporal_compression,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        collate_fn=partial(collate_rtmw, feature_mode="raw"),
    )
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    top5 = 0
    losses = 0.0
    samples = 0
    predictions = []
    input_digest = hashlib.sha256()
    started = time.perf_counter()
    with torch.inference_mode():
        for features, labels, frame_mask in loader:
            input_digest.update(features.contiguous().numpy().tobytes())
            input_digest.update(frame_mask.contiguous().numpy().tobytes())
            features = features.to(args.device, non_blocking=args.device.startswith("cuda"))
            labels = labels.to(args.device, non_blocking=args.device.startswith("cuda"))
            frame_mask = frame_mask.to(args.device, non_blocking=args.device.startswith("cuda"))
            output = model(features, frame_mask)
            logits = _soft_logits(output, weights)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Nonfinite fused logits in condition {name}")
            prediction = logits.argmax(-1)
            truth = labels.cpu().numpy()
            pred = prediction.cpu().numpy()
            predictions.extend({"sample_index": samples + i, "label": int(y), "prediction": int(p)}
                               for i, (y, p) in enumerate(zip(truth, pred)))
            np.add.at(confusion, (truth, pred), 1)
            top5 += int((logits.topk(min(5, args.num_classes), dim=1).indices == labels[:, None]).any(1).sum())
            losses += float(F.cross_entropy(logits, labels, reduction="sum"))
            samples += len(truth)
    correct = int(np.trace(confusion))
    return {
        "name": name,
        "seed": seed,
        "corruption": corruption,
        "samples": samples,
        "correct": correct,
        "errors": samples - correct,
        "top1_percent": 100.0 * correct / samples,
        "top5_percent": 100.0 * top5 / samples,
        "cross_entropy": losses / samples,
        "seconds": time.perf_counter() - started,
        "input_sha256": input_digest.hexdigest(),
        "predictions": predictions,
        "confusion": confusion,
        "per_class_correct": np.diag(confusion).copy(),
        "per_class_support": confusion.sum(1).copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--point-levels", type=float, nargs="*", default=[0.05, 0.10])
    parser.add_argument("--group-levels", type=float, nargs="*", default=[0.05, 0.10])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--temporal-compression",
        choices=("checkpoint", "none", "confidence_weighted"),
        default="checkpoint",
        help="Temporal input construction; checkpoint preserves the training setting.",
    )
    parser.add_argument(
        "--deterministic-temporal-crop",
        choices=("checkpoint", "start", "center", "end"),
        default="checkpoint",
        help="Deterministic crop used after temporal construction.",
    )
    options = parser.parse_args()
    if options.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for corruption evaluation but is unavailable")
    if options.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {options.output_dir}")
    if options.batch_size <= 0 or options.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be nonnegative")

    checkpoint_path = options.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    variant = saved_args["model_variant"]
    # Evaluation accepts only checkpoints produced under the mandatory compiled,
    # random-scratch face-pair training protocol.  The frozen evaluator itself
    # may run eager inference, but it must never silently accept an eager or
    # warm-start checkpoint.
    check_args(saved_args, "checkpoint.args", expected_variant=variant)
    check_compile(checkpoint["compile"], saved_args, "checkpoint.compile")
    classes = {FACE_PAIR_VARIANT: FacePairFusion,
               FACE_PAIR_HAND_MOTION_VARIANT: FacePairHandMotionFusion}
    if FacePairRawXYFusion is not None:
        classes[FACE_PAIR_RAW_XY_VARIANT] = FacePairRawXYFusion
    classes[FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT] = FacePairHandFaceGeometryFusion
    if variant not in classes:
        raise ValueError(f"Expected face-pair checkpoint, got model_variant={variant!r}")
    model = classes[variant](saved_args["num_classes"])
    if checkpoint.get("architecture") != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture does not match current face-pair model")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(options.device).eval()
    weights = _soft_weights(model).detach().cpu()
    archive = (options.archive or Path(saved_args["archive"])).resolve()
    options.split = saved_args["split"]
    options.num_classes = saved_args["num_classes"]
    options.window_size = saved_args["window_size"]
    options.max_persons = saved_args["max_persons"]
    options.score_normalization = saved_args.get("score_normalization", "auto")
    if options.temporal_compression == "checkpoint":
        options.temporal_compression = saved_args.get("temporal_compression", "confidence_weighted")
    if options.deterministic_temporal_crop == "checkpoint":
        options.deterministic_temporal_crop = saved_args.get("deterministic_temporal_crop", "center")
    if options.max_samples < 0:
        raise ValueError("max-samples must be nonnegative")

    conditions = []
    for name, corruption, seed in _condition_specs(options):
        result = _run_condition(model, weights, options, archive, name, corruption, seed)
        conditions.append(result)
        print(json.dumps({key: result[key] for key in ("name", "samples", "correct", "top1_percent", "top5_percent", "seconds")}, ensure_ascii=False), flush=True)
    clean_acc = conditions[0]["top1_percent"]
    for result in conditions:
        result["drop_from_clean_percentage_points"] = clean_acc - result["top1_percent"]
        result["relative_drop_percent"] = 100.0 * (clean_acc - result["top1_percent"]) / clean_acc if clean_acc else None

    options.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    serializable = []
    per_class_rows = []
    for result in conditions:
        confusion = result.pop("confusion")
        predictions = result.pop("predictions")
        correct = result.pop("per_class_correct")
        support = result.pop("per_class_support")
        result["confusion_csv"] = f"confusion_{result['name']}.csv"
        result["predictions_csv"] = f"predictions_{result['name']}.csv"
        _write_csv(options.output_dir / result["predictions_csv"], predictions,
                   ["sample_index", "label", "prediction"])
        rows = [{"true_action": f"A{i + 1:03}", **{f"A{j + 1:03}": int(confusion[i, j]) for j in range(options.num_classes)}} for i in range(options.num_classes)]
        _write_csv(options.output_dir / result["confusion_csv"], rows, list(rows[0]))
        for index in range(options.num_classes):
            per_class_rows.append({"condition": result["name"], "action": f"A{index + 1:03}", "support": int(support[index]), "correct": int(correct[index]), "accuracy_percent": 100.0 * float(correct[index]) / float(support[index]) if support[index] else None})
        serializable.append(result)
    _write_csv(options.output_dir / "per_class_accuracy.csv", per_class_rows, list(per_class_rows[0]))
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "epoch": checkpoint["epoch"],
        "model_variant": variant,
        "architecture": model.ARCHITECTURE,
        "archive": str(archive),
        "soft_routing": {"gamma": 1.0, "hf_weights": [float(value) for value in weights], "formula": "HF_acc/(HF_acc+BF_acc)"},
        "conditions": serializable,
        "protocol": {"split": options.split, "window_size": options.window_size, "temporal_compression": options.temporal_compression, "deterministic_temporal_crop": options.deterministic_temporal_crop, "batch_size": options.batch_size, "num_workers": options.num_workers, "max_samples": options.max_samples, "device": options.device, "validation_augmentation_rebuilt_features": True},
    }
    (options.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(options.output_dir), "clean_top1_percent": clean_acc, "conditions": [{"name": item["name"], "top1_percent": item["top1_percent"], "drop": item["drop_from_clean_percentage_points"]} for item in serializable]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
