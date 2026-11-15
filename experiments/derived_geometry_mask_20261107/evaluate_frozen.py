"""Compiled frozen screen for mask-aware derived geometry variants."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA libraries before torch import.
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.derived_geometry_mask_20261107.model import (
    MaskAwareCenterScaleFusion,
    MaskAwareGeometryOnlyFusion,
)
from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.face_pair_training import compile_model, compile_scope
from isaa.models.face_pair_fusion import FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights
from tools.check_face_pair_checkpoint import check_args, check_compile


CONDITIONS = (
    ("clean", None, 0.0, 1729),
    ("point_0.05", "joint_dropout_probability", 0.05, 2718),
    ("point_0.1", "joint_dropout_probability", 0.10, 2719),
    ("group_0.05", "grouped_joint_dropout_probability", 0.05, 2818),
    ("group_0.1", "grouped_joint_dropout_probability", 0.10, 2819),
)


def load_model(checkpoint, cls, device, compile_mode):
    model = cls(checkpoint["args"]["num_classes"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    return compile_model(model, compile_mode)


def bootstrap(delta, subjects, repeats=20_000, seed=20261107):
    groups, inverse = np.unique(subjects, return_inverse=True)
    if len(groups) < 2:
        raise ValueError("Expected at least two subjects")
    sizes = np.bincount(inverse)
    sums = np.bincount(inverse, weights=delta, minlength=len(groups))
    draws = np.random.default_rng(seed).multinomial(
        len(groups), np.full(len(groups), 1.0 / len(groups)), size=repeats)
    values = 100.0 * (draws @ sums) / (draws @ sizes)
    return np.percentile(values, [2.5, 97.5]).tolist()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mode", choices=("geometry_only", "center_scale"), default="geometry_only")
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    args_saved = checkpoint["args"]
    check_args(args_saved, "checkpoint.args", expected_variant="best-face-pair")
    check_compile(checkpoint["compile"], args_saved, "checkpoint.compile")
    if checkpoint.get("architecture") != FacePairFusion.ARCHITECTURE:
        raise ValueError("Expected the formal best-face-pair checkpoint")
    device = torch.device("cuda")
    weight_model = FacePairFusion(checkpoint["args"]["num_classes"])
    weight_model.load_state_dict(checkpoint["model"], strict=True)
    weights = _soft_weights(weight_model).to(device)
    base = load_model(checkpoint, FacePairFusion, device, args_saved["compile_mode"])
    candidate_cls = (MaskAwareGeometryOnlyFusion if options.mode == "geometry_only"
                     else MaskAwareCenterScaleFusion)
    candidate = load_model(checkpoint, candidate_cls, device, args_saved["compile_mode"])
    manifest = json.loads((options.checkpoint.parent / "split_manifest.json").read_text())
    expected_members = manifest["validation"]["members"]
    results = {}
    rows = []
    args = argparse.Namespace(**args_saved)
    args.archive = str(options.archive.resolve())
    args.test_batch_size = options.batch_size
    args.num_workers = options.num_workers
    args.prefetch_factor = 2
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    for name, key, probability, seed in CONDITIONS:
        torch.manual_seed(seed)
        augmentation = {"random_temporal_crop": False}
        if key is not None:
            augmentation[key] = probability
        dataset = RTMWZipDataset(
            options.archive, split="val", split_protocol=args.split,
            window_size=args.window_size, num_joints=133, num_classes=args.num_classes,
            layout="rtmw_133", max_persons=args.max_persons,
            feature_mode="raw", max_samples=0, augment=key is not None,
            temporal_compression=getattr(args, "temporal_compression", "confidence_weighted"),
            deterministic_temporal_crop=getattr(args, "deterministic_temporal_crop", "center"),
            augmentation_config=augmentation,
        )
        members = list(dataset.members)
        if members != expected_members:
            raise ValueError(f"Validation members differ for {name}")
        loader_args = dict(batch_size=options.batch_size, shuffle=False, drop_last=False,
                           num_workers=options.num_workers, pin_memory=True,
                           collate_fn=partial(collate_rtmw, feature_mode="raw"))
        if options.num_workers:
            loader_args.update(prefetch_factor=2, multiprocessing_context="spawn")
        loader = DataLoader(dataset, **loader_args)
        predictions = {"base": [], options.mode: []}
        labels = []
        started = time.perf_counter()
        with torch.inference_mode():
            for features, batch_labels, frame_mask in loader:
                features = features.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                frame_mask = frame_mask.to(device, non_blocking=True)
                outputs = {
                    "base": base(features, frame_mask),
                    options.mode: candidate(features, frame_mask),
                }
                for model_name, output in outputs.items():
                    logits = _soft_logits(output, weights)
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError(f"Nonfinite logits in {name}/{model_name}")
                    predictions[model_name].extend(logits.argmax(-1).cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())
        labels_np = np.asarray(labels)
        subjects_np = np.asarray([int(re.search(r"P(\d{3})", sample).group(1)) for sample in members])
        base_correct = np.asarray(predictions["base"]) == labels_np
        condition_result = {"samples": len(labels), "seed": seed, "seconds": time.perf_counter() - started,
                            "input_members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
                            "models": {}}
        for model_name in predictions:
            correct = np.asarray(predictions[model_name]) == labels_np
            delta = correct.astype(np.int8) - base_correct.astype(np.int8)
            condition_result["models"][model_name] = {
                "correct": int(correct.sum()), "accuracy_percent": float(100 * correct.mean()),
                "difference_vs_base_pp": float(100 * delta.mean()),
                "corrected": int((delta == 1).sum()), "regressed": int((delta == -1).sum()),
                "subject_bootstrap_95_ci_pp": [0.0, 0.0] if model_name == "base" else bootstrap(delta, subjects_np),
            }
        results[name] = condition_result
        for index, sample in enumerate(members):
            row = {"condition": name, "sample": sample, "label": int(labels_np[index]),
                   "base": int(predictions["base"][index])}
            row[options.mode] = int(predictions[options.mode][index])
            rows.append(row)
        print(json.dumps({"condition": name, "models": condition_result["models"]}), flush=True)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "checkpoint": str(options.checkpoint),
        "checkpoint_sha256": hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
        "validation_members_sha256": hashlib.sha256("\n".join(expected_members).encode()).hexdigest(),
        "samples_per_condition": len(expected_members),
        "models": ["formal_arithmetic_base", f"mask_aware_{options.mode}"],
        "compile": {"required": True, "backend": "inductor", "mode": args_saved["compile_mode"],
                    "scope": "compiled network plus mask-aware derived geometry"},
        "conditions": results,
        "inference_only": True,
        "limitations": "Frozen formal checkpoint screen; corruption masks are synthetic and validation has been reused.",
    }
    (options.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
