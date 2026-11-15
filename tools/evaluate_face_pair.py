"""Evaluate a frozen face-pair checkpoint with the original compiled val protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure the CUDA-wheel library paths before importing torch.
import numpy as np
import torch
from torch.nn import functional as F

from isaa.face_pair_training import COMPILE_SCOPE, _dataset, _loader, compile_model, routing_report
from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_HAND_MOTION_VARIANT,
    FACE_PAIR_VARIANT, FacePairFusion, FacePairHandFaceGeometryFusion,
    FacePairHandMotionFusion,
)
from isaa.utils.seed import seed_everything
from tools.check_face_pair_checkpoint import check_args, check_compile, check_finite, check_manifest, require
from tools.evaluate_best import NTU60_NAMES


OUTPUT_KEYS = {"fused": "logits", "HF": "hand_face_logits", "BF": "body_face_logits"}
METHODS = (*OUTPUT_KEYS, "average")


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected an object in {path}")
    return value


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def routing_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()
            if name.startswith("routing_") or name in
            {"hand_face_weight", "hand_face_temperature", "body_face_temperature"}}


def check_routing_unchanged(model, original):
    current = routing_state(model)
    require(current.keys() == original.keys(), "Routing buffer names changed")
    for name, tensor in current.items():
        require(torch.equal(tensor, original[name]), f"Evaluation mutated routing buffer {name}")


def summarize(confusion, top5, loss_sum, samples):
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = confusion.diagonal()
    recall = np.divide(correct, support, out=np.zeros_like(support, dtype=float), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros_like(predicted, dtype=float), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(recall), where=precision + recall > 0)
    active = support > 0
    require(int(confusion.sum()) == samples, "Confusion matrix sample count differs")
    return {"samples": samples, "correct": int(correct.sum()),
            "errors": samples - int(correct.sum()),
            "top1_percent": float(100 * correct.sum() / samples),
            "top5_percent": float(100 * top5.sum() / samples),
            "ce": float(loss_sum / samples),
            "macro_accuracy_percent": float(100 * recall[active].mean()),
            "macro_f1_percent": float(100 * f1[active].mean())}, (support, correct, recall, precision, f1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda")
    options = parser.parse_args()
    output_dir = options.output_dir.resolve()
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    if options.batch_size is not None:
        require(options.batch_size > 0, "batch-size must be positive")
    if options.num_workers is not None:
        require(options.num_workers >= 0, "num-workers must be nonnegative")
    checkpoint_path = options.checkpoint.resolve()
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    del checkpoint_bytes
    saved_args = checkpoint["args"]
    check_args(saved_args, "checkpoint.args", expected_variant=saved_args["model_variant"])
    check_compile(checkpoint["compile"], saved_args, "checkpoint.compile")
    config = read_json(checkpoint_path.parent / "config.json")
    manifest = read_json(checkpoint_path.parent / "split_manifest.json")
    require(config["args"] == saved_args, "Checkpoint arguments differ from run config")
    require(config["compile"] == checkpoint["compile"], "Checkpoint compile policy differs from run config")
    require(config.get("initialization") == "random_scratch_no_checkpoint", "Run was not initialized from scratch")
    split_counts = check_manifest(manifest, config)
    check_finite(checkpoint["model"], "checkpoint.model")
    check_finite(checkpoint["optimizer"], "checkpoint.optimizer")
    check_finite(checkpoint["metrics"], "checkpoint.metrics")
    check_finite(checkpoint["routing"], "checkpoint.routing")
    model_types = {
        FACE_PAIR_VARIANT: FacePairFusion,
        FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT: FacePairHandFaceGeometryFusion,
        FACE_PAIR_HAND_MOTION_VARIANT: FacePairHandMotionFusion,
    }
    model_type = model_types.get(saved_args["model_variant"])
    require(model_type is not None,
            f"Unsupported face-pair model variant: {saved_args['model_variant']!r}")
    model = model_type(saved_args["num_classes"])
    require(checkpoint.get("architecture") == model.ARCHITECTURE, "Checkpoint architecture differs")
    require(checkpoint["model_config"] == model.model_config == config["model_config"], "Model configuration differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    frozen_routing = routing_state(model)
    expected_routing = routing_report(model, checkpoint["epoch"])
    expected_routing["active"] = checkpoint["epoch"] >= saved_args["pair_calibration_start_epoch"]
    require(checkpoint["routing"] == expected_routing, "Checkpoint routing report differs from buffers")
    require(read_json(checkpoint_path.parent / f"routing_epoch_{checkpoint['epoch']:03}.json") == expected_routing,
            "Immutable epoch routing report differs from checkpoint")
    hf_counts = frozen_routing["routing_hand_face_correct"]
    bf_counts = frozen_routing["routing_body_face_correct"]
    counts = frozen_routing["routing_class_counts"]
    require(bool(((hf_counts >= 0) & (bf_counts >= 0) & (counts >= 0)
                  & (hf_counts <= counts) & (bf_counts <= counts)).all()), "Invalid calibration counts")
    expected_weights = torch.where(hf_counts > bf_counts, 1.0, torch.where(hf_counts < bf_counts, 0.0, 0.5))
    require(torch.equal(expected_weights, frozen_routing["hand_face_weight"]), "Routing selection disagrees with counts")
    require(int(counts.sum()) == (split_counts["calibration"] if expected_routing["active"] else 0),
            "Routing calibration sample count differs")
    require(float(model.hand_face_temperature) == float(model.body_face_temperature) == 1.0,
            "Expected fixed unit temperatures")
    del checkpoint["model"], checkpoint["optimizer"]

    args = argparse.Namespace(**saved_args)
    if options.archive is not None:
        args.archive = str(options.archive.resolve())
    if options.batch_size is not None:
        args.test_batch_size = options.batch_size
    if options.num_workers is not None:
        args.num_workers = options.num_workers
    require(args.test_batch_size > 0 and args.num_workers >= 0, "Invalid loader configuration")
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device(options.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable; no fallback")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    dataset = _dataset(args, "val", False)
    members_sha = hashlib.sha256("\n".join(dataset.members).encode()).hexdigest()
    require(members_sha == manifest["validation"]["sha256"], "Validation member digest differs from training")
    require(list(dataset.members) == manifest["validation"]["members"], "Validation member ordering differs")
    require(len(dataset) == split_counts["validation"] > 0, "Validation sample count differs")
    loader = _loader(args, dataset, device, False)
    model = model.to(device).eval()
    compiled = compile_model(model, args.compile_mode)
    check_routing_unchanged(model, frozen_routing)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print(json.dumps({"checkpoint": str(checkpoint_path), "sha256": checkpoint_sha,
                      "epoch": checkpoint["epoch"], "recorded_val": checkpoint["metrics"]["val"],
                      "samples": len(dataset), "batches": len(loader),
                      "compile": {**COMPILE_SCOPE, "mode": args.compile_mode},
                      "routing": "frozen checkpoint buffers; official val never recalibrates"},
                     ensure_ascii=False), flush=True)
    k = args.num_classes
    confusion = {name: np.zeros((k, k), dtype=np.int64) for name in METHODS}
    top5_counts = {name: np.zeros(k, dtype=np.int64) for name in METHODS}
    loss_sums = {name: 0.0 for name in METHODS}
    normalization_error = {name: 0.0 for name in METHODS}
    error_rows = []
    samples = 0
    with torch.no_grad():
        for step, (features, labels, frame_mask) in enumerate(loader, 1):
            features, labels, frame_mask = (value.to(device, non_blocking=True)
                                            for value in (features, labels, frame_mask))
            output = compiled(features, frame_mask)
            require(all(key in output for key in OUTPUT_KEYS.values()), "Missing model outputs")
            scores = {name: output[key] for name, key in OUTPUT_KEYS.items()}
            scores["average"] = torch.logaddexp(
                F.log_softmax(scores["HF"].float(), dim=-1),
                F.log_softmax(scores["BF"].float(), dim=-1)) - math.log(2.0)
            truth = labels.cpu().numpy()
            require(np.array_equal(truth, dataset.labels[samples:samples + len(truth)]),
                    "Validation labels/member ordering differs")
            predictions, confidences, true_probabilities = {}, {}, {}
            for name, logits in scores.items():
                require(logits.shape == (len(truth), k), f"Invalid {name} output shape")
                require(bool(torch.isfinite(logits).all()), f"Nonfinite {name} logits at batch {step}")
                probabilities = logits.exp() if name in ("fused", "average") else logits.softmax(-1)
                deviation = float((probabilities.sum(-1) - 1).abs().max())
                normalization_error[name] = max(normalization_error[name], deviation)
                require(deviation <= 1e-5, f"Unnormalized {name} probabilities at batch {step}: {deviation}")
                loss_sums[name] += float(F.cross_entropy(logits, labels, reduction="sum"))
                pred = logits.argmax(-1).cpu().numpy()
                hit5 = (logits.topk(min(5, k), dim=1).indices == labels[:, None]).any(1).cpu().numpy()
                np.add.at(confusion[name], (truth, pred), 1)
                np.add.at(top5_counts[name], truth, hit5.astype(np.int64))
                predictions[name] = pred
                confidences[name] = probabilities.max(-1).values.cpu().numpy()
                true_probabilities[name] = probabilities.gather(1, labels[:, None])[:, 0].cpu().numpy()
            for index in range(len(truth)):
                if all(predictions[name][index] == truth[index] for name in METHODS):
                    continue
                label = int(truth[index])
                row = {"sample": dataset.members[samples + index], "true_action": f"A{label + 1:03}",
                       "true_name": NTU60_NAMES[label] if k == 60 else ""}
                for name in METHODS:
                    pred = int(predictions[name][index])
                    row.update({f"{name}_predicted_action": f"A{pred + 1:03}",
                                f"{name}_correct": pred == label,
                                f"{name}_predicted_probability": float(confidences[name][index]),
                                f"{name}_true_probability": float(true_probabilities[name][index])})
                error_rows.append(row)
            samples += len(truth)
            if step == 1 or step % 50 == 0 or step == len(loader):
                accuracy = " ".join(f"{name}={np.trace(confusion[name]) / samples:.6f}" for name in METHODS)
                print(f"progress={step}/{len(loader)} samples={samples} {accuracy} elapsed={time.perf_counter()-started:.1f}s",
                      flush=True)
    check_routing_unchanged(model, frozen_routing)
    require(samples == len(dataset), "Incomplete validation traversal")
    summaries, details = {}, {}
    expected_support = np.bincount(dataset.labels, minlength=k)
    for name in METHODS:
        summaries[name], details[name] = summarize(confusion[name], top5_counts[name], loss_sums[name], samples)
        require(np.array_equal(details[name][0], expected_support), f"{name} class support differs")
        require(sum(not row[f"{name}_correct"] for row in error_rows) == summaries[name]["errors"],
                f"{name} error export count differs")
        rows = [{"true_action": f"A{i + 1:03}", **{f"A{j + 1:03}": int(confusion[name][i, j])
                                                   for j in range(k)}} for i in range(k)]
        write_csv(output_dir / f"confusion_matrix_{name}.csv", rows, list(rows[0]))
    class_rows = []
    for i, route in enumerate(expected_routing["classes"]):
        row = {"action_id": f"A{i + 1:03}", "name": NTU60_NAMES[i] if k == 60 else "",
               "samples": int(expected_support[i]), "checkpoint_selected": route["selected"],
               "checkpoint_HF_weight": route["hand_face_weight"],
               "calibration_samples": route["samples"],
               "calibration_HF_accuracy_percent": 100 * route["hand_face_accuracy"]
                   if route["hand_face_accuracy"] is not None else None,
               "calibration_BF_accuracy_percent": 100 * route["body_face_accuracy"]
                   if route["body_face_accuracy"] is not None else None}
        for name in METHODS:
            support, correct, recall, precision, f1 = details[name]
            row.update({f"{name}_correct": int(correct[i]),
                        f"{name}_accuracy_percent": float(100 * recall[i]) if support[i] else None,
                        f"{name}_top5_percent": float(100 * top5_counts[name][i] / support[i]) if support[i] else None,
                        f"{name}_precision_percent": float(100 * precision[i]),
                        f"{name}_f1_percent": float(100 * f1[i])})
        class_rows.append(row)
    write_csv(output_dir / "per_class_accuracy.csv", class_rows, list(class_rows[0]))
    error_fields = ["sample", "true_action", "true_name"]
    for name in METHODS:
        error_fields.extend(f"{name}_{suffix}" for suffix in
                            ("predicted_action", "correct", "predicted_probability", "true_probability"))
    write_csv(output_dir / "errors.csv", error_rows, error_fields)
    recorded = checkpoint["metrics"]["val"]
    recorded_keys = {"fused": ("acc", "fusion_loss"), "HF": ("hand_face_acc", "hand_face_loss"),
                     "BF": ("body_face_acc", "body_face_loss")}
    require(recorded["samples"] == samples, "Recorded validation sample count differs")
    comparison = {}
    for name, (acc_key, ce_key) in recorded_keys.items():
        comparison[name] = {"recorded_top1_percent": 100 * recorded[acc_key],
                            "difference_percentage_points": summaries[name]["top1_percent"] - 100 * recorded[acc_key],
                            "same_top1_correct_count": summaries[name]["correct"] == int(round(recorded[acc_key] * samples)),
                            "recorded_ce": recorded[ce_key], "ce_difference": summaries[name]["ce"] - recorded[ce_key]}
    total_loss = (saved_args["pair_fusion_loss_weight"] * summaries["fused"]["ce"]
                  + saved_args["pair_head_loss_weight"] * (summaries["HF"]["ce"] + summaries["BF"]["ce"]))
    summary = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
               "epoch": checkpoint["epoch"], "recorded_best_epoch": checkpoint["metrics"]["best_epoch"],
               "model_variant": saved_args["model_variant"], "architecture": model.ARCHITECTURE,
               "partition": "val", "split": args.split, "archive": str(Path(args.archive).resolve()),
               "validation_members_sha256": members_sha, "samples": samples, "methods": summaries,
               "recorded_val_metrics": recorded, "recorded_comparison": comparison,
               "total_training_objective": total_loss, "recorded_total_loss_difference": total_loss - recorded["loss"],
               "routing": {"source": "frozen checkpoint buffers selected on training-subject holdout",
                           "unchanged_after_evaluation": True,
                           "official_validation_used_for_routing": False,
                           "selection_counts": {name: sum(row["selected"] == name for row in expected_routing["classes"])
                                                for name in ("hand_face", "body_face", "average")},
                           "report": expected_routing},
               "average_note": "Equal probability mixture of the same jointly trained HF/BF heads; evaluation comparator only",
               "errors_csv_scope": "Union of samples misclassified by at least one of the four methods",
               "maximum_probability_normalization_error": normalization_error,
               "protocol": {"window_size": args.window_size, "max_persons": args.max_persons,
                            "feature_mode": "raw", "temporal_compression": "confidence_weighted",
                            "augmentation": False, "shuffle": False, "drop_last": False,
                            "batch_size": args.test_batch_size, "num_workers": args.num_workers,
                            "tf32": args.tf32, "max_samples": args.max_samples,
                            "compile": {**COMPILE_SCOPE, "mode": args.compile_mode}},
               "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                           "device": str(device), "gpu": torch.cuda.get_device_name(device)
                               if device.type == "cuda" else None,
                           "seconds": time.perf_counter() - started}}
    check_finite(summary, "evaluation.summary")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "epoch": checkpoint["epoch"], "samples": samples,
                      "methods": summaries, "recorded_comparison": comparison}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
