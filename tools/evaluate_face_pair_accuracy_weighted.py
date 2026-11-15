"""Evaluate a face-pair checkpoint with calibration-accuracy soft routing.

The checkpoint's hard per-class routing is left untouched.  This entry point
uses the saved training-subject calibration counts to form a fixed per-class
weight and evaluates that comparator on the official validation split.
"""
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
from torch import nn
from torch.nn import functional as F

from isaa.face_pair_training import _dataset, _loader, compile_model, compile_scope, routing_report
from isaa.models.face_pair_fusion import FacePairFusion
from isaa.utils.seed import seed_everything
from tools.check_face_pair_checkpoint import (
    FACE_PAIR_MODELS,
    check_args,
    check_compile,
    check_finite,
    check_manifest,
    require,
)
from tools.evaluate_best import NTU60_NAMES
from tools.evaluate_face_pair import (
    check_routing_unchanged,
    read_json,
    summarize,
    write_csv,
)


METHODS = ("HF", "BF", "soft")


def model_type(variant):
    selected = FACE_PAIR_MODELS.get(variant)
    require(selected is not None, f"Unsupported face-pair variant: {variant!r}")
    return selected


def accuracy_weights(hf_correct, bf_correct, class_counts):
    hf_correct, bf_correct, class_counts = (x.float() for x in (hf_correct, bf_correct, class_counts))
    require(bool(((hf_correct >= 0) & (bf_correct >= 0) & (class_counts >= 0)
                  & (hf_correct <= class_counts) & (bf_correct <= class_counts)).all()),
            "Invalid calibration counts")
    hf_accuracy = torch.where(class_counts > 0, hf_correct / class_counts.clamp_min(1),
                              torch.zeros_like(class_counts))
    bf_accuracy = torch.where(class_counts > 0, bf_correct / class_counts.clamp_min(1),
                              torch.zeros_like(class_counts))
    accuracy_sum = hf_accuracy + bf_accuracy
    hf_weight = torch.where(accuracy_sum > 0, hf_accuracy / accuracy_sum,
                            torch.full_like(accuracy_sum, 0.5))
    return hf_accuracy, bf_accuracy, hf_weight, 1.0 - hf_weight


class SoftAccuracyReadout(nn.Module):
    """Compile the checkpoint forward and append fixed soft log-probability fusion."""

    def __init__(self, base: FacePairFusion, hand_face_weight: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("soft_hand_face_weight", hand_face_weight.detach().float().clone())

    def forward(self, features, frame_mask=None):
        output = self.base(features, frame_mask)
        hf = output["hand_face_logits"]
        bf = output["body_face_logits"]
        hf_log_prob = F.log_softmax(hf.float() / self.base.hand_face_temperature, dim=-1)
        bf_log_prob = F.log_softmax(bf.float() / self.base.body_face_temperature, dim=-1)
        weight = self.soft_hand_face_weight.view(1, -1)
        # Arithmetic probability mixture in log space.  Zero weights become
        # -inf in the corresponding branch and are handled by logaddexp.
        bf_weight = 1.0 - weight
        soft = torch.logaddexp(torch.log(weight) + hf_log_prob,
                               torch.log(bf_weight) + bf_log_prob)
        soft = soft - soft.amax(dim=-1, keepdim=True)
        soft = soft - torch.logsumexp(soft, dim=-1, keepdim=True)
        return {"hand_face_logits": hf, "body_face_logits": bf, "soft_logits": soft}


def _routing_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()
            if name.startswith("routing_") or name in {
                "hand_face_weight", "hand_face_temperature", "body_face_temperature"}}


def _write_confusion(path, confusion):
    rows = [{"true_action": f"A{i + 1:03}", **{
        f"A{j + 1:03}": int(confusion[i, j]) for j in range(confusion.shape[1])}}
        for i in range(confusion.shape[0])]
    write_csv(path, rows, list(rows[0]))


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
    variant = saved_args.get("model_variant")
    selected_model = model_type(variant)
    check_args(saved_args, "checkpoint.args", expected_variant=variant)
    check_compile(checkpoint["compile"], saved_args, "checkpoint.compile")

    config = read_json(checkpoint_path.parent / "config.json")
    manifest = read_json(checkpoint_path.parent / "split_manifest.json")
    require(config["args"] == saved_args, "Checkpoint arguments differ from run config")
    require(config["compile"] == checkpoint["compile"], "Checkpoint compile policy differs from run config")
    require(config.get("initialization") == "random_scratch_no_checkpoint",
            "Run was not initialized from scratch")
    split_counts = check_manifest(manifest, config)
    check_finite(checkpoint["model"], "checkpoint.model")
    check_finite(checkpoint["optimizer"], "checkpoint.optimizer")
    check_finite(checkpoint["metrics"], "checkpoint.metrics")
    check_finite(checkpoint["routing"], "checkpoint.routing")

    base = selected_model(saved_args["num_classes"])
    require(checkpoint.get("architecture") == base.ARCHITECTURE, "Checkpoint architecture differs")
    require(checkpoint["model_config"] == base.model_config == config["model_config"],
            "Model configuration differs")
    base.load_state_dict(checkpoint["model"], strict=True)
    frozen_routing = _routing_state(base)
    expected_routing = routing_report(base, checkpoint["epoch"])
    expected_routing["active"] = checkpoint["epoch"] >= saved_args["pair_calibration_start_epoch"]
    require(checkpoint["routing"] == expected_routing, "Checkpoint routing report differs from buffers")
    require(read_json(checkpoint_path.parent / f"routing_epoch_{checkpoint['epoch']:03}.json")
            == expected_routing, "Immutable epoch routing report differs from checkpoint")
    require(float(base.hand_face_temperature) == float(base.body_face_temperature) == 1.0,
            "Expected fixed unit temperatures")

    hf_correct = frozen_routing["routing_hand_face_correct"].float()
    bf_correct = frozen_routing["routing_body_face_correct"].float()
    class_counts = frozen_routing["routing_class_counts"].float()
    hf_accuracy, bf_accuracy, soft_hf_weight, soft_bf_weight = accuracy_weights(
        hf_correct, bf_correct, class_counts)
    require(bool(torch.isfinite(soft_hf_weight).all() and torch.isfinite(soft_bf_weight).all()),
            "Nonfinite soft routing weights")
    require(bool(((soft_hf_weight >= 0) & (soft_hf_weight <= 1)
                  & (soft_bf_weight >= 0) & (soft_bf_weight <= 1)).all()),
            "Soft routing weights outside [0, 1]")
    require(int(class_counts.sum()) == (split_counts["calibration"] if expected_routing["active"] else 0),
            "Routing calibration sample count differs")
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
    require(members_sha == manifest["validation"]["sha256"],
            "Validation member digest differs from training")
    require(list(dataset.members) == manifest["validation"]["members"],
            "Validation member ordering differs")
    require(len(dataset) == split_counts["validation"] > 0, "Validation sample count differs")
    loader = _loader(args, dataset, device, False)

    base = base.to(device).eval()
    wrapper = SoftAccuracyReadout(base, soft_hf_weight).to(device).eval()
    compiled = compile_model(wrapper, args.compile_mode)
    check_routing_unchanged(base, frozen_routing)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print(json.dumps({"checkpoint": str(checkpoint_path), "sha256": checkpoint_sha,
                      "epoch": checkpoint["epoch"], "samples": len(dataset),
                      "batches": len(loader), "compile": {**compile_scope(args), "mode": args.compile_mode},
                      "routing": "soft calibration-accuracy weights; checkpoint buffers frozen"},
                     ensure_ascii=False), flush=True)

    k = args.num_classes
    confusion = {name: np.zeros((k, k), dtype=np.int64) for name in METHODS}
    top5_counts = {name: np.zeros(k, dtype=np.int64) for name in METHODS}
    loss_sums = {name: 0.0 for name in METHODS}
    normalization_error = {name: 0.0 for name in METHODS}
    errors = []
    samples = 0
    with torch.no_grad():
        for step, (features, labels, frame_mask) in enumerate(loader, 1):
            features, labels, frame_mask = (value.to(device, non_blocking=True)
                                            for value in (features, labels, frame_mask))
            output = compiled(features, frame_mask)
            truth = labels.cpu().numpy()
            require(np.array_equal(truth, dataset.labels[samples:samples + len(truth)]),
                    "Validation labels/member ordering differs")
            scores = {"HF": output["hand_face_logits"], "BF": output["body_face_logits"],
                      "soft": output["soft_logits"]}
            predictions = {}
            confidences = {}
            true_probabilities = {}
            for name, logits in scores.items():
                require(logits.shape == (len(truth), k), f"Invalid {name} output shape")
                require(bool(torch.isfinite(logits).all()), f"Nonfinite {name} logits at batch {step}")
                probabilities = logits.exp() if name == "soft" else logits.softmax(-1)
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
                errors.append(row)
            samples += len(truth)
            if step == 1 or step % 50 == 0 or step == len(loader):
                accuracy = " ".join(f"{name}={np.trace(confusion[name]) / samples:.6f}"
                                    for name in METHODS)
                print(f"progress={step}/{len(loader)} samples={samples} {accuracy} "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)

    check_routing_unchanged(base, frozen_routing)
    require(torch.equal(wrapper.soft_hand_face_weight.detach().cpu(), soft_hf_weight),
            "Soft routing weights changed during evaluation")
    require(samples == len(dataset), "Incomplete validation traversal")
    summaries, details = {}, {}
    expected_support = np.bincount(dataset.labels, minlength=k)
    for name in METHODS:
        summaries[name], details[name] = summarize(confusion[name], top5_counts[name],
                                                   loss_sums[name], samples)
        require(np.array_equal(details[name][0], expected_support),
                f"{name} class support differs")
        require(sum(not row[f"{name}_correct"] for row in errors) == summaries[name]["errors"],
                f"{name} error export count differs")
        _write_confusion(output_dir / f"confusion_matrix_{name}.csv", confusion[name])

    class_rows = []
    soft_hf_weight_list = soft_hf_weight.tolist()
    soft_bf_weight_list = soft_bf_weight.tolist()
    for i in range(k):
        row = {"action_id": f"A{i + 1:03}", "name": NTU60_NAMES[i] if k == 60 else "",
               "samples": int(expected_support[i]),
               "calibration_samples": int(class_counts[i]),
               "calibration_HF_correct": int(hf_correct[i]),
               "calibration_BF_correct": int(bf_correct[i]),
               "calibration_HF_accuracy_percent": float(100 * hf_accuracy[i]),
               "calibration_BF_accuracy_percent": float(100 * bf_accuracy[i]),
               "soft_HF_weight": float(soft_hf_weight_list[i]),
               "soft_BF_weight": float(soft_bf_weight_list[i])}
        for name in METHODS:
            support, correct, recall, precision, f1 = details[name]
            row.update({f"{name}_correct": int(correct[i]),
                        f"{name}_accuracy_percent": float(100 * recall[i]) if support[i] else None,
                        f"{name}_top5_percent": float(100 * top5_counts[name][i] / support[i])
                            if support[i] else None,
                        f"{name}_precision_percent": float(100 * precision[i]),
                        f"{name}_f1_percent": float(100 * f1[i])})
        class_rows.append(row)
    write_csv(output_dir / "per_class_accuracy.csv", class_rows, list(class_rows[0]))

    error_fields = ["sample", "true_action", "true_name"]
    for name in METHODS:
        error_fields.extend(f"{name}_{suffix}" for suffix in
                            ("predicted_action", "correct", "predicted_probability",
                             "true_probability"))
    write_csv(output_dir / "errors.csv", errors, error_fields)

    recorded = checkpoint["metrics"]["val"]
    recorded_comparison = {}
    for name, acc_key, ce_key in (("HF", "hand_face_acc", "hand_face_loss"),
                                  ("BF", "body_face_acc", "body_face_loss")):
        recorded_comparison[name] = {
            "recorded_top1_percent": 100 * recorded[acc_key],
            "difference_percentage_points": summaries[name]["top1_percent"]
                - 100 * recorded[acc_key],
            "same_top1_correct_count": summaries[name]["correct"]
                == int(round(recorded[acc_key] * samples)),
            "recorded_ce": recorded[ce_key],
            "ce_difference": summaries[name]["ce"] - recorded[ce_key],
        }

    summary = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "epoch": checkpoint["epoch"], "recorded_best_epoch": checkpoint["metrics"]["best_epoch"],
        "model_variant": saved_args["model_variant"], "architecture": base.ARCHITECTURE,
        "partition": "val", "split": args.split, "archive": str(Path(args.archive).resolve()),
        "validation_members_sha256": members_sha, "samples": samples, "methods": summaries,
        "recorded_val_metrics": recorded, "recorded_comparison": recorded_comparison,
        "soft_routing": {
            "formula": "logaddexp(log(hf_weight)+logp_HF, log(bf_weight)+logp_BF), then global logsumexp normalization; hf_weight=hf_acc/(hf_acc+bf_acc), bf_weight=1-hf_weight, zero_sum=0.5",
            "source": "training-subject calibration counts in checkpoint buffers",
            "official_validation_used_for_routing": False,
            "hf_weights": [float(value) for value in soft_hf_weight_list],
            "bf_weights": [float(value) for value in soft_bf_weight_list],
            "calibration_HF_accuracy": [float(value) for value in hf_accuracy.tolist()],
            "calibration_BF_accuracy": [float(value) for value in bf_accuracy.tolist()],
        },
        "maximum_probability_normalization_error": normalization_error,
        "errors_csv_scope": "Union of samples misclassified by at least one of HF, BF, or soft",
        "protocol": {
            "window_size": args.window_size, "max_persons": args.max_persons,
            "feature_mode": "raw", "temporal_compression": "confidence_weighted",
            "augmentation": False, "shuffle": False, "drop_last": False,
            "batch_size": args.test_batch_size, "num_workers": args.num_workers,
            "tf32": args.tf32, "max_samples": args.max_samples,
            "compile": {**compile_scope(args), "mode": args.compile_mode},
        },
        "runtime": {
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seconds": time.perf_counter() - started,
        },
    }
    check_finite(summary, "evaluation.summary")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "epoch": checkpoint["epoch"],
                      "samples": samples, "methods": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
