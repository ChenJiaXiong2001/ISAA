"""Evaluate one frozen calibration-fitted confusion correction on cached logits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


EXPECTED_CHECKPOINT = "aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7"
EXPECTED_CALIBRATION = "68354a1f6bb0f6046b7ea720a07db89513e29ff8b7717ad36ecb028745d7cf11"
EXPECTED_VALIDATION = "82fd60410fd04e381c65d3bb9f24f18f7993124ee2581902942fbfb4a2ecddc0"
BASELINE_REPORTED_CORRECT = 15536


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_partition(path: Path, section: dict, partition: str, expected: str):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    members = section["members"]
    member_sha = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if (member_sha != expected or section["sha256"] != expected
            or cache["member_sha256"] != expected or cache["checkpoint_sha256"] != EXPECTED_CHECKPOINT
            or cache["partition"] != partition or cache["epoch"] != 83):
        raise ValueError(f"{partition} cache provenance mismatch")
    hf, bf, labels = cache["hf"].float(), cache["bf"].float(), cache["labels"].long()
    if (hf.shape != bf.shape or hf.ndim != 2 or hf.shape[1] != 60
            or len(hf) != len(labels) or len(labels) != len(members)
            or len(set(members)) != len(members) or not torch.isfinite(hf).all()
            or not torch.isfinite(bf).all()):
        raise ValueError(f"{partition} cache contents invalid")
    subjects = []
    for member, label in zip(members, labels.tolist()):
        subject = re.search(r"P(\d{3})", member)
        action = re.search(r"A(\d{3})", member)
        if not subject or not action or int(action.group(1)) != label + 1:
            raise ValueError(f"{partition} action or subject mismatch")
        subjects.append(subject.group(1))
    return hf, bf, labels, members, subjects


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-cache", type=Path, required=True)
    p.add_argument("--validation-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    hf_cal, bf_cal, y_cal, _, subject_cal = load_partition(
        args.calibration_cache, manifest["calibration"], "calibration", EXPECTED_CALIBRATION)
    hf_val, bf_val, y_val, members, subject_val = load_partition(
        args.validation_cache, manifest["validation"], "val", EXPECTED_VALIDATION)
    if set(subject_cal) & set(subject_val):
        raise ValueError("Calibration and validation subjects overlap")
    if digest(args.checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError("Unexpected checkpoint hash")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 83:
        raise ValueError("Unexpected checkpoint epoch")
    state = checkpoint["model"]
    h = state["routing_hand_face_correct"].float()
    b = state["routing_body_face_correct"].float()
    support = state["routing_class_counts"].long()
    if (h.shape != (60,) or b.shape != (60,)
            or not torch.equal(support, torch.bincount(y_cal, minlength=60))
            or (h < 0).any() or (b < 0).any() or (h > support).any() or (b > support).any()):
        raise ValueError("Invalid checkpoint calibration counts")
    recalculated_h = torch.bincount(y_cal[hf_cal.argmax(-1) == y_cal], minlength=60)
    recalculated_b = torch.bincount(y_cal[bf_cal.argmax(-1) == y_cal], minlength=60)
    total = h + b
    weights = torch.where(total > 0, h / total.clamp_min(1), 0.5)

    def fused(hf, bf):
        scores = weights * hf.softmax(-1) + (1 - weights) * bf.softmax(-1)
        return scores / scores.sum(-1, keepdim=True)

    p_cal = fused(hf_cal, bf_cal)
    matrix = 10.0 * torch.eye(60)
    matrix.index_add_(0, y_cal, p_cal)
    matrix /= matrix.sum(0, keepdim=True)
    p_val = fused(hf_val, bf_val)
    candidate = 0.75 * p_val + 0.25 * (p_val @ matrix.T)
    if (not torch.isfinite(candidate).all()
            or not torch.allclose(candidate.sum(-1), torch.ones(len(candidate)), atol=1e-6)):
        raise RuntimeError("Invalid corrected probability")
    before_pred, after_pred = p_val.argmax(-1), candidate.argmax(-1)
    before, after = (before_pred == y_val).numpy(), (after_pred == y_val).numpy()
    diff = after.astype(np.int8) - before.astype(np.int8)
    subject_names, inverse = np.unique(subject_val, return_inverse=True)
    counts = np.bincount(inverse)
    changes = np.bincount(inverse, weights=diff, minlength=len(subject_names))
    rng = np.random.default_rng(20261109)
    draws = rng.multinomial(len(subject_names), np.full(len(subject_names), 1 / len(subject_names)), size=20000)
    boot = 100 * (draws @ changes) / (draws @ counts)
    old_correct, new_correct = int(before.sum()), int(after.sum())
    per_subject = [{"subject": name, "samples": int(count), "net_correct": int(change)}
                   for name, count, change in zip(subject_names, counts, changes)]
    summary = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT, "epoch": 83,
        "calibration_cache_sha256": digest(args.calibration_cache),
        "validation_cache_sha256": digest(args.validation_cache),
        "calibration_members_sha256": EXPECTED_CALIBRATION,
        "validation_members_sha256": EXPECTED_VALIDATION,
        "calibration_samples": len(y_cal), "validation_samples": len(y_val),
        "calibration_subjects": sorted(set(subject_cal)), "validation_subjects": len(subject_names),
        "method": "gamma=1 checkpoint-frozen calibration counts; calibration soft confusion, identity pseudocount=10; 0.75*p+0.25*M*p",
        "calibration_reinference_count_mismatch_classes": {
            "hf": int((recalculated_h != h).sum()), "bf": int((recalculated_b != b).sum())},
        "baseline_reported_correct": BASELINE_REPORTED_CORRECT,
        "cached_baseline_correct": old_correct,
        "cached_baseline_minus_prior_report_correct": old_correct - BASELINE_REPORTED_CORRECT,
        "candidate_correct": new_correct,
        "baseline_accuracy_percent": 100 * float(before.mean()),
        "candidate_accuracy_percent": 100 * float(after.mean()),
        "gain": int((diff == 1).sum()), "loss": int((diff == -1).sum()),
        "difference_pp": 100 * float(diff.mean()),
        "subject_cluster_bootstrap_95_ci_pp": np.quantile(boot, [0.025, 0.975]).tolist(),
        "bootstrap_repeats": 20000, "bootstrap_seed": 20261109,
        "bootstrap_unit": "complete subject; sample-weighted paired difference",
        "baseline_nll": float(F.nll_loss(p_val.log(), y_val)),
        "candidate_nll": float(F.nll_loss(candidate.log(), y_val)),
        "per_subject": per_subject,
        "limitations": [
            "Exploratory: validation data selected the original best epoch and informed historical error diagnosis.",
            "The original model used calibration-subject labels for training-time routing.",
            "This evaluation fixes one candidate and does not select its parameters using validation labels.",
            "Cached logits and original 15536-correct report were generated by separate inference runs.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "paired_samples.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "subject", "label", "baseline_prediction", "candidate_prediction", "baseline_correct", "candidate_correct"])
        writer.writerows(zip(members, subject_val, y_val.tolist(), before_pred.tolist(), after_pred.tolist(),
                             before.astype(int).tolist(), after.astype(int).tolist()))
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
