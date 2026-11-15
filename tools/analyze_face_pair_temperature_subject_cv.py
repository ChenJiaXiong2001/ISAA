"""Screen per-head temperature scaling with leave-one-subject-out calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from tools.face_pair_fusion_sweep import (
    CONFIDENCE_UNIFORM, LogitFusion, class_weights_from_counts, fit_temperature,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    parser.add_argument("--replicates", type=int, default=20000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}")
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    report = json.loads(args.cache_report.read_text(encoding="utf-8"))
    members = report["manifest"]["calibration"]["members"]
    member_hash = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if member_hash != cache["member_sha256"]:
        raise RuntimeError("Calibration members do not match cached logits")
    subject_ids = []
    for member in members:
        match = re.search(r"P(\d{3})", member)
        if not match:
            raise RuntimeError(f"No subject in {member}")
        subject_ids.append(int(match.group(1)))
    subjects = torch.tensor(subject_ids)
    hf, bf, labels = cache["hf"].float(), cache["bf"].float(), cache["labels"].long()
    if (len(members) != len(labels) or hf.shape != bf.shape or hf.shape[0] != len(labels)
            or not torch.isfinite(hf).all() or not torch.isfinite(bf).all()):
        raise RuntimeError("Invalid cached logits or labels")
    observed_subjects = sorted(set(subject_ids))
    if len(observed_subjects) != 2:
        raise RuntimeError("This screening protocol expects exactly two calibration subjects")

    model = LogitFusion().eval()
    rng = np.random.default_rng(args.bootstrap_seed)
    folds, rows = [], []
    with torch.no_grad():
        for held_out in observed_subjects:
            fit = subjects != held_out
            test = ~fit
            train_labels, train_hf, train_bf = labels[fit], hf[fit], bf[fit]
            total = torch.bincount(train_labels, minlength=hf.shape[1])
            correct_hf = torch.bincount(train_labels[train_hf.argmax(-1) == train_labels], minlength=hf.shape[1])
            correct_bf = torch.bincount(train_labels[train_bf.argmax(-1) == train_labels], minlength=hf.shape[1])
            weights = class_weights_from_counts(correct_hf, correct_bf, total, [1.0])
            temperature_hf = fit_temperature(train_hf, train_labels)
            temperature_bf = fit_temperature(train_bf, train_labels)
            test_hf, test_bf, test_labels = hf[test], bf[test], labels[test]
            if (not torch.equal(train_hf.argmax(-1), (train_hf / temperature_hf).argmax(-1))
                    or not torch.equal(train_bf.argmax(-1), (train_bf / temperature_bf).argmax(-1))):
                raise RuntimeError("Positive temperatures changed head accuracy-based weights")
            temperatures = torch.tensor([[1., 1.], [temperature_hf, temperature_bf]])
            probabilities = model(test_hf, test_bf, weights.expand(2, -1), temperatures,
                                  torch.zeros(2), torch.full((2,), CONFIDENCE_UNIFORM))
            if not torch.isfinite(probabilities).all() or not torch.allclose(
                    probabilities.exp().sum(-1), torch.ones(2, len(test_labels)), atol=1e-5):
                raise RuntimeError("Invalid fused probabilities")
            baseline, candidate = probabilities.argmax(-1)
            baseline_ok, candidate_ok = baseline == test_labels, candidate == test_labels
            gain, loss = int((~baseline_ok & candidate_ok).sum()), int((baseline_ok & ~candidate_ok).sum())
            n = len(test_labels)
            bootstrap = rng.multinomial(n, [gain / n, loss / n, (n - gain - loss) / n], size=args.replicates)
            ci = np.percentile(100 * (bootstrap[:, 0] - bootstrap[:, 1]) / n, [2.5, 97.5]).tolist()
            folds.append({"fit_subjects": [s for s in observed_subjects if s != held_out],
                "heldout_subject": held_out, "fit_samples": len(train_labels), "test_samples": n,
                "classes_unseen_in_fit": int((total == 0).sum()),
                "hf_temperature": temperature_hf, "bf_temperature": temperature_bf,
                "baseline_correct": int(baseline_ok.sum()), "candidate_correct": int(candidate_ok.sum()),
                "baseline_top1_percent": 100 * float(baseline_ok.float().mean()),
                "candidate_top1_percent": 100 * float(candidate_ok.float().mean()),
                "baseline_nll": float(F.nll_loss(probabilities[0], test_labels)),
                "candidate_nll": float(F.nll_loss(probabilities[1], test_labels)),
                "gain": gain, "loss": loss, "delta_pp": 100 * (gain - loss) / n,
                "sample_bootstrap_95_ci_pp": ci})
            test_members = [member for member, keep in zip(members, test.tolist()) if keep]
            for member, label, raw_pred, temp_pred in zip(test_members, test_labels.tolist(), baseline.tolist(), candidate.tolist()):
                rows.append({"sample": member, "heldout_subject": held_out, "label": label,
                             "baseline_pred": raw_pred, "temperature_pred": temp_pred})

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"cache_sha256": digest(args.cache), "checkpoint_sha256": cache["checkpoint_sha256"],
        "cache_partition": cache["partition"], "cache_epoch": cache["epoch"],
        "calibration_members_sha256": member_hash, "subjects": observed_subjects,
        "method": "leave-one-subject-out; per-head NLL temperature on fit subject; gamma=1 class accuracy weights from same fit subject",
        "temperature_bounds": [0.1, 10.0], "no_validation_data_used": True,
        "bootstrap_seed": args.bootstrap_seed, "bootstrap_replicates": args.replicates, "folds": folds}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
