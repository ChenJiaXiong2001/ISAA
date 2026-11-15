"""Screen fixed confidence gates with leave-one-subject-out calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.face_pair_fusion_sweep import (
    CONFIDENCE_MAXPROB,
    CONFIDENCE_MARGIN,
    CONFIDENCE_NORMALIZED_ENTROPY,
    CONFIDENCE_UNIFORM,
    LogitFusion,
    class_weights_from_counts,
)


MODES = {
    "maxprob": CONFIDENCE_MAXPROB,
    "margin": CONFIDENCE_MARGIN,
    "normalized_entropy": CONFIDENCE_NORMALIZED_ENTROPY,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    report = json.loads(args.cache_report.read_text(encoding="utf-8"))
    members = report["manifest"]["calibration"]["members"]
    members_hash = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if members_hash != cache["member_sha256"] or len(members) != len(cache["labels"]):
        raise RuntimeError("Calibration cache member manifest mismatch")
    subject_ids = []
    for member in members:
        match = re.search(r"P(\d{3})", member)
        if not match:
            raise RuntimeError(f"Cannot recover subject from {member}")
        subject_ids.append(int(match.group(1)))
    subjects = torch.tensor(subject_ids)
    hf, bf, labels = cache["hf"].float(), cache["bf"].float(), cache["labels"].long()
    observed_subjects = sorted(set(subject_ids))
    if len(observed_subjects) != 2 or hf.shape != bf.shape or hf.shape[0] != len(labels):
        raise RuntimeError("Expected aligned logits and exactly two calibration subjects")

    fusion = LogitFusion().eval()
    rows = []
    folds = []
    with torch.inference_mode():
        for held_out in observed_subjects:
            test = subjects == held_out
            fit = ~test
            fit_y, fit_hf, fit_bf = labels[fit], hf[fit], bf[fit]
            support = torch.bincount(fit_y, minlength=hf.shape[-1])
            hf_correct = torch.bincount(fit_y[fit_hf.argmax(-1) == fit_y], minlength=hf.shape[-1])
            bf_correct = torch.bincount(fit_y[fit_bf.argmax(-1) == fit_y], minlength=hf.shape[-1])
            weights = class_weights_from_counts(hf_correct, bf_correct, support, [1.0])
            test_hf, test_bf, test_y = hf[test], bf[test], labels[test]
            candidates = [("static", 0, 0.0)] + [
                (mode, mode_id, beta)
                for mode, mode_id in MODES.items()
                for beta in (0.5, 1.0, 2.0)
            ]
            batch_weights = weights.expand(len(candidates), -1)
            temperatures = torch.ones((len(candidates), 2))
            betas = torch.tensor([item[2] for item in candidates])
            modes = torch.tensor([CONFIDENCE_UNIFORM if item[1] == 0 else item[1]
                                  for item in candidates])
            log_probs = fusion(test_hf, test_bf, batch_weights, temperatures, betas, modes)
            if not torch.isfinite(log_probs).all() or not torch.allclose(
                    log_probs.exp().sum(-1), torch.ones(len(candidates), len(test_y)), atol=1e-5):
                raise RuntimeError("Invalid fused probabilities")
            baseline_ok = log_probs[0].argmax(-1) == test_y
            fold_candidates = []
            for index, (mode, mode_id, beta) in enumerate(candidates):
                if index == 0:
                    continue
                candidate_ok = log_probs[index].argmax(-1) == test_y
                gain = int((~baseline_ok & candidate_ok).sum())
                loss = int((baseline_ok & ~candidate_ok).sum())
                n = len(test_y)
                delta = 100 * (gain - loss) / n
                nll_base = float(F.nll_loss(log_probs[0], test_y))
                nll_candidate = float(F.nll_loss(log_probs[index], test_y))
                row = {
                    "heldout_subject": held_out,
                    "fit_subjects": ",".join(str(s) for s in observed_subjects if s != held_out),
                    "mode": mode,
                    "beta": beta,
                    "samples": n,
                    "baseline_correct": int(baseline_ok.sum()),
                    "candidate_correct": int(candidate_ok.sum()),
                    "gain": gain,
                    "loss": loss,
                    "delta_pp": delta,
                    "baseline_nll": nll_base,
                    "candidate_nll": nll_candidate,
                    "delta_nll": nll_candidate - nll_base,
                }
                rows.append(row)
                fold_candidates.append(row)
            folds.append({
                "fit_subject": next(s for s in observed_subjects if s != held_out),
                "heldout_subject": held_out,
                "fit_samples": int(fit.sum()),
                "test_samples": int(test.sum()),
                "classes_absent_in_fit": int((support == 0).sum()),
                "baseline_correct": int(baseline_ok.sum()),
                "baseline_nll": float(F.nll_loss(log_probs[0], test_y)),
                "candidates": fold_candidates,
            })

    pooled = []
    for mode, mode_id, beta in [(m, i, b) for m, i in MODES.items() for b in (0.5, 1.0, 2.0)]:
        parts = [row for row in rows if row["mode"] == mode and row["beta"] == beta]
        pooled.append({
            "mode": mode,
            "beta": beta,
            "samples": sum(row["samples"] for row in parts),
            "baseline_correct": sum(row["baseline_correct"] for row in parts),
            "candidate_correct": sum(row["candidate_correct"] for row in parts),
            "gain": sum(row["gain"] for row in parts),
            "loss": sum(row["loss"] for row in parts),
            "delta_pp": sum(row["samples"] * row["delta_pp"] for row in parts)
                        / sum(row["samples"] for row in parts),
            "baseline_nll": sum(row["samples"] * row["baseline_nll"] for row in parts)
                            / sum(row["samples"] for row in parts),
            "candidate_nll": sum(row["samples"] * row["candidate_nll"] for row in parts)
                             / sum(row["samples"] for row in parts),
        })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "folds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "cache_sha256": sha256(args.cache),
        "checkpoint_sha256": cache["checkpoint_sha256"],
        "cache_epoch": cache["epoch"],
        "cache_partition": cache["partition"],
        "members_sha256": members_hash,
        "subjects": observed_subjects,
        "method": "LOSO; fit gamma=1 class accuracy weights on fit subject only; test fixed reliability gates on held-out subject; unit temperatures",
        "modes": list(MODES),
        "betas": [0.5, 1.0, 2.0],
        "no_official_validation_data_used": True,
        "folds": folds,
        "pooled": pooled,
        "limitations": [
            "Only two training calibration subjects are available; two folds cannot estimate new-subject variation.",
            "Reliability candidates were previously explored in an official-validation sweep; this LOSO is a bounded internal check, not an independent method-selection test.",
            "Fold-specific gamma=1 weights make this an end-to-end calibration protocol comparison, not a pure gate-only intervention.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
