"""Evaluate fixed normalized-rank fusion with two-subject LOSO calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.face_pair_fusion_sweep import LogitFusion, class_weights_from_counts


def normalized_average_rank_scores(logits: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return descending average-rank scores normalized to sum to one.

    For each class, score = 1 + number of strictly lower logits + half of
    the other logits tied with it. This assigns the average competition rank
    to ties and makes equal logits exactly equal. Tied final fusion scores are
    resolved by torch.argmax, which selects the lowest class index.
    """
    if logits.ndim != 2 or logits.shape[1] < 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Expected finite N x C logits")
    values = logits.float()
    candidate = values[:, :, None]
    other = values[:, None, :]
    lower = (other < candidate).sum(-1).to(values.dtype)
    equal_count = (other == candidate).sum(-1).to(values.dtype)
    scores = 1.0 + lower + 0.5 * (equal_count - 1.0)
    normalizer = values.shape[1] * (values.shape[1] + 1) / 2.0
    scores = scores / normalizer
    if not torch.allclose(scores.sum(-1), torch.ones(len(scores)), atol=1e-6, rtol=0):
        raise RuntimeError("Normalized rank scores do not sum to one")
    tied_samples = int((equal_count.max(-1).values > 1).sum())
    return scores, tied_samples


def top_tie_sample_count(scores: torch.Tensor) -> int:
    maximum = scores.max(-1, keepdim=True).values
    return int((scores == maximum).sum(-1).gt(1).sum())


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
    member_sha = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if member_sha != cache["member_sha256"] or len(members) != len(cache["labels"]):
        raise RuntimeError("Cached logits do not match calibration member manifest")
    subject_ids = []
    for member in members:
        match = re.search(r"P(\d{3})", member)
        if not match:
            raise RuntimeError(f"Cannot parse subject ID from {member}")
        subject_ids.append(int(match.group(1)))
    subjects = torch.tensor(subject_ids)
    hf, bf = cache["hf"].float(), cache["bf"].float()
    labels = cache["labels"].long()
    observed_subjects = sorted(set(subject_ids))
    if len(observed_subjects) != 2 or hf.shape != bf.shape or hf.shape[0] != len(labels):
        raise RuntimeError("Expected aligned HF/BF logits and exactly two subjects")

    fusion = LogitFusion().eval()
    fold_results, prediction_rows = [], []
    with torch.inference_mode():
        for held_out in observed_subjects:
            test = subjects == held_out
            fit = ~test
            fit_y, fit_hf, fit_bf = labels[fit], hf[fit], bf[fit]
            support = torch.bincount(fit_y, minlength=hf.shape[1])
            hf_correct = torch.bincount(fit_y[fit_hf.argmax(-1) == fit_y], minlength=hf.shape[1])
            bf_correct = torch.bincount(fit_y[fit_bf.argmax(-1) == fit_y], minlength=hf.shape[1])
            weights = class_weights_from_counts(hf_correct, bf_correct, support, [1.0])[0]
            test_hf, test_bf, test_y = hf[test], bf[test], labels[test]

            baseline_log_probs = fusion(
                test_hf, test_bf, weights[None], torch.ones((1, 2)),
                torch.zeros(1), torch.zeros(1, dtype=torch.long),
            )[0]
            hf_rank, hf_tie_samples = normalized_average_rank_scores(test_hf)
            bf_rank, bf_tie_samples = normalized_average_rank_scores(test_bf)
            unnormalized = weights[None] * hf_rank + (1.0 - weights[None]) * bf_rank
            rank_probs = unnormalized / unnormalized.sum(-1, keepdim=True)
            rank_log_probs = rank_probs.clamp_min(torch.finfo(rank_probs.dtype).tiny).log()
            if (not torch.isfinite(rank_log_probs).all()
                    or not torch.allclose(rank_probs.sum(-1), torch.ones(len(test_y)), atol=1e-6)):
                raise RuntimeError("Invalid normalized-rank fusion distribution")

            baseline_pred = baseline_log_probs.argmax(-1)
            rank_pred = rank_probs.argmax(-1)
            baseline_ok = baseline_pred == test_y
            rank_ok = rank_pred == test_y
            gain = int((~baseline_ok & rank_ok).sum())
            loss = int((baseline_ok & ~rank_ok).sum())
            n = len(test_y)
            fold_results.append({
                "fit_subject": next(subject for subject in observed_subjects if subject != held_out),
                "heldout_subject": held_out,
                "fit_samples": int(fit.sum()),
                "test_samples": n,
                "classes_absent_in_fit": int((support == 0).sum()),
                "baseline_correct": int(baseline_ok.sum()),
                "rank_correct": int(rank_ok.sum()),
                "gain": gain,
                "loss": loss,
                "delta_pp": 100.0 * (gain - loss) / n,
                "baseline_probability_nll": float(F.nll_loss(baseline_log_probs, test_y)),
                "rank_pseudo_probability_nll": float(F.nll_loss(rank_log_probs, test_y)),
                "hf_samples_with_exact_logit_ties": hf_tie_samples,
                "bf_samples_with_exact_logit_ties": bf_tie_samples,
                "rank_fusion_samples_with_tied_maximum": top_tie_sample_count(rank_probs),
                "rank_score_definition": "descending average rank; normalized by C*(C+1)/2",
                "rank_fusion_tie_break": "lowest class index via torch.argmax",
            })
            test_members = [member for member, keep in zip(members, test.tolist()) if keep]
            for member, label, pred_base, pred_rank, p_base, p_rank in zip(
                    test_members, test_y.tolist(), baseline_pred.tolist(), rank_pred.tolist(),
                    baseline_log_probs.exp().amax(-1).tolist(), rank_probs.amax(-1).tolist()):
                prediction_rows.append({
                    "sample": member,
                    "heldout_subject": held_out,
                    "label": label,
                    "static_gamma1_prediction": pred_base,
                    "rank_gamma1_prediction": pred_rank,
                    "static_gamma1_confidence": p_base,
                    "rank_gamma1_pseudo_confidence": p_rank,
                })

    total = sum(row["test_samples"] for row in fold_results)
    pooled = {
        "samples": total,
        "baseline_correct": sum(row["baseline_correct"] for row in fold_results),
        "rank_correct": sum(row["rank_correct"] for row in fold_results),
        "gain": sum(row["gain"] for row in fold_results),
        "loss": sum(row["loss"] for row in fold_results),
    }
    pooled["baseline_top1_percent"] = 100.0 * pooled["baseline_correct"] / total
    pooled["rank_top1_percent"] = 100.0 * pooled["rank_correct"] / total
    pooled["delta_pp"] = 100.0 * (pooled["gain"] - pooled["loss"]) / total
    pooled["baseline_probability_nll"] = sum(
        row["test_samples"] * row["baseline_probability_nll"] for row in fold_results
    ) / total
    pooled["rank_pseudo_probability_nll"] = sum(
        row["test_samples"] * row["rank_pseudo_probability_nll"] for row in fold_results
    ) / total

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    result = {
        "cache_sha256": sha256(args.cache),
        "checkpoint_sha256": cache["checkpoint_sha256"],
        "cache_epoch": cache["epoch"],
        "partition": cache["partition"],
        "members_sha256": member_sha,
        "subjects": observed_subjects,
        "method": "two-fold LOSO; estimate gamma=1 class accuracy weights on fit subject; apply same class weights to static probability and normalized average-rank fusion on held-out subject",
        "rank_definition": "per sample and head, descending competition rank with arithmetic-mean ranks for exact logit ties; positive scores from 1 to C normalized by C*(C+1)/2",
        "final_rank_score_ties": "argmax chooses lowest class index; tie counts reported per fold",
        "rank_nll_interpretation": "NLL of the normalized rank-derived pseudo-probability, not a calibrated posterior likelihood",
        "no_official_validation_labels_used": True,
        "folds": fold_results,
        "pooled": pooled,
        "limitations": [
            "Only two training calibration subjects are available; this does not estimate variation across new subjects.",
            "The normalized rank NLL is a comparison score on a pseudo-probability distribution, not a probabilistic posterior claim.",
            "The cache is from epoch 83; results do not establish behavior of other checkpoints.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
