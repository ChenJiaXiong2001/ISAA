"""Read-only routing analysis for frozen pre-compression predictions.

The hidden-point count is the synthetic corruption count emitted by the
evaluator. Rules using it are oracle/validation-only diagnostics unless a
runtime detector can provide the same count.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def _subjects(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = [row["sample"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate sample names")
    values = []
    for name in names:
        match = re.search(r"P([0-9]{3})R[0-9]{3}A[0-9]{3}", name)
        if not match:
            raise ValueError(f"cannot parse subject from {name}")
        values.append(match.group(1))
    subject_ids, group = np.unique(values, return_inverse=True)
    return np.asarray(subject_ids), group, np.bincount(group)


def _bootstrap(delta: np.ndarray, group: np.ndarray, sizes: np.ndarray, repeats: int, seed: int) -> dict:
    subject_delta = np.bincount(group, weights=delta, minlength=len(sizes))
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(sizes), np.full(len(sizes), 1.0 / len(sizes)), size=repeats)
    denominator = draws @ sizes
    distribution = 100.0 * (draws @ subject_delta) / denominator
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    return {
        "difference_pp": float(delta.mean() * 100.0),
        "subject_bootstrap_95_ci_pp": [float(lo), float(hi)],
        "subjects_positive": int((subject_delta > 0).sum()),
        "subjects_negative": int((subject_delta < 0).sum()),
        "subjects_tied": int((subject_delta == 0).sum()),
    }


def _evaluate(base: np.ndarray, pred: np.ndarray, labels: np.ndarray, group: np.ndarray,
              sizes: np.ndarray, repeats: int, seed: int, eligible: np.ndarray | None) -> dict:
    if eligible is None:
        eligible = np.ones(len(labels), dtype=bool)
    routed = np.where(eligible, pred, base)
    base_hit = (base == labels).astype(np.int8)
    hit = (routed == labels).astype(np.int8)
    delta = hit - base_hit
    result = {
        "correct": int(hit.sum()),
        "samples": int(len(labels)),
        "top1_percent": float(hit.mean() * 100.0),
        "difference_pp_vs_unmodified": float(delta.mean() * 100.0),
        "changed": int(np.count_nonzero(routed != base)),
        "corrected": int((delta == 1).sum()),
        "new_errors": int((delta == -1).sum()),
        "eligible_samples": int(eligible.sum()),
        "eligible_percent": float(eligible.mean() * 100.0),
        "subject_bootstrap": _bootstrap(delta, group, sizes, repeats, seed),
    }
    return result


def analyze(path: Path, repeats: int, seed: int) -> dict:
    prediction_path = path / "predictions.csv"
    payload = prediction_path.read_bytes()
    with prediction_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty predictions: {prediction_path}")
    subjects, group, sizes = _subjects(rows)
    labels = np.asarray([int(row["label"]) for row in rows])
    hidden = np.asarray([int(row["hidden_original_points"]) for row in rows])
    output = {
        "analysis": "read-only hidden-count routing exploration",
        "validation_reuse": True,
        "routing_caveat": "hidden_original_points is synthetic corruption metadata; hidden-count rules are oracle diagnostics unless a runtime detector supplies this count",
        "source": {"path": str(prediction_path), "sha256": hashlib.sha256(payload).hexdigest(), "samples": len(rows)},
        "subjects": int(len(subjects)),
        "subject_ids": [f"P{x}" for x in subjects],
        "bootstrap": {"repeats": repeats, "seed": seed, "unit": "complete subject; sample-weighted pooled accuracy"},
        "hidden_count": {"min": int(hidden.min()), "max": int(hidden.max()), "mean": float(hidden.mean()), "positive": int((hidden > 0).sum())},
        "conditions": {},
    }
    for condition in ("clean", "pre_dropout"):
        base = np.asarray([int(row[f"{condition}_unmodified_prediction"]) for row in rows])
        repaired = np.asarray([int(row[f"{condition}_threshold_0.5_max5_prediction"]) for row in rows])
        hf = np.asarray([int(row[f"{condition}_threshold_0.5_max5_HF_prediction"]) for row in rows])
        bf = np.asarray([int(row[f"{condition}_threshold_0.5_max5_BF_prediction"]) for row in rows])
        agree = hf == bf
        rules: dict[str, np.ndarray | None] = {
            "always": np.ones(len(rows), dtype=bool),
            "hidden_gt_0": hidden > 0,
            "hidden_gt_0_and_head_agree": (hidden > 0) & agree,
            "head_agree_only": agree,
        }
        for threshold in (1, 5, 10):
            rules[f"hidden_ge_{threshold}"] = hidden >= threshold
            rules[f"hidden_ge_{threshold}_and_head_agree"] = (hidden >= threshold) & agree
        output["conditions"][condition] = {}
        for index, (name, eligible) in enumerate(rules.items()):
            output["conditions"][condition][name] = _evaluate(
                base, repaired, labels, group, sizes, repeats, seed + index, eligible
            )
        output["conditions"][condition]["head_agreement_diagnostics"] = {
            "repaired_head_agree": int(agree.sum()),
            "repaired_head_disagree": int((~agree).sum()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.evaluation_dir, args.repeats, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
