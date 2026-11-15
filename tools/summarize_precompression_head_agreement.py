"""Summarize a frozen pre-compression evaluation with a head-agreement gate.

The evaluator only emits HF/BF columns when explicitly passed
``--include-head-predictions``. This script is read-only and applies a fixed
gate: use the repaired soft prediction when repaired HF and BF argmax agree;
otherwise fall back to unmodified soft fusion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _stats(rows: list[dict[str, str]], condition: str, method: str, base_method: str = "unmodified"):
    labels = [int(row["label"]) for row in rows]
    base = [int(row[f"{condition}_{base_method}_prediction"]) for row in rows]
    repaired = [int(row[f"{condition}_{method}_prediction"]) for row in rows]
    gated = [
        prediction
        if row[f"{condition}_{method}_HF_prediction"] == row[f"{condition}_{method}_BF_prediction"]
        else original
        for row, prediction, original in zip(rows, repaired, base)
    ]

    def stats(predictions):
        before_correct = sum(prediction == label for prediction, label in zip(base, labels))
        correct = sum(prediction == label for prediction, label in zip(predictions, labels))
        corrected = sum(original != label and prediction == label for original, prediction, label in zip(base, predictions, labels))
        new_errors = sum(original == label and prediction != label for original, prediction, label in zip(base, predictions, labels))
        return {
            "correct": correct,
            "top1_percent": 100.0 * correct / len(labels),
            "difference_pp_vs_unmodified": 100.0 * (correct - before_correct) / len(labels),
            "changed": sum(original != prediction for original, prediction in zip(base, predictions)),
            "corrected": corrected,
            "new_errors": new_errors,
            "net_change": corrected - new_errors,
        }

    return {"unmodified": stats(base), "always_repaired": stats(repaired), "head_agreement_gate": stats(gated)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation_dir = args.evaluation_dir.resolve()
    prediction_path = evaluation_dir / "predictions.csv"
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("empty predictions")
    conditions = ("clean", "pre_dropout")
    methods = sorted(
        key.removeprefix("clean_").removesuffix("_HF_prediction")
        for key in rows[0]
        if key.startswith("clean_") and key.endswith("_HF_prediction")
    )
    required = []
    for condition in conditions:
        for method in methods:
            required.extend(
                [
                    f"{condition}_{method}_prediction",
                    f"{condition}_{method}_HF_prediction",
                    f"{condition}_{method}_BF_prediction",
                ]
            )
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise ValueError(f"missing head columns: {missing[:5]}")
    results = {
        "schema_version": 1,
        "analysis": "frozen pre-compression head-agreement arbitration",
        "read_only": True,
        "source": {"path": str(prediction_path), "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(), "samples": len(rows)},
        "gate": "Use repaired gamma=1 soft prediction only when repaired HF and BF argmax classes agree; otherwise use unmodified soft fusion.",
        "conditions": {},
    }
    for condition in conditions:
        results["conditions"][condition] = {}
        for method in methods:
            if method == "unmodified":
                continue
            results["conditions"][condition][method] = _stats(rows, condition, method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
