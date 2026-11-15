"""Screen a fixed anchor-repair arbitration rule from frozen predictions.

This script is read-only with respect to model artifacts. It consumes paired
predictions produced by an anchor-gap evaluator and compares always-repaired
fusion with a conservative head-agreement gate. No labels are used to choose a
rule; labels are only used to report the frozen calibration result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


CONDITIONS = ("clean", "independent05", "gap2", "gap5", "gap10")
METHODS = (("max1", "repaired"), ("max2", "max2"), ("max5", "max5"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(rows: list[dict[str, str]], condition: str, method_column: str) -> dict[str, object]:
    label_key = "label"
    base_key = f"{condition}_unmodified_soft"
    repaired_key = f"{condition}_{method_column}_soft"
    repaired_hf_key = f"{condition}_{method_column}_HF"
    repaired_bf_key = f"{condition}_{method_column}_BF"

    labels = [int(row[label_key]) for row in rows]
    base = [int(row[base_key]) for row in rows]
    repaired = [int(row[repaired_key]) for row in rows]
    gated = [
        prediction
        if row[repaired_hf_key] == row[repaired_bf_key]
        else original
        for row, prediction, original in zip(rows, repaired, base)
    ]

    def stats(predictions: list[int]) -> dict[str, object]:
        correct = sum(prediction == label for prediction, label in zip(predictions, labels))
        corrected = sum(
            original != label and prediction == label
            for original, prediction, label in zip(base, predictions, labels)
        )
        regressed = sum(
            original == label and prediction != label
            for original, prediction, label in zip(base, predictions, labels)
        )
        changed = sum(original != prediction for original, prediction in zip(base, predictions))
        return {
            "correct": correct,
            "accuracy_percent": 100.0 * correct / len(labels),
            "delta_percentage_points": 100.0 * (correct - sum(x == y for x, y in zip(base, labels))) / len(labels),
            "changed": changed,
            "corrected": corrected,
            "regressed": regressed,
            "net_change": corrected - regressed,
        }

    return {
        "samples": len(labels),
        "unmodified": stats(base),
        "always_repaired": stats(repaired),
        "head_agreement_gate": stats(gated),
        "gate_definition": (
            "Use repaired gamma=1 soft prediction only when repaired HF and BF "
            "argmax classes agree; otherwise use unmodified gamma=1 soft prediction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default="unknown")
    parser.add_argument("--calibration-members-sha256", default="unknown")
    parser.add_argument("--mask-seed", type=int, default=1729)
    parser.add_argument("--fusion", default="gamma=1 HF/BF original-accuracy-proportional soft fusion")
    args = parser.parse_args()
    source = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Prediction file is empty")
    required = {"label"}
    for condition in CONDITIONS:
        required.add(f"{condition}_unmodified_soft")
        for _, method_column in METHODS:
            required.update(
                {
                    f"{condition}_{method_column}_soft",
                    f"{condition}_{method_column}_HF",
                    f"{condition}_{method_column}_BF",
                }
            )
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results: dict[str, object] = {
        "schema_version": 1,
        "analysis": "frozen anchor repair head-agreement arbitration on calibration predictions",
        "read_only": True,
        "source": {"path": str(source), "sha256": sha256(source), "samples": len(rows)},
        "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_members_sha256": args.calibration_members_sha256,
        "mask_seed": args.mask_seed,
        "fusion": args.fusion,
        "rule_status": "fixed before any validation evaluation",
        "rule": {
            "repair": "bounded linear interpolation of shoulder/hip anchors, max gap 5",
            "arbitration": "repaired HF/BF argmax agreement gate; fallback to unmodified soft fusion",
        },
        "conditions": {},
    }
    lines = [
        "# Anchor repair head-agreement gate calibration screen",
        "",
        "This is a read-only analysis of frozen paired predictions. The rule is fixed before any validation evaluation.",
        "",
        f"Source: `{source}`; SHA256 `{results['source']['sha256']}`; samples: {len(rows)}.",
        f"Checkpoint SHA256: `{args.checkpoint_sha256}`.",
        f"Calibration members SHA256: `{args.calibration_members_sha256}`; mask seed: `{args.mask_seed}`.",
        f"Fusion: {args.fusion}.",
        "",
        "Gate: use the repaired gamma=1 soft prediction only when repaired HF and BF argmax classes agree; otherwise use the unmodified gamma=1 soft prediction.",
        "",
        "| condition | method | unmodified | always repaired | always delta (pp) | always corrected/regressed | gated | gated delta (pp) | gated corrected/regressed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        results["conditions"][condition] = {}
        for method_name, method_column in METHODS:
            result = evaluate(rows, condition, method_column)
            results["conditions"][condition][method_name] = result
            always = result["always_repaired"]
            gated = result["head_agreement_gate"]
            base = result["unmodified"]
            lines.append(
                f"| {condition} | {method_name} | {base['accuracy_percent']:.3f}% | "
                f"{always['accuracy_percent']:.3f}% | {always['delta_percentage_points']:+.3f} | "
                f"{always['corrected']}/{always['regressed']} | {gated['accuracy_percent']:.3f}% | "
                f"{gated['delta_percentage_points']:+.3f} | {gated['corrected']}/{gated['regressed']} |"
            )
    lines += [
        "",
        "## Decision",
        "",
        "The max5 head-agreement gate is the candidate to register for a later locked validation traversal. It trades recovery for zero observed calibration regressions in the independent-5%, gap-2, and gap-5 conditions, while leaving clean and gap-10 unchanged on this calibration set.",
        "",
        "This screen does not establish deployment benefit. It uses two training-subject calibration members, and the checkpoint's calibration counts also define the gamma=1 fusion weights. Validation labels were not used to select the rule. A later validation run must use the exact rule and report clean accuracy, each corruption condition, subject-level results, and added two-forward-pass latency.",
    ]
    (output_dir / "screen.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
