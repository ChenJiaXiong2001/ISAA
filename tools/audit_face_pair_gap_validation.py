"""Audit per-class max5 outcomes and paired clean prediction differences."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-dir", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gap_summary = json.loads((args.gap_dir / "summary.json").read_text(encoding="utf-8"))
    compiled_summary = json.loads((args.compiled_dir / "summary.json").read_text(encoding="utf-8"))
    gap = read_rows(args.gap_dir / "predictions.csv")
    compiled = read_rows(args.compiled_dir / "predictions.csv")
    if gap_summary["checkpoint_sha256"] != compiled_summary["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    if (gap_summary["input_sha256"]["clean"] != compiled_summary["input_sha256"] or
            gap_summary["input_sha256"]["frame_mask"] != compiled_summary["frame_mask_sha256"]):
        raise ValueError("Clean inputs or frame masks differ")
    if len(gap) != len(compiled) or len(gap) != gap_summary["samples"]:
        raise ValueError("Sample count mismatch")
    if any(a["sample"] != b["sample"] or a["label"] != b["label"] for a, b in zip(gap, compiled)):
        raise ValueError("Sample or label order mismatch")
    digest = hashlib.sha256("\n".join(row["sample"] for row in gap).encode()).hexdigest()
    if digest != gap_summary["members_sha256"] or digest != compiled_summary["validation_members_sha256"]:
        raise ValueError("Member hash mismatch")
    conditions = {}
    for condition in ("clean", "gap2", "gap5", "gap10"):
        classes = []
        for label in sorted({int(row["label"]) for row in gap}):
            selected = [row for row in gap if int(row["label"]) == label]
            before = [int(row[f"{condition}_unmodified_prediction"]) == label for row in selected]
            after = [int(row[f"{condition}_max5_prediction"]) == label for row in selected]
            corrected = sum(not a and b for a, b in zip(before, after))
            regressed = sum(a and not b for a, b in zip(before, after))
            classes.append({"class_index": label, "action": f"A{label + 1:03}", "samples": len(selected),
                            "corrected": corrected, "new_errors": regressed,
                            "difference_pp": 100 * (corrected - regressed) / len(selected)})
        conditions[condition] = {"classes_improved": sum(row["difference_pp"] > 0 for row in classes),
                                 "classes_unchanged": sum(row["difference_pp"] == 0 for row in classes),
                                 "classes_degraded": sum(row["difference_pp"] < 0 for row in classes),
                                 "classes": classes}
    changes = [{"sample": a["sample"], "label": int(a["label"]),
                "eager_prediction": int(a["clean_unmodified_prediction"]),
                "compiled_prediction": int(b["baseline_prediction"])}
               for a, b in zip(gap, compiled)
               if a["clean_unmodified_prediction"] != b["baseline_prediction"]]
    result = {"members_sha256": digest, "clean_input_sha256": gap_summary["input_sha256"]["clean"],
              "frame_mask_sha256": gap_summary["input_sha256"]["frame_mask"],
              "checkpoint_sha256": gap_summary["checkpoint_sha256"], "samples": len(gap),
              "clean_prediction_disagreements": changes, "conditions": conditions,
              "caveat": "Paired predictions are descriptive; identical input hashes do not isolate the numerical cause of disagreements."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(gap), "clean_prediction_disagreements": changes,
                      "class_counts": {name: {key: value for key, value in details.items() if key != "classes"}
                                       for name, details in conditions.items()}}, indent=2))


if __name__ == "__main__":
    main()
