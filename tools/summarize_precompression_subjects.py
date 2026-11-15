"""Verify paired precompression evidence and summarize per-subject changes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    audit, evaluate = load_json(args.mechanism / "summary.json"), load_json(args.predictions / "summary.json")
    selected = [row["sample"] for row in load_csv(args.selection)]
    excluded = [row["sample"] for row in load_csv(args.excluded)]
    if len(set(selected)) != len(selected) or len(set(excluded)) != len(excluded):
        raise ValueError("Duplicate selected or excluded sample")
    overlap = set(selected) & set(excluded)
    if overlap:
        raise ValueError(f"Excluded sample overlap: {sorted(overlap)[:3]}")
    digest = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    if audit["members_sha256"] != digest or evaluate["members_sha256"] != digest:
        raise ValueError("Member ordering hash mismatch")
    if audit["input_sha256"] != evaluate["input_sha256"]:
        raise ValueError("Mechanism and prediction input hashes differ")
    rows = load_csv(args.predictions / "predictions.csv")
    if [row["sample"] for row in rows] != selected:
        raise ValueError("Prediction member order differs")
    subjects = {}
    for row in rows:
        match = re.search(r"P(\d{3})", row["sample"])
        if match is None:
            raise ValueError(f"Missing subject identifier: {row['sample']}")
        subject = f"P{match.group(1)}"
        group = subjects.setdefault(subject, {"subject": subject, "samples": 0,
            "clean_unmodified_correct": 0, "clean_max5_correct": 0,
            "clean_corrected": 0, "clean_new_errors": 0,
            "pre_dropout_unmodified_correct": 0, "pre_dropout_max5_correct": 0,
            "pre_dropout_corrected": 0, "pre_dropout_new_errors": 0})
        group["samples"] += 1
        for condition in ("clean", "pre_dropout"):
            old = row[f"{condition}_unmodified_prediction"] == row["label"]
            new = row[f"{condition}_max5_prediction"] == row["label"]
            group[f"{condition}_unmodified_correct"] += int(old)
            group[f"{condition}_max5_correct"] += int(new)
            group[f"{condition}_corrected"] += int(not old and new)
            group[f"{condition}_new_errors"] += int(old and not new)
    ordered = [subjects[name] for name in sorted(subjects)]
    for group in ordered:
        for condition in ("clean", "pre_dropout"):
            group[f"{condition}_net_correct"] = group[f"{condition}_corrected"] - group[f"{condition}_new_errors"]
            group[f"{condition}_difference_pp"] = 100 * group[f"{condition}_net_correct"] / group["samples"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_subject.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0])); writer.writeheader(); writer.writerows(ordered)
    summary = {"selected_samples": len(selected), "excluded_samples": len(excluded), "overlap": len(overlap),
               "member_order_sha256": digest, "matching_input_sha256": audit["input_sha256"],
               "subjects": ordered,
               "caveat": "Samples from the same subject are dependent; threshold choice followed earlier calibration exploration and fusion weights use this calibration partition."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
