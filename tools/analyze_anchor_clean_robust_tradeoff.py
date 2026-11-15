"""Describe sample-level clean/anchor-loss tradeoffs in paired predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CONDITIONS = ("clean", "anchor_pre_0.05", "anchor_pre_0.1")


def outcome(row):
    control, candidate = row["control_correct"], row["candidate_correct"]
    return {(0, 0): "both_wrong", (0, 1): "candidate_gain",
            (1, 0): "candidate_loss", (1, 1): "both_correct"}[(control, candidate)]


def summarize(rows):
    counts = Counter(outcome(row["clean"]) for row in rows)
    result = {"samples": len(rows), "clean_outcomes": dict(counts), "anchor_conditions": {}}
    for condition in CONDITIONS[1:]:
        overall = Counter(outcome(row[condition]) for row in rows)
        stratified = {}
        for clean_outcome in ("candidate_loss", "candidate_gain", "both_correct", "both_wrong"):
            selected = [row for row in rows if outcome(row["clean"]) == clean_outcome]
            cases = Counter(outcome(row[condition]) for row in selected)
            stratified[clean_outcome] = {"samples": len(selected), "anchor_outcomes": dict(cases),
                                         "anchor_gain_percent": 100 * cases["candidate_gain"] / len(selected) if selected else None,
                                         "anchor_loss_percent": 100 * cases["candidate_loss"] / len(selected) if selected else None,
                                         "anchor_net_pp": 100 * (cases["candidate_gain"] - cases["candidate_loss"]) / len(selected) if selected else None}
        result["anchor_conditions"][condition] = {"outcomes": dict(overall), "by_clean_outcome": stratified}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    source = args.evaluation_dir / "paired_samples.csv"
    by_condition = {condition: {} for condition in CONDITIONS}
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        for row in reader:
            condition = row["condition"]
            if condition not in by_condition or row["sample"] in by_condition[condition]:
                raise ValueError("Unexpected condition or duplicated sample")
            for key in ("true_class_index", "control_prediction", "candidate_prediction", "control_correct", "candidate_correct"):
                row[key] = int(row[key])
            if row["control_correct"] != int(row["control_prediction"] == row["true_class_index"]):
                raise ValueError("Invalid control correctness")
            if row["candidate_correct"] != int(row["candidate_prediction"] == row["true_class_index"]):
                raise ValueError("Invalid candidate correctness")
            by_condition[condition][row["sample"]] = row
    names = list(by_condition["clean"])
    if len(names) != summary["evaluated_samples_per_condition"]:
        raise ValueError("Unexpected sample count")
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    if digest != summary["full_validation_members_sha256"]:
        raise ValueError("Clean membership hash mismatch")
    for condition in CONDITIONS:
        if list(by_condition[condition]) != names:
            raise ValueError("Condition membership or order mismatch")
    samples, classes, subjects = [], defaultdict(list), defaultdict(list)
    for name in names:
        entries = {condition: by_condition[condition][name] for condition in CONDITIONS}
        label = entries["clean"]["true_class_index"]
        if any(row["true_class_index"] != label for row in entries.values()):
            raise ValueError("Label mismatch across conditions")
        subject = re.search(r"P(\d{3})R\d{3}A\d{3}", name)
        if subject is None:
            raise ValueError("Cannot identify subject")
        samples.append(entries)
        classes[label].append(entries)
        subjects[subject.group(1)].append(entries)
    output = {"control_sha256": summary["control_sha256"], "candidate_sha256": summary["candidate_sha256"],
              "members_sha256": digest, "paired_samples_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "input_conditions": summary["conditions"],
              "confidence_available": any("confidence" in field.lower() or "probability" in field.lower() or "logit" in field.lower() for field in fields),
              "overall": summarize(samples),
              "per_class": [{"class_index": label, "action": f"A{label + 1:03}", **summarize(rows)}
                            for label, rows in sorted(classes.items())],
              "per_subject": [{"subject": f"P{name}", **summarize(rows)} for name, rows in sorted(subjects.items())],
              "caveat": "Descriptive validation-label-conditioned analysis of two 20-epoch checkpoints; not a deployable rule, independent test, or full-training result."}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"overall": output["overall"], "confidence_available": output["confidence_available"]}, indent=2))


if __name__ == "__main__":
    main()
