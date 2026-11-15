"""Exploratory subject and action breakdown of paired predictions."""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def metrics(rows):
    n = len(rows)
    control = sum(int(r["control_correct"]) for r in rows)
    candidate = sum(int(r["candidate_correct"]) for r in rows)
    gained = sum(r["control_correct"] == "0" and r["candidate_correct"] == "1" for r in rows)
    lost = sum(r["control_correct"] == "1" and r["candidate_correct"] == "0" for r in rows)
    return {"samples": n, "control_correct": control, "candidate_correct": candidate,
            "gained": gained, "lost": lost, "difference_pp": 100 * (candidate - control) / n}


def main():
    source, output = map(Path, sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    conditions = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["condition"], row["sample"]
        if key in seen:
            raise ValueError(f"Duplicate {key}")
        seen.add(key)
        match = re.search(r"P(\d{3})R\d{3}A(\d{3})", row["sample"])
        if not match or int(match.group(2)) - 1 != int(row["true_class_index"]):
            raise ValueError(f"Invalid subject or action for {key}")
        row["subject"], row["action"] = match.groups()
        conditions[row["condition"]].append(row)
    names = {tuple(sorted(r["sample"] for r in values)) for values in conditions.values()}
    if len(names) != 1 or set(conditions) != {"clean", "anchor_pre_0.05", "anchor_pre_0.1"}:
        raise ValueError("Condition membership differs")
    result = {"source": str(source), "exploratory": True, "conditions": {}}
    for condition, members in conditions.items():
        subjects, actions = defaultdict(list), defaultdict(list)
        for row in members:
            subjects[row["subject"]].append(row)
            actions[row["action"]].append(row)
        subject_stats = {key: metrics(value) for key, value in sorted(subjects.items())}
        action_stats = {key: metrics(value) for key, value in sorted(actions.items())}
        result["conditions"][condition] = {
            "all": metrics(members), "exclude_p007": metrics([r for r in members if r["subject"] != "007"]),
            "subject_equal_weight_difference_pp": sum(v["difference_pp"] for v in subject_stats.values()) / len(subject_stats),
            "by_subject": subject_stats, "by_action": action_stats}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
