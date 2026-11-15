import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


def summarize(rows, selected, name):
    conditions = ("clean", "group_0.05", "group_0.1")
    by_condition = {condition: {} for condition in conditions}
    for row in rows:
        bucket = by_condition[row["condition"]]
        sample = row["sample"]
        if sample in bucket:
            raise ValueError(f"duplicate sample {sample}")
        bucket[sample] = row
    all_members = set(by_condition["clean"])
    if any(set(bucket) != all_members for bucket in by_condition.values()):
        raise ValueError("conditions do not share identical members")
    if not selected <= all_members:
        raise ValueError("checkpoint selection manifest is not a subset")
    members = sorted(all_members if name == "all" else selected if name == "selection" else all_members - selected)
    subjects = []
    for member in members:
        match = re.search(r"P(\d{3})R\d{3}A\d{3}", member)
        if not match:
            raise ValueError(f"cannot identify subject: {member}")
        subjects.append(match.group(1))
    subject_names, groups = np.unique(subjects, return_inverse=True)
    counts = np.bincount(groups)
    rng = np.random.default_rng(20261109)
    draws = rng.multinomial(len(subject_names), np.full(len(subject_names), 1 / len(subject_names)), size=20000)
    denominator = draws @ counts

    def interval(values):
        values = np.asarray(values, dtype=np.int8)
        totals = np.bincount(groups, weights=values, minlength=len(subject_names))
        return {"difference_pp": float(values.mean() * 100),
                "subject_cluster_bootstrap_95_ci_pp": np.percentile(100 * (draws @ totals) / denominator, [2.5, 97.5]).tolist()}

    results, differences = {}, {}
    for condition in conditions:
        entries = by_condition[condition]
        control = np.array([int(entries[m]["control_correct"]) for m in members], dtype=np.int8)
        candidate = np.array([int(entries[m]["candidate_correct"]) for m in members], dtype=np.int8)
        diff = candidate - control
        differences[condition] = diff
        results[condition] = {"control_top1_percent": float(control.mean() * 100),
                              "candidate_top1_percent": float(candidate.mean() * 100),
                              "candidate_minus_control": interval(diff)}
    for condition in conditions[1:]:
        results[condition]["robustness_difference_in_differences"] = interval(differences[condition] - differences["clean"])
    return {"samples": len(members), "subjects": len(subject_names),
            "subject_counts": dict(zip(subject_names.tolist(), counts.tolist())), "conditions": results}


source, manifest, output = map(Path, sys.argv[1:])
with source.open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
selected_members = json.loads(manifest.read_text(encoding="utf-8"))["validation"]["members"]
if len(selected_members) != len(set(selected_members)):
    raise ValueError("duplicate selection-manifest entries")
result = {"source": str(source), "selection_manifest": str(manifest), "bootstrap_repeats": 20000,
          "bootstrap_unit": "subject, sample-weighted", "strata": {}}
for name in ("all", "selection", "remaining"):
    result["strata"][name] = summarize(rows, set(selected_members), name)
output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
