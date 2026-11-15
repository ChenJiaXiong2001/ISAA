"""Subject-cluster paired bootstrap with and without epoch-selection samples."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


def summarize(rows, selected, stratum):
    names = ("clean", "anchor_pre_0.05", "anchor_pre_0.1")
    by_condition = {name: {} for name in names}
    for row in rows:
        table = by_condition[row["condition"]]
        if row["sample"] in table:
            raise ValueError("Duplicate sample in condition")
        table[row["sample"]] = row
    all_members = set(by_condition["clean"])
    if not selected <= all_members or any(set(table) != all_members for table in by_condition.values()):
        raise ValueError("Condition or epoch-selection members differ")
    members = sorted(all_members if stratum == "all" else selected if stratum == "selection" else all_members - selected)
    subjects = []
    for member in members:
        match = re.search(r"P(\d{3})R\d{3}A\d{3}", member)
        if not match:
            raise ValueError(f"Cannot identify subject: {member}")
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

    results, diffs = {}, {}
    for name in names:
        table = by_condition[name]
        if any(table[m]["true_class_index"] != by_condition["clean"][m]["true_class_index"] for m in members):
            raise ValueError("Labels differ across conditions")
        control = np.array([int(table[m]["control_correct"]) for m in members], dtype=np.int8)
        candidate = np.array([int(table[m]["candidate_correct"]) for m in members], dtype=np.int8)
        diffs[name] = candidate - control
        results[name] = {"control_top1_percent": float(control.mean() * 100),
                         "candidate_top1_percent": float(candidate.mean() * 100),
                         "candidate_minus_control": interval(diffs[name])}
        if name != "clean":
            results[name]["robustness_difference_in_differences"] = interval(diffs[name] - diffs["clean"])
    return {"samples": len(members), "subjects": len(subject_names),
            "subject_counts": dict(zip(subject_names.tolist(), counts.tolist())), "conditions": results}


def main():
    source, manifest, output = map(Path, sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = json.loads(manifest.read_text(encoding="utf-8"))["validation"]["members"]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate selection members")
    result = {"source": str(source), "selection_manifest": str(manifest), "bootstrap_repeats": 20000,
              "bootstrap_unit": "subject, sample-weighted", "strata": {}}
    for name in ("all", "selection", "remaining"):
        result["strata"][name] = summarize(rows, set(selected), name)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
