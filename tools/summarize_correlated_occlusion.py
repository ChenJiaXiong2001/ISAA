"""Paired subject-cluster summaries for the fixed correlated occlusion test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

METHODS = ("original_arithmetic", "original_geometric_mix", "max5_arithmetic", "max5_geometric_mix")


def summarize(rows, repeats=20000, seed=20261106):
    if not rows or repeats < 1:
        raise ValueError("Need samples and positive bootstrap repeats")
    names = [row["sample"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate samples")
    subjects = []
    for name in names:
        match = re.search(r"P(\d{3})R\d{3}A\d{3}", name)
        if not match:
            raise ValueError(f"Missing subject in {name}")
        subjects.append("P" + match.group(1))
    eligible = np.array([str(row["eligible"]).lower() in ("true", "1") for row in rows])
    labels = np.array([int(row["label"]) for row in rows])
    hits = {(condition, method): np.array([int(row[f"{condition}_{method}"]) for row in rows]) == labels
            for condition in ("clean", "occlusion") for method in METHODS}
    result = {"repeats": repeats, "seed": seed, "bootstrap_unit": "complete subject; sample-weighted accuracy",
              "eligible_samples": int(eligible.sum()), "samples": len(rows), "subsets": {}}
    subject_array = np.array(subjects)
    result["eligibility_per_subject"] = [
        {"subject": str(subject), "samples": int((subject_array == subject).sum()),
         "eligible": int(((subject_array == subject) & eligible).sum()),
         "eligible_percent": float(100 * eligible[subject_array == subject].mean())}
        for subject in np.unique(subject_array)]
    for subset, selected in (("all", np.ones(len(rows), dtype=bool)), ("eligible", eligible)):
        if not selected.any():
            result["subsets"][subset] = {"samples": 0}
            continue
        subject_names, groups = np.unique(np.array(subjects)[selected], return_inverse=True)
        nsubjects = len(subject_names)
        sizes = np.bincount(groups)
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(nsubjects, np.full(nsubjects, 1 / nsubjects), size=repeats)
        denominators = draws @ sizes

        def stat(delta):
            values = delta[selected].astype(np.int64)
            totals = np.bincount(groups, weights=values, minlength=nsubjects)
            interval = np.percentile(100 * (draws @ totals) / denominators, [2.5, 97.5]).tolist() if nsubjects >= 2 else None
            return {"difference_pp": float(100 * values.mean()), "subject_bootstrap_95_percent_ci_pp": interval,
                    "per_subject": [{"subject": str(s), "samples": int(n), "net_correct": int(d),
                                     "difference_pp": float(100 * d / n)}
                                    for s, n, d in zip(subject_names, sizes, totals)]}

        out = {"samples": int(selected.sum()), "subjects": nsubjects, "conditions": {},
               "occlusion_minus_clean": {}, "difference_in_differences": {}}
        for condition in ("clean", "occlusion"):
            base = hits[condition, "original_arithmetic"]
            out["conditions"][condition] = {}
            for method in METHODS:
                candidate = hits[condition, method]
                delta = candidate.astype(np.int8) - base.astype(np.int8)
                entry = stat(delta)
                entry.update(correct=int(candidate[selected].sum()), accuracy_percent=float(100 * candidate[selected].mean()),
                             corrected=int(((delta == 1) & selected).sum()), new_errors=int(((delta == -1) & selected).sum()))
                out["conditions"][condition][method] = entry
        base_shift = hits["occlusion", "original_arithmetic"].astype(np.int8) - hits["clean", "original_arithmetic"].astype(np.int8)
        for method in METHODS:
            shift = hits["occlusion", method].astype(np.int8) - hits["clean", method].astype(np.int8)
            out["occlusion_minus_clean"][method] = stat(shift)
            out["difference_in_differences"][method] = stat(shift - base_shift)
        result["subsets"][subset] = out
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    with (args.evaluation_dir / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    digest = hashlib.sha256("\n".join(row["sample"] for row in rows).encode()).hexdigest()
    if len(rows) != summary["samples"] or digest != summary["members_sha256"]:
        raise ValueError("Predictions do not match evaluation membership")
    result = summarize(rows, args.repeats, args.seed)
    if result["eligible_samples"] != summary["eligible_samples"]:
        raise ValueError("Eligible count mismatch")
    result.update(members_sha256=digest, input_sha256=summary["input_sha256"],
                  caveat="Fixed synthetic stress test on reused subjects; bootstrap does not establish detector or domain generalization.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
