"""Cluster-bootstrap paired max5 changes by complete NTU subject."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def summarize(rows, repeats, seed):
    if not rows or repeats < 1:
        raise ValueError("Need samples and positive bootstrap repeats")
    names = [row["sample"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate sample")
    subjects = []
    for name in names:
        match = re.search(r"P(\d{3})R\d{3}A\d{3}", name)
        if not match:
            raise ValueError(f"Missing subject in {name}")
        subjects.append(match.group(1))
    subject_names, group = np.unique(subjects, return_inverse=True)
    size = np.bincount(group)
    nsubjects = len(subject_names)
    if nsubjects < 2:
        raise ValueError("Need at least two distinct subjects")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(nsubjects, np.full(nsubjects, 1 / nsubjects), size=repeats)
    sampled_size = draws @ size
    labels = np.array([int(row["label"]) for row in rows])
    result = {"samples": len(rows), "subjects": nsubjects, "repeats": repeats,
              "seed": seed, "bootstrap_unit": "complete subject; sample-weighted overall accuracy",
              "conditions": {}}
    for condition in ("clean", "pre_dropout"):
        before = np.array([int(row[f"{condition}_unmodified_prediction"]) for row in rows]) == labels
        after = np.array([int(row[f"{condition}_max5_prediction"]) for row in rows]) == labels
        delta = after.astype(np.int8) - before.astype(np.int8)
        subject_delta = np.bincount(group, weights=delta, minlength=nsubjects)
        interval = np.percentile(100 * (draws @ subject_delta) / sampled_size, [2.5, 97.5])
        result["conditions"][condition] = {
            "unmodified_correct": int(before.sum()), "max5_correct": int(after.sum()),
            "corrected": int((delta == 1).sum()), "new_errors": int((delta == -1).sum()),
            "difference_pp": float(100 * delta.mean()), "subject_bootstrap_95_percent_ci_pp": interval.tolist(),
            "equal_subject_mean_difference_pp": float(100 * np.mean(subject_delta / size)),
            "per_subject": [{"subject": f"P{name}", "samples": int(count),
                             "net_correct": int(change), "difference_pp": float(100 * change / count)}
                            for name, count, change in zip(subject_names, size, subject_delta)]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261105)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    summary = json.loads((options.evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    with (options.evaluation_dir / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    digest = hashlib.sha256("\n".join(row["sample"] for row in rows).encode()).hexdigest()
    if len(rows) != summary["samples"] or digest != summary["members_sha256"]:
        raise ValueError("Predictions do not match evaluation membership")
    result = summarize(rows, options.repeats, options.seed)
    result["members_sha256"] = digest
    result["input_sha256"] = summary["input_sha256"]
    result["caveat"] = "Fixed checkpoint and synthetic mask; cluster bootstrap does not remove validation reuse or account for detector/domain shifts."
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
