"""Subject-cluster bootstrap for validation max-gap paired predictions."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


METHODS = ("repaired", "max2", "max5")
CONDITIONS = ("clean", "independent05", "gap2", "gap5", "gap10")


def subject_groups(rows):
    names = [row["sample"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate samples")
    subjects = []
    for name in names:
        match = re.search(r"P(\d{3})R\d{3}A\d{3}", name)
        if not match:
            raise ValueError(f"No NTU subject identifier in {name}")
        subjects.append(match.group(1))
    _, group = np.unique(subjects, return_inverse=True)
    return group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", default=20000, type=int)
    parser.add_argument("--seed", default=20261105, type=int)
    args = parser.parse_args()
    if args.output.exists() or args.replicates < 1:
        raise ValueError("Output must be new and replicates positive")
    with args.predictions.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty predictions")
    group = subject_groups(rows)
    subjects = int(group.max()) + 1
    if subjects < 2:
        raise ValueError("Need at least two subjects")
    sizes = np.bincount(group)
    rng = np.random.default_rng(args.seed)
    draws = rng.multinomial(subjects, np.full(subjects, 1 / subjects), size=args.replicates)
    sampled_sizes = draws @ sizes
    labels = np.asarray([int(row["label"]) for row in rows])
    result = {"samples": len(rows), "subjects": subjects, "replicates": args.replicates,
              "bootstrap_seed": args.seed,
              "interval_scope": "subject bootstrap over validation members; synthetic corruption remains fixed per sample",
              "conditions": {}}
    for condition in CONDITIONS:
        result["conditions"][condition] = {}
        for method in METHODS:
            before = np.asarray([int(row[f"{condition}_unmodified_soft"]) for row in rows]) == labels
            after = np.asarray([int(row[f"{condition}_{method}_soft"]) for row in rows]) == labels
            changes = after.astype(np.int64) - before.astype(np.int64)
            subject_changes = np.bincount(group, weights=changes, minlength=subjects)
            interval = np.percentile(100 * (draws @ subject_changes) / sampled_sizes, [2.5, 97.5])
            result["conditions"][condition][method] = {
                "gain": int((changes == 1).sum()), "loss": int((changes == -1).sum()),
                "delta_percentage_points": float(100 * changes.mean()),
                "subject_bootstrap_95_percent_ci_pp": interval.tolist()}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
