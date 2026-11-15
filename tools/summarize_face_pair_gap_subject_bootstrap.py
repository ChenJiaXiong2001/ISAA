"""Paired subject bootstrap for the fixed max-5 anchor repair."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def summarize(rows, replicates=20000, seed=20261105):
    if not rows or replicates < 1:
        raise ValueError("Expected samples and a positive replicate count")
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
    subject_count = int(group.max()) + 1
    if subject_count < 2:
        raise ValueError("Subject resampling requires at least two subjects")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(subject_count, np.full(subject_count, 1 / subject_count), size=replicates)
    sampled_sizes = draws @ np.bincount(group)
    labels = np.asarray([int(row["label"]) for row in rows])
    result = {"samples": len(rows), "subjects": subject_count, "replicates": replicates,
              "bootstrap_seed": seed, "conditions": {}}
    for condition in ("clean", "independent05", "gap2", "gap5", "gap10"):
        before = np.asarray([int(row[f"{condition}_unmodified_soft"]) for row in rows]) == labels
        after = np.asarray([int(row[f"{condition}_max5_soft"]) for row in rows]) == labels
        changes = after.astype(np.int64) - before.astype(np.int64)
        subject_changes = np.bincount(group, weights=changes, minlength=subject_count)
        interval = np.percentile(100 * (draws @ subject_changes) / sampled_sizes, [2.5, 97.5])
        result["conditions"][condition] = {
            "gain": int((changes == 1).sum()), "loss": int((changes == -1).sum()),
            "delta_percentage_points": float(100 * changes.mean()),
            "subject_bootstrap_95_percent_ci_pp": interval.tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summarize(rows, args.replicates, args.seed), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
