"""Compare anchor repair with paired resampling of complete subjects."""
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
    subjects = []
    seen = set()
    for row in rows:
        sample = row["sample"]
        if sample in seen:
            raise ValueError(f"Duplicate sample: {sample}")
        seen.add(sample)
        match = re.search(r"P(\d{3})R\d{3}A\d{3}", sample)
        if not match:
            raise ValueError(f"No NTU subject identifier in {sample}")
        subjects.append(match.group(1))

    _, group = np.unique(subjects, return_inverse=True)
    subject_count = int(group.max()) + 1
    if subject_count < 2:
        raise ValueError("Subject resampling requires at least two subjects")
    group_sizes = np.bincount(group)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(subject_count, np.full(subject_count, 1 / subject_count), size=replicates)
    sampled_sizes = draws @ group_sizes
    labels = np.asarray([int(row["label"]) for row in rows])
    result = {"samples": len(rows), "subjects": subject_count, "replicates": replicates,
              "bootstrap_seed": seed, "conditions": {}}
    for condition in ("clean", "point05", "point10"):
        before = np.asarray([int(row[f"{condition}_unmodified"]) for row in rows]) == labels
        after = np.asarray([int(row[f"{condition}_repaired"]) for row in rows]) == labels
        changes = after.astype(np.int64) - before.astype(np.int64)
        group_changes = np.bincount(group, weights=changes, minlength=subject_count)
        sampled_delta = 100 * (draws @ group_changes) / sampled_sizes
        interval = np.percentile(sampled_delta, [2.5, 97.5])
        result["conditions"][condition] = {
            "gain": int((changes == 1).sum()), "loss": int((changes == -1).sum()),
            "delta_percentage_points": float(100 * changes.mean()),
            "subject_bootstrap_95_percent_ci_pp": interval.tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    options = parser.parse_args()
    with options.predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rendered = json.dumps(summarize(rows, options.replicates, options.bootstrap_seed), indent=2) + "\n"
    if options.output:
        with options.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
