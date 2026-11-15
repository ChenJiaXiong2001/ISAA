"""Paired sample-level comparisons for the anchor-gap calibration experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", default=20000, type=int)
    parser.add_argument("--seed", default=20261105, type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    with args.predictions.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["sample"] for row in rows}) != len(rows):
        raise ValueError("Expected unique, nonempty sample rows")
    labels = np.array([int(row["label"]) for row in rows])
    rng = np.random.default_rng(args.seed)
    result = {"samples": len(rows), "replicates": args.replicates, "seed": args.seed,
              "interval_scope": "sample bootstrap; does not account for subject clustering or selection on this calibration set",
              "conditions": {}}
    for condition in ("clean", "independent05", "gap2", "gap5", "gap10"):
        comparisons = {}
        for reference, candidate in (("unmodified", "repaired"), ("repaired", "max2"),
                                     ("repaired", "max5"), ("unmodified", "max2"),
                                     ("unmodified", "max5")):
            before = np.array([int(row[f"{condition}_{reference}_soft"]) for row in rows]) == labels
            after = np.array([int(row[f"{condition}_{candidate}_soft"]) for row in rows]) == labels
            gain = int((~before & after).sum())
            loss = int((before & ~after).sum())
            n = len(rows)
            draws = rng.multinomial(n, [gain / n, loss / n, (n - gain - loss) / n], size=args.replicates)
            interval = np.percentile(100 * (draws[:, 0] - draws[:, 1]) / n, [2.5, 97.5])
            comparisons[f"{reference}_to_{candidate}"] = {
                "gain": gain, "loss": loss, "delta_pp": 100 * (gain - loss) / n,
                "bootstrap_95_ci_pp": interval.tolist()}
        result["conditions"][condition] = comparisons
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
