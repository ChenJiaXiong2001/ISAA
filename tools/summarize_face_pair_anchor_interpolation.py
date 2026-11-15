"""Summarize paired anchor-repair predictions with bootstrap intervals."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    options = parser.parse_args()
    if options.replicates < 1:
        raise ValueError("replicates must be positive")
    with options.predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["sample"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique sample rows")
    rng = np.random.default_rng(options.bootstrap_seed)
    result = {"samples": len(rows), "replicates": options.replicates,
              "bootstrap_seed": options.bootstrap_seed, "conditions": {}}
    for condition in ("clean", "point05", "point10"):
        labels = np.asarray([int(row["label"]) for row in rows])
        before = np.asarray([int(row[f"{condition}_unmodified"]) for row in rows]) == labels
        after = np.asarray([int(row[f"{condition}_repaired"]) for row in rows]) == labels
        gain = int((~before & after).sum())
        loss = int((before & ~after).sum())
        unchanged = len(rows) - gain - loss
        draws = rng.multinomial(len(rows), [gain / len(rows), loss / len(rows),
                                           unchanged / len(rows)], size=options.replicates)
        interval = np.percentile(100.0 * (draws[:, 0] - draws[:, 1]) / len(rows), [2.5, 97.5])
        result["conditions"][condition] = {
            "unmodified_correct": int(before.sum()), "repaired_correct": int(after.sum()),
            "gain": gain, "loss": loss, "delta_percentage_points": 100.0 * (gain - loss) / len(rows),
            "bootstrap_95_percent_ci_pp": interval.tolist(),
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if options.output is not None:
        if options.output.exists():
            raise FileExistsError(f"Output already exists: {options.output}")
        options.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
