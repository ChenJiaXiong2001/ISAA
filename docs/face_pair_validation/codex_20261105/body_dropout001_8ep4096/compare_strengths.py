"""Recompute paired comparisons against 0.005 and 0.02 from saved outputs."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tools.compare_face_pair_corruption import paired_interval

DIRECTORY = Path(__file__).resolve().parent
CONDITIONS = ("clean", "point_0.05", "point_0.1")


def read_experiment(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "paired_samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return summary, rows


def compare(reference_name, candidate, reference):
    a, a_rows = candidate
    b, b_rows = reference
    assert a["validation_sha256"] == b["validation_sha256"]
    for a_condition, b_condition in zip(a["conditions"], b["conditions"]):
        for key in ("condition", "input_sha256", "validity_and_frame_mask_sha256"):
            assert a_condition[key] == b_condition[key], key
    comparisons = []
    for condition in CONDITIONS:
        a_condition = [row for row in a_rows if row["condition"] == condition]
        b_condition = [row for row in b_rows if row["condition"] == condition]
        assert [(x["sample"], x["true_class_index"]) for x in a_condition] == [
            (x["sample"], x["true_class_index"]) for x in b_condition]
        diff = np.array([int(x["candidate_correct"]) - int(y["candidate_correct"])
                         for x, y in zip(a_condition, b_condition)], dtype=np.int8)
        comparisons.append({"condition": condition, "samples": len(diff),
            "difference_pp": float(diff.mean() * 100), "gains": int((diff > 0).sum()),
            "losses": int((diff < 0).sum()),
            "bootstrap_95_percent_ci_pp": paired_interval(diff, seed=20261105)})
    return {"candidate": "dropout_0.01", "reference": reference_name,
            "bootstrap_replicates": 20000, "conditions": comparisons}


if __name__ == "__main__":
    candidate = read_experiment(DIRECTORY)
    result = [compare(name, candidate, read_experiment(DIRECTORY.parent / folder))
              for name, folder in (("dropout_0.005", "body_dropout0005_8ep4096"),
                                   ("dropout_0.02", "body_dropout002_8ep4096"))]
    (DIRECTORY / "strength_comparisons.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
