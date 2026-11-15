"""Paired cached-candidate comparison after input, label and control checks."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "precompression_anchor_paired_p025_20ep_20261106"))
from summarize import summarize


def load(path):
    summary = json.loads((path / "summary.json").read_text())
    with (path / "paired_samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {(row["condition"], row["sample"]): row for row in rows}
    if len(rows) != len(keyed):
        raise ValueError("Duplicate sample")
    return summary, keyed


def main():
    fixed_dir, anneal_dir, manifest_path, output = map(Path, sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)
    fixed, a = load(fixed_dir)
    anneal, b = load(anneal_dir)
    for key in ("control_sha256", "control_hf_weights", "full_validation_members_sha256", "seed", "mask_scheme"):
        if fixed[key] != anneal[key]:
            raise ValueError(f"Mismatched {key}")
    x = {row["condition"]: row for row in fixed["conditions"]}
    y = {row["condition"]: row for row in anneal["conditions"]}
    if x.keys() != y.keys() or a.keys() != b.keys():
        raise ValueError("Mismatched conditions or samples")
    for condition in x:
        for key in ("input_sha256", "frame_mask_sha256", "samples"):
            if x[condition][key] != y[condition][key]:
                raise ValueError(f"Mismatched {condition} {key}")
    rows, disagreements = [], {condition: 0 for condition in x}
    for key in a:
        left, right = a[key], b[key]
        if left["true_class_index"] != right["true_class_index"]:
            raise ValueError(f"Mismatched label for {key}")
        disagreements[key[0]] += int(left["control_prediction"] != right["control_prediction"])
        rows.append({**right, "control_correct": left["candidate_correct"]})
    selected = set(json.loads(manifest_path.read_text())["validation"]["members"])
    result = {"candidate": "annealing", "control": "fixed_p025_clean50",
              "shared_control_prediction_disagreements": disagreements,
              "candidate_sha256": anneal["candidate_sha256"], "control_sha256": fixed["candidate_sha256"],
              "bootstrap_unit": "subject, sample-weighted", "bootstrap_repeats": 20000,
              "strata": {name: summarize(rows, selected, name) for name in ("all", "selection", "remaining")}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
