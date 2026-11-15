"""Compare two candidates only after checking their shared inputs and control."""
import csv
import json
import sys
from pathlib import Path

from summarize import summarize


def load(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "paired_samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {(row["condition"], row["sample"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate condition and sample")
    return summary, keyed


def main():
    p025_dir, p05_dir, manifest_path, output = map(Path, sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)
    p025, left = load(p025_dir)
    p05, right = load(p05_dir)
    for key in ("control_sha256", "control_hf_weights", "full_validation_members_sha256", "seed", "mask_scheme"):
        if p025[key] != p05[key]:
            raise ValueError(f"Different {key}")
    lc = {item["condition"]: item for item in p025["conditions"]}
    rc = {item["condition"]: item for item in p05["conditions"]}
    if lc.keys() != rc.keys() or left.keys() != right.keys():
        raise ValueError("Different conditions or samples")
    for condition in lc:
        for key in ("input_sha256", "frame_mask_sha256", "samples"):
            if lc[condition][key] != rc[condition][key]:
                raise ValueError(f"Different {condition} {key}")
    rows = []
    control_disagreements = {condition: {"prediction_count": 0, "correctness_count": 0, "examples": []}
                             for condition in lc}
    for key in left:
        a, b = left[key], right[key]
        if a["true_class_index"] != b["true_class_index"]:
            raise ValueError(f"Different label for {key}")
        if a["control_prediction"] != b["control_prediction"]:
            d = control_disagreements[key[0]]
            d["prediction_count"] += 1
            d["correctness_count"] += int(a["control_correct"] != b["control_correct"])
            if len(d["examples"]) < 5:
                d["examples"].append(key[1])
        rows.append({**a, "control_correct": b["candidate_correct"]})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest["validation"]["members"]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate selection members")
    result = {"candidate": "p025_clean50", "control": "p05_clean75",
              "p025_candidate_sha256": p025["candidate_sha256"],
              "p05_candidate_sha256": p05["candidate_sha256"],
              "shared_control_sha256": p025["control_sha256"],
              "shared_control_prediction_disagreements": control_disagreements,
              "bootstrap_repeats": 20000, "bootstrap_unit": "subject, sample-weighted",
              "strata": {name: summarize(rows, set(selected), name)
                         for name in ("all", "selection", "remaining")}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
