"""Compare a new candidate to the prior p025 run after checking shared inputs."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from shared_weights import predictions

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    summary = json.loads((path / "summary.json").read_text())
    with (path / "paired_samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {(row["condition"], row["sample"]): row for row in rows}
    if len(rows) != len(keyed):
        raise ValueError("Duplicate condition/member")
    return summary, rows, keyed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition-set", choices=("standard", "anchor-pre"), required=True)
    opts = parser.parse_args()
    output = opts.new_dir / "vs_p025_clean50_reused.json"
    if output.exists():
        raise FileExistsError(output)
    old, _, old_rows = load(opts.old_dir)
    new, rows, new_rows = load(opts.new_dir)
    for key in ("control_sha256", "control_hf_weights", "full_validation_members_sha256"):
        if old[key] != new[key]:
            raise ValueError(f"Different {key}")
    if hashlib.sha256(opts.old_checkpoint.read_bytes()).hexdigest() != old["candidate_sha256"]:
        raise ValueError("Old candidate checkpoint hash mismatch")
    if old_rows.keys() != new_rows.keys():
        raise ValueError("Different members or conditions")
    left = {item["condition"]: item for item in old["conditions"]}
    right = {item["condition"]: item for item in new["conditions"]}
    if left.keys() != right.keys():
        raise ValueError("Different condition summaries")
    for condition in left:
        for key in ("samples", "input_sha256", "frame_mask_sha256" if opts.condition_set == "anchor-pre" else "validity_and_frame_mask_sha256"):
            if left[condition][key] != right[condition][key]:
                raise ValueError(f"Different {condition} {key}")
        if opts.condition_set == "standard" and left[condition]["seed"] != right[condition]["seed"]:
            raise ValueError(f"Different {condition} seed")
    if opts.condition_set == "anchor-pre" and (old["seed"] != new["seed"] or old["mask_scheme"] != new["mask_scheme"]):
        raise ValueError("Different original-frame masking protocol")
    with np.load(opts.new_dir / "paired_branch_logits.npz", allow_pickle=False) as logits:
        candidate = logits["candidate"]
    if candidate.shape[:2] != (len(rows), 2):
        raise ValueError("Mismatched logit count")
    own_pred = predictions(candidate, new["candidate_hf_weights"])
    shared_pred = predictions(candidate, old["candidate_hf_weights"])
    if not np.array_equal(own_pred, [int(row["candidate_prediction"]) for row in rows]):
        raise ValueError("New candidate predictions cannot be reconstructed")
    own_rows, shared_rows = [], []
    disagreements = dict.fromkeys(left, 0)
    for i, row in enumerate(rows):
        prev = old_rows[(row["condition"], row["sample"])]
        if row["true_class_index"] != prev["true_class_index"]:
            raise ValueError("Mismatched label")
        disagreements[row["condition"]] += int(row["control_prediction"] != prev["control_prediction"])
        comparison = {**row, "control_correct": prev["candidate_correct"]}
        own_rows.append(comparison)
        shared_rows.append({**comparison, "candidate_correct": str(int(shared_pred[i] == int(row["true_class_index"])) )})
    dirname = "hand_motion_paired_p025_20ep_20261106" if opts.condition_set == "standard" else "precompression_anchor_paired_p025_20ep_20261106"
    spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "experiments" / dirname / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = set(json.loads(opts.manifest.read_text())["validation"]["members"])
    old_weights, new_weights = np.array(old["candidate_hf_weights"]), np.array(new["candidate_hf_weights"])
    result = {"old_candidate_sha256": old["candidate_sha256"], "new_candidate_sha256": new["candidate_sha256"],
              "shared_clean_control_prediction_disagreements": disagreements,
              "per_class_hf_weight_difference": (new_weights-old_weights).tolist(),
              "changed_class_count": int(np.count_nonzero(new_weights != old_weights)),
              "mean_absolute_weight_difference": float(np.mean(np.abs(new_weights-old_weights))),
              "bootstrap_unit": "subject, sample-weighted", "bootstrap_repeats": 20000,
              "separate_weights": module.summarize(own_rows, selected, "all"),
              "shared_old_weights": module.summarize(shared_rows, selected, "all")}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
