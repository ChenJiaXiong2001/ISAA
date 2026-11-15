"""Evaluate saved branch logits with both models using control weights."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def predictions(logits, weights):
    raw = torch.from_numpy(logits)
    hf, bf = F.log_softmax(raw[:, 0].float(), dim=-1), F.log_softmax(raw[:, 1].float(), dim=-1)
    w = torch.as_tensor(weights, dtype=torch.float32).view(1, -1)
    return torch.logaddexp(torch.log(w) + hf, torch.log1p(-w) + bf).argmax(dim=-1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition-set", choices=("standard", "anchor-pre"), required=True)
    opts = parser.parse_args()
    output = opts.evaluation_dir / "shared_weights_sensitivity.json"
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((opts.evaluation_dir / "summary.json").read_text())
    with (opts.evaluation_dir / "paired_samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(opts.evaluation_dir / "paired_branch_logits.npz", allow_pickle=False) as archive:
        control, candidate = archive["control"], archive["candidate"]
    if control.shape != candidate.shape or control.shape[:2] != (len(rows), 2):
        raise ValueError("Logits and rows have different shapes")
    truth = np.array([int(row["true_class_index"]) for row in rows])
    own_control = predictions(control, summary["control_hf_weights"])
    own_candidate = predictions(candidate, summary["candidate_hf_weights"])
    if not np.array_equal(own_control, [int(row["control_prediction"]) for row in rows]):
        raise ValueError("Reconstructed control predictions differ")
    if not np.array_equal(own_candidate, [int(row["candidate_prediction"]) for row in rows]):
        raise ValueError("Reconstructed candidate predictions differ")
    shared_candidate = predictions(candidate, summary["control_hf_weights"])
    for index, row in enumerate(rows):
        row["candidate_correct"] = str(int(shared_candidate[index] == truth[index]))
    dirname = "hand_motion_paired_p025_20ep_20261106" if opts.condition_set == "standard" else "precompression_anchor_paired_p025_20ep_20261106"
    spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "experiments" / dirname / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = set(json.loads(opts.manifest.read_text())["validation"]["members"])
    control_weights = np.asarray(summary["control_hf_weights"])
    candidate_weights = np.asarray(summary["candidate_hf_weights"])
    result = {"control_sha256": summary["control_sha256"], "candidate_sha256": summary["candidate_sha256"],
              "shared_weights": "control checkpoint training-calibration weights",
              "changed_class_count": int(np.count_nonzero(control_weights != candidate_weights)),
              "mean_absolute_weight_difference": float(np.mean(np.abs(control_weights - candidate_weights))),
              "per_class_hf_weight_difference": (candidate_weights - control_weights).tolist(),
              "bootstrap_unit": "subject, sample-weighted", "bootstrap_repeats": 20000,
              "strata": {name: module.summarize(rows, selected, name) for name in ("all", "selection", "remaining")}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["strata"]["all"], indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
