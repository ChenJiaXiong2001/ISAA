"""Evaluate the frozen p025/clean75 screen against checked controls."""
import argparse
import importlib.util
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "hand_motion_paired_p025_20ep_20261106"))
import evaluate as paired

ORIGINAL_VALIDATE = paired.validate_checkpoints


def validate_checkpoints(control, candidate, manifests):
    left, right = control["args"], candidate["args"]
    left_settings = (left.get("precompression_anchor_dropout_probability"),
                     left.get("precompression_anchor_clean_sample_probability"))
    right_settings = (right.get("precompression_anchor_dropout_probability"),
                      right.get("precompression_anchor_clean_sample_probability"))
    if left_settings not in ((0, 0), (0.025, 0.5)) or right_settings != (0.025, 0.75):
        raise ValueError("Expected clean or p025/clean50 control and p025/clean75 candidate")
    copied_left, copied_right = dict(control), dict(candidate)
    copied_left["args"], copied_right["args"] = dict(left), dict(right)
    copied_left["args"]["precompression_anchor_dropout_probability"] = 0
    copied_left["args"]["precompression_anchor_clean_sample_probability"] = 0
    copied_right["args"]["precompression_anchor_clean_sample_probability"] = 0.5
    return ORIGINAL_VALIDATE(copied_left, copied_right, manifests)


paired.validate_checkpoints = validate_checkpoints


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--condition-set", required=True, choices=("standard", "anchor-pre"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--output-dir", str(args.output_dir), *rest]
    records = [[], []]
    original_logits = paired._soft_logits

    def capture(output, weights):
        index = (len(records[0]) + len(records[1])) % 2
        records[index].append(np.stack((output["hand_face_logits"].detach().float().cpu().numpy(),
                                        output["body_face_logits"].detach().float().cpu().numpy()), axis=1))
        return original_logits(output, weights)

    if args.condition_set == "standard":
        paired._soft_logits = capture
        paired.main()
    else:
        path = ROOT / "experiments" / "precompression_anchor_paired_p025_20ep_20261106" / "evaluate.py"
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("anchor_pre_evaluator", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._soft_logits = capture
        module.main()
    arrays = [np.concatenate(items, axis=0) for items in records]
    with (args.output_dir / "paired_samples.csv").open(encoding="utf-8", newline="") as stream:
        row_count = sum(1 for _ in csv.DictReader(stream))
    if arrays[0].shape != arrays[1].shape or arrays[0].shape[0] != row_count:
        raise RuntimeError("Captured logits do not match paired row count")
    np.savez_compressed(args.output_dir / "paired_branch_logits.npz", control=arrays[0], candidate=arrays[1])


if __name__ == "__main__":
    main()
