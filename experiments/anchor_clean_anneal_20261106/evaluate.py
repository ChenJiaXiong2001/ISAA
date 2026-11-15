"""Run existing paired evaluators with a checked annealing checkpoint."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "hand_motion_paired_p025_20ep_20261106"))
import evaluate as paired
from curriculum import SCHEDULE


def validate_checkpoints(control, candidate, manifests):
    left, right = control["args"], candidate["args"]
    if left.get("anchor_clean_curriculum") is not None or right.get("anchor_clean_curriculum") != SCHEDULE:
        raise ValueError("Expected fixed control and recorded annealing curriculum")
    left_settings = (left.get("precompression_anchor_dropout_probability"),
                     left.get("precompression_anchor_clean_sample_probability"))
    if left_settings not in ((0, 0), (0.025, 0.5)):
        raise ValueError("Control must be clean or fixed p025/clean50")
    # Reuse all existing checks after validating the only intentional protocol
    # differences; the unmodified checkpoints remain the ones evaluated.
    copied_left, copied_right = dict(control), dict(candidate)
    copied_left["args"], copied_right["args"] = dict(left), dict(right)
    copied_left["args"]["precompression_anchor_dropout_probability"] = 0
    copied_left["args"]["precompression_anchor_clean_sample_probability"] = 0
    del copied_right["args"]["anchor_clean_curriculum"]
    return ORIGINAL_VALIDATE(copied_left, copied_right, manifests)


ORIGINAL_VALIDATE = paired.validate_checkpoints
paired.validate_checkpoints = validate_checkpoints


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--condition-set", required=True, choices=("standard", "anchor-pre"))
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.condition_set == "standard":
        paired.main()
        return
    path = ROOT / "experiments" / "precompression_anchor_paired_p025_20ep_20261106" / "evaluate.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("anchor_pre_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
