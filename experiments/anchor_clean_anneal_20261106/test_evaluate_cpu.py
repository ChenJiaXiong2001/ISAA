"""Validate compatibility using the already trained control checkpoints."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaa.train  # Configure CUDA library paths before importing torch.
from curriculum import SCHEDULE

spec = importlib.util.spec_from_file_location("anneal_evaluator", Path(__file__).with_name("evaluate.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    args = parser.parse_args()
    clean, _, _ = module.paired.load_checkpoint(args.clean, "cpu", compile_for_inference=False)
    fixed, _, _ = module.paired.load_checkpoint(args.fixed, "cpu", compile_for_inference=False)
    manifests = [json.loads((path.parent / "split_manifest.json").read_text())
                 for path in (args.clean, args.fixed)]
    candidate = dict(fixed)
    candidate["args"] = dict(fixed["args"], anchor_clean_curriculum=dict(SCHEDULE))
    assert len(module.validate_checkpoints(clean, candidate, manifests)) == 4096
    assert len(module.validate_checkpoints(fixed, candidate, manifests[::-1])) == 4096
    candidate["args"]["lr"] = 0.1
    try:
        module.validate_checkpoints(clean, candidate, manifests)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched learning rate was accepted")
    print("clean/fixed controls accepted; mismatched learning rate rejected")


if __name__ == "__main__":
    main()
