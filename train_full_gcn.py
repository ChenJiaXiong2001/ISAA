"""Train the full body/hand/face GCN version with the NTU60 XSub preset."""
import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def training_arguments(overrides):
    """Apply preset defaults while allowing ordinary training flag overrides."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--model-variant", choices=("body-local-hand-ctr-wide-relative-full",))
    selector.parse_known_args(overrides)
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    defaults = {
        "--model-variant": "body-local-hand-ctr-wide-relative-full",
        "--archive": "data/ntu60_skeletons_rtmw.zip",
        "--feature-mode": "raw",
        "--split": "xsub60",
        "--window-size": "64",
        "--max-persons": "2",
        "--batch-size": "32",
        "--test-batch-size": "32",
        "--epochs": "65",
        "--lr": "0.1",
        "--momentum": "0.9",
        "--weight-decay": "0.0004",
        "--warmup-epochs": "5",
        "--lr-steps": ("35", "55"),
        "--seed": "1",
        "--num-workers": "8",
        "--device": "auto",
        "--save-dir": "outputs/full_gcn_ntu60_xsub",
    }
    # Fine-tuning reconstructs the exact backbone configuration in the checkpoint.
    if "--init-checkpoint" not in supplied:
        defaults["--gcn-config"] = str(PROJECT_ROOT / "configs" / "full_gcn.json")
    arguments = []
    for flag, value in defaults.items():
        if flag not in supplied:
            arguments.append(flag)
            arguments.extend(value if isinstance(value, tuple) else (value,))
    if not supplied.intersection({"--compile", "--no-compile"}):
        arguments.append("--compile")
    return arguments + list(overrides)


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
