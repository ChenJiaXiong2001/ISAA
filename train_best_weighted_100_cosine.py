"""Train the original pooled 100-frame baseline for 100 epochs with batch cosine."""
import argparse
import sys

from train_best_weighted_100 import training_arguments as pooled_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if supplied.intersection({"--lr-steps", "--lr-decay"}):
        raise ValueError("Batch cosine replaces the old milestone schedule")
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--lr-scheduler", choices=("warmup_cosine",))
    selector.add_argument("--grad-accum-steps", type=int, choices=(1,))
    selector.parse_known_args(overrides)
    defaults = {"--epochs": "100", "--lr-scheduler": "warmup_cosine",
                "--warmup-start-lr": "0.001", "--min-lr": "0.0001",
                "--grad-accum-steps": "1",
                "--save-dir": "outputs/best_weighted_100_cosine_ntu60_xsub"}
    arguments = []
    for flag, value in defaults.items():
        if flag not in supplied:
            arguments.extend((flag, value))
    arguments = pooled_arguments(arguments + list(overrides))
    # The original preset injects milestones; they do not apply to this run.
    position = arguments.index("--lr-steps")
    del arguments[position:position + 3]
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main
    main()
