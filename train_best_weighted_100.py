"""Train the original best pooled model with weighted full-sequence compression."""
import argparse
import sys

from train_full_gcn import training_arguments as full_training_arguments


def training_arguments(overrides):
    """Keep the best architecture and schedule; change only the time protocol."""
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if supplied.intersection({"--model-variant", "--gcn-config", "--interaction-config",
                              "--init-checkpoint", "--resume-checkpoint", "--ablation-stage",
                              "--npy-dir"}):
        raise ValueError("best-weighted-100 trains the original best model from scratch using full RTMW ZIP sequences")
    if "--no-compile" in supplied:
        raise ValueError("best-weighted-100 requires torch.compile; --no-compile is not allowed")
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--window-size", type=int, choices=(100,))
    selector.add_argument("--temporal-compression", choices=("confidence_weighted",))
    selector.add_argument("--feature-mode", choices=("raw",))
    selector.parse_known_args(overrides)
    arguments = full_training_arguments(overrides)
    arguments[arguments.index("--model-variant") + 1] = "body-local-hand-ctr-wide-relative"
    position = arguments.index("--gcn-config")
    del arguments[position:position + 2]
    if "--window-size" not in supplied:
        arguments[arguments.index("--window-size") + 1] = "100"
    if "--temporal-compression" not in supplied:
        arguments.extend(("--temporal-compression", "confidence_weighted"))
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = "outputs/best_weighted_100_ntu60_xsub"
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
