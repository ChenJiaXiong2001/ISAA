"""Train the best no-pool model with confidence-weighted compression to 100 frames."""

import sys

from train_best_no_pool import training_arguments as base_training_arguments


def training_arguments(overrides):
    """Keep the original preset intact while selecting the new time protocol."""
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    arguments = base_training_arguments(overrides)
    if "--window-size" not in supplied:
        arguments[arguments.index("--window-size") + 1] = "100"
    if "--temporal-compression" not in supplied:
        arguments.extend(("--temporal-compression", "confidence_weighted"))
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
