"""Train pooled 100-frame baseline with a two-hand relation residual."""
import sys

from train_best_weighted_100_cosine import training_arguments as baseline_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if "--model-variant" in supplied:
        raise ValueError("The hand-relation entry fixes its model variant")
    arguments = list(overrides)
    if "--save-dir" not in supplied:
        arguments.extend(("--save-dir", "outputs/best_hand_relation_100_cosine_ntu60_xsub"))
    arguments = baseline_arguments(arguments)
    position = arguments.index("--model-variant")
    arguments[position + 1] = "best-hand-relation-fusion"
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
