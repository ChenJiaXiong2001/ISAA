"""Train the additive hand-motion-only pair model."""
import sys

from train_best_face_pair_100 import training_arguments as pair_arguments


def training_arguments(overrides):
    arguments = pair_arguments(overrides)
    arguments[arguments.index("--model-variant") + 1] = "best-face-pair-hand-motion"
    supplied = {value.split("=", 1)[0] for value in overrides if value.startswith("--")}
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = "outputs/best_face_pair_hand_motion_100_ntu60_xsub"
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
