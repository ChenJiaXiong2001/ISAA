"""Train original best + action-selected hand/body interactions from scratch."""
from pathlib import Path
import sys

from train_best_hand_body_proximity import training_arguments as proximity_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if "--init-checkpoint" in supplied:
        raise ValueError("This experiment trains from scratch; --init-checkpoint is not allowed")
    flags = list(overrides)
    if not supplied.intersection({"--compile", "--no-compile"}):
        flags.append("--no-compile")
    arguments = proximity_arguments(flags)
    arguments[arguments.index("--model-variant") + 1] = "best-action-hand-body"
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = "outputs/best_action_hand_body_ntu60_xsub"
    if "--interaction-config" not in supplied:
        arguments[arguments.index("--interaction-config") + 1] = str(
            Path(__file__).resolve().parent / "configs" / "best_action_hand_body.json")
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main
    main()
