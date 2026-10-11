"""Train original best + action-selected hand/body interactions from scratch."""
from pathlib import Path
import sys

from train_best_hand_body_proximity import training_arguments as proximity_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if "--init-checkpoint" in supplied:
        raise ValueError("This experiment trains from scratch; --init-checkpoint is not allowed")
    if "--no-compile" in supplied:
        raise ValueError("best-action-hand-body requires torch.compile; --no-compile is not allowed")
    arguments = proximity_arguments(overrides)
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
