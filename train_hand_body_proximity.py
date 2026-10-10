"""Train the additive hand/body proximity variant using ordinary main flags."""
from pathlib import Path
import sys

from train_full_gcn import training_arguments as full_training_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if "--model-variant" in supplied:
        raise ValueError("This preset selects hand-body-proximity; use main.py for other variants")
    arguments = full_training_arguments(overrides)
    position = arguments.index("--model-variant")
    arguments[position+1] = "hand-body-proximity"
    if "--save-dir" not in supplied:
        position = arguments.index("--save-dir")
        arguments[position+1] = "outputs/hand_body_proximity_ntu60_xsub"
    if "--init-checkpoint" not in supplied and "--interaction-config" not in supplied:
        arguments.extend(("--interaction-config", str(Path(__file__).resolve().parent / "configs" / "hand_body_proximity.json")))
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
