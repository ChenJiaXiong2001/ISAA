"""Train the pooled torso/hand model with learned hand-joint weights."""
import sys

from train_best_weighted_100_cosine import training_arguments as cosine_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    forbidden = {"--model-variant", "--gcn-config", "--interaction-config",
                 "--init-checkpoint", "--resume-checkpoint", "--npy-dir"}
    if supplied.intersection(forbidden):
        raise ValueError("body-hand-weighted-pool trains from random initialization")
    if "--no-compile" in supplied:
        raise ValueError("body-hand-weighted-pool requires torch.compile; --no-compile is not allowed")
    arguments = cosine_arguments(overrides)
    arguments[arguments.index("--model-variant") + 1] = "body-hand-weighted-pool"
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = "outputs/body_hand_weighted_pool_100_ntu60_xsub"
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main
    main()
