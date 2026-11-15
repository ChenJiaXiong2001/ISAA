"""Discover a global raw-joint subset, or train its compact model from scratch."""
import sys


def training_arguments(overrides):
    flags = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if "--no-compile" in flags:
        raise ValueError("Input joint selection requires torch.compile; --no-compile is not allowed")
    if flags.intersection({"--model-variant", "--init-checkpoint", "--resume-checkpoint", "--npy-dir"}):
        raise ValueError("This entry trains independent selection/compact models from random initialization")
    return ["--model-variant", "best-input-joint-selection", *overrides]


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main
    main()
