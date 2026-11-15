"""Train Hand-Face and Body-Face classifiers with holdout-based class selection."""
import sys

from train_best_weighted_100_cosine import training_arguments as baseline_arguments


def training_arguments(overrides):
    """Keep baseline inputs and schedule; add the two pair classifiers from scratch."""
    supplied = {value.split("=", 1)[0] for value in overrides if value.startswith("--")}
    if "--no-compile" in supplied:
        raise ValueError("best-face-pair requires torch.compile; --no-compile is not allowed")
    if supplied.intersection({"--model-variant", "--init-checkpoint", "--resume-checkpoint", "--npy-dir"}):
        raise ValueError("best-face-pair trains from random initialization using full RTMW ZIP sequences")
    defaults = {"--lr": "0.025", "--max-grad-norm": "5.0"}
    arguments = [item for flag, value in defaults.items() if flag not in supplied
                 for item in (flag, value)]
    arguments = baseline_arguments(arguments + list(overrides))
    arguments[arguments.index("--model-variant") + 1] = "best-face-pair"
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = "outputs/best_face_pair_100_ntu60_xsub"
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
