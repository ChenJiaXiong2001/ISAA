"""Train the pooled 100-frame baseline with matched torso/hand width.

The first 55 epochs are clean.  After the train-accuracy trigger (or epoch
70 fallback), geometric and joint-dropout augmentation ramps to its maximum
by epoch 70.  This is a new scratch experiment; the original entry remains
unchanged.
"""
import argparse
import sys

from train_best_weighted_100_cosine import training_arguments as cosine_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    forbidden = {"--gcn-config", "--interaction-config", "--init-checkpoint",
                 "--resume-checkpoint", "--npy-dir", "--no-compile"}
    if supplied.intersection(forbidden):
        raise ValueError("same-channel-late-aug is scratch-only and requires torch.compile")
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--late-augmentation-start", type=int)
    selector.add_argument("--late-augmentation-force-epoch", type=int)
    selector.add_argument("--late-augmentation-trigger", type=float)
    selector.parse_known_args(overrides)
    defaults = {
        "--epochs": "100",
        "--late-augmentation-start": "55",
        "--late-augmentation-force-epoch": "70",
        "--late-augmentation-trigger": "0.985",
        "--weight-decay": "0.0007",
        "--max-grad-norm": "5",
        "--save-dir": "outputs/best_weighted_100_same_channel_late_aug_ntu60_xsub",
    }
    args = []
    for flag, value in defaults.items():
        if flag not in supplied:
            args.extend((flag, value))
    combined = args + list(overrides)
    result = cosine_arguments(combined)
    result[result.index("--model-variant") + 1] = "body-local-hand-ctr-wide-relative-same-channel-late-aug"
    return result


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main
    main()
