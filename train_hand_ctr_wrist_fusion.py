"""Train the face-light hand-CTR wrist-injection model."""

import sys

def training_arguments(overrides):
    """Apply the fixed 100-frame/100-epoch per-update cosine protocol."""
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    forbidden = {
        "--model-variant", "--window-size", "--temporal-compression", "--epochs",
        "--lr", "--lr-scheduler", "--warmup-epochs", "--warmup-start-lr", "--min-lr",
        "--lr-steps", "--lr-decay", "--compile-mode", "--max-grad-norm",
        "--grad-accum-steps", "--init-checkpoint", "--resume-checkpoint", "--npy-dir",
        "--no-compile",
    }
    if supplied & forbidden:
        raise ValueError(
            "train_hand_ctr_wrist_fusion fixes model, 100-frame compression, 100 epochs, "
            "warmup cosine scheduling, random initialization, and torch.compile; "
            f"conflicting flags: {sorted(supplied & forbidden)}"
        )
    arguments = [
        "--model-variant", "hand-ctr-wrist-fusion",
        "--window-size", "100",
        "--temporal-compression", "confidence_weighted",
        "--epochs", "100",
        "--lr-scheduler", "warmup_cosine",
        "--warmup-start-lr", "0.001",
        "--min-lr", "0.0001",
        "--compile",
        "--save-dir", "outputs/hand_ctr_wrist_t100_e100_cosine_ntu60_xsub",
    ]
    return arguments + list(overrides)


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
