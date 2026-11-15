"""Train Hand-Face and Body-Face classifiers with holdout-based class selection."""
import sys

from train_full_gcn import training_arguments as full_training_arguments
from isaa.models.face_pair_fusion import FACE_PAIR_RAW_XY_VARIANT, FACE_PAIR_VARIANT


def training_arguments(overrides):
    """Keep baseline inputs and schedule; add the two pair classifiers from scratch."""
    supplied = {value.split("=", 1)[0] for value in overrides if value.startswith("--")}
    if "--no-compile" in supplied:
        raise ValueError("face-pair training requires torch.compile; --no-compile is not allowed")
    if supplied.intersection({"--init-checkpoint", "--resume-checkpoint", "--npy-dir"}):
        raise ValueError("face-pair training uses random initialization and full RTMW ZIP sequences")
    selected_variant = FACE_PAIR_VARIANT
    if "--model-variant" in supplied:
        selected_variant = overrides[overrides.index("--model-variant") + 1]
        if selected_variant not in {FACE_PAIR_VARIANT, FACE_PAIR_RAW_XY_VARIANT}:
            raise ValueError("this entry only supports face-pair variants")
    defaults = {"--lr": "0.025", "--max-grad-norm": "5.0"}
    arguments = [item for flag, value in defaults.items() if flag not in supplied
                 for item in (flag, value)]
    # Start from the generic parser preset so this entry can select either
    # face-pair architecture without the original pooled preset rejecting the
    # explicit model variant.
    # The full-GCN preset supplies the common NTU60/compile defaults; remove
    # the explicit model flag before invoking its narrow selector, then put the
    # requested face-pair variant back into the resulting argv.
    forwarded = [value for index, value in enumerate(arguments + list(overrides))
                 if value != "--model-variant" and
                 not (index > 0 and (arguments + list(overrides))[index - 1] == "--model-variant")]
    arguments = full_training_arguments(forwarded)
    arguments = [value for index, value in enumerate(arguments)
                 if not (value == "--gcn-config" or
                         (index > 0 and arguments[index - 1] == "--gcn-config"))]
    if "--window-size" not in supplied:
        arguments[arguments.index("--window-size") + 1] = "100"
    if "--temporal-compression" not in supplied:
        arguments.extend(("--temporal-compression", "confidence_weighted"))
    if "--lr-scheduler" not in supplied:
        arguments.extend(("--lr-scheduler", "warmup_cosine"))
    if "--grad-accum-steps" not in supplied:
        arguments.extend(("--grad-accum-steps", "1"))
    if "--lr-steps" in arguments:
        position = arguments.index("--lr-steps")
        del arguments[position:position + 3]
    if "--epochs" not in supplied:
        arguments[arguments.index("--epochs") + 1] = "100"
    arguments[arguments.index("--model-variant") + 1] = selected_variant
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = (
            "outputs/best_face_pair_raw_xy_100_ntu60_xsub"
            if selected_variant == FACE_PAIR_RAW_XY_VARIANT
            else "outputs/best_face_pair_100_ntu60_xsub")
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
