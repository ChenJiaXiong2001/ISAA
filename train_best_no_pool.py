"""Train ordered no-pool readouts with required compilation and stable SGD defaults."""
import sys

from train_full_gcn import training_arguments as full_training_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if supplied.intersection({"--model-variant", "--gcn-config", "--interaction-config", "--init-checkpoint"}):
        raise ValueError("This preset trains the best backbone with new ordered readouts from scratch")
    if "--no-compile" in supplied:
        raise ValueError("best-no-pool requires torch.compile; --no-compile is not allowed")
    arguments = full_training_arguments(overrides)
    arguments[arguments.index("--model-variant") + 1] = "body-local-hand-ctr-wide-relative-no-pool"
    position = arguments.index("--gcn-config")
    del arguments[position:position + 2]
    if "--lr" not in supplied:
        arguments[arguments.index("--lr") + 1] = "0.025"
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = "outputs/best_no_pool_ntu60_xsub"
    return arguments


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
