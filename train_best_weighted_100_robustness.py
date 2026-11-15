"""Train the pooled 100-frame baseline for 150 epochs with 20% grouped dropout."""
import sys

from train_best_weighted_100_cosine import training_arguments as cosine_arguments


def training_arguments(overrides):
    supplied = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    if supplied.intersection({"--model-variant", "--gcn-config", "--interaction-config",
                              "--init-checkpoint", "--resume-checkpoint", "--ablation-stage",
                              "--npy-dir", "--no-compile"}):
        raise ValueError("baseline robustness keeps the canonical model, trains from scratch, and requires torch.compile")
    arguments = list(overrides)
    if "--epochs" not in supplied:
        arguments.extend(("--epochs", "150"))
    if "--joint-dropout-probability" not in supplied:
        arguments.extend(("--joint-dropout-probability", "0.20"))
    if "--grouped-joint-dropout-probability" not in supplied:
        arguments.extend(("--grouped-joint-dropout-probability", "0.20"))
    if "--save-dir" not in supplied:
        arguments.extend(("--save-dir", "outputs/best_weighted_100_robustness_ntu60_xsub"))
    return cosine_arguments(arguments)


if __name__ == "__main__":
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main

    main()
