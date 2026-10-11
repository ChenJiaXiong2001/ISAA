"""Validate epoch-end continuation without accepting baseline initialization."""
from pathlib import Path
import math

import torch


def load_resume_checkpoint(path: Path, model, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != model.ARCHITECTURE:
        raise ValueError("Resume requires a matching no-pool architecture, not original best weights")
    saved = checkpoint.get("args", {})
    for key in ("model_variant", "num_classes", "window_size", "max_persons", "split", "feature_mode"):
        if saved.get(key) != getattr(args, key):
            raise ValueError(f"Resume {key} mismatch: checkpoint={saved.get(key)!r}, requested={getattr(args, key)!r}")
    if not isinstance(checkpoint.get("epoch"), int) or not 0 < checkpoint["epoch"] < args.epochs:
        raise ValueError("Resume epoch must be completed and smaller than --epochs (total target epochs)")
    if "optimizer" not in checkpoint:
        raise ValueError("Resume requires the saved SGD optimizer state")
    if not isinstance(checkpoint.get("val_accuracy"), (int, float)) or not math.isfinite(checkpoint["val_accuracy"]):
        raise ValueError("Resume requires finite epoch-end validation accuracy")
    tensors = list(checkpoint["model"].values()) + [v for state in checkpoint["optimizer"]["state"].values()
                                                  for v in state.values() if torch.is_tensor(v)]
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("Resume checkpoint contains nonfinite model or optimizer tensors")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint
