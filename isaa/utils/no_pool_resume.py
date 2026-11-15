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


def load_batch_cosine_resume_checkpoint(path: Path, model, args):
    """Restore a matching pooled cosine run for an explicitly extended budget."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != model.ARCHITECTURE:
        raise ValueError("Cosine continuation requires the matching pooled architecture")
    saved = checkpoint.get("args", {})
    for key in ("model_variant", "num_classes", "window_size", "max_persons", "split", "feature_mode",
                "temporal_compression", "lr_scheduler"):
        if saved.get(key) != getattr(args, key):
            raise ValueError(f"Resume {key} mismatch: checkpoint={saved.get(key)!r}, requested={getattr(args, key)!r}")
    if not isinstance(checkpoint.get("epoch"), int) or not 0 < checkpoint["epoch"] < args.epochs:
        raise ValueError("Cosine continuation epoch must be completed and smaller than the new total")
    schedule = checkpoint.get("lr_schedule", {})
    completed = schedule.get("completed_updates")
    if not isinstance(completed, int) or completed <= 0:
        raise ValueError("Cosine continuation requires saved completed optimizer updates")
    if "optimizer" not in checkpoint:
        raise ValueError("Cosine continuation requires the saved SGD optimizer state")
    if not isinstance(checkpoint.get("val_accuracy"), (int, float)) or not math.isfinite(checkpoint["val_accuracy"]):
        raise ValueError("Cosine continuation requires finite epoch-end validation accuracy")
    tensors = list(checkpoint["model"].values()) + [v for state in checkpoint["optimizer"]["state"].values()
                                                  for v in state.values() if torch.is_tensor(v)]
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("Cosine continuation checkpoint contains nonfinite tensors")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint
