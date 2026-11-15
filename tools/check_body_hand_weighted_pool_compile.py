"""CUDA compilation smoke test for the pooled two-branch model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isaa.train  # Establish the same CUDA wheel library path as the training entry.
import torch
from torch.nn import functional as F

from isaa.models.body_hand_weighted_pool import BodyHandWeightedPoolFusion


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this compile probe")
    torch._dynamo.config.suppress_errors = False
    model = BodyHandWeightedPoolFusion(7, window_size=100, max_persons=2).cuda().train()
    compiled = torch.compile(model, mode="default")
    x = torch.randn(1, 3, 100, 133, 2, device="cuda") * 0.05
    x[:, 2] = 1
    logits = compiled(x)
    loss = F.cross_entropy(logits, torch.tensor([3], device="cuda"))
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("Non-finite compiled forward")
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or non-finite compiled gradient: {name}")
    print(f"CUDA_COMPILED_FORWARD_BACKWARD_OK logits={tuple(logits.shape)} loss={loss.item():.6f}", flush=True)


if __name__ == "__main__":
    main()
