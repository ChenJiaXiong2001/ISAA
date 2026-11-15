"""Explicit CUDA/Inductor smoke test; never use eager as a fallback."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import isaa.train  # Configure CUDA wheel paths before importing torch.
import torch
from torch.nn import functional as F

from isaa.face_pair_training import compile_model
from model import BodyFaceInteractionFusion


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = BodyFaceInteractionFusion(6).cuda().train()
    body = torch.randn(2, 8, model.interaction_body.in_features, device="cuda")
    face = torch.randn(2, 8, model.interaction_face.in_features, device="cuda")
    valid = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    # fullgraph=True is specific to the new interaction; the baseline's face
    # LocalSTBlock eager boundary is deliberately retained in the full model.
    interaction = torch.compile(model.interaction_logits, backend="inductor", fullgraph=True)
    interaction(body, face, valid, valid, 2, 1).sum().backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    compiled = compile_model(model, "default")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.rand(2, 3, 16, 133, 2, device="cuda")
    mask = torch.ones(2, 16, device="cuda", dtype=torch.bool)
    labels = torch.tensor([0, 1], device="cuda")
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = compiled(x, mask)
        loss = F.cross_entropy(result["body_face_logits"], labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite compiled loss")
        loss.backward()
        grad = model.interaction_readout.weight.grad if step == 0 else model.interaction_body.weight.grad
        if grad is None or not bool(torch.isfinite(grad).all()) or not bool(grad.abs().sum() > 0):
            raise RuntimeError(f"missing or nonfinite interaction gradient at step {step}")
        optimizer.step()
        torch.cuda.synchronize()
    print("CUDA Inductor fullgraph interaction and model forward/backward passed; "
          "existing face LocalSTBlock eager boundary retained")


if __name__ == "__main__":
    main()
