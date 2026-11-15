"""Fail-fast CUDA Inductor smoke for derived-geometry dropout."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.derived_geometry_dropout_20261107.model import DerivedGeometryDropoutFacePairFusion
from isaa.face_pair_training import compile_model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch._dynamo.config.suppress_errors = False
    model = DerivedGeometryDropoutFacePairFusion(60).cuda().train()
    compiled = compile_model(model, "default")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.rand(2, 3, 8, 133, 1, device="cuda")
    x[:, 2] = 1
    mask = torch.ones(2, 8, dtype=torch.bool, device="cuda")
    rows = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = compiled(x, mask)
        loss = F.cross_entropy(output["logits"], torch.tensor([1, 2], device="cuda"))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        if model.hand_face_head.weight.grad is None:
            raise RuntimeError("missing head gradient")
        optimizer.step()
        rows.append({"step": step, "loss": float(loss.detach().cpu())})
    path = Path(__file__).with_name("cuda_compile_smoke.json")
    path.write_text(json.dumps({"status": "ok", "backend": "inductor", "required": True, "training_only_dropout_probability": 0.025, "steps": rows}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
