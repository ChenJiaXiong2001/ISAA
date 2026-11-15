"""Fail-fast CUDA Inductor smoke test for the interaction candidate.

The test uses synthetic data and performs two optimizer updates.  It does not
load a checkpoint or start training on the dataset.  Compilation errors are
allowed to propagate; there is no eager fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.hand_face_interaction_20261107.model import HandFaceTemporalInteractionFusion
from isaa.face_pair_training import compile_model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the compiled interaction smoke test")
    torch._dynamo.config.suppress_errors = False
    device = torch.device("cuda")
    model = HandFaceTemporalInteractionFusion(60).to(device)
    compiled = compile_model(model, "default")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.rand(2, 3, 8, 133, 1, device=device)
    x[:, 2] = 1
    frame_mask = torch.ones(2, 8, dtype=torch.bool, device=device)
    rows = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = compiled(x, frame_mask)
        labels = torch.tensor([1, 2], device=device)
        loss = torch.nn.functional.cross_entropy(output["logits"], labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        interaction_readout_grad = model.interaction_readout.weight.grad
        interaction_hand_grad = model.interaction_hand.weight.grad
        interaction_face_grad = model.interaction_face.weight.grad
        if interaction_readout_grad is None or not bool(torch.isfinite(interaction_readout_grad).all()):
            raise RuntimeError(f"missing/nonfinite interaction_readout gradient at step {step}")
        # The projections receive a gradient after the zero-initialized
        # readout has had one update; this proves they are inside the graph.
        if step == 1:
            for name, grad in (("interaction_hand", interaction_hand_grad),
                               ("interaction_face", interaction_face_grad)):
                if grad is None or not bool(torch.isfinite(grad).all()) or float(grad.abs().sum()) == 0:
                    raise RuntimeError(f"missing/nonfinite {name} gradient at step {step}")
        optimizer.step()
        rows.append({"step": step, "loss": float(loss.detach().cpu()),
                     "readout_grad_l1": float(interaction_readout_grad.abs().sum().detach().cpu()),
                     "hand_grad_l1": None if interaction_hand_grad is None else float(interaction_hand_grad.abs().sum().detach().cpu()),
                     "face_grad_l1": None if interaction_face_grad is None else float(interaction_face_grad.abs().sum().detach().cpu())})
    output_path = Path(__file__).with_name("cuda_compile_smoke.json")
    output_path.write_text(json.dumps({
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "compile": {"backend": "inductor", "mode": "default", "required": True,
                    "suppress_errors": False, "whole_model_eager_fallback": False,
                    "interaction_inside_compiled_scope": True,
                    "inherited_eager_boundary": "face LocalSTBlock.forward"},
        "steps": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output_path), "steps": rows}, indent=2))


if __name__ == "__main__":
    main()

