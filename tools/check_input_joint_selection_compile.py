"""Real CUDA/Inductor train/eval checks for both newly added stages."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Configure the existing wheel CUDA library bootstrap before importing torch.
from isaa.train import _prefer_bundled_cuda_libraries
import torch
from isaa.input_joint_training import COMPILE_SCOPE, compile_model
from isaa.models.input_joint_selection import InputJointSelectionFusion, OPTIONAL_BODY


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This acceptance check requires real CUDA")
    torch.manual_seed(1)
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = True
    search = InputJointSelectionFusion(60)
    # Representative strict subset, not a claimed learned result.
    search.gates.mu.data.fill_(-1.)
    search.gates.mu.data[:len(OPTIONAL_BODY):2] = .7
    search.gates.mu.data[len(OPTIONAL_BODY)::2] = .7
    subset = search.export_selection()
    results = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "window_size": args.window_size,
               "compile": {**COMPILE_SCOPE, "mode": args.compile_mode}, "stages": {}}
    for stage in ("select", "compact"):
        print(f"Checking {stage}", flush=True)
        model = (InputJointSelectionFusion(60) if stage == "select" else
                 InputJointSelectionFusion(60, stage="compact", subset=subset)).to(args.device)
        if stage == "select":
            model.gates.mu.data.fill_(.6)
        compiled = compile_model(model, args.compile_mode)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        start = time.monotonic()
        for step in range(2):
            compiled.train()
            if stage == "select":
                # Validate warmup->stochastic transition without an eager path.
                model.gates.noise_scale.fill_(float(step))
            x = torch.randn(args.batch_size, 3, args.window_size,
                            len(model.selection["input_indices"]), 2, device=args.device) * .05
            x[:, 2] = 1.
            x[0, :, :, :, 1] = 0.
            frame_mask = torch.ones(args.batch_size, args.window_size, dtype=torch.bool, device=args.device)
            frame_mask[0, -3:] = False
            optimizer.zero_grad(set_to_none=True)
            logits, penalty = compiled(x, frame_mask, return_selection=True)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(args.batch_size, device=args.device) % 60) + .1 * penalty
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            if stage == "select" and (model.gates.mu.grad is None or model.gates.mu.grad.abs().sum() == 0):
                raise AssertionError("Compiled selector received no gradient")
            optimizer.step()
            torch.cuda.synchronize()
            print(f"{stage} compiled train backward step={step} loss={loss.item():.5f}", flush=True)
        compiled.eval()
        with torch.no_grad():
            # A different batch size also exercises the final partial val batch.
            predicted = compiled(x[:1], frame_mask[:1])
            reference = model(x[:1], frame_mask[:1])
        torch.testing.assert_close(predicted, reference, rtol=3e-3, atol=3e-4)
        torch.cuda.synchronize()
        results["stages"][stage] = dict(compiled_forward_backward=True, train_steps=2,
                                             eval_matches_eager=True, loss=loss.item(), gradient_norm=norm.item(),
                                             raw_input_count=len(model.selection["input_indices"]),
                                             body_nodes=len(model.selection["body_indices"]),
                                             hand_nodes=len(model.selection["hand_local_indices"]),
                                             elapsed_seconds=time.monotonic() - start)
        del compiled, optimizer, model, logits, penalty, loss, predicted, reference
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
