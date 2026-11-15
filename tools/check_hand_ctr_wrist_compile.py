"""Run a real compiled forward/backward/optimizer smoke test for the hand-wrist model."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the training module first so the project's bundled CUDA libraries are
# selected before importing torch.
from isaa.train import collate_rtmw  # noqa: F401
import torch

from isaa.models.hand_ctr_wrist_fusion import HandCTRWristFusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=16)
    options = parser.parse_args()
    device = torch.device(options.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A real CUDA device is required for this compile check")

    torch.manual_seed(17)
    model = HandCTRWristFusion(num_classes=60).to(device).train()
    compile_options = dict(torch._inductor.list_mode_options(options.compile_mode))
    compile_options["triton.cudagraphs"] = False
    model.enable_staged_compile(compile_options, microbatch=1)
    compiled = model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    batch = options.batch_size
    x = torch.randn(batch, 3, options.frames, 133, 2, device=device) * 0.05
    x[:, 2] = torch.rand(batch, options.frames, 133, 2, device=device)
    frame_mask = torch.ones(batch, options.frames, device=device, dtype=torch.bool)
    labels = torch.arange(batch, device=device) % 60

    for update in range(1):
        optimizer.zero_grad(set_to_none=True)
        logits = compiled(x, frame_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite compiled loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Non-finite gradient: " + name)
        required_prefixes = ("body_blocks", "hand_blocks", "head_encoder",
                             "hand_projection", "head_projection", "classifier")
        for prefix in required_prefixes:
            if not any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                       for name, parameter in model.named_parameters()
                       if name.startswith(prefix)):
                raise RuntimeError("Missing gradient: " + prefix)
        optimizer.step()
        torch.cuda.synchronize()
        print(f"compiled update={update + 1} loss={float(loss.detach()):.6f}", flush=True)

    altered = x.detach().clone()
    altered[:, :, :, 23:91] = torch.randn_like(altered[:, :, :, 23:91]) * 100
    with torch.no_grad():
        torch.testing.assert_close(compiled(x, frame_mask), compiled(altered, frame_mask), rtol=0, atol=0)
    print("PASSED: CUDA torch.compile forward, backward, optimizer updates, and face-independence check", flush=True)


if __name__ == "__main__":
    main()
