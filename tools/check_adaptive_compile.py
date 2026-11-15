"""Exercise real Inductor CUDA sandwich backward and accumulation without dataset training."""
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaa.train import collate_rtmw  # Bootstrap CUDA libraries before torch.
import torch
from isaa.models.class_adaptive_gcn import ClassAdaptiveGCN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-size", action="store_true", help="64/128/256, batch32, T64, M2")
    parser.add_argument("--compile-mode", default="default")
    options = parser.parse_args()
    device = torch.device(options.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the CUDA Graph regression")
    torch.manual_seed(17)
    maxima = (64, 128, 256) if options.full_size else (8, 16, 32)
    batch, frames, persons = (32, 64, 2) if options.full_size else (2, 9, 2)
    model = ClassAdaptiveGCN(60, stage_maxima=maxima).to(device).train()
    model.enable_compile(True, mode=options.compile_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    x = torch.randn(batch, 3, frames, 133, persons, device=device) * .01
    x[:, 2] = 1
    mask = torch.ones(batch, frames, device=device, dtype=torch.bool)
    labels = torch.arange(batch, device=device) % 60
    started = time.perf_counter()
    for update_index in range(2):
        optimizer.zero_grad(set_to_none=True)
        for microbatch in range(2):
            for profile, compile, bn in ((model.max_profile, True, True),
                                         (model.min_profile, True, False),
                                         (tuple(max(8, cap // 2) for cap in model.max_profile), False, False)):
                print(f"check update={update_index} microbatch={microbatch} forward widths={profile}", flush=True)
                logits = model.execute(x, mask, profile, compiled=compile, update_bn=bn)
                loss = torch.nn.functional.cross_entropy(logits, labels) / 6
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                print(f"check backward starts loss={float(loss.detach()):.5f}", flush=True)
                loss.backward()
                print("check backward finished", flush=True)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Nonfinite gradient: " + name)
        if model.classifier.weight.grad is None or not model.classifier.weight.grad.abs().sum():
            raise RuntimeError("Classifier gradients missing")
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"PASSED: Inductor sandwich backward, two accumulated updates; "
          f"full_size={options.full_size} seconds={time.perf_counter()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
