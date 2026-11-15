"""Real CUDA Inductor F/B across conditional routing phases and hard evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaa.train import _prefer_bundled_cuda_libraries
import torch
from isaa.input_joint_training_v4 import compile_model, COMPILE_SCOPE, make_optimizers, parse_args
from isaa.models.input_joint_selection_v4 import InputJointSelectionV4Fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-size', type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required; no fallback')
    torch.set_num_threads(4)
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = True
    model = InputJointSelectionV4Fusion(60).cuda()
    compiled = compile_model(model, 'default')
    backbone, router = make_optimizers(model, parse_args([]))
    x = torch.randn(2, 3, args.window_size, 129, 2, device='cuda') * .05
    x[:, 2] = 1.
    x[0, :, :, :, 1] = 0.
    x[1, :, :, 4] = float('nan')
    frame = torch.ones(2, args.window_size, device='cuda', dtype=torch.bool)
    frame[0, -3:] = False
    results = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                   compile=COMPILE_SCOPE, window_size=args.window_size, training=[])
    start = time.perf_counter()
    for epoch in (1, 6, 10):
        model.gates.begin_epoch(epoch, 5, 5)
        compiled.train()
        backbone.zero_grad(set_to_none=True)
        router.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in model.gates.parameters()]
        logits, cost, aux = compiled(x, frame, return_gate_values=True)
        assert aux['sampled_gate_values'].shape == (2, 40)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2], device='cuda'))
        loss = loss + (.01 if epoch > 5 else 0.) * cost
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        norm = float(torch.nn.utils.clip_grad_norm_(model.gates.parameters(), float('inf'), error_if_nonfinite=True))
        if epoch == 1:
            assert norm == 0., 'Warmup router gradient must be zero'
        else:
            assert norm > 0., 'Compiled router has no active gradient'
        backbone.step()
        if epoch > 5:
            router.step()
        changed = any(not torch.equal(old, p) for old, p in zip(before, model.gates.parameters()))
        assert changed == (epoch > 5), 'Warmup/update optimizer behavior incorrect'
        if epoch == 10:
            assert set(aux['sampled_gate_values'].detach().flatten().tolist()).issubset({0., 1.})
        torch.cuda.synchronize()
        row = dict(epoch=epoch, loss=float(loss.detach()), router_gradient_norm=norm,
                   router_updated=changed, compiled_forward_backward=True, gate_count=40, face_token_count=6)
        results['training'].append(row)
        print(json.dumps(row), flush=True)
    # No sparsity loss here: classification alone must reopen every group.
    compiled.train()
    model.gates.noise_scale.zero_()
    with torch.no_grad():
        model.gates.router[-1].bias.fill_(-10.)
    backbone.zero_grad(set_to_none=True)
    router.zero_grad(set_to_none=True)
    logits, _, aux = compiled(x, frame, return_gate_values=True)
    logits.square().mean().backward()
    assert (aux['sampled_gate_values'].detach() == 0).all()
    closed_grads = [float(model.gates.node_embedding.grad[lo:hi].abs().sum()) for lo,hi in ((0,14),(14,34),(34,40))]
    assert all(v > 0 for v in closed_grads), closed_grads
    results['closed_classification_gradient_body_hand_face'] = closed_grads
    with torch.no_grad():
        model.gates.router[-1].bias.fill_(.5)
    compiled.eval()
    for hard in (False, True):
        model.gates.hard_eval.fill_(hard)
        with torch.no_grad():
            predicted = compiled(x[:1], frame[:1])
            reference = model(x[:1], frame[:1])
        torch.testing.assert_close(predicted, reference, rtol=3e-3, atol=3e-4)
        print('compiled partial-batch eval matches eager; hard=', hard, flush=True)
    with torch.no_grad():
        missing = torch.full_like(x[:1], float('nan'))
        assert torch.isfinite(compiled(missing, frame[:1])).all()
    results.update(partial_batch_soft_hard_eval_matches_eager=True, all_missing_finite=True,
                   elapsed_seconds=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
