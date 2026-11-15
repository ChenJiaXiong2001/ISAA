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
from isaa.input_joint_training_v5 import compile_model, COMPILE_SCOPE, make_optimizers, parse_args
from isaa.models.input_joint_selection_v5 import InputJointSelectionV5Fusion


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
    model = InputJointSelectionV5Fusion(60).cuda()
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
    training_args=parse_args([])
    from isaa.input_joint_training_v5 import gate_weight
    for epoch in (1, 6, 16):
        model.gates.begin_epoch(epoch, 5, 5)
        compiled.train()
        backbone.zero_grad(set_to_none=True)
        router.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in model.gates.parameters()]
        logits, cost, aux = compiled(x, frame, return_gate_values=True)
        assert aux['sampled_gate_values'].shape == (2, 40)
        ce = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2], device='cuda'))
        sparse = gate_weight(epoch,training_args)*cost
        loss = ce + sparse+.01*aux['logit_guard_loss']
        if epoch>5:
            bias=model.gates.router[-1].bias
            ce_probe=torch.autograd.grad(ce,bias,retain_graph=True)[0]
            cost_probe=torch.autograd.grad(sparse,bias,retain_graph=True)[0]
            assert torch.isfinite(ce_probe).all() and torch.isfinite(cost_probe).all()
            assert ce_probe.abs().sum()>0
            if epoch>15:
                assert cost_probe.abs().sum()>0
        torch.testing.assert_close(cost,(aux['gate_values']*model.gates.cost).sum(-1).mean()/60.)
        if epoch>5:
            assert aux['gate_values'].min()>=.05
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
        if epoch > 5:
            assert set(aux['sampled_gate_values'].detach().flatten().tolist()).issubset({0., 1.})
        torch.cuda.synchronize()
        row = dict(epoch=epoch, loss=float(loss.detach()), router_gradient_norm=norm,
                   router_updated=changed, compiled_forward_backward=True, gate_count=40, face_token_count=6)
        results['training'].append(row)
        print(json.dumps(row), flush=True)
    # No sparsity loss here: classification alone must reopen every group.
    compiled.train()
    with torch.no_grad():
        model.gates.router[-1].bias.fill_(-40.)
    backbone.zero_grad(set_to_none=True)
    router.zero_grad(set_to_none=True)
    logits, _, aux = compiled(x, frame, return_gate_values=True)
    logits.square().mean().backward()
    assert aux['gate_values'].min()>=.05
    assert aux['gate_logits'].min()>=-4.00001
    closed_grads = [float(model.gates.node_embedding.grad[lo:hi].abs().sum()) for lo,hi in ((0,14),(14,34),(34,40))]
    assert all(v > 0 for v in closed_grads), closed_grads
    results['low_probability_classification_gradient_body_hand_face'] = closed_grads
    results['cost_matches_binary_probability']=True
    results['retained_graph_gradient_diagnostics']=True
    with torch.no_grad():
        model.gates.router[-1].bias.fill_(2.)
    compiled.eval()
    for hard in (False, True):
        model.gates.hard_eval.fill_(hard)
        with torch.no_grad():
            predicted = compiled(x[:1], frame[:1])
            reference = model(x[:1], frame[:1])
        torch.testing.assert_close(predicted, reference, rtol=3e-3, atol=3e-4)
        print('compiled partial-batch eval matches eager; hard=', hard, flush=True)
    control_outputs=[]
    for mode in (1,2):
        model.gates.forced_mode.fill_(mode)
        with torch.no_grad():
            y=compiled(x[:1],frame[:1])
        assert torch.isfinite(y).all()
        control_outputs.append(y)
    model.gates.forced_mode.zero_()
    results['forced_open_closed_control_delta']=float((control_outputs[0]-control_outputs[1]).abs().max())
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
