"""Check real CUDA Inductor training and live count-buffer updates for soft routing."""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA libraries before importing torch.
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch._inductor.compile_fx import compile_fx

from experiments.training_route_mismatch_20261106.soft_route import SoftRouteFacePair
from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.train import collate_rtmw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'data/ntu60_skeletons_rtmw.zip')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or not hasattr(torch, 'compile'):
        raise RuntimeError('CUDA and torch.compile are required')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
    if not hasattr(torch._dynamo.config, 'fail_on_recompile_limit_hit'):
        raise RuntimeError('Cannot enforce failure on Dynamo recompile limit')
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    dataset = RTMWZipDataset(args.archive, split='train', split_protocol='xsub60', window_size=100,
        num_joints=133, num_classes=60, layout='rtmw_133', max_persons=2, feature_mode='raw',
        augment=False, temporal_compression='confidence_weighted', max_samples=2)
    loader = DataLoader(dataset, batch_size=2, num_workers=0,
                        collate_fn=partial(collate_rtmw, feature_mode='raw'))
    x, labels, mask = (item.cuda() for item in next(iter(loader)))
    model = SoftRouteFacePair(60).cuda().train()
    graphs = []

    def recording_inductor(graph_module, example_inputs, **kwargs):
        graphs.append(str(graph_module.graph))
        patches = kwargs.pop('options', None)
        if kwargs:
            raise RuntimeError(f'Unexpected backend options: {kwargs}')
        return compile_fx(graph_module, example_inputs, config_patches=patches)

    options = dict(torch._inductor.list_mode_options('default'))
    options['triton.cudagraphs'] = False
    compiled = torch.compile(model, backend=recording_inductor, options=options)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if n.startswith(('hand_face_head.', 'body_face_head.'))}
    started = time.perf_counter()
    print('START_COMPILED_TRAIN', flush=True)
    out = compiled(x, mask)
    loss = (F.cross_entropy(out['logits'], labels) + .5 * F.cross_entropy(out['hand_face_logits'], labels)
            + .5 * F.cross_entropy(out['body_face_logits'], labels))
    if not torch.isfinite(loss) or not all(torch.isfinite(v).all() for v in out.values()):
        raise RuntimeError('Nonfinite forward')
    loss.backward()
    gradients = {}
    for group, prefix in {'body': 'body_blocks.', 'hand': 'hand_ctr_blocks.', 'face': 'face_st_blocks.',
                          'hf_head': 'hand_face_head.', 'bf_head': 'body_face_head.'}.items():
        params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if not params or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError(f'Missing or nonfinite gradients: {group}')
        gradients[group] = sum(float(p.grad.float().square().sum()) for p in params) ** .5
        if gradients[group] <= 0:
            raise RuntimeError(f'Zero gradient: {group}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    if any(torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise RuntimeError('A head parameter did not update')
    torch.cuda.synchronize()
    print('COMPILED_TRAIN_OK', json.dumps(gradients), flush=True)
    model.eval()
    with torch.no_grad():
        prior = compiled(x, mask)['logits']
        ids = torch.arange(60, device='cuda')
        model.set_routing_counts(ids % 3, (ids + 1) % 3, torch.full_like(ids, 10))
        updated = compiled(x, mask)
        reference = model(x, mask)
        for key in reference:
            torch.testing.assert_close(updated[key], reference[key], rtol=3e-3, atol=5e-3)
            torch.testing.assert_close(updated[key].softmax(-1), reference[key].softmax(-1), rtol=3e-3, atol=3e-4)
            if not torch.equal(updated[key].argmax(-1), reference[key].argmax(-1)):
                raise RuntimeError(f'Compiled top1 mismatch: {key}')
        delta = float((updated['logits'] - prior).abs().max())
        if delta < 1e-5:
            raise RuntimeError('Compiled output did not respond to changed counts')
    text = '\n'.join(graphs)
    evidence = {op: op in text for op in ('hand_face_head', 'body_face_head', 'log_softmax', 'logaddexp', 'logsumexp',
                                           'routing_hand_face_correct', 'routing_body_face_correct')}
    if not all(evidence.values()):
        raise RuntimeError(f'Compiled graph is missing expected operations: {evidence}')
    torch.cuda.synchronize()
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'backend': 'inductor', 'compile_mode': 'default', 'suppress_errors': False,
        'inherited_eager_boundary': 'face LocalSTBlock.forward', 'input_shape': list(x.shape),
        'members': dataset.members, 'loss': float(loss.detach()), 'gradient_norms': gradients,
        'both_heads_updated': True, 'counts_updated_compiled_max_delta': delta,
        'updated_compiled_matches_eager': True, 'compiled_graph_evidence': evidence,
        'graph_count': len(graphs), 'elapsed_seconds': time.perf_counter() - started,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    args.output.with_suffix('.graphs.txt').write_text(text, encoding='utf-8')
    print('CUDA_SOFT_ROUTE_OK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
