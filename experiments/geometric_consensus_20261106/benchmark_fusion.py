"""CUDA-event benchmark on resident inputs with the original frozen network."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA libraries before importing torch.
import numpy as np
import torch
from torch.nn import functional as F

from isaa.face_pair_training import _dataset, _loader, compile_model, compile_scope
from isaa.models.face_pair_fusion import FacePairFusion
from tools.check_face_pair_checkpoint import check_args, check_compile

EXPECTED = 'aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7'


def original(hf, bf, w):
    hlog, blog = F.log_softmax(hf, -1), F.log_softmax(bf, -1)
    arithmetic = torch.logaddexp(w.log() + hlog, torch.log1p(-w) + blog)
    return arithmetic - torch.logsumexp(arithmetic, -1, keepdim=True)


def geometric(hf, bf, w):
    hlog, blog = F.log_softmax(hf, -1), F.log_softmax(bf, -1)
    arithmetic = torch.logaddexp(w.log() + hlog, torch.log1p(-w) + blog)
    arithmetic = arithmetic - torch.logsumexp(arithmetic, -1, keepdim=True)
    geo = w * hlog + (1 - w) * blog
    geo = geo - torch.logsumexp(geo, -1, keepdim=True)
    return torch.logaddexp(arithmetic, geo) - math.log(2.0)


def measure(fn, args, repeat):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        value = fn(*args)
    stop.record()
    stop.synchronize()
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite output')
    return start.elapsed_time(stop) / repeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=30)
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--network-groups', type=int, default=12)
    parser.add_argument('--network-repeat', type=int, default=4)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    if (not torch.cuda.is_available() or min(options.groups, options.repeat,
            options.network_groups, options.network_repeat) <= 0):
        raise ValueError('CUDA and positive groups/repeat are required')
    device = torch.device('cuda')
    if hashlib.sha256(options.checkpoint.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Unexpected checkpoint')
    checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=True)
    saved = checkpoint['args']
    check_args(saved, 'checkpoint.args', expected_variant='best-face-pair')
    check_compile(checkpoint['compile'], saved, 'checkpoint.compile')
    if checkpoint['epoch'] != 83 or not saved['compile']:
        raise ValueError('Expected compiled epoch-83 checkpoint')
    model = FacePairFusion(saved['num_classes'])
    if checkpoint['architecture'] != model.ARCHITECTURE:
        raise ValueError('Unexpected model architecture')
    model.load_state_dict(checkpoint['model'], strict=True)
    counts = model.routing_class_counts.float()
    ha = torch.where(counts > 0, model.routing_hand_face_correct.float() / counts.clamp_min(1), 0)
    ba = torch.where(counts > 0, model.routing_body_face_correct.float() / counts.clamp_min(1), 0)
    w = torch.where(ha + ba > 0, ha / (ha + ba), 0.5).to(device)
    args = argparse.Namespace(**saved)
    args.archive = str(options.archive.resolve())
    args.test_batch_size = 32
    args.num_workers = 0
    data = _dataset(args, 'val', False)
    manifest = json.loads((options.checkpoint.parent / 'split_manifest.json').read_text(encoding='utf-8'))
    if list(data.members) != manifest['validation']['members']:
        raise ValueError('Validation manifest mismatch')
    loader = _loader(args, data, device, False)
    features, _, frame_mask = next(iter(loader))
    input_sha = hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest()
    features, frame_mask = features.to(device), frame_mask.to(device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    compiled = compile_model(model.to(device).eval(), args.compile_mode)

    def forward_original(x, mask):
        out = compiled(x, mask)
        return original(out['hand_face_logits'], out['body_face_logits'], w)

    def forward_geometric(x, mask):
        out = compiled(x, mask)
        return geometric(out['hand_face_logits'], out['body_face_logits'], w)

    measurements = []
    with torch.inference_mode():
        for batch in (1, 32):
            x, mask = features[:batch], frame_mask[:batch]
            # Warm both shapes and both paths before timing compilation-free runs.
            for _ in range(5):
                forward_original(x, mask)
                forward_geometric(x, mask)
            torch.cuda.synchronize()
            heads = compiled(x, mask)
            hf, bf = heads['hand_face_logits'], heads['body_face_logits']
            for _ in range(100):
                original(hf, bf, w)
                geometric(hf, bf, w)
            torch.cuda.synchronize()
            values = {'original': [], 'geometric': []}
            for group in range(options.groups):
                order = ('original', 'geometric') if group % 2 == 0 else ('geometric', 'original')
                for name in order:
                    fn = original if name == 'original' else geometric
                    values[name].append(measure(fn, (hf, bf, w), options.repeat))
            networks = {'original': [], 'geometric': []}
            for group in range(options.network_groups):
                order = ('original', 'geometric') if group % 2 == 0 else ('geometric', 'original')
                for name in order:
                    fn = forward_original if name == 'original' else forward_geometric
                    networks[name].append(measure(fn, (x, mask), options.network_repeat))
            measurements.append({'batch_size': batch,
                'original_median_ms': statistics.median(values['original']),
                'original_p95_ms': float(np.percentile(values['original'], 95)),
                'geometric_median_ms': statistics.median(values['geometric']),
                'geometric_p95_ms': float(np.percentile(values['geometric'], 95)),
                'paired_delta_median_ms': statistics.median([b - a for a, b in zip(values['original'], values['geometric'])]),
                'groups_ms_per_call': values,
                'network_original_median_ms': statistics.median(networks['original']),
                'network_original_p95_ms': float(np.percentile(networks['original'], 95)),
                'network_geometric_median_ms': statistics.median(networks['geometric']),
                'network_geometric_p95_ms': float(np.percentile(networks['geometric'], 95)),
                'network_paired_delta_median_ms': statistics.median([b - a for a, b in zip(networks['original'], networks['geometric'])]),
                'network_groups_ms_per_call': networks})
            print(json.dumps({'completed_batch': batch, 'measurements': measurements[-1]}), flush=True)
    summary = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'checkpoint_sha256': EXPECTED, 'first_32_input_sha256': input_sha,
        'compile': {**compile_scope(args), 'mode': args.compile_mode},
        'groups': options.groups, 'repetitions_per_group': options.repeat, 'warmup_pairs': 100,
        'network_groups': options.network_groups, 'network_repetitions_per_group': options.network_repeat,
        'network_warmup_pairs_per_batch': 5,
        'measurements': measurements,
        'limitations': 'CUDA-event timings use resident real validation inputs and eager fusion operations. '
                       'Network timing includes compiled forward and fusion, excluding detector, CPU input preprocessing, '
                       'device transfer, batching latency and deployment runtime. No simultaneous GPU jobs assumed.'}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
