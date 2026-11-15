"""Same-input CUDA comparison of original and fixed geometric-consensus fusion."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaa.train  # Set CUDA library paths before importing torch.
import numpy as np
import torch
from torch.nn import functional as F

from isaa.face_pair_training import _dataset, _loader, compile_model, compile_scope
from isaa.models.face_pair_fusion import FacePairFusion
from isaa.utils.seed import seed_everything
from tools.check_face_pair_checkpoint import check_args, check_compile

EXPECTED = 'aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7'


def fuse(hf, bf, w):
    hlog, blog = F.log_softmax(hf.float(), -1), F.log_softmax(bf.float(), -1)
    arithmetic = torch.logaddexp(w.log() + hlog, torch.log1p(-w) + blog)
    arithmetic = arithmetic - torch.logsumexp(arithmetic, -1, keepdim=True)
    geometric = w * hlog + (1 - w) * blog
    geometric = geometric - torch.logsumexp(geometric, -1, keepdim=True)
    return arithmetic, torch.logaddexp(arithmetic, geometric) - math.log(2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if options.batch_size <= 0 or options.num_workers < 0 or not torch.cuda.is_available():
        raise ValueError('CUDA, positive batch size, and nonnegative workers required')
    checkpoint_sha = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != EXPECTED:
        raise ValueError('Wrong checkpoint')
    checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=True)
    saved = checkpoint['args']
    check_args(saved, 'checkpoint.args', expected_variant='best-face-pair')
    check_compile(checkpoint['compile'], saved, 'checkpoint.compile')
    if checkpoint['epoch'] != 83 or not saved['compile']:
        raise ValueError('Expected compiled epoch-83 checkpoint')
    manifest = json.loads((options.checkpoint.parent / 'split_manifest.json').read_text(encoding='utf-8'))
    model = FacePairFusion(saved['num_classes'])
    if checkpoint['architecture'] != model.ARCHITECTURE:
        raise ValueError('Architecture mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    if float(model.hand_face_temperature) != 1 or float(model.body_face_temperature) != 1:
        raise ValueError('Expected unit temperatures')
    frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
    counts = model.routing_class_counts.float()
    ha = torch.where(counts > 0, model.routing_hand_face_correct.float() / counts.clamp_min(1), 0)
    ba = torch.where(counts > 0, model.routing_body_face_correct.float() / counts.clamp_min(1), 0)
    weights = torch.where(ha + ba > 0, ha / (ha + ba), 0.5).to('cuda')
    args = argparse.Namespace(**saved)
    args.archive = str(options.archive.resolve())
    args.test_batch_size = options.batch_size
    args.num_workers = options.num_workers
    seed_everything(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    data = _dataset(args, 'val', False)
    members = list(data.members)
    member_sha = hashlib.sha256('\n'.join(members).encode()).hexdigest()
    if members != manifest['validation']['members'] or member_sha != manifest['validation']['sha256'] or len(members) != 16306:
        raise ValueError('Full validation manifest mismatch')
    loader = _loader(args, data, torch.device('cuda'), False)
    model = model.to('cuda').eval()
    compiled = compile_model(model, args.compile_mode)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    input_hash, frame_hash = hashlib.sha256(), hashlib.sha256()
    rows, subjects = [], []
    count = gain = loss = 0
    nll = [0.0, 0.0]
    initial = time.perf_counter()
    print(json.dumps({'checkpoint_sha256': checkpoint_sha, 'members_sha256': member_sha,
        'samples': len(members), 'compile': {**compile_scope(args), 'mode': args.compile_mode},
        'fusion_scope': 'Post-forward GPU operations, outside compiled network'}), flush=True)
    with torch.inference_mode():
        for step, (features, labels, frame_mask) in enumerate(loader, 1):
            input_hash.update(features.contiguous().numpy().tobytes())
            frame_hash.update(frame_mask.contiguous().numpy().tobytes())
            truth = labels.numpy()
            if not np.array_equal(truth, np.asarray(data.labels[count:count + len(truth)])):
                raise ValueError('Label ordering differs')
            output = compiled(features.to('cuda', non_blocking=True), frame_mask.to('cuda', non_blocking=True))
            a, c = fuse(output['hand_face_logits'], output['body_face_logits'], weights)
            if not torch.isfinite(a).all() or not torch.isfinite(c).all():
                raise ValueError('Nonfinite fusion')
            if not torch.allclose(a.exp().sum(-1), torch.ones(len(truth), device='cuda'), atol=2e-6):
                raise ValueError('Unnormalized arithmetic')
            if not torch.allclose(c.exp().sum(-1), torch.ones(len(truth), device='cuda'), atol=2e-6):
                raise ValueError('Unnormalized candidate')
            nll[0] += float(F.nll_loss(a, labels.to('cuda'), reduction='sum'))
            nll[1] += float(F.nll_loss(c, labels.to('cuda'), reduction='sum'))
            pa, pc = a.argmax(-1).cpu().numpy(), c.argmax(-1).cpu().numpy()
            for j, label in enumerate(truth):
                sample = members[count + j]
                subject = re.search(r'P(\d{3})', sample)
                action = re.search(r'A(\d{3})', sample)
                if not subject or not action or int(action.group(1)) != int(label) + 1:
                    raise ValueError('Sample identifier mismatch')
                subjects.append(int(subject.group(1)))
                rows.append([sample, int(subject.group(1)), int(label), int(pa[j]), int(pc[j])])
            gain += int(((pa != truth) & (pc == truth)).sum())
            loss += int(((pa == truth) & (pc != truth)).sum())
            count += len(truth)
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f'progress={step}/{len(loader)} samples={count} gain={gain} loss={loss} elapsed={time.perf_counter() - initial:.1f}s', flush=True)
    if count != 16306 or any(not torch.equal(value.detach().cpu(), frozen[key]) for key, value in model.state_dict().items()):
        raise ValueError('Incomplete traversal or altered model state')
    truth = np.array([row[2] for row in rows])
    before = np.array([row[3] for row in rows])
    after = np.array([row[4] for row in rows])
    changes = (after == truth).astype(np.int64) - (before == truth).astype(np.int64)
    unique, group = np.unique(subjects, return_inverse=True)
    if len(unique) != 20:
        raise ValueError('Expected 20 subjects')
    sizes, net = np.bincount(group), np.bincount(group, weights=changes)
    rng = np.random.default_rng(20261106)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=20000)
    ci = np.percentile(100 * (draws @ net) / (draws @ sizes), [2.5, 97.5]).tolist()
    groups = [{'subject': int(subject), 'samples': int(sizes[i]), 'gain': int((changes[group == i] == 1).sum()),
        'loss': int((changes[group == i] == -1).sum()), 'difference_pp': float(100 * net[i] / sizes[i])}
        for i, subject in enumerate(unique)]
    result = {'checkpoint_sha256': checkpoint_sha, 'validation_members_sha256': member_sha,
        'input_sha256': input_hash.hexdigest(), 'frame_mask_sha256': frame_hash.hexdigest(),
        'samples': count, 'subjects': len(unique), 'baseline_correct': int((before == truth).sum()),
        'candidate_correct': int((after == truth).sum()), 'baseline_top1_percent': float(100 * (before == truth).mean()),
        'candidate_top1_percent': float(100 * (after == truth).mean()), 'gain': gain, 'loss': loss,
        'difference_pp': float(100 * changes.mean()), 'subject_cluster_bootstrap_95_ci_pp': ci,
        'bootstrap_replicates': 20000, 'bootstrap_seed': 20261106,
        'baseline_nll': nll[0] / count, 'candidate_nll': nll[1] / count,
        'compile': {**compile_scope(args), 'mode': args.compile_mode},
        'fusion_scope': 'Post-forward GPU operations, outside compiled network',
        'fusion': 'same batch of logits; frozen gamma=1 accuracy weights; candidate 50% arithmetic and 50% geometric',
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'seconds': time.perf_counter() - initial,
        'limitations': 'Candidate fixed after exploration of two training calibration subjects; official validation has been reused.'}
    with (options.output_dir / 'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample', 'subject', 'label', 'baseline_prediction', 'candidate_prediction'])
        writer.writerows(rows)
    (options.output_dir / 'subjects.json').write_text(json.dumps(groups, indent=2) + '\n', encoding='utf-8')
    (options.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
