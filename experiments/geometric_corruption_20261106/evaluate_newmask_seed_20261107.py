"""Compare frozen arithmetic and geometric fusion under fixed corruptions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA wheel libraries before importing torch.
import numpy as np
import torch
from torch.nn import functional as F

from experiments.geometric_consensus_20261106.evaluate_cuda import EXPECTED, fuse
from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.face_pair_training import _loader, compile_model, compile_scope
from isaa.models.face_pair_fusion import FacePairFusion
from isaa.utils.seed import seed_everything
from tools.check_face_pair_checkpoint import check_args, check_compile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--smoke', action='store_true', help='CPU eager smoke test on 8 validation samples per condition')
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if options.batch_size <= 0 or options.num_workers < 0:
        raise ValueError('Invalid batch size or worker count')
    if not options.smoke and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full compiled evaluation')
    device = torch.device('cpu' if options.smoke else 'cuda')
    digest = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED:
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
    original_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    counts = model.routing_class_counts.float()
    h, b = model.routing_hand_face_correct.float(), model.routing_body_face_correct.float()
    if bool(((counts < 0) | (h < 0) | (b < 0) | (h > counts) | (b > counts)).any()):
        raise ValueError('Invalid calibration counts')
    weights = torch.where(h + b > 0, h / (h + b), .5).to(device)
    args = argparse.Namespace(**saved)
    args.archive = str(options.archive.resolve())
    args.test_batch_size = options.batch_size
    args.num_workers = 0 if options.smoke else options.num_workers
    torch.set_num_threads(4)
    if not options.smoke:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    model = model.to(device).eval()
    network = model if options.smoke else compile_model(model, args.compile_mode)
    conditions = (('point_0.05', 'joint_dropout_probability', .05, 2718),
                  ('point_0.1', 'joint_dropout_probability', .1, 2719),
                  ('group_0.05', 'grouped_joint_dropout_probability', .05, 2818),
                  ('group_0.1', 'grouped_joint_dropout_probability', .1, 2819))
    options.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for name, key, probability, seed in conditions:
        seed_everything(seed)
        data = RTMWZipDataset(options.archive, split='val', split_protocol=args.split,
            window_size=args.window_size, num_joints=133, num_classes=args.num_classes,
            layout='rtmw_133', max_persons=args.max_persons, feature_mode='raw', augment=True,
            score_normalization=getattr(args, 'score_normalization', 'auto'),
            temporal_compression=getattr(args, 'temporal_compression', 'confidence_weighted'),
            deterministic_temporal_crop=getattr(args, 'deterministic_temporal_crop', 'center'),
            augmentation_config={'random_temporal_crop': False, key: probability},
            max_samples=8 if options.smoke else 0)
        members = list(data.members)
        expected = manifest['validation']['members']
        if members != expected[:len(members)] or len(members) != (8 if options.smoke else 16306):
            raise ValueError('Validation members do not match manifest')
        member_hash = hashlib.sha256('\n'.join(members).encode()).hexdigest()
        if not options.smoke and member_hash != manifest['validation']['sha256']:
            raise ValueError('Validation member digest mismatch')
        loader = _loader(args, data, device, False)
        input_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
        rows, groups, nll, offset = [], [], [0., 0.], 0
        start = time.perf_counter()
        with torch.inference_mode():
            for step, (features, labels, mask) in enumerate(loader, 1):
                input_hash.update(features.contiguous().numpy().tobytes())
                mask_hash.update(mask.contiguous().numpy().tobytes())
                truth = labels.numpy()
                if not np.array_equal(truth, np.asarray(data.labels[offset:offset + len(truth)])):
                    raise ValueError('Label ordering mismatch')
                output = network(features.to(device), mask.to(device))
                a, c = fuse(output['hand_face_logits'], output['body_face_logits'], weights)
                if not torch.isfinite(a).all() or not torch.isfinite(c).all():
                    raise ValueError('Nonfinite fused output')
                if not torch.allclose(a.exp().sum(-1), torch.ones(len(truth), device=device), atol=2e-6):
                    raise ValueError('Arithmetic output not normalized')
                if not torch.allclose(c.exp().sum(-1), torch.ones(len(truth), device=device), atol=2e-6):
                    raise ValueError('Candidate output not normalized')
                nll[0] += float(F.nll_loss(a, labels.to(device), reduction='sum'))
                nll[1] += float(F.nll_loss(c, labels.to(device), reduction='sum'))
                pa, pc = a.argmax(-1).cpu().numpy(), c.argmax(-1).cpu().numpy()
                for i, label in enumerate(truth):
                    sample = members[offset + i]
                    match = re.search(r'P(\d{3}).*A(\d{3})', sample)
                    if not match or int(match.group(2)) != int(label) + 1:
                        raise ValueError('Invalid member identity')
                    subject = int(match.group(1))
                    groups.append(subject)
                    rows.append([sample, subject, int(label), int(pa[i]), int(pc[i])])
                offset += len(truth)
                if step == 1 or step % 50 == 0 or step == len(loader):
                    print(f'{name} batch={step}/{len(loader)} samples={offset} elapsed={time.perf_counter()-start:.1f}s', flush=True)
        if offset != len(members):
            raise ValueError('Incomplete traversal')
        truth = np.array([row[2] for row in rows])
        old, new = np.array([row[3] for row in rows]), np.array([row[4] for row in rows])
        changes = (new == truth).astype(np.int64) - (old == truth).astype(np.int64)
        unique, group = np.unique(groups, return_inverse=True)
        if not options.smoke and len(unique) != 20:
            raise ValueError('Expected 20 subjects')
        sizes = np.bincount(group)
        net = np.bincount(group, weights=changes)
        rng = np.random.default_rng(20261106)
        draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=20000)
        ci = np.percentile(100 * (draws @ net) / (draws @ sizes), [2.5, 97.5]).tolist()
        result = {'condition': name, 'seed': seed, 'input_sha256': input_hash.hexdigest(),
            'frame_mask_sha256': mask_hash.hexdigest(), 'samples': offset,
            'baseline_correct': int((old == truth).sum()), 'candidate_correct': int((new == truth).sum()),
            'baseline_top1_percent': float(100 * (old == truth).mean()),
            'candidate_top1_percent': float(100 * (new == truth).mean()),
            'difference_pp': float(100 * changes.mean()), 'gain': int((changes == 1).sum()),
            'loss': int((changes == -1).sum()), 'subject_bootstrap_95_ci_pp': ci,
            'baseline_nll': nll[0] / offset, 'candidate_nll': nll[1] / offset,
            'seconds': time.perf_counter() - start}
        with (options.output_dir / f'{name}.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample', 'subject', 'label', 'baseline_prediction', 'candidate_prediction'])
            writer.writerows(rows)
        summaries.append(result)
        print(json.dumps(result), flush=True)
    if any(not torch.equal(value.detach().cpu(), original_state[key]) for key, value in model.state_dict().items()):
        raise ValueError('Model state changed during evaluation')
    result = {'checkpoint_sha256': digest, 'smoke_only': options.smoke,
        'compile': 'CPU eager smoke only' if options.smoke else {**compile_scope(args), 'mode': args.compile_mode},
        'fusion': 'fixed 50% arithmetic / 50% geometric with original gamma=1 calibration weights; same model output per pair',
        'corruption': 'synthetic point/group dropout after temporal compression; independent of real detector failure patterns',
        'selection': '50% mixture chosen on two training calibration subjects; validation has been used in prior research',
        'conditions': summaries}
    (options.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
