"""Evaluate the fixed geometric mixture on the frozen epoch-83 logit cache."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


CHECKPOINT_HASH = 'aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    if sha(args.checkpoint) != CHECKPOINT_HASH:
        raise ValueError('Unexpected checkpoint SHA256')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    report = json.loads(args.report.read_text(encoding='utf-8'))
    members = report['manifest']['validation']['members']
    member_hash = hashlib.sha256('\n'.join(members).encode()).hexdigest()
    if (checkpoint['epoch'] != 83 or cache['partition'] != 'val' or cache['epoch'] != 83
            or cache['checkpoint_sha256'] != CHECKPOINT_HASH or cache['member_sha256'] != member_hash):
        raise ValueError('Unexpected checkpoint, cache, or validation members')
    hf, bf, labels = cache['hf'].float(), cache['bf'].float(), cache['labels'].long()
    if hf.shape != bf.shape or hf.shape != (16306, 60) or labels.shape != (16306,) or len(set(members)) != 16306:
        raise ValueError('Unexpected logits or member shape')
    if not torch.isfinite(hf).all() or not torch.isfinite(bf).all():
        raise ValueError('Nonfinite logits')
    subjects = []
    for member, label in zip(members, labels.tolist()):
        subject = re.search(r'P(\d{3})', member)
        action = re.search(r'A(\d{3})', member)
        if not subject or not action or int(action.group(1)) != label + 1:
            raise ValueError('Invalid validation label or subject')
        subjects.append(int(subject.group(1)))
    state = checkpoint['model']
    counts = state['routing_class_counts'].float()
    hc = state['routing_hand_face_correct'].float()
    bc = state['routing_body_face_correct'].float()
    if bool(((counts < 0) | (hc < 0) | (bc < 0) | (hc > counts) | (bc > counts)).any()):
        raise ValueError('Invalid frozen calibration counts')
    ha = torch.where(counts > 0, hc / counts.clamp_min(1), torch.zeros_like(counts))
    ba = torch.where(counts > 0, bc / counts.clamp_min(1), torch.zeros_like(counts))
    w = torch.where(ha + ba > 0, ha / (ha + ba), torch.full_like(ha, 0.5))
    hlog, blog = F.log_softmax(hf, -1), F.log_softmax(bf, -1)
    arithmetic = torch.logaddexp(w.log() + hlog, torch.log1p(-w) + blog)
    arithmetic = arithmetic - torch.logsumexp(arithmetic, -1, keepdim=True)
    geo = w * hlog + (1 - w) * blog
    geo = geo - torch.logsumexp(geo, -1, keepdim=True)
    candidate = torch.logaddexp(arithmetic, geo) - math.log(2.0)
    if not torch.isfinite(arithmetic).all() or not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite fusion scores')
    if not torch.allclose(arithmetic.exp().sum(-1), torch.ones(len(labels)), atol=2e-6):
        raise ValueError('Arithmetic not normalized')
    if not torch.allclose(candidate.exp().sum(-1), torch.ones(len(labels)), atol=2e-6):
        raise ValueError('Candidate not normalized')
    pred_a, pred_c = arithmetic.argmax(-1), candidate.argmax(-1)
    changes = ((pred_c == labels).to(torch.int64) - (pred_a == labels).to(torch.int64)).numpy()
    unique, group = np.unique(subjects, return_inverse=True)
    if len(unique) != 20:
        raise ValueError('Expected 20 validation subjects')
    sizes = np.bincount(group)
    net = np.bincount(group, weights=changes, minlength=len(unique))
    rng = np.random.default_rng(20261106)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=20000)
    ci = np.percentile(100 * (draws @ net) / (draws @ sizes), [2.5, 97.5]).tolist()
    groups = []
    for index, subject in enumerate(unique):
        mask = group == index
        groups.append({'subject': int(subject), 'samples': int(mask.sum()),
            'baseline_correct': int((pred_a.numpy()[mask] == labels.numpy()[mask]).sum()),
            'candidate_correct': int((pred_c.numpy()[mask] == labels.numpy()[mask]).sum()),
            'gain': int((changes[mask] == 1).sum()), 'loss': int((changes[mask] == -1).sum()),
            'difference_pp': float(100 * changes[mask].mean())})
    summary = {'checkpoint_sha256': CHECKPOINT_HASH, 'validation_cache_sha256': sha(args.cache),
        'validation_members_sha256': member_hash, 'samples': len(labels), 'subjects': len(unique),
        'fusion_weights': 'Frozen checkpoint calibration counts; gamma=1; no refit',
        'candidate': '50% normalized arithmetic plus 50% normalized weighted geometric pool',
        'baseline_correct': int((pred_a == labels).sum()), 'candidate_correct': int((pred_c == labels).sum()),
        'baseline_top1_percent': float(100 * (pred_a == labels).float().mean()),
        'candidate_top1_percent': float(100 * (pred_c == labels).float().mean()),
        'difference_pp': float(100 * changes.mean()), 'gain': int((changes == 1).sum()),
        'loss': int((changes == -1).sum()), 'subject_cluster_bootstrap_95_ci_pp': ci,
        'bootstrap_replicates': 20000, 'bootstrap_seed': 20261106,
        'baseline_nll': float(F.nll_loss(arithmetic, labels)), 'candidate_nll': float(F.nll_loss(candidate, labels)),
        'numerics': 'log-softmax and logaddexp for arithmetic and final mixture; weighted log-softmax for geometric; no probability clipping',
        'low_probability': {'minimum_head_log_probability': float(torch.minimum(hlog.min(), blog.min())),
            'minimum_arithmetic_log_probability': float(arithmetic.min()), 'minimum_candidate_log_probability': float(candidate.min()),
            'zero_hf_weights': int((w == 0).sum()), 'unit_hf_weights': int((w == 1).sum())},
        'limitations': 'Exploratory 50% mixture was chosen after screening two calibration subjects; '
            'the same validation set has been used in earlier investigations. Cache logits may differ from fresh GPU inference.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'predictions.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample', 'subject', 'label', 'baseline_prediction', 'candidate_prediction'])
        writer.writerows(zip(members, subjects, labels.tolist(), pred_a.tolist(), pred_c.tolist()))
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (args.output_dir / 'subjects.json').write_text(json.dumps(groups, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
