"""Screen fixed head-weight shrinkage with leave-one-event-out calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if cache['checkpoint_sha256'] != checkpoint_hash or cache['partition'] != 'calibration' or cache['epoch'] != 20:
        raise ValueError('Unexpected cache provenance')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    members = cache['members']
    if hashlib.sha256('\n'.join(members).encode()).hexdigest() != cache['member_sha256']:
        raise ValueError('Invalid member hash')
    hf, bf, y = cache['hf'], cache['bf'], cache['labels'].long()
    if hf.shape != bf.shape or hf.shape != (len(members), 60) or len(set(members)) != len(members):
        raise ValueError('Unexpected cache shape or duplicated member')
    event_ids, subjects = [], []
    for member, label in zip(members, y.tolist()):
        match = re.search(r'(S\d{3})C\d{3}(P\d{3})(R\d{3})(A\d{3})', member)
        if not match or int(match.group(4)[1:]) != label + 1:
            raise ValueError('Invalid event or label')
        event_ids.append(''.join(match.groups()))
        subjects.append(match.group(2))
    if set(subjects) != {'P025'}:
        raise ValueError('Expected the documented single-subject calibration')
    hp, bp = F.softmax(hf, -1), F.softmax(bf, -1)
    hc, bc = hf.argmax(-1).eq(y), bf.argmax(-1).eq(y)
    alphas = (0., .5, 1., 2., 5.)
    predictions = torch.empty((len(alphas), len(y)), dtype=torch.long)
    weights = torch.empty((len(alphas), len(y), 60))
    for event in sorted(set(event_ids)):
        held = torch.tensor([item == event for item in event_ids])
        fit = ~held
        total = torch.bincount(y[fit], minlength=60).float()
        hcorrect = torch.bincount(y[fit & hc], minlength=60).float()
        bcorrect = torch.bincount(y[fit & bc], minlength=60).float()
        for index, alpha in enumerate(alphas):
            denominator = hcorrect + bcorrect + 2 * alpha
            w = torch.where(denominator > 0, (hcorrect + alpha) / denominator.clamp_min(1e-12),
                            torch.full_like(total, .5))
            scores = hp[held] * w + bp[held] * (1 - w)
            predictions[index, held] = scores.argmax(-1)
            weights[index, held] = w
    baseline = predictions[0].eq(y)
    unique, groups = np.unique(event_ids, return_inverse=True)
    sizes = np.bincount(groups)
    rng = np.random.default_rng(20261106)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=20000)
    results = []
    for alpha, pred in zip(alphas, predictions):
        correct = pred.eq(y)
        changes = (correct.long() - baseline.long()).numpy()
        group_delta = np.bincount(groups, weights=changes, minlength=len(unique))
        boot = 100 * (draws @ group_delta) / (draws @ sizes)
        fused = hp * weights[len(results)] + bp * (1 - weights[len(results)])
        fused = fused / fused.sum(-1, keepdim=True)
        results.append({'alpha': alpha, 'correct': int(correct.sum()), 'samples': len(y),
            'accuracy_percent': 100 * float(correct.float().mean()), 'nll': float(F.nll_loss(fused.log(), y)),
            'gain': int((changes == 1).sum()), 'loss': int((changes == -1).sum()),
            'difference_pp': 100 * float(changes.mean()), 'event_bootstrap_95_ci_pp': np.percentile(boot, [2.5, 97.5]).tolist()})
    state = checkpoint['model']
    count = state['routing_class_counts'].long()
    stored_h, stored_b = state['routing_hand_face_correct'].long(), state['routing_body_face_correct'].long()
    observed_count = torch.bincount(y, minlength=60)
    observed_h = torch.bincount(y[hc], minlength=60)
    observed_b = torch.bincount(y[bc], minlength=60)
    result = {'checkpoint_sha256': checkpoint_hash, 'cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        'member_sha256': cache['member_sha256'], 'samples': len(y), 'events': len(unique), 'subjects': sorted(set(subjects)),
        'method': 'Leave one S/P/R/A event out, including every camera view; fit all class weights on remaining calibration events. Symmetric equal head pseudocount alpha.',
        'results': results, 'checkpoint_calibration_count_differences': {'total': int((observed_count-count).abs().sum()),
            'hf': int((observed_h-stored_h).abs().sum()), 'bf': int((observed_b-stored_b).abs().sum())},
        'limitations': 'One subject and one setup only; model weights and frozen baseline were already trained/calibrated using this partition. '
            'Event bootstrap does not quantify across-subject generalization. Most classes have only two events and heldout classes have roughly one event for estimating their weights. '
            'Screened alphas share the same data; no official validation labels were read. CPU inference can differ numerically from GPU training.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
