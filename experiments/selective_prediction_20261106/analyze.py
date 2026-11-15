"""Evaluate fixed confidence-quantile rejection on cached logits, CPU only."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(accept, correct):
    n = len(correct)
    accepted = int(accept.sum())
    errors = ~correct
    caught = int((~accept & errors).sum())
    return dict(samples=n, accepted=accepted, coverage_pct=100 * accepted / n,
                accepted_correct=int((accept & correct).sum()),
                accepted_accuracy_pct=100 * (accept & correct).sum() / accepted if accepted else None,
                accepted_errors=int((accept & errors).sum()),
                rejected_correct=int((~accept & correct).sum()), rejected_errors=caught,
                all_errors=int(errors.sum()),
                errors_rejected_pct=100 * caught / errors.sum() if errors.sum() else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration-cache', type=Path, required=True)
    p.add_argument('--validation-cache', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    o = p.parse_args()
    if o.output_dir.exists():
        raise FileExistsError(o.output_dir)
    torch.set_num_threads(1)
    cal = torch.load(o.calibration_cache, map_location='cpu', weights_only=True)
    val = torch.load(o.validation_cache, map_location='cpu', weights_only=True)
    manifest = json.loads(o.manifest.read_text(encoding='utf-8'))['manifest']
    expected = 'aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7'
    if cal['checkpoint_sha256'] != expected or val['checkpoint_sha256'] != expected:
        raise ValueError('wrong checkpoint')
    for cache, name in ((cal, 'calibration'), (val, 'validation')):
        members = manifest[name]['members']
        if hashlib.sha256('\n'.join(members).encode()).hexdigest() != cache['member_sha256']:
            raise ValueError('manifest mismatch: ' + name)
        if len(members) != len(cache['labels']) or not torch.isfinite(cache['hf']).all() or not torch.isfinite(cache['bf']).all():
            raise ValueError('invalid cache: ' + name)
    y = cal['labels'].long()
    classes = cal['hf'].shape[1]
    count = torch.bincount(y, minlength=classes)
    h = torch.bincount(y[cal['hf'].argmax(-1) == y], minlength=classes).float()
    b = torch.bincount(y[cal['bf'].argmax(-1) == y], minlength=classes).float()
    ha, ba = h / count.clamp_min(1), b / count.clamp_min(1)
    w = torch.where(ha + ba > 0, ha / (ha + ba), 0.5)

    def fused(cache):
        hlog = torch.log_softmax(cache['hf'].float(), -1)
        blog = torch.log_softmax(cache['bf'].float(), -1)
        ar = torch.logaddexp(w.log() + hlog, torch.log1p(-w) + blog)
        ar = ar - torch.logsumexp(ar, -1, keepdim=True)
        geo = w * hlog + (1 - w) * blog
        geo = geo - torch.logsumexp(geo, -1, keepdim=True)
        return torch.logaddexp(ar, geo) - np.log(2.0)

    c, v = fused(cal), fused(val)
    if not torch.isfinite(c).all() or not torch.isfinite(v).all():
        raise ValueError('nonfinite fused probabilities')
    c_scores = c.exp().max(-1).values.numpy()
    scores, preds = v.exp().max(-1)
    scores, preds = scores.numpy(), preds.numpy()
    labels = val['labels'].numpy()
    correct = preds == labels
    members = manifest['validation']['members']
    subjects = np.array([int(re.search(r'P(\d{3})', item).group(1)) for item in members])
    unique, ids = np.unique(subjects, return_inverse=True)
    if len(unique) != 20:
        raise ValueError('expected 20 validation subjects')
    rng = np.random.default_rng(20261106)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=20000)
    sizes = np.bincount(ids)
    result = dict(checkpoint_sha256=expected, calibration_sha256=sha(o.calibration_cache),
                  validation_sha256=sha(o.validation_cache), manifest_sha256=sha(o.manifest),
                  all_samples=int(len(labels)), all_correct=int(correct.sum()),
                  all_accuracy_pct=float(100 * correct.mean()), calibration_samples=len(y),
                  bootstrap='20000 draws of 20 validation subjects, fixed threshold; seed 20261106',
                  limitations='cached CUDA logits; two calibration subjects reused to fit weights; validation reused; not blind; rejection needs fallback',
                  points=[])
    records = []
    for q in (0.05, 0.10):
        threshold = float(np.quantile(c_scores, q))
        accept = scores >= threshold
        point = stats(accept, correct)
        point.update(calibration_rejection_quantile=q, threshold=threshold,
                     calibration_coverage_pct=float(100 * (c_scores >= threshold).mean()))
        accepted = np.bincount(ids, weights=accept)
        acorrect = np.bincount(ids, weights=accept & correct)
        aggregate_n = draws @ sizes
        aggregate_accept = draws @ accepted
        point['subject_bootstrap_coverage_95_ci_pct'] = np.percentile(100 * aggregate_accept / aggregate_n, [2.5, 97.5]).tolist()
        point['subject_bootstrap_accepted_accuracy_95_ci_pct'] = np.percentile(100 * (draws @ acorrect) / aggregate_accept, [2.5, 97.5]).tolist()
        point['subjects'] = [dict(subject=int(s), **stats(accept[ids == i], correct[ids == i])) for i, s in enumerate(unique)]
        point['macro_subject_coverage_pct'] = float(np.mean([r['coverage_pct'] for r in point['subjects']]))
        point['macro_subject_accepted_accuracy_pct'] = float(np.mean([r['accepted_accuracy_pct'] for r in point['subjects']]))
        point['classes'] = [dict(action=f'A{k + 1:03d}', **stats(accept[labels == k], correct[labels == k])) for k in range(classes)]
        result['points'].append(point)
        records.append(accept)
    o.output_dir.mkdir(parents=True, exist_ok=False)
    (o.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (o.output_dir / 'predictions.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample', 'subject', 'label', 'prediction', 'confidence', 'accept_q05', 'accept_q10'])
        writer.writerows(zip(members, subjects, labels, preds, scores, *records))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
