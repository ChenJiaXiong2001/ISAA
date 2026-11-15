"""Compare fixed disagreement-aware and maximum-probability rejection."""
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


def evaluate(cache, weights):
    h = torch.softmax(cache['hf'].double(), -1).numpy()
    b = torch.softmax(cache['bf'].double(), -1).numpy()
    ar = weights * h + (1 - weights) * b
    ar /= ar.sum(axis=1, keepdims=True)
    geo = np.exp(weights * np.log(h) + (1 - weights) * np.log(b))
    geo /= geo.sum(axis=1, keepdims=True)
    mixed = (ar + geo) / 2
    mid = (h + b) / 2
    js = (np.sum(h * np.log(h / mid), axis=1) + np.sum(b * np.log(b / mid), axis=1)) / (2 * np.log(2))
    confidence = mixed.max(axis=1)
    return mixed.argmax(axis=1), confidence, confidence * (1 - js), js, h.argmax(1) != b.argmax(1)


def metric(accepted, correct):
    return dict(accepted=int(accepted.sum()), rejected=int((~accepted).sum()),
        coverage_pct=float(100 * accepted.mean()),
        accepted_correct=int((accepted & correct).sum()),
        accepted_errors=int((accepted & ~correct).sum()),
        accepted_accuracy_pct=float(100 * (accepted & correct).sum() / accepted.sum()),
        rejected_errors=int((~accepted & ~correct).sum()),
        rejected_correct=int((~accepted & correct).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration-cache', type=Path, required=True)
    p.add_argument('--validation-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    o = p.parse_args()
    if o.output_dir.exists():
        raise FileExistsError(o.output_dir)
    torch.set_num_threads(1)
    cal = torch.load(o.calibration_cache, map_location='cpu', weights_only=True)
    val = torch.load(o.validation_cache, map_location='cpu', weights_only=True)
    manifest = json.loads(o.manifest.read_text(encoding='utf-8'))['manifest']
    checkpoint = 'aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7'
    if sha(o.checkpoint) != checkpoint:
        raise ValueError('checkpoint digest mismatch')
    for cache, part in ((cal, 'calibration'), (val, 'validation')):
        members = manifest[part]['members']
        expected_partition = 'calibration' if part == 'calibration' else 'val'
        if (cache['checkpoint_sha256'] != checkpoint or cache['epoch'] != 83 or
                cache['partition'] != expected_partition or
                hashlib.sha256('\n'.join(members).encode()).hexdigest() != cache['member_sha256']):
            raise ValueError('checkpoint or member mismatch')
        if len(members) != len(cache['labels']) or not torch.isfinite(cache['hf']).all() or not torch.isfinite(cache['bf']).all():
            raise ValueError('invalid cache')
        if any(int(re.search(r'A(\d{3})', member).group(1)) != int(label) + 1
               for member, label in zip(members, cache['labels'])):
            raise ValueError('manifest and cache labels mismatch: ' + part)
    saved = torch.load(o.checkpoint, map_location='cpu', weights_only=True)['model']
    nclass = cal['hf'].shape[1]
    hcorrect = saved['routing_hand_face_correct'].numpy()
    bcorrect = saved['routing_body_face_correct'].numpy()
    weights = np.divide(hcorrect, hcorrect + bcorrect, out=np.full(nclass, 0.5), where=hcorrect + bcorrect > 0)
    _, cconfidence, ccandidate, _, _ = evaluate(cal, weights)
    pred, confidence, candidate, js, disagree = evaluate(val, weights)
    truth = val['labels'].numpy()
    members = manifest['validation']['members']
    subjects = np.array([int(re.search(r'P(\d{3})', m).group(1)) for m in members])
    actions = np.array([int(re.search(r'A(\d{3})', m).group(1)) - 1 for m in members])
    if not np.array_equal(actions, truth):
        raise ValueError('labels do not match manifest')
    unique, groups = np.unique(subjects, return_inverse=True)
    if len(unique) != 20 or len(truth) != 16306:
        raise ValueError('unexpected validation size')
    correct = pred == truth
    draws = np.random.default_rng(20261106).multinomial(20, np.full(20, 0.05), size=20000)
    counts = np.bincount(groups)
    result = dict(checkpoint_sha256=checkpoint, calibration_sha256=sha(o.calibration_cache),
        validation_sha256=sha(o.validation_cache), manifest_sha256=sha(o.manifest),
        samples=len(truth), baseline_correct=int(correct.sum()), subjects=20, weight_source='epoch-83 checkpoint routing counts',
        definition='fused maximum probability * (1 - Jensen-Shannon divergence / ln(2))',
        agreement=dict(disagreement_samples=int(disagree.sum()),
                       errors_with_argmax_disagreement=int((disagree & ~correct).sum()),
                       errors_with_argmax_agreement=int((~disagree & ~correct).sum())), points=[])
    rows = [dict(sample=member, subject=int(subject), label=int(label), prediction=int(pr),
                 confidence=float(co), candidate_score=float(sc), js_divergence=float(j),
                 heads_disagree=bool(dis), correct=bool(ok))
            for member, subject, label, pr, co, sc, j, dis, ok in
            zip(members, subjects, truth, pred, confidence, candidate, js, disagree, correct)]
    for quantile in (0.05, 0.10):
        base_threshold = float(np.quantile(cconfidence, quantile))
        candidate_threshold = float(np.quantile(ccandidate, quantile))
        base = confidence >= base_threshold
        fixed = candidate >= candidate_threshold
        nreject = int((~base).sum())
        matched = np.ones(len(truth), dtype=bool)
        matched[np.argsort(candidate, kind='stable')[:nreject]] = False
        # Positive means candidate sends an additional erroneous prediction to review.
        diff = ((~matched & ~correct).astype(int) - (~base & ~correct).astype(int))
        net = np.bincount(groups, weights=diff, minlength=20)
        size = draws @ counts
        point = dict(calibration_quantile=quantile, base_threshold=base_threshold,
            candidate_threshold=candidate_threshold, base=metric(base, correct),
            fixed_candidate=metric(fixed, correct), matched_candidate=metric(matched, correct),
            matched_additional_errors_rejected=int(diff.sum()),
            base_accepted_errors_now_rejected=int((base & ~matched & ~correct).sum()),
            base_rejected_errors_now_accepted=int((~base & matched & ~correct).sum()),
            subject_bootstrap_additional_rejected_errors_pp_ci=np.percentile(100 * (draws @ net) / size, [2.5, 97.5]).tolist(),
            subjects=[dict(subject=int(s), samples=int(counts[i]), net_additional_errors_rejected=int(net[i]),
                           baseline=metric(base[groups == i], correct[groups == i]),
                           matched=metric(matched[groups == i], correct[groups == i])) for i,s in enumerate(unique)],
            classes=[dict(action=f'A{k+1:03d}', samples=int((truth == k).sum()),
                          net_additional_errors_rejected=int(diff[truth == k].sum())) for k in range(nclass)])
        result['points'].append(point)
        for i, row in enumerate(rows):
            key = f'q{int(quantile*100):02d}'
            row['baseline_accept_' + key] = bool(base[i])
            row['fixed_candidate_accept_' + key] = bool(fixed[i])
            row['matched_candidate_accept_' + key] = bool(matched[i])
    o.output_dir.mkdir(parents=True, exist_ok=False)
    (o.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (o.output_dir / 'predictions.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
