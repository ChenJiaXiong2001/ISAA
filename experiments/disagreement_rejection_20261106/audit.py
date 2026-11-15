"""Independent torch-space audit of checkpoint weights and rejection scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration-cache', type=Path, required=True)
    p.add_argument('--validation-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    o = p.parse_args()
    if o.output.exists():
        raise FileExistsError(o.output)
    torch.set_num_threads(1)
    cal = torch.load(o.calibration_cache, map_location='cpu', weights_only=True)
    val = torch.load(o.validation_cache, map_location='cpu', weights_only=True)
    checkpoint = torch.load(o.checkpoint, map_location='cpu', weights_only=True)
    if sha(o.checkpoint) != cal['checkpoint_sha256'] or sha(o.checkpoint) != val['checkpoint_sha256']:
        raise ValueError('Checkpoint digest differs from cache')
    state = checkpoint['model']
    total = state['routing_class_counts'].double()
    hf = state['routing_hand_face_correct'].double()
    bf = state['routing_body_face_correct'].double()
    saved = torch.where(hf + bf > 0, hf / (hf + bf), 0.5)
    y = cal['labels'].long()
    nclass = cal['hf'].shape[1]
    cache_h = torch.bincount(y[cal['hf'].argmax(-1) == y], minlength=nclass).double()
    cache_b = torch.bincount(y[cal['bf'].argmax(-1) == y], minlength=nclass).double()
    cache_n = torch.bincount(y, minlength=nclass).double()
    cache_weights = torch.where(cache_h + cache_b > 0, cache_h / (cache_h + cache_b), 0.5)
    weights = saved

    def scores(cache):
        h = F.log_softmax(cache['hf'].double(), -1)
        b = F.log_softmax(cache['bf'].double(), -1)
        ar = torch.logaddexp(weights.log() + h, torch.log1p(-weights) + b)
        ar -= torch.logsumexp(ar, -1, keepdim=True)
        geo = weights * h + (1 - weights) * b
        geo -= torch.logsumexp(geo, -1, keepdim=True)
        mixed = (ar.exp() + geo.exp()) / 2
        mid = (h.exp() + b.exp()) / 2
        js = ((torch.special.entr(mid).sum(-1) -
               (torch.special.entr(h.exp()).sum(-1) + torch.special.entr(b.exp()).sum(-1)) / 2) / np.log(2))
        conf, pred = mixed.max(-1)
        return pred.numpy(), conf.numpy(), (conf * (1 - js)).numpy()

    _, cal_conf, cal_score = scores(cal)
    pred, conf, candidate = scores(val)
    summary = json.loads((o.result_dir / 'summary.json').read_text())
    rows = list(csv.DictReader((o.result_dir / 'predictions.csv').open(encoding='utf-8', newline='')))
    yval = val['labels'].numpy()
    subjects = np.array([int(row['subject']) for row in rows])
    assert np.array_equal(pred, np.array([int(row['prediction']) for row in rows]))
    assert np.max(np.abs(conf - np.array([float(row['confidence']) for row in rows]))) < 1e-12
    assert np.max(np.abs(candidate - np.array([float(row['candidate_score']) for row in rows]))) < 1e-12
    for q, point in zip((0.05, 0.10), summary['points']):
        assert abs(float(np.quantile(cal_conf, q)) - point['base_threshold']) < 1e-12
        assert abs(float(np.quantile(cal_score, q)) - point['candidate_threshold']) < 1e-12
        base = conf >= point['base_threshold']
        candidate_fixed = candidate >= point['candidate_threshold']
        matched = np.ones(len(yval), dtype=bool)
        matched[np.argsort(candidate, kind='stable')[:int((~base).sum())]] = False
        err = pred != yval
        assert int((~base & err).sum()) == point['base']['rejected_errors']
        assert int((~candidate_fixed & err).sum()) == point['fixed_candidate']['rejected_errors']
        assert int((~matched & err).sum()) == point['matched_candidate']['rejected_errors']
        difference = (~matched & err).astype(int) - (~base & err).astype(int)
        for item in point['subjects']:
            assert int(difference[subjects == item['subject']].sum()) == item['net_additional_errors_rejected']
        for item in point['classes']:
            assert int(difference[yval == int(item['action'][1:]) - 1].sum()) == item['net_additional_errors_rejected']
    out = dict(checkpoint_sha256=sha(o.checkpoint), cache_class_counts_match=bool(torch.equal(cache_n, total)),
        cache_head_correct_counts_match=dict(hf=bool(torch.equal(cache_h, hf)), bf=bool(torch.equal(cache_b, bf))),
        weights_exact_match=bool(torch.equal(cache_weights, saved)),
        num_weight_differences=int((cache_weights != saved).sum()), max_abs_weight_diff=float((cache_weights - saved).abs().max()),
        tested_weight_source='epoch-83 checkpoint routing counts',
        differing_classes=[dict(action=f'A{i+1:03d}', cache_weight=float(cache_weights[i]), checkpoint_weight=float(saved[i]),
                                cache_hf_correct=int(cache_h[i]), cache_bf_correct=int(cache_b[i]),
                                saved_hf_correct=int(hf[i]), saved_bf_correct=int(bf[i]))
                           for i in range(nclass) if cache_weights[i] != saved[i]],
        independent_recalculation='PASS: all predictions and scores via torch log-softmax and entropy, both thresholds and 20 subjects / 60 classes')
    o.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
