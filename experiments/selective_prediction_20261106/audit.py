"""Independently recompute fusion, thresholds, and CSV aggregates on CPU."""
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--calibration-cache', type=Path, required=True)
parser.add_argument('--validation-cache', type=Path, required=True)
parser.add_argument('--result-dir', type=Path, required=True)
options = parser.parse_args()
cal = torch.load(options.calibration_cache, map_location='cpu', weights_only=True)
val = torch.load(options.validation_cache, map_location='cpu', weights_only=True)
summary = json.loads((options.result_dir / 'summary.json').read_text())
rows = list(csv.DictReader((options.result_dir / 'predictions.csv').open(newline='', encoding='utf-8')))
labels = cal['labels'].numpy()
hands, bodies = cal['hf'].numpy(), cal['bf'].numpy()
weights = np.array([((hands.argmax(1) == labels) & (labels == i)).sum() /
                    max(1, (((hands.argmax(1) == labels) & (labels == i)).sum()
                           + ((bodies.argmax(1) == labels) & (labels == i)).sum()))
                    for i in range(hands.shape[1])])
if not np.all(np.bincount(labels, minlength=len(weights)) > 0):
    raise ValueError('This independent calculation expects all classes represented')


def calc(cache):
    def softmax(logits):
        logits = logits.numpy().astype(np.float64)
        values = np.exp(logits - logits.max(axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)
    h, b = softmax(cache['hf']), softmax(cache['bf'])
    ar = weights * h + (1 - weights) * b
    ar /= ar.sum(axis=1, keepdims=True)
    geo = np.exp(weights * np.log(h) + (1 - weights) * np.log(b))
    geo /= geo.sum(axis=1, keepdims=True)
    result = (ar + geo) / 2
    return result.argmax(axis=1), result.max(axis=1)


_, cal_conf = calc(cal)
pred, confidence = calc(val)
truth = val['labels'].numpy()
assert len(rows) == len(truth) == 16306
assert np.array_equal(pred, np.array([int(row['prediction']) for row in rows]))
assert np.array_equal(truth, np.array([int(row['label']) for row in rows]))
assert np.max(np.abs(confidence - np.array([float(row['confidence']) for row in rows]))) < 2e-6
for q, point in zip((0.05, 0.10), summary['points']):
    threshold = float(np.quantile(cal_conf, q))
    assert abs(threshold - point['threshold']) < 2e-6
    accepted = confidence >= threshold
    correct = pred == truth
    assert int(accepted.sum()) == point['accepted']
    assert int((accepted & correct).sum()) == point['accepted_correct']
    assert int((~accepted & ~correct).sum()) == point['rejected_errors']
    for group, field, identifier in ((point['subjects'], 'subject', 'subject'),
                                     (point['classes'], 'action', 'label')):
        for item in group:
            match = (np.array([int(row[identifier]) for row in rows]) ==
                     (item[field] if field == 'subject' else int(item[field][1:]) - 1))
            assert int(match.sum()) == item['samples']
            assert int((match & accepted).sum()) == item['accepted']
            assert int((match & accepted & correct).sum()) == item['accepted_correct']
print('PASS: independent float64 probability fusion, two quantiles, all predictions, '
      'all 20 subjects, all 60 classes, and aggregate counts')
