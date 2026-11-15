"""Pair clean and corrupted predictions to compare degradation from clean."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 16306 or len({row['sample'] for row in rows}) != len(rows):
        raise ValueError('Expected 16,306 unique members')
    return rows


def delta(rows):
    return np.array([(int(row['candidate_prediction']) == int(row['label'])) -
                     (int(row['baseline_prediction']) == int(row['label'])) for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--corruption-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    clean = load(args.clean)
    identity = [(row['sample'], row['subject'], row['label']) for row in clean]
    subjects = np.array([int(row['subject']) for row in clean])
    unique, groups = np.unique(subjects, return_inverse=True)
    if len(unique) != 20:
        raise ValueError('Expected 20 subjects')
    rng = np.random.default_rng(20261106)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=20000)
    sizes = np.bincount(groups)
    clean_delta = delta(clean)
    results = []
    for condition in ('point_0.05', 'point_0.1', 'group_0.05', 'group_0.1'):
        path = args.corruption_dir / f'{condition}.csv'
        corrupted = load(path)
        if [(row['sample'], row['subject'], row['label']) for row in corrupted] != identity:
            raise ValueError(f'Member or label mismatch for {condition}')
        corrupted_delta = delta(corrupted)
        contrast = corrupted_delta - clean_delta
        totals = np.bincount(groups, weights=contrast)
        ci = np.percentile(100 * (draws @ totals) / (draws @ sizes), [2.5, 97.5]).tolist()
        results.append({'condition': condition, 'difference_in_accuracy_gains_pp': float(100 * contrast.mean()),
            'drop_reduction_pp': float(100 * contrast.mean()), 'subject_bootstrap_95_ci_pp': ci,
            'baseline_clean_accuracy': float(100 * np.mean([int(row['baseline_prediction']) == int(row['label']) for row in clean])),
            'candidate_clean_accuracy': float(100 * np.mean([int(row['candidate_prediction']) == int(row['label']) for row in clean]))})
    result = {'method': '(candidate_corrupted − baseline_corrupted) − (candidate_clean − baseline_clean); positive means less degradation from clean',
        'clean_prediction_sha256': hashlib.sha256(args.clean.read_bytes()).hexdigest(),
        'conditions': results, 'limitations': 'Clean and corruption use separate inference passes; model and members are fixed. '
        'Synthetic corruptions and repeated validation-set use limit interpretation.'}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
