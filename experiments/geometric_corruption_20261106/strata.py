"""Summarize paired fusion predictions by subject and action."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--corruption-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = {'clean': args.clean, **{name: args.corruption_dir / f'{name}.csv' for name in
        ('point_0.05', 'point_0.1', 'group_0.05', 'group_0.1')}}
    result, identity = {}, None
    for name, path in files.items():
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        current = [(row['sample'], row['subject'], row['label']) for row in rows]
        if len(rows) != 16306 or len(set(current)) != 16306 or (identity is not None and identity != current):
            raise ValueError('Sample identities differ')
        identity = current
        subjects = np.array([int(row['subject']) for row in rows])
        classes = np.array([int(row['label']) for row in rows])
        truth = classes
        old = np.array([int(row['baseline_prediction']) for row in rows])
        new = np.array([int(row['candidate_prediction']) for row in rows])
        changes = (new == truth).astype(int) - (old == truth).astype(int)

        def summarize(mask):
            delta = changes[mask]
            return {'n': int(mask.sum()), 'gain': int((delta == 1).sum()), 'loss': int((delta == -1).sum()),
                    'difference_pp': float(100 * delta.mean())}

        sub = {str(i): summarize(subjects == i) for i in sorted(set(subjects))}
        cat = {f'A{i+1:03d}': summarize(classes == i) for i in sorted(set(classes))}
        result[name] = {'total': summarize(np.ones(len(rows), dtype=bool)),
            'excluding_P007': summarize(subjects != 7),
            'subject_direction_counts': {key: sum((x['difference_pp'] > 0 if key == 'positive' else
                x['difference_pp'] < 0 if key == 'negative' else x['difference_pp'] == 0) for x in sub.values())
                for key in ('positive', 'negative', 'zero')},
            'subject_macro_mean_pp': float(np.mean([x['difference_pp'] for x in sub.values()])),
            'subjects': sub, 'classes': cat,
            'worst_classes': sorted(({'class': key, **value} for key, value in cat.items()),
                                    key=lambda item: item['difference_pp'])[:5]}
    payload = {'conditions': result, 'limitations': 'Post-hoc descriptions, not multiple-testing-adjusted findings; '
        'same official validation set reused; synthetic corruption.'}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, value in result.items():
        print(json.dumps({'condition': name, 'total': value['total'], 'excluding_P007': value['excluding_P007'],
            'subject_macro_mean_pp': value['subject_macro_mean_pp'],
            'subject_direction_counts': value['subject_direction_counts'], 'worst_classes': value['worst_classes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
