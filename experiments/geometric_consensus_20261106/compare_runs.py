"""Compare paired fusion predictions across cached and CUDA inference runs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 16306 or len(set(x['sample'] for x in rows)) != len(rows):
        raise ValueError('Expected full unique validation predictions')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before, after = read(args.cache), read(args.cuda)
    if [x['sample'] for x in before] != [x['sample'] for x in after]:
        raise ValueError('Sample ordering differs')
    groups = defaultdict(lambda: {'samples': 0, 'gain': 0, 'loss': 0})
    differences = {'baseline': [], 'candidate': []}
    for x, y in zip(before, after):
        if x['label'] != y['label'] or x['subject'] != y['subject']:
            raise ValueError('Labels or subject differ')
        label = int(y['label'])
        subject = int(y['subject'])
        groups[subject]['samples'] += 1
        old_ok = int(y['baseline_prediction']) == label
        new_ok = int(y['candidate_prediction']) == label
        groups[subject]['gain'] += int(new_ok and not old_ok)
        groups[subject]['loss'] += int(old_ok and not new_ok)
        for name in ('baseline', 'candidate'):
            key = name + '_prediction'
            if x[key] != y[key]:
                differences[name].append({'sample': x['sample'], 'label': label,
                    'cache_prediction': int(x[key]), 'cuda_prediction': int(y[key]),
                    'cache_correct': int(x[key]) == label, 'cuda_correct': int(y[key]) == label})
    if set(groups) != {3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40}:
        raise ValueError('Unexpected subject identifiers')
    subjects = []
    for subject, counts in sorted(groups.items()):
        subjects.append({'subject': subject, **counts,
            'difference_pp': 100 * (counts['gain'] - counts['loss']) / counts['samples']})
    others = [x for x in subjects if x['subject'] != 7]
    n = sum(x['samples'] for x in others)
    gain = sum(x['gain'] for x in others)
    loss = sum(x['loss'] for x in others)
    result = {'subjects': subjects, 'equal_subject_average_pp': sum(x['difference_pp'] for x in subjects) / len(subjects),
        'excluding_p007': {'samples': n, 'gain': gain, 'loss': loss, 'difference_pp': 100 * (gain - loss) / n},
        'run_differences': differences,
        'interpretation': 'Exploratory subgroup results, not independent confirmations or multiplicity-adjusted.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
