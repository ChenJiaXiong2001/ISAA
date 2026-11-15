"""Summarize paired prediction changes by true action."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    grouped = defaultdict(lambda: {'samples': 0, 'gain': 0, 'loss': 0})
    with args.predictions.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            label = int(row['label'])
            baseline = int(row['baseline_prediction']) == label
            candidate = int(row['candidate_prediction']) == label
            grouped[label]['samples'] += 1
            grouped[label]['gain'] += int(candidate and not baseline)
            grouped[label]['loss'] += int(baseline and not candidate)
    if len(grouped) != 60 or sum(x['samples'] for x in grouped.values()) != 16306:
        raise ValueError('Expected full 60-class validation set')
    rows = [{'action': f'A{label + 1:03}', **grouped[label],
             'difference_pp': 100 * (grouped[label]['gain'] - grouped[label]['loss']) / grouped[label]['samples']}
            for label in sorted(grouped)]
    result = {'classes': rows, 'exploratory': True,
              'note': 'Per-class comparisons are descriptive and not multiplicity-adjusted.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Negative:', [x for x in rows if x['gain'] < x['loss']])
    print('Largest positive:', sorted(rows, key=lambda x: -x['difference_pp'])[:5])


if __name__ == '__main__':
    main()
