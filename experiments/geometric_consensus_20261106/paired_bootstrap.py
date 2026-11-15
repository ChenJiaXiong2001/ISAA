"""Compute subject-stratified sample bootstrap from fixed paired counts."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.summary.read_text(encoding='utf-8'))
    rng = np.random.default_rng(20261106)
    records = []
    for strength in report['strengths']:
        folds = [x for x in report['folds'] if x['strength'] == strength]
        if {x['heldout_subject'] for x in folds} != {9, 18}:
            raise ValueError('Unexpected folds')
        n = sum(x['samples'] for x in folds)
        gain, loss = sum(x['gain'] for x in folds), sum(x['loss'] for x in folds)
        difference = np.zeros(20000, dtype=np.int64)
        for x in folds:
            count = x['samples']
            draw = rng.multinomial(count, [x['gain'] / count, x['loss'] / count,
                                           (count - x['gain'] - x['loss']) / count], size=20000)
            difference += draw[:, 0] - draw[:, 1]
        records.append({'strength': strength, 'n': n, 'gain': gain, 'loss': loss,
            'difference_pp': 100 * (gain - loss) / n,
            'subject_stratified_sample_bootstrap_95_ci_pp': np.percentile(100 * difference / n, [2.5, 97.5]).tolist()})
    result = {'source': str(args.summary), 'replicates': 20000, 'seed': 20261106, 'results': records,
              'limitation': 'Fixed two subjects; sample resampling does not quantify new-subject uncertainty. '
                            'Multiple strengths were explored on the same data.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
