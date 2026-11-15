"""Compare the cached baseline against the original HF/BF/soft error export."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--errors', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.errors.open(encoding='utf-8-sig', newline='') as stream:
        original = {r['sample']: r for r in csv.DictReader(stream)}
    with args.predictions.open(encoding='utf-8-sig', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    if len(predictions) != 16306 or len(set(r['sample'] for r in predictions)) != 16306:
        raise ValueError('Expected 16306 unique predictions')
    if not set(original).issubset(r['sample'] for r in predictions):
        raise ValueError('Original errors contain unknown members')
    diffs = []
    for row in predictions:
        label, cached = int(row['label']), int(row['baseline_prediction'])
        old = original.get(row['sample'])
        previous = int(old['soft_predicted_action'][1:]) - 1 if old else label
        if old and int(old['true_action'][1:]) - 1 != label:
            raise ValueError('True label differs')
        if previous != cached:
            diffs.append({'sample': row['sample'], 'label': label, 'original_prediction': previous,
                          'cached_prediction': cached, 'original_correct': previous == label,
                          'cached_correct': cached == label})
    summary = {'samples': len(predictions), 'changed_predictions': len(diffs),
        'corrected': sum(not r['original_correct'] and r['cached_correct'] for r in diffs),
        'regressed': sum(r['original_correct'] and not r['cached_correct'] for r in diffs),
        'changes': diffs,
        'interpretation': 'Original export records all samples where at least one head or soft prediction was incorrect; omitted members are correct for all heads.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
