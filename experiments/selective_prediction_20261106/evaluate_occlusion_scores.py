"""Reuse fixed occlusion evaluation and save maximum fusion probabilities."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import evaluate_face_pair_correlated_occlusion as evaluator


def main():
    if '--output-dir' not in sys.argv:
        raise ValueError('An output directory is required')
    output_dir = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    captured = []
    original = evaluator.scores

    def recording_scores(output, weights):
        arithmetic, mixed = original(output, weights)
        score, prediction = mixed.exp().max(-1)
        captured.append((prediction.cpu().tolist(), score.cpu().tolist()))
        return arithmetic, mixed

    evaluator.scores = recording_scores
    evaluator.main()
    with (output_dir / 'predictions.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = ('clean_original', 'clean_max5', 'occlusion_original', 'occlusion_max5')
    values = {name: [] for name in names}
    if len(captured) % len(names):
        raise ValueError('Unexpected number of model calls')
    for i, (predictions, scores) in enumerate(captured):
        values[names[i % len(names)]].extend(zip(predictions, scores))
    if any(len(v) != len(rows) for v in values.values()):
        raise ValueError('Score count differs from predictions')
    for i, row in enumerate(rows):
        for name in names:
            prediction, score = values[name][i]
            if prediction != int(row[name + '_geometric_mix']):
                raise ValueError('Score and prediction order differ')
            row[name + '_confidence'] = score
    with (output_dir / 'confidence.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['sample', 'label'] + [name + '_confidence' for name in names]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
