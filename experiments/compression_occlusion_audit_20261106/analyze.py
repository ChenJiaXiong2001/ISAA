"""Join existing calibration metadata and quantify occlusion/compression strata."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lengths', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with args.lengths.open(encoding='utf-8', newline='') as f:
        lengths = list(csv.DictReader(f))
    with args.predictions.open(encoding='utf-8', newline='') as f:
        predictions = list(csv.DictReader(f))
    if len(lengths) != 3191 or len(predictions) != 3191:
        raise ValueError('expected full calibration partition')
    if [r['sample'] for r in lengths] != [r['sample'] for r in predictions]:
        raise ValueError('member order differs')
    members_sha = hashlib.sha256('\n'.join(r['sample'] for r in lengths).encode()).hexdigest()
    if members_sha != '68354a1f6bb0f6046b7ea720a07db89513e29ff8b7717ad36ecb028745d7cf11':
        raise ValueError('unexpected manifest')
    rows = []
    for length, pred in zip(lengths, predictions):
        frames = int(length['raw_frames'])
        start = int(pred['start'])
        if pred['eligible'] != 'True' or start < 0 or start + 5 > frames:
            raise ValueError('invalid eligible occlusion window')
        n = min(frames, 100)
        bins = [(i * frames // n, (i + 1) * frames // n) for i in range(n)]
        if frames <= 100:
            bins = [(i, i + 1) for i in range(frames)]
        overlap = [max(0, min(b, start + 5) - max(a, start)) for a, b in bins]
        affected = [k for k in range(n) if overlap[k] > 0]
        whole = [k for k in affected if overlap[k] == bins[k][1] - bins[k][0]]
        if sum(overlap) != 5:
            raise ValueError('bin overlap does not cover five frames')
        row = dict(sample=length['sample'], subject=int(length['subject']), frames=frames,
            group='1_100' if frames <= 100 else '101_199' if frames < 200 else '200_plus',
            start=start, original_occlusion_percent=500 / frames,
            affected_bins=len(affected), fully_occluded_bins=len(whole),
            partially_occluded_bins=len(affected)-len(whole),
            output_frames=n, max_bin_length=max(b-a for a,b in bins),
            label=int(pred['label']))
        for condition in ('clean', 'occlusion'):
            for method in ('original_arithmetic', 'original_geometric_mix'):
                key = condition + '_' + method
                row[key] = int(pred[key])
        rows.append(row)
    results = []
    for group in ('1_100', '101_199', '200_plus'):
        for subject in ('all', 9, 18):
            subset = [r for r in rows if r['group'] == group and (subject == 'all' or r['subject'] == subject)]
            if not subset:
                continue
            n = len(subset)
            entry = dict(group=group, subject=subject, samples=n,
                frame_range=[min(r['frames'] for r in subset), max(r['frames'] for r in subset)],
                mean_original_occluded_percent=sum(r['original_occlusion_percent'] for r in subset)/n,
                samples_any_partial_bin=sum(r['partially_occluded_bins'] > 0 for r in subset),
                affected_bins=sum(r['affected_bins'] for r in subset),
                fully_occluded_bins=sum(r['fully_occluded_bins'] for r in subset),
                partially_occluded_bins=sum(r['partially_occluded_bins'] for r in subset),
                partial_bins_histogram=dict(sorted(Counter(r['partially_occluded_bins'] for r in subset).items())))
            for method in ('original_arithmetic', 'original_geometric_mix'):
                clean = [r['clean_' + method] == r['label'] for r in subset]
                damaged = [r['occlusion_' + method] == r['label'] for r in subset]
                entry[method] = dict(clean_correct=sum(clean), occluded_correct=sum(damaged),
                    clean_accuracy_percent=100*sum(clean)/n, occluded_accuracy_percent=100*sum(damaged)/n,
                    drop_pp=100*(sum(damaged)-sum(clean))/n,
                    corrected=sum(not a and b for a,b in zip(clean, damaged)),
                    new_errors=sum(a and not b for a,b in zip(clean, damaged)))
            results.append(entry)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'per_sample.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(samples=len(rows), members_sha256=members_sha, lengths_sha256=sha(args.lengths),
        predictions_sha256=sha(args.predictions), groups=results,
        caveat='Synthetic sample-selected raw occlusion, two training calibration subjects; no causal compression estimate; validation not used.')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
