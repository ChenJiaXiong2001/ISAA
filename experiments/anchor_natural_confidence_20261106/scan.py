"""Scan natural calibration shoulder/hip confidence and repair perturbation."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress
from tools.evaluate_face_pair_anchor_confidence_repair import apply_score_threshold, interpolate_short_anchor_gaps

ANCHORS = [5, 6, 11, 12]
THRESHOLDS = (0.5, 0.9)


def center_and_scale(features, frames):
    anchors = features[0, :, :, ANCHORS, :]
    valid = (anchors[2] > 0) & frames[:, None, None]
    xy = anchors[:2]
    center = (xy * valid[None]).sum(2) / valid.sum(1).clamp_min(1)[None]
    shoulder_valid, hip_valid = valid[:, :2], valid[:, 2:]
    shoulder = (xy[:, :, :2] * shoulder_valid[None]).sum(2) / shoulder_valid.sum(1).clamp_min(1)[None]
    hip = (xy[:, :, 2:] * hip_valid[None]).sum(2) / hip_valid.sum(1).clamp_min(1)[None]
    scale = torch.linalg.vector_norm(shoulder - hip, dim=0).clamp_min(1e-3)
    comparable = shoulder_valid.any(1) & hip_valid.any(1) & frames[:, None]
    return center, scale, comparable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))['calibration']
    members = manifest['members']
    digest = hashlib.sha256('\n'.join(members).encode()).hexdigest()
    if digest != manifest['sha256'] or len(members) != 3191 or len(set(members)) != 3191:
        raise ValueError('calibration manifest mismatch')
    data = RTMWZipDataset(args.archive, split='train', split_protocol='xsub60', window_size=100,
        num_joints=133, num_classes=60, layout='rtmw_133', max_persons=2,
        feature_mode='raw', augment=False, temporal_compression='confidence_weighted')
    index = {name: i for i, name in enumerate(data.members)}
    if len(index) != len(data.members) or not set(members).issubset(index):
        raise ValueError('dataset membership mismatch')
    rows = []
    center_changes = {str(t): [] for t in THRESHOLDS}
    scale_changes = {str(t): [] for t in THRESHOLDS}
    for n, member in enumerate(members, 1):
        raw, frame = data._read_raw_sample(index[member])
        compressed, compressed_frame = _confidence_weighted_temporal_compress(raw, frame, 100)
        subject = int(re.search(r'P(\d{3})', member).group(1))
        row = dict(sample=member, subject=subject, raw_frames=int(raw.shape[1]), compressed=bool(raw.shape[1] > 100))
        for stage, x, mask in (('raw', raw, frame), ('compressed', compressed, compressed_frame)):
            score = x[2, :, ANCHORS, :]
            people = (score > 0).any(1) & mask[:, None]
            valid = people[:, None, :].expand_as(score)
            row[stage + '_person_frames'] = int(people.sum())
            row[stage + '_positive'] = int(((score > 0) & valid).sum())
            row[stage + '_zero'] = int(((score == 0) & valid).sum())
            for t in THRESHOLDS:
                row[f'{stage}_low_{t}'] = int(((score > 0) & (score <= t) & valid).sum())
        clean = compressed.clone()
        clean[:2] *= clean[2:3] > 0
        clean.masked_fill_(~compressed_frame[None, :, None, None], 0)
        original = clean[None]
        base, _ = interpolate_short_anchor_gaps(original, compressed_frame[None], 5)
        center0, scale0, comparable0 = center_and_scale(base, compressed_frame)
        for t in THRESHOLDS:
            low = (original[0, 2, :, ANCHORS, :] > 0) & (original[0, 2, :, ANCHORS, :] <= t) & compressed_frame[:, None, None]
            if not low.any():
                row.update({f'restored_{t}': 0, f'removed_{t}': 0, f'center_changed_{t}': 0,
                            f'scale_changed_{t}': 0})
                continue
            thresholded, changed = apply_score_threshold(original, t)
            if changed != int(low.sum()):
                raise ValueError('threshold count mismatch')
            repaired, _ = interpolate_short_anchor_gaps(thresholded, compressed_frame[None], 5)
            new_score = repaired[0, 2, :, ANCHORS, :]
            row[f'restored_{t}'] = int((low & (new_score > 0)).sum())
            row[f'removed_{t}'] = int((low & (new_score <= 0)).sum())
            center1, scale1, comparable1 = center_and_scale(repaired, compressed_frame)
            comparable = comparable0 & comparable1
            center_delta = torch.linalg.vector_norm(center1 - center0, dim=0)
            shift = center_delta > 1e-7
            scale_delta = (scale1 - scale0).abs()
            scale_shift = comparable & (scale_delta > 1e-7)
            row[f'center_changed_{t}'] = int(shift.sum())
            row[f'scale_changed_{t}'] = int(scale_shift.sum())
            if bool((shift & comparable).any()):
                center_changes[str(t)].extend((center_delta[shift & comparable] / scale0[shift & comparable]).tolist())
            if bool(scale_shift.any()):
                scale_changes[str(t)].extend((scale_delta[scale_shift] / scale0[scale_shift]).tolist())
        rows.append(row)
        if n == 1 or n % 100 == 0 or n == len(members):
            print(f'processed={n}/{len(members)}', flush=True)
    summary = dict(members_sha256=digest, samples=len(rows), subjects=sorted(set(r['subject'] for r in rows)),
        compressed_clips=sum(r['compressed'] for r in rows),
        definitions=dict(low='0 < normalized score <= fixed threshold',
                         valid_person_frame='valid frame with one or more positive shoulder/hip anchors',
                         changed='absolute difference > 1e-7; scale requires shoulder and hip available in both cases'),
        totals={}, subjects_detail=[], percentiles={})
    keys = [k for k in rows[0] if k not in ('sample', 'subject', 'raw_frames', 'compressed')]
    summary['totals'] = {k: sum(r[k] for r in rows) for k in keys}
    for stage in ('raw', 'compressed'):
        for t in THRESHOLDS:
            key = f'{stage}_low_{t}'
            summary['totals'][key + '_clips'] = sum(r[key] > 0 for r in rows)
    for t in THRESHOLDS:
        summary['totals'][f'center_changed_{t}_clips'] = sum(r[f'center_changed_{t}'] > 0 for r in rows)
        summary['totals'][f'scale_changed_{t}_clips'] = sum(r[f'scale_changed_{t}'] > 0 for r in rows)
        summary['percentiles'][str(t)] = {}
        for label, parts in (('center_shift_over_original_scale', center_changes[str(t)]),
                             ('absolute_scale_change_over_original_scale', scale_changes[str(t)])):
            summary['percentiles'][str(t)][label] = dict(n=len(parts),
                quantiles=torch.quantile(torch.tensor(parts), torch.tensor([0.5, 0.9, 0.99])).tolist() if parts else None)
    for subject in summary['subjects']:
        subset = [r for r in rows if r['subject'] == subject]
        entry = dict(subject=subject, samples=len(subset))
        for k in keys:
            entry[k] = sum(r[k] for r in subset)
        for t in THRESHOLDS:
            entry[f'compressed_low_{t}_clips'] = sum(r[f'compressed_low_{t}'] > 0 for r in subset)
        summary['subjects_detail'].append(entry)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'per_sample.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
