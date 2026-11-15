"""CPU-only geometry proxy for fixed confidence-weighted temporal smoothing."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress


def smooth(x, frame):
    out = x.clone()
    valid = (x[2] > 0) & frame[:, None, None]
    eligible = valid[2:] & valid[1:-1] & valid[:-2]
    weight = torch.stack((x[2, :-2], 2*x[2, 1:-1], x[2, 2:]), dim=0)
    xy = torch.stack((x[:2, :-2], x[:2, 1:-1], x[:2, 2:]), dim=0)
    avg = (xy * weight[:, None]).sum(0) / weight.sum(0).clamp_min(1e-8)[None]
    out[:2, 1:-1] = torch.where(eligible[None], avg, x[:2, 1:-1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--occlusion-script', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('fixed_correlated_occlusion', args.occlusion_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))['calibration']
    full = manifest['members']
    if hashlib.sha256('\n'.join(full).encode()).hexdigest() != manifest['sha256'] or len(full) != 3191:
        raise ValueError('unexpected calibration manifest')
    members = [full[i * len(full) // 256] for i in range(256)]
    if len(set(members)) != 256:
        raise ValueError('unexpected calibration members')
    data = RTMWZipDataset(args.archive, split='train', split_protocol='xsub60', window_size=100,
        num_joints=133, num_classes=60, layout='rtmw_133', max_persons=2,
        feature_mode='raw', augment=False, temporal_compression='confidence_weighted')
    index = {name: i for i, name in enumerate(data.members)}
    groups = ('affected', 'other', 'affected_anchor', 'affected_body_other', 'affected_hand')
    rows, point_values = [], {name: [] for name in groups}
    for i, member in enumerate(members):
        raw, raw_mask = data._read_raw_sample(index[member])
        damaged, info = module.corrupt(raw, raw_mask, member)
        clean, frame = _confidence_weighted_temporal_compress(raw, raw_mask, 100)
        corrupt, frame2 = _confidence_weighted_temporal_compress(damaged, raw_mask, 100)
        if not torch.equal(frame, frame2) or not info['eligible']:
            raise ValueError('bad sample frame mask or occlusion')
        clean[:2] *= clean[2:3] > 0
        corrupt[:2] *= corrupt[2:3] > 0
        clean_smooth, corrupt_smooth = smooth(clean, frame), smooth(corrupt, frame)
        if not torch.equal(clean[2], clean_smooth[2]) or not torch.equal(corrupt[2], corrupt_smooth[2]):
            raise ValueError('scores changed')
        # Model uses equal-weight shoulder/hip centroids and requires both groups here.
        score = clean[2, :, [5,6,11,12], :]
        valid = (score > 0) & frame[:, None, None]
        xy = clean[:2, :, [5,6,11,12], :]
        shoulder = (xy[:, :, :2] * valid[None, :, :2]).sum(2) / valid[:, :2].sum(1).clamp_min(1)[None]
        hip = (xy[:, :, 2:] * valid[None, :, 2:]).sum(2) / valid[:, 2:].sum(1).clamp_min(1)[None]
        scale = torch.linalg.vector_norm(shoulder - hip, dim=0).clamp_min(1e-3)
        scale_valid = valid[:, :2].any(1) & valid[:, 2:].any(1)
        shared = (clean[2] > 0) & (corrupt[2] > 0) & frame[:, None, None] & scale_valid[:, None, :]
        targets = torch.zeros_like(shared)
        n = clean.shape[1]
        for t in range(n):
            lo = t * raw.shape[1] // n if raw.shape[1] > 100 else t
            hi = (t + 1) * raw.shape[1] // n if raw.shape[1] > 100 else t + 1
            if lo < info['start'] + 5 and hi > info['start']:
                targets[t, list(module.REGIONS[info['side']]), info['person']] = True
        d0 = torch.linalg.vector_norm(corrupt[:2] - clean[:2], dim=0) / scale[:, None, :]
        d1 = torch.linalg.vector_norm(corrupt_smooth[:2] - clean[:2], dim=0) / scale[:, None, :]
        dc = torch.linalg.vector_norm(clean_smooth[:2] - clean[:2], dim=0) / scale[:, None, :]
        row = {'sample': member, 'raw_frames': raw.shape[1]}
        anchor_nodes = torch.zeros_like(shared); anchor_nodes[:, [5, 6, 11, 12], :] = True
        hand_nodes = torch.zeros_like(shared); hand_nodes[:, 91:133, :] = True
        masks = (('affected', shared & targets), ('other', shared & ~targets),
                 ('affected_anchor', shared & targets & anchor_nodes),
                 ('affected_body_other', shared & targets & ~anchor_nodes & ~hand_nodes),
                 ('affected_hand', shared & targets & hand_nodes))
        for name, mask in masks:
            values = torch.stack((d0[mask], d1[mask], dc[mask]), dim=1)
            row[name + '_n'] = len(values)
            row[name + '_baseline_sum'] = float(values[:, 0].sum())
            row[name + '_candidate_sum'] = float(values[:, 1].sum())
            row[name + '_clean_smoothing_sum'] = float(values[:, 2].sum())
            row[name + '_improved'] = int((values[:, 1] < values[:, 0] - 1e-6).sum())
            row[name + '_worsened'] = int((values[:, 1] > values[:, 0] + 1e-6).sum())
            point_values[name].append(values.numpy())
        rows.append(row)
        if (i+1) % 50 == 0:
            print(f'processed={i+1}/{len(members)}', flush=True)
    summary = {'samples': len(rows), 'members_sha256': hashlib.sha256('\n'.join(members).encode()).hexdigest(),
        'occlusion_script_sha256': hashlib.sha256(args.occlusion_script.read_bytes()).hexdigest(), 'groups': {}}
    for name in groups:
        values = np.concatenate(point_values[name])
        summary['groups'][name] = {'point_frames': len(values),
            'clips_with_eligible_points': sum(r[name + '_n'] > 0 for r in rows),
            'baseline_mean_distance': float(values[:, 0].mean()) if len(values) else None,
            'candidate_mean_distance': float(values[:, 1].mean()) if len(values) else None,
            'clean_smoothing_mean_distance': float(values[:, 2].mean()) if len(values) else None,
            'candidate_closer': sum(r[name + '_improved'] for r in rows),
            'candidate_farther': sum(r[name + '_worsened'] for r in rows),
            'clips_candidate_closer_total': sum(r[name+'_candidate_sum'] < r[name+'_baseline_sum']-1e-6 for r in rows),
            'clips_candidate_farther_total': sum(r[name+'_candidate_sum'] > r[name+'_baseline_sum']+1e-6 for r in rows)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'per_sample.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
