"""Independently audit calibration counts and low-anchor interpolation on CPU."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress

ANCHORS = [5, 6, 11, 12]


def repair_numpy(data, valid_frame):
    result = data.copy()
    for joint in range(4):
        for person in range(result.shape[3]):
            valid = (data[2, :, joint, person] > 0) & valid_frame
            for time in np.flatnonzero(~valid & valid_frame):
                left = time - 1
                while left >= 0 and not valid[left]:
                    left -= 1
                right = time + 1
                while right < len(valid) and not valid[right]:
                    right += 1
                if left >= 0 and right < len(valid) and right - left - 1 <= 5 and valid_frame[left:right + 1].all():
                    alpha = np.float32(time - left) / np.float32(right - left)
                    result[:, time, joint, person] = data[:, left, joint, person] + alpha * (
                        data[:, right, joint, person] - data[:, left, joint, person])
    return result


def geometry(data, valid_frame):
    valid = (data[2] > 0) & valid_frame[:, None, None]
    xy = data[:2]
    center = (xy * valid[None]).sum(axis=2) / np.maximum(valid.sum(axis=1), 1)[None]
    shoulders, hips = valid[:, :2], valid[:, 2:]
    shoulder = (xy[:, :, :2] * shoulders[None]).sum(axis=2) / np.maximum(shoulders.sum(axis=1), 1)[None]
    hip = (xy[:, :, 2:] * hips[None]).sum(axis=2) / np.maximum(hips.sum(axis=1), 1)[None]
    scale = np.maximum(np.sqrt(np.sum((shoulder - hip) ** 2, axis=0)), 1e-3)
    comparable = np.any(shoulders, axis=1) & np.any(hips, axis=1) & valid_frame[:, None]
    return center, scale, comparable


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--result-dir', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    rows = list(csv.DictReader((args.result_dir / 'per_sample.csv').open(encoding='utf-8', newline='')))
    names = json.loads(args.manifest.read_text(encoding='utf-8'))['calibration']['members']
    assert names == [r['sample'] for r in rows]
    data = RTMWZipDataset(args.archive, split='train', split_protocol='xsub60', window_size=100,
        num_joints=133, num_classes=60, layout='rtmw_133', max_persons=2,
        feature_mode='raw', augment=False, temporal_compression='confidence_weighted')
    index = {name: i for i, name in enumerate(data.members)}
    for j, row in enumerate(rows, 1):
        raw, frame = data._read_raw_sample(index[row['sample']])
        out, mask = _confidence_weighted_temporal_compress(raw, frame, 100)
        for prefix, features, valid_frame in (('raw', raw, frame), ('compressed', out, mask)):
            score = features[2, :, ANCHORS, :].numpy()
            present = np.any(score > 0, axis=1) & valid_frame.numpy()[:, None]
            s = score.transpose(0, 2, 1)[present].flatten()
            assert len(s) == 4 * int(row[prefix + '_person_frames'])
            assert int(np.count_nonzero(s > 0)) == int(row[prefix + '_positive'])
            assert int(np.count_nonzero(s == 0)) == int(row[prefix + '_zero'])
            for t in (0.5, 0.9):
                assert int(np.count_nonzero((s > 0) & (s <= t))) == int(row[f'{prefix}_low_{t}'])
        scores = out[2, :, ANCHORS, :].numpy()
        valid_frame = mask.numpy()
        anchors = out[:, :, ANCHORS, :].numpy().copy()
        anchors[:2] *= anchors[2:3] > 0
        anchors *= valid_frame[None, :, None, None]
        base = repair_numpy(anchors, valid_frame)
        center0, scale0, comparable0 = geometry(base, valid_frame)
        for t in (0.5, 0.9):
            flagged = (scores > 0) & (scores <= t) & valid_frame[:, None, None]
            restored = 0
            for ti, joint, person in np.argwhere(flagged):
                valid = (scores[:, joint, person] > t) & valid_frame
                left = ti - 1
                while left >= 0 and not valid[left]:
                    left -= 1
                right = ti + 1
                while right < len(valid) and not valid[right]:
                    right += 1
                if left >= 0 and right < len(valid) and right - left - 1 <= 5 and valid_frame[left:right + 1].all():
                    restored += 1
            assert restored == int(row[f'restored_{t}'])
            assert int(flagged.sum()) - restored == int(row[f'removed_{t}'])
            if flagged.any():
                thresholded = anchors.copy()
                thresholded[:, flagged] = 0
                candidate = repair_numpy(thresholded, valid_frame)
                center1, scale1, comparable1 = geometry(candidate, valid_frame)
                center_delta = np.sqrt(np.sum((center1 - center0) ** 2, axis=0))
                changed_center = int(np.count_nonzero(center_delta > 1e-7))
                changed_scale = int(np.count_nonzero(comparable0 & comparable1 & (np.abs(scale1 - scale0) > 1e-7)))
                assert changed_center == int(row[f'center_changed_{t}']), (row['sample'], t, changed_center, row[f'center_changed_{t}'])
                assert changed_scale == int(row[f'scale_changed_{t}']), (row['sample'], t, changed_scale, row[f'scale_changed_{t}'])
        if j % 500 == 0:
            print(f'audited={j}/{len(rows)}', flush=True)
    print('PASS: every sample score counts, scalar interpolation eligibility, torso center and scale change counts')


if __name__ == '__main__':
    main()
