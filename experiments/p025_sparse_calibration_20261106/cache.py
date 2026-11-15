"""Cache p025 training-calibration logits without altering the checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA wheel paths before importing torch.
import torch
from torch.utils.data import DataLoader, Dataset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import FACE_PAIR_HAND_MOTION_VARIANT, FacePairHandMotionFusion
from isaa.train import collate_rtmw


class MemberSubset(Dataset):
    def __init__(self, dataset, members):
        index = {name: i for i, name in enumerate(dataset.members)}
        if len(index) != len(dataset.members) or any(name not in index for name in members):
            raise ValueError('Calibration members do not uniquely match training dataset')
        self.dataset = dataset
        self.indices = [index[name] for name in members]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint['args']
    if checkpoint['epoch'] != 20 or cfg['model_variant'] != FACE_PAIR_HAND_MOTION_VARIANT:
        raise ValueError('Unexpected checkpoint variant or epoch')
    if cfg.get('precompression_anchor_dropout_probability') != .025 or cfg.get('precompression_anchor_clean_sample_probability') != .5:
        raise ValueError('Expected p025/clean50 experiment')
    manifest = json.loads((args.checkpoint.parent / 'split_manifest.json').read_text(encoding='utf-8'))
    section = manifest['calibration']
    members = section['members']
    member_hash = hashlib.sha256('\n'.join(members).encode()).hexdigest()
    if len(members) != section['count'] or member_hash != section['sha256'] or len(set(members)) != len(members):
        raise ValueError('Calibration manifest mismatch')
    model = FacePairHandMotionFusion(cfg['num_classes'])
    if checkpoint['architecture'] != model.ARCHITECTURE:
        raise ValueError('Unexpected architecture')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    dataset = RTMWZipDataset(args.archive, split='train', split_protocol=cfg['split'],
        window_size=cfg['window_size'], num_joints=133, num_classes=cfg['num_classes'],
        layout='rtmw_133', max_persons=cfg['max_persons'], feature_mode='raw', augment=False,
        score_normalization=cfg.get('score_normalization', 'auto'),
        temporal_compression=cfg.get('temporal_compression', 'confidence_weighted'),
        deterministic_temporal_crop=cfg.get('deterministic_temporal_crop', 'center'))
    loader = DataLoader(MemberSubset(dataset, members), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=partial(collate_rtmw, feature_mode='raw'))
    hf, bf, labels = [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for i, (features, target, mask) in enumerate(loader, 1):
            output = model(features, mask)
            h, b = output['hand_face_logits'].float(), output['body_face_logits'].float()
            if not torch.isfinite(h).all() or not torch.isfinite(b).all():
                raise RuntimeError('Nonfinite logits')
            hf.append(h); bf.append(b); labels.append(target)
            print(f'batch={i}/{len(loader)} samples={sum(map(len, labels))} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    cache = {'schema_version': 1, 'hf': torch.cat(hf), 'bf': torch.cat(bf), 'labels': torch.cat(labels),
        'members': members, 'member_sha256': member_hash, 'checkpoint_sha256': digest(args.checkpoint),
        'epoch': checkpoint['epoch'], 'partition': 'calibration', 'device': 'cpu',
        'note': 'Frozen eager CPU inference; checkpoint training used mandatory torch.compile.'}
    if len(cache['labels']) != len(members):
        raise RuntimeError('Incomplete calibration')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(cache, args.output_dir / 'calibration_logits.pt')
    summary = {key: cache[key] for key in ('member_sha256', 'checkpoint_sha256', 'epoch', 'partition', 'device', 'note')}
    summary.update(samples=len(members), seconds=time.perf_counter()-start)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
