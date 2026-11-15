"""Evaluate official CTR-GCN on paired NTU60 XSub samples and compare per class."""
import csv
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/home/xiong/Desktop/ISAA')
OFFICIAL = Path('/home/xiong/Desktop/CTR-GCN')
sys.path.insert(0, str(OFFICIAL))
from model.ctrgcn import Model
from feeders.tools import valid_crop_resize

DATA = OFFICIAL / 'data/ntu60_preprocessed/extracted/NTU-RGB-D/xsub'
MEMBERS = ROOT / 'tools/original_ctrgcn_pair_members_20261108.csv'
ERRORS = ROOT / 'experiments/face_pair_stable_20261103/verification/best_epoch83_accuracy_weighted/errors.csv'
CKPT = ROOT / 'runtime_ctrgcn25_raw.pt'
OUT = ROOT / 'outputs/original_ctrgcn_facepair_centered_20261108'
PATTERN = re.compile(r'S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}')

def identity(s):
    match = PATTERN.search(s)
    if not match:
        raise ValueError(f'Bad sample identifier: {s}')
    return match.group()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

with MEMBERS.open(encoding='utf-8-sig', newline='') as f:
    members = list(csv.DictReader(f))
with ERRORS.open(encoding='utf-8-sig', newline='') as f:
    errors = list(csv.DictReader(f))
with (DATA / 'val_label.pkl').open('rb') as f:
    names, labels = pickle.load(f)
if len(members) != 16306 or len(set(identity(r['sample']) for r in members)) != 16306:
    raise RuntimeError('RTMW member count or uniqueness mismatch')
index = {identity(name): (i, int(label)) for i, (name, label) in enumerate(zip(names, labels))}
if len(index) != len(names):
    raise RuntimeError('Duplicate official input sample identifiers')
errors_by_id = {identity(r['sample']): r for r in errors}
if len(errors_by_id) != len(errors):
    raise RuntimeError('Duplicate face-pair error records')
paired = []
for row in members:
    ident = identity(row['sample'])
    if ident not in index:
        continue
    pos, target = index[ident]
    if target != int(row['label']):
        raise RuntimeError(f'Label mismatch: {ident}')
    err = errors_by_id.get(ident)
    if err and int(err['true_action'][1:]) - 1 != target:
        raise RuntimeError(f'Error label mismatch: {ident}')
    fused = int(err['soft_predicted_action'][1:]) - 1 if err else target
    paired.append((ident, pos, target, fused))
if not paired:
    raise RuntimeError('No matching samples')

class Samples(Dataset):
    def __init__(self):
        self.data = np.load(DATA / 'val_data.npy', mmap_mode='r')
    def __len__(self):
        return len(paired)
    def __getitem__(self, i):
        x = np.array(self.data[paired[i][1]], dtype=np.float32)
        present = np.any(x != 0, axis=(0, 2))
        first = np.flatnonzero(present[:, 0])
        if not len(first):
            raise RuntimeError(f'No first person: {paired[i][0]}')
        origin = x[:, first[0], 1, 0].copy()
        x = (x - origin[:, None, None, None]) * present[None, :, None, :]
        valid = np.sum(x.sum(0).sum(-1).sum(-1) != 0)
        if not valid:
            raise RuntimeError(f'Empty skeleton: {paired[i][0]}')
        return valid_crop_resize(x, valid, [0.95], 64), i

if OUT.exists():
    raise RuntimeError(f'Output already exists: {OUT}')
OUT.mkdir(parents=True)
model = Model(num_class=60, num_point=25, num_person=2,
              graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
state = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(state, strict=True)
model.cuda().eval()
preds = np.empty(len(paired), dtype=np.int64)
loader = DataLoader(Samples(), batch_size=64, num_workers=6, pin_memory=True)
with torch.inference_mode():
    for batch, ids in loader:
        preds[ids.numpy()] = model(batch.cuda(non_blocking=True)).argmax(1).cpu().numpy()
        if int(ids[-1]) % 2048 < 64:
            print(f'{int(ids[-1])+1}/{len(paired)}', flush=True)
class_stats = {i: [0, 0, 0] for i in range(60)}
with (OUT / 'paired_predictions.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sample','label','ctrgcn_prediction','face_pair_prediction'])
    for (ident, _, label, fused), pred in zip(paired, preds):
        writer.writerow([ident, label, int(pred), fused])
        class_stats[label][0] += 1
        class_stats[label][1] += int(pred == label)
        class_stats[label][2] += int(fused == label)
with (OUT / 'per_class.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['action_id','samples','ctrgcn_correct','face_pair_correct','ctrgcn_accuracy_percent','face_pair_accuracy_percent','delta_pp'])
    for label, (n, ctr, face) in class_stats.items():
        if not n:
            raise RuntimeError(f'No matched samples for class {label}')
        writer.writerow([f'A{label+1:03}', n, ctr, face, 100*ctr/n, 100*face/n, 100*(face-ctr)/n])
total = len(paired)
summary = {'official_input_samples':len(names),'face_pair_input_samples':len(members),
           'paired_samples':total,'unmatched_face_pair':len(members)-total,
           'ctrgcn_correct':sum(v[1] for v in class_stats.values()),
           'face_pair_correct':sum(v[2] for v in class_stats.values()),
           'checkpoint_sha256':digest(CKPT),'official_data':'Kinect 3D 25 joints, preprocessed',
           'face_pair_data':'RTMW 2D 133 joints', 'ctrgcn_sampling':'first-frame joint-2 origin, valid_crop_resize [0.95], 64 frames',
           'face_pair_errors_sha256':digest(ERRORS),'members_sha256':digest(MEMBERS)}
summary['ctrgcn_top1_percent'] = 100*summary['ctrgcn_correct']/total
summary['face_pair_top1_percent'] = 100*summary['face_pair_correct']/total
(OUT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False), flush=True)
