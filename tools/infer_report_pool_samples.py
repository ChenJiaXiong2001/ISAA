"""Read-only checkpoint inference for the report's three named validation videos.

Run on the training server. No training or checkpoint changes occur here.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/home/xiong/Desktop/ISAA')
sys.path.insert(0, str(ROOT))
from isaa.train import collate_rtmw
import torch
from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion

ckpath = ROOT / 'outputs/best_weighted_100_cosine_ntu60_xsub/20261014-074736-ab97a386/best.pt'
checkpoint = torch.load(ckpath, map_location='cpu', weights_only=False)
args = checkpoint['args']
model = BodyLocalHandCTRWideRelativeFusion(num_classes=args['num_classes'])
assert checkpoint['architecture'] == model.ARCHITECTURE
model.load_state_dict(checkpoint['model'], strict=True)
model.cuda().eval()
torch.backends.cuda.matmul.allow_tf32 = args.get('tf32', True)
torch.backends.cudnn.allow_tf32 = args.get('tf32', True)
ds = RTMWZipDataset(
    ROOT / args['archive'], split='val', split_protocol=args['split'],
    window_size=args['window_size'], num_joints=133, num_classes=60,
    layout='rtmw_133', max_persons=2, feature_mode=args['feature_mode'],
    temporal_compression=args.get('temporal_compression', 'none'), augment=False)
rows = []
for action in (10, 12, 29):
    suffix = f'S001C001P007R001A{action:03}_rgb.npz'
    idx = next(i for i, name in enumerate(ds.members) if name.endswith(suffix))
    x, y, mask = collate_rtmw([ds[idx]], feature_mode=args['feature_mode'])
    with torch.inference_mode():
        probs = model(x.cuda(), mask.cuda()).softmax(-1)[0].cpu()
    values, ids = probs.topk(5)
    rows.append({'sample': ds.members[idx], 'true_action': f'A{action:03}',
                 'recognized_action': f'A{int(ids[0])+1:03}',
                 'confidence': float(values[0]),
                 'top5': [{'action':f'A{int(k)+1:03}', 'probability':float(v)}
                          for k,v in zip(ids, values)]})
result = {'checkpoint':str(ckpath), 'checkpoint_sha256':hashlib.sha256(ckpath.read_bytes()).hexdigest(),
          'architecture': model.ARCHITECTURE, 'protocol': 'same dataset preparation as pool evaluation; one full-sequence compressed clip; offline inference; CUDA eager evaluation only',
          'samples': rows}
out = ROOT / 'outputs/report_paper_reference_20261019'
out.mkdir(exist_ok=True)
(out/'pool_sample_recognition.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
