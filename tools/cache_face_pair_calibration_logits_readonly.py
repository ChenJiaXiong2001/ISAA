"""Extract frozen HF/BF logits on the checkpoint's calibration subjects."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train
import torch
from torch.utils.data import DataLoader, Dataset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.train import collate_rtmw


class MemberSubset(Dataset):
    def __init__(self, dataset, members):
        by_name = {name: i for i, name in enumerate(dataset.members)}
        if len(by_name) != len(dataset.members) or any(name not in by_name for name in members):
            raise ValueError("Calibration members do not uniquely match training dataset")
        self.dataset, self.indices = dataset, [by_name[name] for name in members]
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.dataset[self.indices[i]]


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--device",default="cpu")
    a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); saved=ck["args"]
    if saved["model_variant"] != FACE_PAIR_VARIANT: raise ValueError("Expected best-face-pair checkpoint")
    model=FacePairFusion(saved["num_classes"])
    if ck["architecture"] != model.ARCHITECTURE: raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(ck["model"],strict=True); model.eval().to(a.device)
    if a.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    manifest=json.loads(a.manifest.read_text(encoding="utf-8")); section=manifest["calibration"]
    members=section["members"]
    digest=hashlib.sha256("\n".join(members).encode()).hexdigest()
    if digest != section["sha256"] or len(members) != section["count"]: raise ValueError("Calibration manifest mismatch")
    ds=RTMWZipDataset(a.archive,split="train",split_protocol=saved["split"],window_size=saved["window_size"],
       num_joints=133,num_classes=saved["num_classes"],layout="rtmw_133",max_persons=saved["max_persons"],
       score_normalization=saved.get("score_normalization","auto"),feature_mode="raw",augment=False,
       temporal_compression=saved.get("temporal_compression","confidence_weighted"),
       deterministic_temporal_crop=saved.get("deterministic_temporal_crop","center"))
    subset=MemberSubset(ds,members)
    loader=DataLoader(subset,batch_size=a.batch_size,shuffle=False,num_workers=0,
       collate_fn=partial(collate_rtmw,feature_mode="raw"))
    hfs=[]; bfs=[]; labels=[]; started=time.perf_counter()
    with torch.inference_mode():
       for step,(x,y,mask) in enumerate(loader,1):
          out=model(x.to(a.device),mask.to(a.device))
          hf=out["hand_face_logits"].float().cpu(); bf=out["body_face_logits"].float().cpu()
          if not bool(torch.isfinite(hf).all() and torch.isfinite(bf).all()): raise RuntimeError("Nonfinite logits")
          hfs.append(hf); bfs.append(bf); labels.append(y.cpu())
          if step==1 or step%10==0 or step==len(loader): print(f"batch={step}/{len(loader)} samples={sum(map(len,labels))} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    cache={"schema_version":1,"hf":torch.cat(hfs),"bf":torch.cat(bfs),"labels":torch.cat(labels),
       "members":members,"member_sha256":digest,"checkpoint_sha256":hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
       "epoch":ck["epoch"],"partition":"calibration","device":a.device}
    a.output_dir.mkdir(parents=True)
    torch.save(cache,a.output_dir/"calibration_logits.pt")
    summary={"checkpoint":str(a.checkpoint),"checkpoint_sha256":cache["checkpoint_sha256"],
       "samples":len(members),"members_sha256":digest,"subjects":sorted(set(x.split("P")[-1][:3] for x in members)),
       "logits_shape":list(cache["hf"].shape),"device":a.device,"seconds":time.perf_counter()-started,
       "read_only_frozen_inference":True,"partition":"training calibration only"}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
