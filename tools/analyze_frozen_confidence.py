"""Descriptive confidence diagnostics for cached HF/BF logits."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from face_pair_fusion_sweep import LogitFusion, class_weights_from_counts

def conf(x):
    p = x.softmax(-1)
    top = p.topk(2, -1).values
    return {"maxprob": top[:, 0], "margin": top[:, 0] - top[:, 1],
            "entropy": -(p * p.clamp_min(1e-30).log()).sum(-1)}

def stat(name, m, hc, bc, fc, c_h, c_b, c_f):
    n = int(m.sum())
    if n == 0: return {"n": 0}
    d = {"n": n, "hf_conf_gt_bf": int((c_h["maxprob"][m] > c_b["maxprob"][m]).sum()),
         "hf_margin_gt_bf": int((c_h["margin"][m] > c_b["margin"][m]).sum())}
    for k in ("maxprob", "margin", "entropy"):
        d[f"hf_{k}_mean"] = float(c_h[k][m].mean()); d[f"bf_{k}_mean"] = float(c_b[k][m].mean())
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--calibration-cache",type=Path,required=True); ap.add_argument("--validation-cache",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); a=ap.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    cal=torch.load(a.calibration_cache,map_location="cpu",weights_only=True); val=torch.load(a.validation_cache,map_location="cpu",weights_only=True)
    hc_logits,bc_logits,yc=cal["hf"].float(),cal["bf"].float(),cal["labels"].long(); hf,bf,y=val["hf"].float(),val["bf"].float(),val["labels"].long(); C=hf.shape[1]
    w=class_weights_from_counts(torch.bincount(yc[hc_logits.argmax(1)==yc],minlength=C),torch.bincount(yc[bc_logits.argmax(1)==yc],minlength=C),torch.bincount(yc,minlength=C),[1.0])
    with torch.inference_mode(): fused=LogitFusion()(hf,bf,w,torch.ones((1,2)),torch.zeros(1),torch.zeros(1,dtype=torch.long))[0]
    hp,bp,fp=hf.argmax(1),bf.argmax(1),fused.argmax(1); hc=hp==y; bc=bp==y; fc=fp==y
    ch,cb,cf=conf(hf),conf(bf),conf(fused)
    masks={"hf_only_correct":hc&~bc,"bf_only_correct":~hc&bc,"both_wrong":~hc&~bc,"misfused_hf":hc&~fc,"misfused_bf":bc&~fc,"fused_correct":fc,"fused_wrong":~fc}
    out={"samples":len(y),"strata":{n:stat(n,m,hc,bc,fc,ch,cb,cf) for n,m in masks.items()}}
    # A fixed, purely descriptive dynamic preference: route to the branch with larger confidence.
    for k in ("maxprob","margin"):
        route=torch.where(ch[k]>=cb[k],hp,bp); out[f"route_{k}"]={"correct":int((route==y).sum()),"accuracy":float((route==y).float().mean()*100),"gain_pp":float(((route==y).float().mean()-fc.float().mean())*100)}
    a.output_dir.mkdir(parents=True); (a.output_dir/"summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
