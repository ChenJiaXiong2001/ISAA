"""Describe one-subject count instability using the same calibration cache."""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(); p.add_argument('--calibration-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); o=p.parse_args()
    rows=list(csv.DictReader((o.calibration_dir/'paired_samples.csv').open(encoding='utf-8',newline='')))
    arr=np.load(o.calibration_dir/'paired_branch_logits.npz'); clean=[i for i,r in enumerate(rows) if r['condition']=='clean']; rows=[rows[i] for i in clean]
    labels=np.array([int(r['true_class_index']) for r in rows]); rep=np.array([int(re.search(r'R(\d{3})',r['sample']).group(1)) for r in rows]); out={}
    for model in ('control','candidate'):
        pred=arr[model][clean].argmax(-1); m={}
        for hold in (1,2):
            fit=rep!=hold; h=np.bincount(labels[fit][pred[fit,0]==labels[fit]],minlength=60); b=np.bincount(labels[fit][pred[fit,1]==labels[fit]],minlength=60); n=np.bincount(labels[fit],minlength=60)
            w=np.divide(h,h+b,out=np.full(60,.5),where=h+b>0)
            m[str(hold)]={'fit_samples':int(fit.sum()),'class_count_min':int(n.min()),'class_count_max':int(n.max()),'classes_no_fit':int((n==0).sum()),'classes_head_extreme':int(((h==0)|(b==0)) .sum()),'hf_correct_total':int(h.sum()),'bf_correct_total':int(b.sum()),'weights':w.tolist()}
        a=np.array(m['1']['weights']); b=np.array(m['2']['weights']); d=np.abs(a-b)
        m['between_fold_weight_abs_diff']={'mean':float(d.mean()),'median':float(np.median(d)),'max':float(d.max()),'classes_changed':int((d>1e-12).sum()),'classes_diff_ge_025':int((d>=.25).sum()),'classes_diff_ge_05':int((d>=.5).sum())}
        m['largest_changes']=[{'action':f'A{i+1:03d}','fold1':float(a[i]),'fold2':float(b[i]),'abs_diff':float(d[i])} for i in np.argsort(-d)[:15]]
        out[model]=m
    o.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
