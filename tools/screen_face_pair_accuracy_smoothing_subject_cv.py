"""Screen fixed pseudocount smoothing for gamma=1 weights across subjects."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.face_pair_fusion_sweep import CONFIDENCE_UNIFORM, LogitFusion


ALPHAS = (0.5, 1.0, 2.0, 5.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    parser.add_argument("--replicates", type=int, default=20000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}")
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    report = json.loads(args.cache_report.read_text(encoding="utf-8"))
    members = report["manifest"]["calibration"]["members"]
    digest = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if digest != cache["member_sha256"]:
        raise RuntimeError("Cached logits and calibration members differ")
    subject_ids = []
    for member in members:
        match = re.search(r"P(\d{3})", member)
        if not match:
            raise RuntimeError(f"Subject ID missing from {member}")
        subject_ids.append(int(match.group(1)))
    subjects = torch.tensor(subject_ids)
    hf, bf, labels = cache["hf"].float(), cache["bf"].float(), cache["labels"].long()
    unique = sorted(set(subject_ids))
    if len(unique) != 2 or hf.shape != bf.shape or hf.shape[0] != len(labels):
        raise RuntimeError("Unexpected calibration cache shape or subject count")

    baseline_name = "pseudocount_0"
    candidate_names = [f"pseudocount_{alpha:g}" for alpha in ALPHAS]
    names = [baseline_name, *candidate_names]
    model = LogitFusion().eval()
    rng = np.random.default_rng(args.bootstrap_seed)
    folds, rows = [], []
    with torch.no_grad():
        for held_out in unique:
            fit, test = subjects != held_out, subjects == held_out
            fit_labels, fit_hf, fit_bf = labels[fit], hf[fit], bf[fit]
            total = torch.bincount(fit_labels, minlength=hf.shape[-1]).float()
            correct_hf = torch.bincount(fit_labels[fit_hf.argmax(-1) == fit_labels], minlength=hf.shape[-1]).float()
            correct_bf = torch.bincount(fit_labels[fit_bf.argmax(-1) == fit_labels], minlength=hf.shape[-1]).float()
            weights = [torch.where(correct_hf + correct_bf > 0,
                       correct_hf / (correct_hf + correct_bf), torch.full_like(total, 0.5))]
            for alpha in ALPHAS:
                smooth_hf = (correct_hf + alpha) / (total + 2 * alpha)
                smooth_bf = (correct_bf + alpha) / (total + 2 * alpha)
                weights.append(torch.where(smooth_hf + smooth_bf > 0,
                                           smooth_hf / (smooth_hf + smooth_bf),
                                           torch.full_like(total, 0.5)))
            weights_tensor = torch.stack(weights)
            y = labels[test]
            logp = model(hf[test], bf[test], weights_tensor,
                         torch.ones((len(names), 2)), torch.zeros(len(names)),
                         torch.full((len(names),), CONFIDENCE_UNIFORM))
            predictions = logp.argmax(-1)
            if not torch.isfinite(logp).all():
                raise RuntimeError("Nonfinite fused predictions")
            n = len(y)
            fold = {"fit_subject": next(s for s in unique if s != held_out),
                    "heldout_subject": held_out, "fit_samples": int(fit.sum()),
                    "test_samples": n, "candidates": {}}
            for index, name in enumerate(names):
                right = predictions[index] == y
                base_right = predictions[0] == y
                gain = int((~base_right & right).sum())
                loss = int((base_right & ~right).sum())
                draws = rng.multinomial(n, [gain/n, loss/n, (n-gain-loss)/n],
                                        size=args.replicates)
                ci = np.percentile(100*(draws[:,0]-draws[:,1])/n,[2.5,97.5]).tolist()
                fold["candidates"][name] = {"correct": int(right.sum()),
                    "top1_percent": 100*float(right.float().mean()),
                    "nll": float(F.nll_loss(logp[index], y)), "gain": gain, "loss": loss,
                    "delta_pp": 100*(gain-loss)/n, "sample_bootstrap_95_ci_pp": ci,
                    "mean_abs_weight_shift": float((weights_tensor[index]-weights_tensor[0]).abs().mean())}
                test_names = [member for member, keep in zip(members,test.tolist()) if keep]
                for member,label,base_pred,pred in zip(test_names,y.tolist(),predictions[0].tolist(),predictions[index].tolist()):
                    rows.append({"sample":member,"fit_subject":fold["fit_subject"],
                                 "heldout_subject":held_out,"label":label,"candidate":name,
                                 "baseline_pred":base_pred,"prediction":pred})
            folds.append(fold)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/"predictions.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={"cache_sha256":hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "checkpoint_sha256":cache["checkpoint_sha256"],"cache_epoch":cache["epoch"],
        "calibration_members_sha256":digest,"subjects":unique,
        "method":"LOSO; fit-subject HF/BF accuracy weights; equal Beta(alpha,alpha) pseudocount on each head correct count; no heldout selection",
        "fixed_alphas":list(ALPHAS),"no_official_validation_used":True,
        "bootstrap_seed":args.bootstrap_seed,"bootstrap_replicates":args.replicates,"folds":folds}
    (args.output_dir/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
