"""Analyze branch/fusion error strata from cached HF/BF logits.

This is a diagnostic only: class priors use calibration labels, while all
reported strata are computed on a separate validation cache.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from face_pair_fusion_sweep import LogitFusion, class_weights_from_counts


def ranks(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=1, descending=True)
    return (order == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1


def stats(mask: torch.Tensor, y: torch.Tensor, hf: torch.Tensor, bf: torch.Tensor,
          fused: torch.Tensor, hf_rank: torch.Tensor, bf_rank: torch.Tensor,
          fused_rank: torch.Tensor) -> dict:
    n = int(mask.sum())
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "fraction": n / len(y),
        "hf_correct": int((hf.argmax(1)[mask] == y[mask]).sum()),
        "bf_correct": int((bf.argmax(1)[mask] == y[mask]).sum()),
        "fused_correct": int((fused.argmax(1)[mask] == y[mask]).sum()),
        "hf_top5": int((hf_rank[mask] <= 5).sum()),
        "bf_top5": int((bf_rank[mask] <= 5).sum()),
        "fused_top5": int((fused_rank[mask] <= 5).sum()),
        "hf_rank_median": float(hf_rank[mask].float().median()),
        "bf_rank_median": float(bf_rank[mask].float().median()),
        "fused_rank_median": float(fused_rank[mask].float().median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-cache", type=Path, required=True)
    ap.add_argument("--validation-cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cal = torch.load(args.calibration_cache, map_location="cpu", weights_only=True)
    val = torch.load(args.validation_cache, map_location="cpu", weights_only=True)
    hf_c, bf_c, y_c = cal["hf"].float(), cal["bf"].float(), cal["labels"].long()
    hf, bf, y = val["hf"].float(), val["bf"].float(), val["labels"].long()
    classes = hf.shape[1]
    w = class_weights_from_counts(
        torch.bincount(y_c[hf_c.argmax(1) == y_c], minlength=classes),
        torch.bincount(y_c[bf_c.argmax(1) == y_c], minlength=classes),
        torch.bincount(y_c, minlength=classes), [1.0],
    )
    with torch.inference_mode():
        fused = LogitFusion()(hf, bf, w, torch.ones((1, 2)), torch.zeros(1), torch.zeros(1, dtype=torch.long))[0]
    hp, bp, fp = hf.argmax(1), bf.argmax(1), fused.argmax(1)
    hr, br, fr = ranks(hf, y), ranks(bf, y), ranks(fused, y)
    hc, bc, fc = hp == y, bp == y, fp == y
    masks = {
        "all": torch.ones_like(y, dtype=torch.bool),
        "both_heads_correct": hc & bc,
        "hf_only_correct": hc & ~bc,
        "bf_only_correct": ~hc & bc,
        "both_heads_wrong": ~hc & ~bc,
        "fused_correct": fc,
        "fused_wrong": ~fc,
        "misfused_hf_correct": hc & ~fc,
        "misfused_bf_correct": bc & ~fc,
        "both_wrong_true_in_either_top5": (~hc & ~bc) & ((hr <= 5) | (br <= 5)),
        "both_wrong_true_in_both_top5": (~hc & ~bc) & (hr <= 5) & (br <= 5),
        "both_wrong_true_in_neither_top5": (~hc & ~bc) & (hr > 5) & (br > 5),
        "fused_wrong_true_in_either_top5": (~fc) & ((hr <= 5) | (br <= 5)),
    }
    out = {
        "checkpoint_sha256": val.get("checkpoint_sha256"),
        "calibration_samples": len(y_c),
        "validation_samples": len(y),
        "overall": {
            "hf_correct": int(hc.sum()), "bf_correct": int(bc.sum()),
            "fused_correct": int(fc.sum()),
            "union_oracle_correct": int((hc | bc).sum()),
            "union_oracle_gain_pp": float(100 * ((hc | bc).float().mean() - fc.float().mean())),
            "misfused_hf_correct": int((hc & ~fc).sum()),
            "misfused_bf_correct": int((bc & ~fc).sum()),
        },
        "strata": {name: stats(mask, y, hf, bf, fused, hr, br, fr) for name, mask in masks.items()},
    }
    focus = {}
    for cls in (11, 12, 29, 41, 48):
        mask = y == cls - 1
        focus[f"A{cls:03d}"] = {
            "n": int(mask.sum()),
            "hf_correct": int(hc[mask].sum()), "bf_correct": int(bc[mask].sum()),
            "fused_correct": int(fc[mask].sum()),
            "both_wrong": int(((~hc & ~bc)[mask]).sum()),
            "fused_wrong_true_in_either_top5": int(((((~fc) & ((hr <= 5) | (br <= 5)))[mask])).sum()),
            "hf_rank_median": float(hr[mask].float().median()),
            "bf_rank_median": float(br[mask].float().median()),
            "fused_rank_median": float(fr[mask].float().median()),
        }
    out["focus_actions"] = focus
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
