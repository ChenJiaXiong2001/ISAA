"""Audit cached HF/BF/soft errors to prioritize structural input experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.face_pair_fusion_sweep import LogitFusion, class_weights_from_counts


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=True)
    validation = torch.load(args.validation_cache, map_location="cpu", weights_only=True)
    if calibration["checkpoint_sha256"] != validation["checkpoint_sha256"]:
        raise RuntimeError("Calibration and validation caches use different checkpoints")
    if calibration["hf"].shape[1] != validation["hf"].shape[1]:
        raise RuntimeError("Class count mismatch")
    cal_hf, cal_bf, cal_y = calibration["hf"], calibration["bf"], calibration["labels"]
    val_hf, val_bf, val_y = validation["hf"].float(), validation["bf"].float(), validation["labels"].long()
    classes = val_hf.shape[1]
    weights = class_weights_from_counts(
        torch.bincount(cal_y[cal_hf.argmax(-1) == cal_y], minlength=classes),
        torch.bincount(cal_y[cal_bf.argmax(-1) == cal_y], minlength=classes),
        torch.bincount(cal_y, minlength=classes), [1.0],
    )[0]
    with torch.inference_mode():
        fused = LogitFusion().eval()(val_hf, val_bf, weights[None], torch.ones((1, 2)),
                                     torch.zeros(1), torch.zeros(1, dtype=torch.long))[0]
    hf_pred, bf_pred, fused_pred = val_hf.argmax(-1), val_bf.argmax(-1), fused.argmax(-1)
    hf_top5, bf_top5, fused_top5 = val_hf.topk(5, -1).indices, val_bf.topk(5, -1).indices, fused.topk(5, -1).indices
    rows, pair_counts = [], {}
    for cls in range(classes):
        mask = val_y == cls
        errors = fused_pred[mask] != cls
        wrong = fused_pred[mask][errors].bincount(minlength=classes)
        wrong[cls] = 0
        top_wrong = int(wrong.argmax()) if int(wrong.sum()) else -1
        rows.append({
            "action_id": f"A{cls + 1:03d}",
            "samples": int(mask.sum()),
            "hf_correct": int((hf_pred[mask] == cls).sum()),
            "bf_correct": int((bf_pred[mask] == cls).sum()),
            "fused_correct": int((fused_pred[mask] == cls).sum()),
            "fused_top5": int((fused_top5[mask] == cls).any(-1).sum()),
            "fused_near_miss_top5": int(((fused_pred[mask] != cls) & (fused_top5[mask] == cls).any(-1)).sum()),
            "both_heads_wrong": int(((hf_pred[mask] != cls) & (bf_pred[mask] != cls)).sum()),
            "hf_only_correct": int(((hf_pred[mask] == cls) & (bf_pred[mask] != cls)).sum()),
            "bf_only_correct": int(((hf_pred[mask] != cls) & (bf_pred[mask] == cls)).sum()),
            "top_fused_error": f"A{top_wrong + 1:03d}" if top_wrong >= 0 else "",
            "top_fused_error_count": int(wrong[top_wrong]) if top_wrong >= 0 else 0,
        })
        for pred in fused_pred[mask][errors].tolist():
            pair_counts[(cls + 1, pred + 1)] = pair_counts.get((cls + 1, pred + 1), 0) + 1

    overall = {
        "samples": len(val_y),
        "hf_correct": int((hf_pred == val_y).sum()),
        "bf_correct": int((bf_pred == val_y).sum()),
        "fused_correct": int((fused_pred == val_y).sum()),
        "hf_only_correct": int(((hf_pred == val_y) & (bf_pred != val_y)).sum()),
        "bf_only_correct": int(((hf_pred != val_y) & (bf_pred == val_y)).sum()),
        "both_heads_correct": int(((hf_pred == val_y) & (bf_pred == val_y)).sum()),
        "both_heads_wrong": int(((hf_pred != val_y) & (bf_pred != val_y)).sum()),
        "fused_top5_correct": int((fused_top5 == val_y[:, None]).any(-1).sum()),
        "fused_near_miss": int(((fused_pred != val_y) & (fused_top5 == val_y[:, None]).any(-1)).sum()),
    }
    top_pairs = [
        {"true": f"A{true:03d}", "predicted": f"A{pred:03d}", "count": count}
        for (true, pred), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))[:50]
    ]
    focus = {key: next(row for row in rows if row["action_id"] == key)
             for key in ("A011", "A012", "A029", "A041", "A048")}
    result = {
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "calibration_cache_sha256": digest(args.calibration_cache),
        "validation_cache_sha256": digest(args.validation_cache),
        "calibration_samples": len(cal_y),
        "validation_samples": len(val_y),
        "method": "frozen cached epoch-83 logits; gamma=1 class weights computed from training calibration cache; validation used for post-hoc error audit only",
        "overall": overall,
        "focus_actions": focus,
        "top_confusion_pairs": top_pairs,
        "structural_hypotheses": [
            "A011/A012/A029 form a hand-centric confusion cluster (reading, writing, phone/tablet); explicit hand-to-face geometry and temporal hand displacement are plausible input features to test.",
            "The current model supplies torso-relative coordinates and cross-hand distance/direction to independent HF/BF encoders, then combines pooled features with a linear head; it has no explicit hand-face interaction channel or hand velocity channel.",
            "Many errors remain in fused top-5, so a margin/representation experiment is more plausible than a fusion-weight change; this audit does not establish that the proposed features improve accuracy.",
        ],
        "selection_warning": "No candidate was selected by validation accuracy; this is diagnosis for a future fixed short training protocol.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "top_confusions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["true", "predicted", "count"])
        writer.writeheader()
        writer.writerows(top_pairs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
