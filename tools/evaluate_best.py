"""Evaluate a saved RTMW hand-CTR checkpoint with the original validation protocol."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Import before torch: this entry point fixes the wheel's CUDA library paths.
from isaa.train import collate_rtmw
import numpy as np
import torch
from torch.utils.data import DataLoader
from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.body_local_fusion import (
    BodyLocalHandCTRWideRelativeFusion,
    BodyLocalHandCTRWideRelativeRoutedFusion,
)
from isaa.utils.seed import seed_everything
from isaa.models.class_hand_routed import CLASS_HAND_VARIANT, BodyLocalClassHandRoutedFusion

NTU60_NAMES = [
    "喝水", "吃饭或吃零食", "刷牙", "梳头", "掉落物品", "捡起物品", "扔东西",
    "坐下", "站起", "拍手", "阅读", "写字", "撕纸", "穿外套", "脱外套",
    "穿鞋", "脱鞋", "戴眼镜", "摘眼镜", "戴帽子", "摘帽子", "挥手",
    "踢腿或踢脚", "踢东西", "手伸进口袋", "单脚跳", "跳起", "打电话",
    "玩手机或平板", "使用键盘", "指向某物", "自拍", "看时间", "搓手",
    "点头或鞠躬", "摇头", "擦脸", "敬礼", "合十", "交叉双臂",
    "打喷嚏或咳嗽", "踉跄", "跌倒", "头痛", "胸痛", "背痛", "颈部疼痛",
    "恶心或呕吐", "使用扇子", "拳击或拍打他人", "踢他人", "推他人",
    "拍他人后背", "指向他人", "拥抱", "给他人物品", "触碰他人口袋",
    "握手", "一起走近", "一起走开",
]

def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--hand-mode", choices=("adaptive", "all", "none"), default="adaptive")
    options = parser.parse_args()
    checkpoint_path = options.checkpoint.resolve()
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    classes = {
        "body-local-hand-ctr-wide-relative": BodyLocalHandCTRWideRelativeFusion,
        "body-local-hand-ctr-wide-relative-routed": BodyLocalHandCTRWideRelativeRoutedFusion,
        CLASS_HAND_VARIANT: BodyLocalClassHandRoutedFusion,
    }
    model_class = classes[args["model_variant"]]
    routed = model_class is BodyLocalHandCTRWideRelativeRoutedFusion
    class_routed = model_class is BodyLocalClassHandRoutedFusion
    seed_everything(args.get("seed", 1))
    device = torch.device(options.device)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)
    model = (model_class(num_classes=args["num_classes"],
                         hand_threshold=args.get("hand_route_threshold", 0.5),
                         confidence_threshold=args.get("body_confidence_threshold", 0.8),
                         body_temperature=args.get("body_probability_temperature", 1.0))
             if class_routed else model_class(num_classes=args["num_classes"]))
    if checkpoint.get("architecture") != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture does not match this model version")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    archive = options.archive or Path(args["archive"])
    if not archive.is_absolute():
        archive = ROOT / archive
    dataset = RTMWZipDataset(
        archive, split="val", split_protocol=args["split"],
        window_size=args["window_size"], num_joints=133,
        num_classes=args["num_classes"], layout="rtmw_133",
        max_persons=args["max_persons"], feature_mode=args["feature_mode"],
        max_samples=args.get("max_samples", 0), augment=False,
    )
    batch_size = options.batch_size or args["test_batch_size"]
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=options.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=options.num_workers > 0,
        prefetch_factor=args.get("prefetch_factor", 4) if options.num_workers > 0 else None,
        collate_fn=partial(collate_rtmw, feature_mode=args["feature_mode"]),
    )
    k = args["num_classes"]
    confusion = np.zeros((k, k), dtype=np.int64)
    top5_counts = np.zeros(k, dtype=np.int64)
    route_keys = ["hand_gate", "face_gate", "hand_quality", "face_quality", "coarse_uncertainty"] if routed else []
    if class_routed:
        route_keys = ["hand_called", "hand_requirement", "body_confidence"]
    route_sums = {name: np.zeros(k, dtype=np.float64) for name in route_keys}
    errors = []
    offset = 0
    loss_sum = 0.
    started = time.perf_counter()
    print(json.dumps({"checkpoint": str(checkpoint_path), "epoch": checkpoint["epoch"],
                      "recorded_top1": checkpoint["val_accuracy"], "samples": len(dataset),
                      "batches": len(loader), "eager": True}, ensure_ascii=False), flush=True)
    with torch.inference_mode():
        for step, (x, labels, mask) in enumerate(loader, 1):
            x, labels, mask = (value.to(device, non_blocking=device.type == "cuda") for value in (x, labels, mask))
            if class_routed:
                output = model(x, mask, hand_mode=options.hand_mode, return_routing=True)
            else:
                output = model(x, mask, return_routing=True) if routed else model(x, mask)
            logits = output["logits"] if routed or class_routed else output
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Nonfinite logits at batch {step}")
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            probs = logits.softmax(-1)
            predicted = logits.argmax(-1)
            top5 = logits.topk(min(5, k), dim=1).indices
            truth = labels.cpu().numpy()
            pred = predicted.cpu().numpy()
            hit5 = (top5 == labels[:, None]).any(1).cpu().numpy()
            np.add.at(confusion, (truth, pred), 1)
            np.add.at(top5_counts, truth, hit5.astype(np.int64))
            route_means = {}
            for name in route_keys:
                values = output[name].float().reshape(len(truth), -1).mean(1).cpu().numpy()
                route_means[name] = values
                np.add.at(route_sums[name], truth, values)
            confidence = probs.max(1).values.cpu().numpy()
            true_confidence = probs.gather(1, labels[:, None])[:, 0].cpu().numpy()
            for j in np.flatnonzero(truth != pred):
                errors.append({
                    "sample": dataset.members[offset + j], "true_action": f"A{truth[j]+1:03}",
                    "predicted_action": f"A{pred[j]+1:03}",
                    "true_name": NTU60_NAMES[truth[j]] if k == 60 else "",
                    "predicted_name": NTU60_NAMES[pred[j]] if k == 60 else "",
                    "predicted_probability": float(confidence[j]),
                    "true_probability": float(true_confidence[j]),
                    **{name: float(values[j]) for name, values in route_means.items()},
                })
            offset += len(truth)
            if step % 50 == 0 or step == len(loader):
                print(f"progress={step}/{len(loader)} samples={offset} top1={np.trace(confusion)/offset:.6f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    support = confusion.sum(1)
    correct = confusion.diagonal()
    predicted_counts = confusion.sum(0)
    assert offset == len(dataset) == confusion.sum()
    assert np.array_equal(support, np.bincount(dataset.labels, minlength=k))
    assert len(errors) == offset - correct.sum()
    rows = []
    for i in range(k):
        recall = float(correct[i] / support[i]) if support[i] else None
        precision = float(correct[i] / predicted_counts[i]) if predicted_counts[i] else 0.
        f1 = 2 * precision * recall / (precision + recall) if recall and precision else 0.
        wrong = confusion[i].copy()
        wrong[i] = 0
        confused_with = sorted(np.flatnonzero(wrong), key=lambda j: (-wrong[j], j))[:3]
        rows.append({
            "action_id": f"A{i+1:03}", "name": NTU60_NAMES[i] if k == 60 else "",
            "samples": int(support[i]), "correct": int(correct[i]), "errors": int(support[i]-correct[i]),
            "accuracy_percent": 100 * recall if recall is not None else None,
            "top5_percent": float(100 * top5_counts[i] / support[i]) if support[i] else None,
            "precision_percent": 100 * precision, "f1_percent": 100 * f1,
            "main_confusions": "; ".join(f"A{j+1:03}:{wrong[j]}" for j in confused_with),
            **{f"mean_{name}": float(values[i] / support[i]) if support[i] else None for name, values in route_sums.items()},
        })
    detail_name = f"best_details_{options.hand_mode}" if class_routed else "best_details"
    out = options.output_dir or checkpoint_path.parent / detail_name
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_class_accuracy.csv", rows, list(rows[0]))
    confusion_rows = [{"true_action": f"A{i+1:03}", **{f"A{j+1:03}": int(confusion[i,j]) for j in range(k)}} for i in range(k)]
    write_csv(out / "confusion_matrix.csv", confusion_rows, list(confusion_rows[0]))
    error_fields = ["sample", "true_action", "predicted_action", "true_name", "predicted_name",
                    "predicted_probability", "true_probability", *route_keys]
    write_csv(out / "errors.csv", errors, error_fields)
    pairs = [
        {"true_action": f"A{i+1:03}", "true_name": NTU60_NAMES[i] if k == 60 else "",
         "predicted_action": f"A{j+1:03}", "predicted_name": NTU60_NAMES[j] if k == 60 else "",
         "count": int(confusion[i,j]), "fraction_of_true_percent": float(100*confusion[i,j]/support[i])}
        for i in range(k) for j in range(k) if i != j and confusion[i,j]
    ]
    pairs.sort(key=lambda p: (-p["count"], p["true_action"], p["predicted_action"]))
    write_csv(out / "confusion_pairs.csv", pairs, ["true_action", "true_name", "predicted_action", "predicted_name", "count", "fraction_of_true_percent"])
    ranked = sorted((row for row in rows if row["samples"]), key=lambda r: r["accuracy_percent"])
    actual = int(correct.sum()) / offset
    summary = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "model_variant": args["model_variant"], "architecture": model.ARCHITECTURE,
        "epoch": checkpoint["epoch"], "best_epoch": checkpoint.get("best_epoch"),
        "split": args["split"], "partition": "val", "archive": str(archive.resolve()),
        "samples": offset, "correct": int(correct.sum()), "errors": len(errors),
        "top1_percent": 100*actual, "top5_percent": float(100*top5_counts.sum()/offset),
        "macro_accuracy_percent": float(np.mean([r["accuracy_percent"] for r in ranked])),
        "macro_f1_percent": float(np.mean([r["f1_percent"] for r in ranked])),
        "loss": loss_sum / offset, "recorded_top1_percent": 100*checkpoint["val_accuracy"],
        "difference_percentage_points": 100*(actual-checkpoint["val_accuracy"]),
        "same_top1_correct_count": int(round(checkpoint["val_accuracy"]*offset)) == int(correct.sum()),
        "protocol": {"window_size": args["window_size"], "max_persons": args["max_persons"],
                     "crop": "center", "augmentation": False, "batch_size": batch_size,
                     "compile": False, "tf32": args.get("tf32", True), "max_samples": args.get("max_samples", 0)},
        "runtime": {"torch": torch.__version__, "device": str(device), "seconds": time.perf_counter()-started},
        "mean_routing": {name: float(values.sum()/offset) for name, values in route_sums.items()},
        "worst_10": ranked[:10], "best_10": list(reversed(ranked[-10:])),
        "top_confusions": pairs[:20],
    }
    if class_routed:
        summary["hand_mode"] = options.hand_mode
        summary["hand_call_rate"] = float(route_sums["hand_called"].sum() / offset)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Artifacts saved to {out}", flush=True)

if __name__ == "__main__":
    main()
