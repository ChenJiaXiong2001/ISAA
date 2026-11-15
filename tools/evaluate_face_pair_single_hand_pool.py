"""Compare frozen original and valid-hand pooling on training calibration subjects."""
import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA wheel library paths before torch import.
import torch
from torch.utils.data import DataLoader, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.face_pair_training import compile_model
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.models.rtmw_local_ctr import _downsample_mask
from isaa.train import collate_rtmw
from tools.check_face_pair_checkpoint import check_args, check_compile, check_manifest
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights
from tools.face_pair_single_hand_experiment import ValidHandMeanFacePair


def occlude_raw_hand(raw, member, side, duration):
    """Hide person 0's hand before compression, choosing start by member hash."""
    if side not in ("left", "right") or duration not in (5, 20, 0):
        raise ValueError("side must be left/right and duration must be 5, 20 or 0 (all)")
    if raw.ndim != 4 or raw.shape[0] != 3 or raw.shape[2] != 133 or raw.shape[1] < 1:
        raise ValueError("Expected 3 x T x 133 x M raw input with at least one frame")
    length = raw.shape[1] if duration == 0 else min(duration, raw.shape[1])
    digest = hashlib.sha256(("single-hand-v1:" + member).encode()).digest()
    start = int.from_bytes(digest[:8], "big") % (raw.shape[1] - length + 1)
    first, last = (91, 112) if side == "left" else (112, 133)
    out = raw.clone()
    out[:, start:start + length, first:last, 0] = 0
    return out


def hand_visibility_at_pool(x, frame_mask, strides):
    """Return per-sample actual hand pool visibility, excluding absent people."""
    batch, _, time, _, people = x.shape
    mask = (x[:, 2:3] > 0) & torch.isfinite(x[:, :3]).all(1, keepdim=True)
    mask = mask & frame_mask[:, None, :, None, None].bool()
    mask = mask.permute(0, 4, 1, 2, 3).reshape(batch * people, 1, time, 133)
    left, right = mask[..., 91:112], mask[..., 112:133]
    present = mask.any(-1, keepdim=True)
    for stride in strides:
        left = _downsample_mask(left, stride)
        right = _downsample_mask(right, stride)
        present = _downsample_mask(present, stride)
    left, right, present = (value.any(-1).squeeze(1).reshape(batch, people, -1)
                            for value in (left, right, present))
    return {"both": (left & right & present).sum((1, 2)),
            "left_only": (left & ~right & present).sum((1, 2)),
            "right_only": (~left & right & present).sum((1, 2)),
            "neither": (~left & ~right & present).sum((1, 2))}


class OccludedCalibration(RTMWZipDataset):
    def __init__(self, *args, side=None, duration=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.occlusion_side = side
        self.occlusion_duration = duration

    def _read_raw_sample(self, index):
        raw, mask = super()._read_raw_sample(index)
        if self.occlusion_side is not None:
            raw = occlude_raw_hand(raw, self.members[index], self.occlusion_side, self.occlusion_duration)
        return raw, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-samples", type=int, default=0,
                        help="Use the first N calibration samples for a compiled smoke test; 0 uses all")
    parser.add_argument("--cpu-smoke", action="store_true",
                        help="Explicitly run at most 8 real calibration samples on CPU eager for plumbing only")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size <= 0 or args.num_workers < 0 or args.smoke_samples < 0:
        raise ValueError("Invalid batch size, worker count or smoke sample count")
    if args.cpu_smoke and not 1 <= args.smoke_samples <= 8:
        raise ValueError("CPU is restricted to an explicit smoke test of 1-8 samples")
    if not args.cpu_smoke and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requires torch.compile; no CPU fallback")
    device = torch.device("cpu" if args.cpu_smoke else "cuda")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    if saved["model_variant"] != FACE_PAIR_VARIANT:
        raise ValueError("Only the original best-face-pair checkpoint is supported")
    check_args(saved, "checkpoint.args", expected_variant=FACE_PAIR_VARIANT)
    check_compile(checkpoint["compile"], saved, "checkpoint.compile")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = saved.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = saved.get("tf32", True)
    if checkpoint["architecture"] != FacePairFusion.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads((args.checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    if config["args"] != saved or config["compile"] != checkpoint["compile"]:
        raise ValueError("Checkpoint and saved run configuration differ")
    check_manifest(manifest, config)
    all_members = manifest["calibration"]["members"]
    digest = hashlib.sha256("\n".join(all_members).encode()).hexdigest()
    if digest != manifest["calibration"]["sha256"] or len(set(all_members)) != len(all_members):
        raise ValueError("Calibration manifest digest or uniqueness mismatch")
    members = all_members[:args.smoke_samples] if args.smoke_samples else all_members
    models = []
    for cls in (FacePairFusion, ValidHandMeanFacePair):
        model = cls(saved["num_classes"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model = model.to(device).eval()
        models.append(model)
    weights = _soft_weights(models[0])
    if int(models[0].routing_class_counts.sum()) != len(all_members):
        raise ValueError("Checkpoint calibration counts do not match manifest")
    if not torch.equal(weights, _soft_weights(models[1])):
        raise RuntimeError("Frozen class weights differ")
    compiled = models if args.cpu_smoke else [compile_model(model, saved["compile_mode"]) for model in models]
    hand_strides = [block.stride for block in models[0].hand_ctr_blocks]
    conditions = [("clean", None, None)] + [
        (f"{side}_{duration or 'all'}", side, duration)
        for side in ("left", "right") for duration in (5, 20, 0)]
    rows, condition_hashes, visibility, pool_visibility, drift = [], {}, {}, {}, {}
    for condition, side, duration in conditions:
        dataset = OccludedCalibration(
            args.archive, split="train", split_protocol=saved["split"],
            window_size=saved["window_size"], num_joints=133, num_classes=saved["num_classes"],
            layout="rtmw_133", max_persons=saved["max_persons"], feature_mode="raw",
            max_samples=saved["max_samples"], augment=False,
            temporal_compression=saved["temporal_compression"], side=side, duration=duration)
        index = {member: i for i, member in enumerate(dataset.members)}
        if not set(members).issubset(index):
            raise ValueError("Calibration manifest members absent from training archive")
        indices = [index[member] for member in members]
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=partial(collate_rtmw, feature_mode="raw"),
                            multiprocessing_context="spawn" if args.num_workers else None)
        hasher, offset = hashlib.sha256(), 0
        visible_counts = {"both": 0, "left_only": 0, "right_only": 0, "neither": 0}
        pooled_counts = {name: 0 for name in visible_counts}
        drift_stats = {key: {"max_abs": 0.0, "sum_abs": 0.0, "elements": 0,
                             "argmax_disagreements": 0} for key in ("hf", "bf", "soft")}
        for x, labels, mask in loader:
            hasher.update(x.contiguous().numpy().tobytes())
            hasher.update(mask.contiguous().numpy().tobytes())
            left = (x[:, 2, :, 91:112, 0] > 0).any(-1) & mask
            right = (x[:, 2, :, 112:133, 0] > 0).any(-1) & mask
            visible_counts["both"] += int((left & right).sum())
            visible_counts["left_only"] += int((left & ~right).sum())
            visible_counts["right_only"] += int((~left & right).sum())
            visible_counts["neither"] += int((~left & ~right & mask).sum())
            pool_counts = hand_visibility_at_pool(x, mask, hand_strides)
            for key in pooled_counts:
                pooled_counts[key] += int(pool_counts[key].sum())
            x, mask = x.to(device), mask.to(device)
            with torch.inference_mode():
                outputs = [model(x, mask) for model in compiled]
                if not all(bool(torch.isfinite(value).all()) for out in outputs for value in out.values()):
                    raise RuntimeError(f"Nonfinite model output in {condition}")
                reference, shared = outputs
                shared_original = {"hand_face_logits": shared["original_hand_face_logits"],
                                   "body_face_logits": shared["body_face_logits"]}
                reference_soft = _soft_logits(reference, weights)
                original_soft = _soft_logits(shared_original, weights)
                candidate_soft = _soft_logits(shared, weights)
                for name, a, b in (("hf", reference["hand_face_logits"], shared_original["hand_face_logits"]),
                                   ("bf", reference["body_face_logits"], shared["body_face_logits"]),
                                   ("soft", reference_soft, original_soft)):
                    delta = (a - b).abs()
                    stat = drift_stats[name]
                    stat["max_abs"] = max(stat["max_abs"], delta.max().item())
                    stat["sum_abs"] += delta.sum().item()
                    stat["elements"] += delta.numel()
                    stat["argmax_disagreements"] += int((a.argmax(-1) != b.argmax(-1)).sum())
                soft = [reference_soft.argmax(-1).cpu(), original_soft.argmax(-1).cpu(),
                        candidate_soft.argmax(-1).cpu()]
                heads = [{key: out[key].argmax(-1).cpu() for key in ("hand_face_logits", "body_face_logits")}
                         for out in (reference, shared_original, shared)]
            for i, label in enumerate(labels.tolist()):
                member = members[offset + i]
                subject = re.search(r"P(\d{3})", member)
                if subject is None:
                    raise ValueError(f"Cannot identify subject in {member}")
                row = {"condition": condition, "sample": member, "subject": int(subject.group(1)), "label": label}
                row.update({f"pool_{key}_frames": int(value[i]) for key, value in pool_counts.items()})
                for j, name in enumerate(("reference", "baseline", "valid_mean")):
                    row[f"{name}_soft"] = int(soft[j][i])
                    row[f"{name}_hf"] = int(heads[j]["hand_face_logits"][i])
                    row[f"{name}_bf"] = int(heads[j]["body_face_logits"][i])
                rows.append(row)
            offset += len(labels)
        if offset != len(members):
            raise RuntimeError("Incomplete traversal")
        condition_hashes[condition] = hasher.hexdigest()
        visibility[condition] = visible_counts
        pool_visibility[condition] = pooled_counts
        drift[condition] = drift_stats
        print(f"condition={condition} samples={offset}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = {}
    for condition, _, _ in conditions:
        group = [row for row in rows if row["condition"] == condition]
        b = [row["baseline_soft"] == row["label"] for row in group]
        a = [row["valid_mean_soft"] == row["label"] for row in group]
        results[condition] = {"baseline_correct": sum(b), "valid_mean_correct": sum(a),
                              "gain": sum(x and not y for x, y in zip(a, b)),
                              "loss": sum(y and not x for x, y in zip(a, b))}
    summary = {"checkpoint": str(args.checkpoint.resolve()),
               "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "calibration_members_sha256": digest, "samples_per_condition": len(members),
               "evaluated_members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
               "smoke_samples": args.smoke_samples,
               "conditions": [condition[0] for condition in conditions],
               "results": results,
               "input_and_frame_mask_sha256": condition_hashes,
               "person0_postcompression_frame_visibility": visibility,
               "hand_pool_visibility_present_person_frames": pool_visibility,
               "independent_reference_vs_shared_original_drift": drift,
               "pool_comparison": "same compiled graph and shared features; independent original retained as reference",
               "hand_encoder_strides": hand_strides,
               "hand_mask_rule": "stride 1 passes mask unchanged; stride >1 uses max-pooling bins",
               "weights": weights.cpu().tolist(), "routing": "fixed gamma=1 checkpoint counts",
               "scope": "training calibration only; no official validation",
               "occlusion": "person 0; raw frames before temporal compression; stable member hash start",
               "checkpoint_training_compile": checkpoint["compile"],
               "evaluation": {"device": str(device), "compile": not args.cpu_smoke,
                              "mode": "explicit_cpu_eager_smoke_only" if args.cpu_smoke
                                      else "mandatory_inductor"}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
