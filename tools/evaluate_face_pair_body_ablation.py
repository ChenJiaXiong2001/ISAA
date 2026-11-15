"""Evaluate frozen face-pair predictions under matched body-point ablations."""
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
import isaa.train  # Configure CUDA wheel paths before importing torch.
import torch
from torch.utils.data import DataLoader, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights


ANCHORS = (5, 6, 11, 12)
NONANCHORS = tuple(i for i in range(23) if i not in ANCHORS)
BODY = tuple(range(23))


def derive_eight_channel_input(features, frame_mask):
    """Reproduce FacePairFusion's raw-input torso and cross-hand features."""
    batch, _, time, joints, people = features.shape
    mask = (features[:, 2:3] > 0) & torch.isfinite(features[:, :3]).all(1, keepdim=True)
    mask = mask & frame_mask[:, None, :, None, None].bool()
    x = features.permute(0, 4, 1, 2, 3).reshape(batch * people, 3, time, joints)
    mask = mask.permute(0, 4, 1, 2, 3).reshape(batch * people, 1, time, joints)
    x = x.masked_fill(~mask, 0)
    torso_ids = torch.tensor(ANCHORS, device=x.device)
    torso_xy = x[:, :2].index_select(-1, torso_ids)
    torso_valid = mask.index_select(-1, torso_ids).to(x.dtype)
    center = (torso_xy * torso_valid).sum(-1, keepdim=True) / torso_valid.sum(-1, keepdim=True).clamp_min(1)
    shoulder_w, hip_w = torso_valid[..., :2], torso_valid[..., 2:]
    shoulder = (torso_xy[..., :2] * shoulder_w).sum(-1) / shoulder_w.sum(-1).clamp_min(1)
    hip = (torso_xy[..., 2:] * hip_w).sum(-1) / hip_w.sum(-1).clamp_min(1)
    scale = torch.linalg.vector_norm(shoulder - hip, dim=1).clamp_min(1e-3)
    pair_valid = mask[..., 91:112] & mask[..., 112:133]
    relative = x[:, :2] - center
    vector = relative[:, :, :, 112:133] - relative[:, :, :, 91:112]
    pair_distance = torch.linalg.vector_norm(vector, dim=1)
    distance = (pair_distance / scale[..., None]) * pair_valid[:, 0].to(x.dtype)
    direction = vector / pair_distance[:, None].clamp_min(1e-6) * pair_valid.to(x.dtype)
    distances = x.new_zeros((batch * people, 1, time, joints))
    directions = x.new_zeros((batch * people, 2, time, joints))
    distances[..., 91:112] = distances[..., 112:133] = distance[:, None]
    directions[..., 91:112] = direction
    directions[..., 112:133] = -direction
    expanded = torch.cat((x, relative, distances, directions), dim=1)
    return expanded.reshape(batch, people, 8, time, joints).permute(0, 2, 3, 4, 1).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-seed", type=int, default=1729)
    parser.add_argument("--sample-seed", type=int, default=20261105)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--isolate-anchor", action="store_true",
                        help="Compare anchor ablation with clean local derived features preserved")
    opts = parser.parse_args()
    if opts.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {opts.output_dir}")
    if opts.sample_count <= 0 or opts.batch_size <= 0 or opts.num_workers < 0:
        raise ValueError("Invalid sample count, batch size, or worker count")
    if opts.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    checkpoint = torch.load(opts.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args["model_variant"] != FACE_PAIR_VARIANT:
        raise ValueError("Expected best-face-pair checkpoint")
    model = FacePairFusion(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(opts.device).eval()
    weights = _soft_weights(model).detach()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)

    dataset = RTMWZipDataset(opts.archive, split="val", split_protocol=args["split"],
        window_size=args["window_size"], num_joints=133, num_classes=args["num_classes"],
        layout="rtmw_133", max_persons=args["max_persons"], feature_mode="raw",
        augment=False, temporal_compression=args["temporal_compression"])
    sample_generator = torch.Generator().manual_seed(opts.sample_seed)
    indices = torch.randperm(len(dataset), generator=sample_generator)[:opts.sample_count].sort().values.tolist()
    members = [dataset.members[i] for i in indices]
    members_digest = hashlib.sha256("\n".join(members).encode()).hexdigest()
    loader_args = dict(batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                       collate_fn=partial(collate_rtmw, feature_mode="raw"))
    if opts.num_workers:
        loader_args.update(prefetch_factor=2, multiprocessing_context="spawn")
    loader = DataLoader(Subset(dataset, indices), **loader_args)
    conditions = [("clean", (), 0.0, False)] + [(f"{name}_{level:g}", joints, level, False)
                  for level in (0.05, 0.10)
                  for name, joints in (("anchor", ANCHORS), ("nonanchor", NONANCHORS), ("body", BODY))]
    if opts.isolate_anchor:
        conditions = [("clean", (), 0.0, False), ("clean8", (), 0.0, True)]
        conditions += [(f"anchor_{level:g}", ANCHORS, level, False) for level in (0.05, 0.10)]
        conditions += [(f"anchor_preserved_{level:g}", ANCHORS, level, True) for level in (0.05, 0.10)]
    generator = torch.Generator().manual_seed(opts.mask_seed)
    per_sample = []
    counts = {name: {method: 0 for method in ("HF", "BF", "soft")} for name, _, _, _ in conditions}
    hidden_counts = {name: 0 for name, _, _, _ in conditions}
    clean_max_abs_error = {key: 0.0 for key in ("logits", "hand_face_logits", "body_face_logits")}
    clean_prediction_disagreements = {key: 0 for key in clean_max_abs_error}
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (features, labels, frame_mask) in enumerate(loader, 1):
            features = features.to(opts.device)
            labels = labels.to(opts.device)
            frame_mask = frame_mask.to(opts.device)
            random = torch.rand((features.shape[0], features.shape[2], 23, features.shape[4]), generator=generator)
            random = random.to(opts.device)
            clean_eight = derive_eight_channel_input(features, frame_mask) if opts.isolate_anchor else None
            batch_rows = [{"sample": members[len(per_sample) + i], "label": int(y)}
                          for i, y in enumerate(labels.cpu().tolist())]
            clean_output = None
            for name, joints, probability, preserve_local in conditions:
                source = clean_eight if preserve_local else features
                if joints:
                    selected = torch.zeros(23, device=opts.device, dtype=torch.bool)
                    selected[list(joints)] = True
                    hidden = (random < probability) & selected[None, None, :, None] & (features[:, 2, :, :23, :] > 0)
                    altered = source.clone()
                    altered[:, :3, :, :23, :] = altered[:, :3, :, :23, :].masked_fill(hidden[:, None], 0)
                    hidden_counts[name] += int(hidden.sum())
                else:
                    altered = source
                output = model(altered, frame_mask)
                if name == "clean":
                    clean_output = output
                elif name == "clean8":
                    for key in clean_max_abs_error:
                        error = float((output[key] - clean_output[key]).abs().max())
                        clean_max_abs_error[key] = max(clean_max_abs_error[key], error)
                        clean_prediction_disagreements[key] += int((output[key].argmax(-1) != clean_output[key].argmax(-1)).sum())
                        if not torch.allclose(output[key], clean_output[key], rtol=1e-4, atol=1e-4):
                            raise RuntimeError(f"Clean 3/8 {key} mismatch: max absolute error {error}")
                for method, logits in (("HF", output["hand_face_logits"]),
                                       ("BF", output["body_face_logits"]),
                                       ("soft", _soft_logits(output, weights))):
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError(f"Nonfinite logits in {name} {method}")
                    predicted = logits.argmax(-1)
                    counts[name][method] += int((predicted == labels).sum())
                    for row, value in zip(batch_rows, predicted.cpu().tolist()):
                        row[f"{name}_{method}"] = value
            per_sample.extend(batch_rows)
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(per_sample)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if len(per_sample) != len(indices):
        raise RuntimeError("Incomplete dataset traversal")
    opts.output_dir.mkdir(parents=True, exist_ok=False)
    with (opts.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    result = {"checkpoint": str(opts.checkpoint),
        "checkpoint_sha256": hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest(),
        "epoch": checkpoint["epoch"], "samples": len(indices), "validation_members_sha256": members_digest,
        "sample_seed": opts.sample_seed, "mask_seed": opts.mask_seed, "batch_size": opts.batch_size,
        "mask_scheme": "same independent point/frame/person random draws reused across subsets and levels",
        "sampling": "random subset of full validation, sorted by original member index",
        "joint_subsets": {"anchor": ANCHORS, "nonanchor": NONANCHORS, "body": BODY},
        "conditions": {name: {"hidden_valid_point_frames": hidden_counts[name],
            "methods": {method: {"correct": value, "top1_percent": 100 * value / len(indices)}
                        for method, value in values.items()}}
            for name, values in counts.items()},
        "inference": "frozen eager model, HF/BF and gamma=1 from checkpoint calibration counts",
        "seconds": time.perf_counter()-started}
    if opts.isolate_anchor:
        result["isolation"] = {"method": "clean 8-channel torso/cross-hand features retained while raw body points are masked",
            "clean_3_vs_8_max_absolute_logit_error": clean_max_abs_error,
            "clean_3_vs_8_prediction_disagreements": clean_prediction_disagreements,
            "clean_logits_allclose_rtol": 1e-4, "clean_logits_allclose_atol": 1e-4,
            "caveat": "Preserved clean local features are an oracle intervention, not an available inference method."}
    (opts.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
