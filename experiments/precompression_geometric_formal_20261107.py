"""Formal precompression anchor-dropout screen with fixed geometric fusion."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # noqa: E402
from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress
from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask
from isaa.face_pair_training import compile_model
from isaa.models.face_pair_fusion import FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_corruption import _soft_weights

ANCHORS = (5, 6, 11, 12)


def member_seed(name: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big") % (2**63 - 1)


def tensor_hash(digest, *values):
    for value in values:
        digest.update(value.contiguous().cpu().numpy().tobytes())


def interpolate_short_anchor_gaps(features, frame_mask, max_gap):
    anchors = features[:, :, :, list(ANCHORS), :]
    batch, _, time, joints, people = anchors.shape
    position = torch.arange(time, device=features.device).view(1, time, 1, 1).expand(batch, time, joints, people)
    frames = frame_mask.bool()
    valid = (anchors[:, 2] > 0) & frames[:, :, None, None]
    previous = torch.where(valid, position, -1).cummax(dim=1).values
    after = torch.where(valid, position, time).flip(1).cummin(dim=1).values.flip(1)
    gap = after - previous - 1
    invalid_prefix = F.pad((~frames).long().cumsum(dim=1), (1, 0))[:, :, None, None].expand(batch, time + 1, joints, people)
    lo = previous.clamp(0, time - 1)
    hi = after.clamp(0, time - 1)
    invalid_between = invalid_prefix.gather(1, hi + 1) - invalid_prefix.gather(1, lo)
    fill = (~valid & frames[:, :, None, None] & (previous >= 0) & (after < time) & (gap <= max_gap) & (invalid_between == 0))
    gather_shape = (-1, 3, -1, -1, -1)
    left = anchors.gather(2, lo[:, None].expand(*gather_shape))
    right = anchors.gather(2, hi[:, None].expand(*gather_shape))
    alpha = ((position - previous).float() / (after - previous).clamp_min(1).float())[:, None]
    replacement = left + alpha * (right - left)
    result = features.clone()
    result[:, :, :, list(ANCHORS), :] = torch.where(fill[:, None], replacement, anchors)
    return result, int(fill.sum())


class PairedPrecompression(torch.utils.data.Dataset):
    def __init__(self, dataset, members, probability, seed):
        self.dataset, self.members = dataset, members
        self.probability, self.seed = probability, seed
        index = {name: i for i, name in enumerate(dataset.members)}
        self.indices = [index[name] for name in members]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        raw, frame = self.dataset._read_raw_sample(self.indices[i])
        rng = torch.Generator().manual_seed(member_seed(self.members[i], self.seed))
        anchor = raw[:, :, list(ANCHORS), :]
        hidden = (torch.rand(anchor[2].shape, generator=rng) < self.probability)
        hidden &= (anchor[2] > 0) & frame[:, None, None]
        corrupt = raw.clone()
        corrupt[:, :, list(ANCHORS), :] = anchor.masked_fill(hidden[None], 0)
        outputs = []
        for value in (raw, corrupt):
            value, mask = _confidence_weighted_temporal_compress(value, frame, self.dataset.window_size)
            value, mask = _temporal_crop_or_pad_sample_with_mask(value, mask, self.dataset.window_size, random_start=False, deterministic_crop=self.dataset.deterministic_temporal_crop)
            value[:2] *= value[2:3] > 0
            outputs.append(value.masked_fill(~mask[None, :, None, None], 0).contiguous())
        return outputs[0], outputs[1], int(self.dataset.labels[self.indices[i]]), mask, int(hidden.sum()), raw.shape[1]


def geometric_fusion(output, weights):
    hand = F.log_softmax(output["hand_face_logits"].float(), -1)
    body = F.log_softmax(output["body_face_logits"].float(), -1)
    arithmetic = torch.logaddexp(weights.log() + hand, torch.log1p(-weights) + body)
    arithmetic = arithmetic - torch.logsumexp(arithmetic, -1, keepdim=True)
    geometric = weights * hand + (1.0 - weights) * body
    geometric = geometric - torch.logsumexp(geometric, -1, keepdim=True)
    return arithmetic, torch.logaddexp(arithmetic, geometric) - np.log(2.0)


def main() -> None:
    checkpoint_path = Path("outputs/best_face_pair_stable_100_20261103/20261103-052121-a73ca490/best.pt")
    archive = Path("data/ntu60_skeletons_rtmw.zip")
    output_dir = Path(__import__("os").environ.get("PRECOMPRESSION_OUTPUT_DIR", "outputs/precompression_geometric_max5_formal_20261107"))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    model = FacePairFusion(saved["num_classes"])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model = model.to(device).eval()
    weights = _soft_weights(model).to(device)
    model = compile_model(model, saved["compile_mode"])
    manifest = json.loads((checkpoint_path.parent / "split_manifest.json").read_text(encoding="utf-8"))["validation"]
    members = manifest["members"]
    dataset = RTMWZipDataset(
        archive, split="val", split_protocol=saved["split"], window_size=saved["window_size"],
        num_joints=133, num_classes=saved["num_classes"], layout="rtmw_133",
        max_persons=saved["max_persons"], score_normalization=saved.get("score_normalization", "auto"),
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted",
        deterministic_temporal_crop=saved.get("deterministic_temporal_crop", "center"),
    )
    probability = float(__import__("os").environ.get("PRECOMPRESSION_PROBABILITY", "0.05"))
    paired = PairedPrecompression(dataset, members, probability, 1729)
    loader = DataLoader(
        paired, batch_size=64, shuffle=False, num_workers=8, prefetch_factor=2,
        multiprocessing_context="spawn", collate_fn=None,
    )
    names = ("clean_unmodified", "clean_max5", "pre_dropout_unmodified", "pre_dropout_max5")
    rows = []
    hashes = {name: hashlib.sha256() for name in ("clean", "corrupted", "frame_mask")}
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (clean, corrupted, labels, frame_mask, hidden, original_frames) in enumerate(loader, 1):
            tensor_hash(hashes["clean"], clean)
            tensor_hash(hashes["corrupted"], corrupted)
            tensor_hash(hashes["frame_mask"], frame_mask)
            clean, corrupted, labels, frame_mask = (value.to(device) for value in (clean, corrupted, labels, frame_mask))
            repaired = interpolate_short_anchor_gaps(corrupted, frame_mask, 5)[0]
            tensors = (clean, repaired, corrupted, repaired)
            sample_offset = len(rows) // 4
            for condition, tensor in zip(names, tensors):
                output = model(tensor, frame_mask)
                arithmetic, geometric = geometric_fusion(output, weights)
                truth = labels.cpu().numpy()
                arithmetic_prediction = arithmetic.argmax(-1).cpu().numpy()
                geometric_prediction = geometric.argmax(-1).cpu().numpy()
                for index, label in enumerate(truth):
                    rows.append({
                        "sample": members[sample_offset + index], "label": int(label), "condition": condition,
                        "arithmetic_prediction": int(arithmetic_prediction[index]),
                        "geometric_prediction": int(geometric_prediction[index]),
                        "hidden_original_points": int(hidden[index]), "original_frames": int(original_frames[index]),
                    })
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f"progress={step}/{len(loader)} samples={len(rows)//4}", flush=True)
    subjects = np.asarray([int(re.search(r"P(\d{3})", row["sample"]).group(1)) for row in rows[::4]])
    unique, group = np.unique(subjects, return_inverse=True)
    draws = np.random.default_rng(20261107).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=20_000)
    result = {}
    for condition in names:
        block = [row for row in rows if row["condition"] == condition]
        arithmetic = np.asarray([int(row["arithmetic_prediction"]) == int(row["label"]) for row in block])
        geometric = np.asarray([int(row["geometric_prediction"]) == int(row["label"]) for row in block])
        delta = geometric.astype(np.int8) - arithmetic.astype(np.int8)
        group_delta = np.bincount(group, weights=delta, minlength=len(unique))
        group_size = np.bincount(group, minlength=len(unique))
        result[condition] = {
            "arithmetic_correct": int(arithmetic.sum()), "geometric_correct": int(geometric.sum()),
            "difference_pp": float(100 * delta.mean()), "corrected": int((delta == 1).sum()),
            "regressed": int((delta == -1).sum()),
            "subject_bootstrap_95_ci_pp": np.percentile(100 * (draws @ group_delta) / (draws @ group_size), [2.5, 97.5]).tolist(),
        }
    output_dir.mkdir(parents=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest["sha256"], "samples": len(members), "probability": 0.05,
        "seed": 1729, "probability": probability, "repair": "max5 precompression anchor gap interpolation",
        "fusion": "formal gamma=1 arithmetic vs strength 0.5 arithmetic-geometric mixture",
        "compile": {"enabled": True, "backend": "inductor", "mode": saved["compile_mode"]},
        "input_sha256": {name: digest.hexdigest() for name, digest in hashes.items()},
        "metrics": result, "seconds": time.perf_counter() - started,
        "caveat": "synthetic precompression anchor dropout; validation reused",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
