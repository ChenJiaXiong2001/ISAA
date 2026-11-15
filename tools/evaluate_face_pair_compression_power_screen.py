"""Screen confidence-power temporal compression on a frozen face-pair model.

This is an inference-only input construction screen.  It keeps the model,
gamma=1 calibration counters, member order, and labels fixed while replacing
the coordinate weight in each compression bin with ``positive_score ** p``.
The output confidence remains the ordinary mean score used by the baseline.
No training or validation recalibration is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA wheel paths before importing torch.

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights


def compress_score_power(raw, frame_mask, target_frames, power):
    """Compress one raw 3 x T x N x M sample with a fixed score exponent."""
    if raw.ndim != 4 or raw.size(0) < 3:
        raise ValueError(f"Expected 3 x T x N x M raw input, got {tuple(raw.shape)}")
    frames = raw.size(1)
    if frames <= target_frames:
        return raw.contiguous(), frame_mask.bool().contiguous()
    if not math.isfinite(power) or power < 0:
        raise ValueError("power must be finite and nonnegative")
    starts = torch.floor(torch.arange(target_frames, dtype=torch.float64, device=raw.device)
                         * frames / target_frames).long()
    ends = torch.floor(torch.arange(1, target_frames + 1, dtype=torch.float64,
                                    device=raw.device) * frames / target_frames).long()
    scores = raw[2:3].clamp(0.0, 1.0)
    valid = scores > 0
    if power == 0:
        weights = valid.to(raw.dtype)
    else:
        weights = scores.pow(power)
    coords = raw[:2]
    coord_prefix = torch.cat((coords.new_zeros(2, 1, *coords.shape[2:]),
                              (coords * weights).cumsum(1)), 1)
    weight_prefix = torch.cat((weights.new_zeros(1, 1, *weights.shape[2:]),
                               weights.cumsum(1)), 1)
    score_prefix = torch.cat((scores.new_zeros(1, 1, *scores.shape[2:]),
                              scores.cumsum(1)), 1)
    coord_sum = coord_prefix.index_select(1, ends) - coord_prefix.index_select(1, starts)
    weight_sum = weight_prefix.index_select(1, ends) - weight_prefix.index_select(1, starts)
    coords_out = coord_sum / weight_sum.clamp_min(1e-6)
    coords_out = torch.where(weight_sum > 1e-6, coords_out, torch.zeros_like(coords_out))
    lengths = (ends - starts).to(raw.dtype).view(1, target_frames, 1, 1)
    scores_out = (score_prefix.index_select(1, ends) - score_prefix.index_select(1, starts)) / lengths
    compressed = torch.cat((coords_out, scores_out), 0).contiguous()
    mask = frame_mask.flatten().bool().to(raw.device)
    mask_prefix = torch.cat((torch.zeros(1, dtype=torch.int64, device=raw.device),
                             mask.to(torch.int64).cumsum(0)))
    compressed_mask = (mask_prefix.index_select(0, ends) - mask_prefix.index_select(0, starts) > 0)
    return compressed, compressed_mask.contiguous()


class PowerDataset(Dataset):
    def __init__(self, dataset, indices, power):
        self.dataset = dataset
        self.indices = list(indices)
        self.power = float(power)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        raw, file_mask = self.dataset._read_raw_sample(index)
        raw, frame_mask = compress_score_power(raw, file_mask, self.dataset.window_size, self.power)
        raw, frame_mask = _temporal_crop_or_pad_sample_with_mask(
            raw, frame_mask, self.dataset.window_size, random_start=False,
            deterministic_crop=self.dataset.deterministic_temporal_crop)
        raw[:2] *= raw[2:3] > 0
        raw = raw.masked_fill(~frame_mask[None, :, None, None], 0).contiguous()
        return raw, torch.tensor(int(self.dataset.labels[index]), dtype=torch.long), frame_mask


def member_digest(members):
    return hashlib.sha256("\n".join(members).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--powers", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_samples <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid sample, batch, or worker count")
    powers = [float(value) for value in args.powers]
    if not powers or len(set(powers)) != len(powers) or any(not math.isfinite(x) or x < 0 for x in powers):
        raise ValueError("powers must be unique finite nonnegative values")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    if saved["model_variant"] != FACE_PAIR_VARIANT:
        raise ValueError("This screen expects the standard best-face-pair checkpoint")
    model = FacePairFusion(saved["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(args.device).eval()
    weights = _soft_weights(model).to(args.device)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    members = manifest["validation"]["members"]
    if member_digest(members) != manifest["validation"]["sha256"]:
        raise ValueError("Validation manifest digest mismatch")
    members = members[:args.max_samples]
    dataset = RTMWZipDataset(
        args.archive, split="val", split_protocol=saved["split"],
        window_size=saved["window_size"], num_joints=133, num_classes=saved["num_classes"],
        layout="rtmw_133", max_persons=saved["max_persons"], feature_mode="raw",
        augment=False, temporal_compression="none", deterministic_temporal_crop="center",
        max_samples=len(members),
    )
    if list(dataset.members) != members:
        raise ValueError("Dataset member order differs from manifest prefix")
    loader_options = dict(batch_size=args.batch_size, shuffle=False, drop_last=False,
                          num_workers=args.num_workers,
                          collate_fn=partial(collate_rtmw, feature_mode="raw"))
    if args.num_workers:
        loader_options.update(prefetch_factor=2, multiprocessing_context="spawn")

    results = {}
    prediction_rows = []
    started = time.perf_counter()
    for power in powers:
        current = PowerDataset(dataset, range(len(members)), power)
        loader = DataLoader(current, **loader_options)
        correct = {"HF": 0, "BF": 0, "soft": 0}
        rows = 0
        digest = hashlib.sha256()
        mode_started = time.perf_counter()
        with torch.inference_mode():
            for step, (features, labels, frame_mask) in enumerate(loader, 1):
                digest.update(features.contiguous().cpu().numpy().tobytes())
                digest.update(frame_mask.contiguous().cpu().numpy().tobytes())
                features, labels, frame_mask = (value.to(args.device) for value in (features, labels, frame_mask))
                output = model(features, frame_mask)
                scores = {"HF": output["hand_face_logits"], "BF": output["body_face_logits"],
                          "soft": _soft_logits(output, weights)}
                if not all(bool(torch.isfinite(value).all()) for value in scores.values()):
                    raise RuntimeError(f"Nonfinite scores at power={power}, batch={step}")
                predictions = {name: value.argmax(-1) for name, value in scores.items()}
                for name, prediction in predictions.items():
                    correct[name] += int((prediction == labels).sum())
                for index in range(len(labels)):
                    prediction_rows.append({"power": power, "sample": members[rows + index],
                                            "label": int(labels[index]),
                                            **{f"{name}_prediction": int(prediction[index])
                                               for name, prediction in predictions.items()}})
                rows += len(labels)
        results[str(power)] = {
            "power": power, "samples": rows, "input_sha256": digest.hexdigest(),
            "metrics": {name: {"correct": value, "top1_percent": 100.0 * value / rows}
                         for name, value in correct.items()},
            "seconds": time.perf_counter() - mode_started,
        }
        print(json.dumps(results[str(power)]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["power", "sample", "label", "HF_prediction", "BF_prediction", "soft_prediction"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(prediction_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "manifest_validation_sha256": manifest["validation"]["sha256"],
        "evaluated_members_sha256": member_digest(members), "samples": len(members),
        "powers": powers, "weight_source": "frozen checkpoint gamma=1 calibration counters",
        "coordinate_weight": "positive score raised to p; output score remains ordinary bin mean",
        "results": results, "seconds": time.perf_counter() - started,
        "screen_only": True, "no_training": True,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
