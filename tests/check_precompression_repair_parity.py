"""Check evaluation-wrapper parity against real RTMW calibration samples."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from tools.evaluate_face_pair_precompression_repair import PairedPrecompression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-a", type=Path, required=True)
    parser.add_argument("--prediction-b", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.samples < 4 or args.samples % 2:
        raise ValueError("samples must be an even number >= 4")
    calibration = json.loads(args.manifest.read_text(encoding="utf-8"))["calibration"]["members"]
    dataset = RTMWZipDataset(args.archive, split="train", split_protocol="xsub60", window_size=100,
        num_joints=133, num_classes=60, layout="rtmw_133", max_persons=2,
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted")
    index = {name: i for i, name in enumerate(dataset.members)}
    short, long = [], []
    for name in calibration:
        raw, _ = dataset._read_raw_sample(index[name])
        group = long if raw.shape[1] > 100 else short
        if len(group) < args.samples // 2:
            group.append(name)
        if len(short) == len(long) == args.samples // 2:
            break
    if len(short) != args.samples // 2 or len(long) != args.samples // 2:
        raise AssertionError("Insufficient short or long clips")
    members = short + long
    paired = PairedPrecompression(dataset, members, 0.05, 1729)
    lengths = []
    for i in range(len(paired)):
        clean, _, label, mask, _, length = paired[i]
        original, original_label, original_mask = dataset[paired.indices[i]]
        if not (torch.equal(clean, original) and torch.equal(mask, original_mask) and label == int(original_label)):
            raise AssertionError(f"Data parity failed: {members[i]}")
        lengths.append(length)
    if not (any(x <= 100 for x in lengths) and any(x > 100 for x in lengths)):
        raise AssertionError("Parity sample must include both compressed and padded clips")
    with args.prediction_a.open(encoding="utf-8", newline="") as stream:
        a = list(csv.DictReader(stream))
    with args.prediction_b.open(encoding="utf-8", newline="") as stream:
        b = list(csv.DictReader(stream))
    if a != b:
        raise AssertionError("Per-sample predictions differ across batch sizes")
    print(json.dumps({"samples": len(members), "lengths": lengths,
                      "clean_tensor_mask_label_parity": True,
                      "cross_batch_prediction_parity": True}))


if __name__ == "__main__":
    main()
