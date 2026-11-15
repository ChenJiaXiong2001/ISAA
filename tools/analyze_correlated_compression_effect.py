"""Measure how fixed raw-frame occlusion survives compression, without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from tools.evaluate_face_pair_anchor_gaps import interpolate_short_anchor_gaps
from tools.evaluate_face_pair_anchor_interpolation import ANCHORS
from tools.evaluate_face_pair_correlated_occlusion import CHECKPOINT_SHA256, Paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Wrong checkpoint")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["args"]
    section = json.loads(args.manifest.read_text(encoding="utf-8"))["calibration"]
    members = section["members"]
    if hashlib.sha256("\n".join(members).encode()).hexdigest() != section["sha256"]:
        raise ValueError("Manifest digest mismatch")
    data = RTMWZipDataset(args.archive, split="train", split_protocol=saved["split"],
        window_size=saved["window_size"], num_joints=133, num_classes=saved["num_classes"],
        layout="rtmw_133", max_persons=saved["max_persons"], score_normalization=saved.get("score_normalization", "auto"),
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted",
        deterministic_temporal_crop=saved.get("deterministic_temporal_crop", "center"))
    paired = Paired(data, members)
    counts = {key: 0 for key in ("samples_with_any_compressed_change", "changed_joint_frames",
              "samples_with_new_zero_anchor", "new_zero_anchor_frames", "samples_with_extra_repaired_anchor",
              "extra_repaired_anchor_frames", "clean_repaired_anchor_frames", "occluded_repaired_anchor_frames")}
    torch.set_num_threads(1)
    for index in range(len(paired)):
        clean, damaged, _, mask, _ = paired[index]
        changed = (clean != damaged).any(dim=0)
        counts["changed_joint_frames"] += int(changed.sum())
        counts["samples_with_any_compressed_change"] += int(changed.any())
        new_zero = (clean[2, :, list(ANCHORS), :] > 0) & (damaged[2, :, list(ANCHORS), :] <= 0)
        counts["new_zero_anchor_frames"] += int(new_zero.sum())
        counts["samples_with_new_zero_anchor"] += int(new_zero.any())
        fixed_clean, count_clean = interpolate_short_anchor_gaps(clean.unsqueeze(0), mask.unsqueeze(0), 5)
        fixed_damaged, count_damaged = interpolate_short_anchor_gaps(damaged.unsqueeze(0), mask.unsqueeze(0), 5)
        clean_filled = (clean[2, :, list(ANCHORS), :] <= 0) & (fixed_clean[0, 2, :, list(ANCHORS), :] > 0)
        damaged_filled = (damaged[2, :, list(ANCHORS), :] <= 0) & (fixed_damaged[0, 2, :, list(ANCHORS), :] > 0)
        extra = damaged_filled & ~clean_filled
        counts["clean_repaired_anchor_frames"] += count_clean
        counts["occluded_repaired_anchor_frames"] += count_damaged
        counts["extra_repaired_anchor_frames"] += int(extra.sum())
        counts["samples_with_extra_repaired_anchor"] += int(extra.any())
        if (index + 1) % 500 == 0 or index + 1 == len(paired):
            print(f"samples={index + 1}/{len(paired)}", flush=True)
    result = {"samples": len(paired), "members_sha256": section["sha256"], **counts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
