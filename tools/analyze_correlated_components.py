"""Count actual body/hand components hidden by the fixed raw-frame masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from tools.evaluate_face_pair_correlated_occlusion import CHECKPOINT_SHA256, LENGTH, Paired, REGIONS, corrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("calibration", "validation"), default="calibration")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Wrong checkpoint")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["args"]
    section = json.loads(args.manifest.read_text(encoding="utf-8"))[args.partition]
    members = section["members"]
    if hashlib.sha256("\n".join(members).encode()).hexdigest() != section["sha256"]:
        raise ValueError("Manifest digest mismatch")
    data = RTMWZipDataset(args.archive, split="train" if args.partition == "calibration" else "val", split_protocol=saved["split"],
        window_size=saved["window_size"], num_joints=133, num_classes=saved["num_classes"],
        layout="rtmw_133", max_persons=saved["max_persons"], score_normalization=saved.get("score_normalization", "auto"),
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted",
        deterministic_temporal_crop=saved.get("deterministic_temporal_crop", "center"))
    paired = Paired(data, members)
    counts = {"eligible": 0, "body_and_hand_both_any": 0, "body_and_hand_each_frame": 0,
              "no_body": 0, "no_hand": 0, "body_hidden_point_frames": 0, "hand_hidden_point_frames": 0}
    torch.set_num_threads(1)
    for index, member in enumerate(members):
        raw, frames = data._read_raw_sample(paired.indices[index])
        _, info = corrupt(raw, frames, member)
        if info["eligible"]:
            counts["eligible"] += 1
            points = raw[2, info["start"]:info["start"] + LENGTH, list(REGIONS[info["side"]]), info["person"]] > 0
            body, hand = points[:, :4], points[:, 4:]
            b, h = int(body.sum()), int(hand.sum())
            counts["body_and_hand_both_any"] += int(b > 0 and h > 0)
            counts["body_and_hand_each_frame"] += int(bool((body.any(dim=1) & hand.any(dim=1)).all()))
            counts["no_body"] += int(b == 0)
            counts["no_hand"] += int(h == 0)
            counts["body_hidden_point_frames"] += b
            counts["hand_hidden_point_frames"] += h
        if (index + 1) % 500 == 0 or index + 1 == len(members):
            print(f"samples={index + 1}/{len(members)}", flush=True)
    result = {"samples": len(members), "partition": args.partition, "members_sha256": section["sha256"], **counts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
