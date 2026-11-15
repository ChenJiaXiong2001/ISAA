"""CPU-only manifest and deterministic mask audit, without model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from tools.evaluate_face_pair_correlated_occlusion import CHECKPOINT_SHA256, Paired, corrupt
from tools.evaluate_face_pair_precompression_repair import select_members, tensor_hash
from isaa.data.rtmw_zip_dataset import RTMWZipDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_samples < 0:
        raise ValueError("Invalid sample count")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Wrong checkpoint")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["args"]
    section = json.loads(args.manifest.read_text(encoding="utf-8"))[args.partition]
    full = section["members"]
    if hashlib.sha256("\n".join(full).encode()).hexdigest() != section["sha256"]:
        raise ValueError("Manifest digest mismatch")
    members, _ = select_members(full, None, args.max_samples)
    data = RTMWZipDataset(args.archive, split="train" if args.partition == "calibration" else "val",
        split_protocol=saved["split"], window_size=saved["window_size"], num_joints=133,
        num_classes=saved["num_classes"], layout="rtmw_133", max_persons=saved["max_persons"],
        score_normalization=saved.get("score_normalization", "auto"), feature_mode="raw", augment=False,
        temporal_compression="confidence_weighted", deterministic_temporal_crop=saved.get("deterministic_temporal_crop", "center"))
    paired = Paired(data, members)
    hashes = {name: hashlib.sha256() for name in ("clean", "corrupted", "frame_mask")}
    rows = []
    torch.set_num_threads(1)
    for index, member in enumerate(members):
        clean, damaged, _, mask, info = paired[index]
        for name, value in (("clean", clean), ("corrupted", damaged), ("frame_mask", mask)):
            tensor_hash(hashes[name], value.unsqueeze(0))
        if index < 20:
            raw, frames = data._read_raw_sample(paired.indices[index])
            first, first_info = corrupt(raw, frames, member)
            second, second_info = corrupt(raw, frames, member)
            if not torch.equal(first, second) or first_info != second_info or info != first_info:
                raise ValueError(f"Mask is not repeatable for {member}")
        rows.append({"sample": member, **info})
        if (index + 1) % 200 == 0 or index + 1 == len(members):
            print(f"samples={index + 1}/{len(members)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {"samples": len(rows), "partition": args.partition,
              "members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
              "input_sha256": {name: digest.hexdigest() for name, digest in hashes.items()},
              "eligible_samples": sum(row["eligible"] for row in rows),
              "hidden_points": sum(row["hidden_points"] for row in rows), "repeatability_checked": min(20, len(rows))}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
