"""Select remaining long calibration clips by original length only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train
import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from tools.evaluate_face_pair_precompression_repair import select_members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["calibration"]
    members = manifest["members"]
    if hashlib.sha256("\n".join(members).encode()).hexdigest() != manifest["sha256"]:
        raise ValueError("Calibration manifest digest mismatch")
    exclude, exclusion_source = select_members(members, args.exclude_file, 0)
    exclusion = set(exclude)
    dataset = RTMWZipDataset(args.archive, split="train", split_protocol="xsub60", window_size=100,
        num_joints=133, num_classes=60, layout="rtmw_133", max_persons=2,
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted")
    index = {name: i for i, name in enumerate(dataset.members)}
    if len(index) != len(dataset.members):
        raise ValueError("Duplicate archive members")
    scanned, selected, excluded_seen = [], [], []
    for i, name in enumerate(members, 1):
        raw, _ = dataset._read_raw_sample(index[name])
        length = int(raw.shape[1])
        row = {"sample": name, "original_frames": length, "excluded": int(name in exclusion)}
        scanned.append(row)
        if name in exclusion:
            excluded_seen.append(row)
        elif length > 100:
            selected.append({"sample": name, "original_frames": length})
        if i % 200 == 0 or i == len(members):
            print(f"scanned={i}/{len(members)} remaining_long={len(selected)}", flush=True)
    if len(excluded_seen) != len(exclusion) or any(row["original_frames"] <= 100 for row in excluded_seen):
        raise ValueError("Exclusion list contains missing or non-long samples")
    if not selected:
        raise ValueError("No remaining long clips")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("members.csv", selected), ("all_lengths.csv", scanned)):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    list_path = args.output_dir / "members.csv"
    summary = {"partition": "calibration", "manifest_sha256": manifest["sha256"],
               "scanned": len(scanned), "excluded": len(exclude), "exclusion_source": exclusion_source,
               "selected": len(selected), "selected_101_199": sum(100 < r["original_frames"] < 200 for r in selected),
               "selected_200_plus": sum(r["original_frames"] >= 200 for r in selected),
               "minimum_length": min(r["original_frames"] for r in selected),
               "maximum_length": max(r["original_frames"] for r in selected),
               "members_file_sha256": hashlib.sha256(list_path.read_bytes()).hexdigest(),
               "members_sha256": hashlib.sha256("\n".join(r["sample"] for r in selected).encode()).hexdigest(),
               "selection": "manifest order, original length > 100, exclusion list removed; labels and predictions not inspected"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
