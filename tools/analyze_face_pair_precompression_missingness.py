"""Contrast original-frame and post-compression anchor dropout on fixed clips."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress


ANCHORS = (5, 6, 11, 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {options.output_dir}")
    torch.set_num_threads(1)
    with options.samples.open(encoding="utf-8", newline="") as stream:
        names = [row["sample"] for row in csv.DictReader(stream)]
    if not names or len(set(names)) != len(names):
        raise ValueError("Empty or duplicate sample list")
    dataset = RTMWZipDataset(options.archive, split="train", split_protocol="xsub60", window_size=100,
        num_joints=133, num_classes=60, layout="rtmw_133", max_persons=2,
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted")
    index = {member: i for i, member in enumerate(dataset.members)}
    if len(index) != len(dataset.members):
        raise RuntimeError("Duplicate training members")
    generators = {"pre": torch.Generator().manual_seed(options.seed),
                  "post": torch.Generator().manual_seed(options.seed + 1)}
    totals = {group: {"samples": 0, "valid_positive_output_anchors": 0,
                      "pre_affected_output_bins": 0,
                      "pre_zero": 0, "pre_still_positive_with_lower_score": 0,
                      "pre_still_positive_xy_changed_gt_0_001": 0,
                      "pre_still_positive_xy_changed_gt_0_01": 0, "post_zero": 0}
              for group in ("all", "compressed", "not_compressed")}
    rows = []
    for name in names:
        raw, frame = dataset._read_raw_sample(index[name])
        clean, clean_mask = _confidence_weighted_temporal_compress(raw, frame, 100)
        pre = raw.clone()
        anchor = pre[:, :, list(ANCHORS), :]
        pre_draws = torch.rand(anchor[2].shape, generator=generators["pre"])
        hidden = (pre_draws < 0.05) & (anchor[2] > 0) & frame[:, None, None]
        anchor = anchor.masked_fill(hidden[None], 0)
        pre[:, :, list(ANCHORS), :] = anchor
        compressed_pre, _ = _confidence_weighted_temporal_compress(pre, frame, 100)
        clean_anchor = clean[:, :, list(ANCHORS), :]
        pre_anchor = compressed_pre[:, :, list(ANCHORS), :]
        post_draws = torch.rand(clean_anchor[2].shape, generator=generators["post"])
        present = (clean_anchor[2] > 0).any(dim=1) & clean_mask[:, None]
        eligible = (clean_anchor[2] > 0) & present[:, None, :]
        if raw.shape[1] <= 100:
            affected = hidden
        else:
            starts = torch.arange(100, dtype=torch.int64) * raw.shape[1] // 100
            ends = torch.arange(1, 101, dtype=torch.int64) * raw.shape[1] // 100
            affected = torch.stack([hidden[start:end].any(dim=0) for start, end in zip(starts, ends)])
        affected &= eligible
        zero = affected & (pre_anchor[2] == 0)
        reduced = affected & (pre_anchor[2] > 0) & (pre_anchor[2] < clean_anchor[2] - 1e-4)
        distance = torch.linalg.vector_norm(pre_anchor[:2] - clean_anchor[:2], dim=0)
        moved_001 = affected & (pre_anchor[2] > 0) & (distance > 0.001)
        moved_01 = affected & (pre_anchor[2] > 0) & (distance > 0.01)
        post_zero = eligible & (post_draws < 0.05)
        row = {"sample": name, "original_frames": raw.shape[1], "did_compress": raw.shape[1] > 100,
               "valid_positive_output_anchors": int(eligible.sum()),
               "pre_affected_output_bins": int(affected.sum()), "pre_zero": int(zero.sum()),
               "pre_still_positive_with_lower_score": int(reduced.sum()),
               "pre_still_positive_xy_changed_gt_0_001": int(moved_001.sum()),
               "pre_still_positive_xy_changed_gt_0_01": int(moved_01.sum()), "post_zero": int(post_zero.sum())}
        rows.append(row)
        for group in ("all", "compressed" if row["did_compress"] else "not_compressed"):
            totals[group]["samples"] += 1
            for key in totals[group]:
                if key != "samples":
                    totals[group][key] += row[key]
    result = {"samples": len(names), "selected_members_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "probability": 0.05, "pre_seed": options.seed, "post_seed": options.seed + 1,
        "mask_scheme": "independent anchor point/frame/person zeroing before versus after compression; distinct fixed random draws",
        "score_population": "positive anchor point-frames after clean compression for valid frame/person with >=1 positive anchor",
        "totals": totals, "caveat": "Input perturbation counts only, not model accuracy or an empirical detector failure process. Pre-compression changes are counted only in bins containing a deliberately hidden point; coordinate units are those of the archive. Thresholds reduce sensitivity to prefix-sum floating-point noise."}
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (options.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
