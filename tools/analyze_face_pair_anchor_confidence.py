"""Summarize RTMW anchor scores before and after temporal compression."""
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=20261105)
    opts = parser.parse_args()
    if opts.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {opts.output_dir}")
    if opts.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    torch.set_num_threads(1)
    manifest = json.loads(opts.manifest.read_text(encoding="utf-8"))
    names = manifest["calibration"]["members"]
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    if digest != manifest["calibration"]["sha256"]:
        raise RuntimeError("Calibration manifest digest mismatch")
    dataset = RTMWZipDataset(opts.archive, split="train", split_protocol="xsub60", window_size=100,
        num_joints=133, num_classes=60, layout="rtmw_133", max_persons=2,
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted")
    index = {member: i for i, member in enumerate(dataset.members)}
    if len(index) != len(dataset.members) or len(set(names)) != len(names):
        raise RuntimeError("Duplicate dataset or manifest members")
    generator = torch.Generator().manual_seed(opts.sample_seed)
    positions = torch.randperm(len(names), generator=generator)[:opts.sample_count].tolist()
    selected = [names[i] for i in positions]
    score_parts = {"raw": [], "compressed": []}
    rows = []
    for name in selected:
        raw, frame_mask = dataset._read_raw_sample(index[name])
        compressed, compressed_mask = _confidence_weighted_temporal_compress(raw, frame_mask, 100)
        rows.append({"sample": name, "original_frames": raw.shape[1],
                     "compressed_frames": compressed.shape[1], "did_compress": raw.shape[1] > 100})
        for tag, data, frames in (("raw", raw, frame_mask), ("compressed", compressed, compressed_mask)):
            scores = data[2, :, list(ANCHORS), :]
            present = (scores > 0).any(dim=1) & frames[:, None]
            score_parts[tag].append(scores.permute(0, 2, 1)[present].reshape(-1))
    thresholds = (0.01, 0.05, 0.1, 0.2, 0.5)
    summaries = {}
    for tag, parts in score_parts.items():
        scores = torch.cat(parts)
        positive = scores[scores > 0]
        if not positive.numel():
            raise RuntimeError(f"No positive anchor scores in {tag}")
        quantiles = torch.quantile(positive, torch.tensor([0.01, 0.05, 0.10, 0.50])).tolist()
        summaries[tag] = {"anchor_point_frame_scores": scores.numel(),
            "zero_scores": int((scores == 0).sum()), "positive_scores": positive.numel(),
            "positive_below_threshold_counts": {str(t): int((positive < t).sum()) for t in thresholds},
            "positive_score_quantiles": dict(zip(("0.01", "0.05", "0.10", "0.50"), quantiles))}
    lengths = torch.tensor([row["original_frames"] for row in rows], dtype=torch.float64)
    compressed_count = sum(row["did_compress"] for row in rows)
    result = {"samples": len(rows), "sample_seed": opts.sample_seed,
        "parent_calibration_members_sha256": digest,
        "selected_members_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "score_population": "four anchor scores for each valid frame/person having >=1 positive anchor; excludes padding persons and frames",
        "scores": "RTMW loader's auto-normalized score channel, before and after confidence-weighted compression",
        "sequence_frames": {"minimum": int(lengths.min()), "median": float(lengths.median()),
                            "maximum": int(lengths.max()), "over_100_count": compressed_count,
                            "over_100_fraction": compressed_count / len(rows)},
        "score_summary": summaries,
        "caveat": "Only a random calibration subset of this archive; not a general detector-confidence distribution."}
    opts.output_dir.mkdir(parents=True, exist_ok=False)
    with (opts.output_dir / "samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (opts.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
