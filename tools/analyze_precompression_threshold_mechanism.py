"""Audit threshold decisions on paired original-frame anchor dropout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure the project environment before importing torch.
import torch

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress
from tools.evaluate_face_pair_anchor_confidence_repair import apply_score_threshold, interpolate_short_anchor_gaps
from tools.evaluate_face_pair_precompression_repair import ANCHORS, member_seed, select_members, tensor_hash


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--members-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--probability", type=float, default=0.05)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--threshold", type=float, action="append", help="Fixed confidence threshold; repeatable")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0 <= args.probability <= 1 or args.max_gap < 1:
        raise ValueError("Invalid probability or maximum gap")
    thresholds = args.threshold if args.threshold is not None else [0.95, 0.98]
    if not thresholds or any(not 0 < t < 1 for t in thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be unique and in (0, 1)")
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["calibration"]
    if hashlib.sha256("\n".join(manifest["members"]).encode()).hexdigest() != manifest["sha256"]:
        raise ValueError("Calibration manifest digest mismatch")
    members, source = select_members(manifest["members"], args.members_file, 0)
    dataset = RTMWZipDataset(args.archive, split="train", split_protocol="xsub60", window_size=100,
        num_joints=133, num_classes=60, layout="rtmw_133", max_persons=2,
        feature_mode="raw", augment=False, temporal_compression="confidence_weighted")
    index = {name: i for i, name in enumerate(dataset.members)}
    if len(index) != len(dataset.members):
        raise ValueError("Duplicate dataset members")
    totals = Counter()
    per_sample = []
    hashes = {name: hashlib.sha256() for name in ("clean", "corrupted", "frame_mask")}
    for name in members:
        raw, frame = dataset._read_raw_sample(index[name])
        frames = raw.shape[1]
        if frames <= 100:
            raise ValueError(f"Expected a compressed long clip: {name}")
        anchor = raw[:, :, list(ANCHORS), :]
        generator = torch.Generator().manual_seed(member_seed(name, args.seed))
        hidden = (torch.rand(anchor[2].shape, generator=generator) < args.probability)
        hidden &= (anchor[2] > 0) & frame[:, None, None]
        corrupted = raw.clone()
        corrupted[:, :, list(ANCHORS), :] = anchor.masked_fill(hidden[None], 0)
        clean, mask = _confidence_weighted_temporal_compress(raw, frame, 100)
        corrupted, mask2 = _confidence_weighted_temporal_compress(corrupted, frame, 100)
        if not torch.equal(mask, mask2):
            raise RuntimeError("Compression frame masks differ")
        for value in (clean, corrupted):
            value[:2] *= value[2:3] > 0
            value.masked_fill_(~mask[None, :, None, None], 0)
        tensor_hash(hashes["clean"], clean)
        tensor_hash(hashes["corrupted"], corrupted)
        tensor_hash(hashes["frame_mask"], mask)
        affected = torch.stack([hidden[t * frames // 100:(t + 1) * frames // 100].any(0) for t in range(100)])
        clean_score = clean[2, :, list(ANCHORS), :]
        corrupt_score = corrupted[2, :, list(ANCHORS), :]
        eligible = (clean_score > 0) & mask[:, None, None]
        affected &= eligible
        affected_positive = affected & (corrupt_score > 0)
        affected_zero = affected & (corrupt_score == 0)
        unaffected_positive = eligible & ~affected & (corrupt_score > 0)
        row = {"sample": name, "original_frames": frames, "hidden_original_points": int(hidden.sum()),
               "clean_positive": int(eligible.sum()), "affected_bins": int(affected.sum()),
               "affected_zero": int(affected_zero.sum()), "affected_positive": int(affected_positive.sum()),
               "unaffected_positive": int(unaffected_positive.sum())}
        for threshold in thresholds:
            tag = f"t{threshold:g}"
            flagged = (corrupt_score > 0) & (corrupt_score <= threshold) & mask[:, None, None]
            clean_flagged = eligible & (clean_score <= threshold)
            thresholded, _ = apply_score_threshold(corrupted[None], threshold)
            repaired, _ = interpolate_short_anchor_gaps(thresholded, mask[None], args.max_gap)
            restored = repaired[0, 2, :, list(ANCHORS), :] > 0
            true_flag = affected_positive & flagged
            false_flag = unaffected_positive & flagged
            row.update({f"{tag}_clean_flagged": int(clean_flagged.sum()),
                        f"{tag}_affected_positive_flagged": int(true_flag.sum()),
                        f"{tag}_unaffected_positive_flagged": int(false_flag.sum()),
                        f"{tag}_affected_positive_flagged_restored": int((true_flag & restored).sum()),
                        f"{tag}_unaffected_positive_flagged_restored": int((false_flag & restored).sum())})
        totals.update({key: value for key, value in row.items() if key not in ("sample", "original_frames")})
        per_sample.append(row)
    result = {"partition": "calibration", "samples": len(members), "member_source": source,
              "members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
              "manifest_sha256": manifest["sha256"], "input_sha256": {k: v.hexdigest() for k, v in hashes.items()},
              "seed": args.seed, "probability": args.probability, "max_gap": args.max_gap,
              "definitions": {"affected": "output bin contains at least one deliberately hidden valid anchor point",
                 "affected_positive": "affected, clean score > 0, corrupted score > 0, valid output frame",
                 "unaffected_positive": "no deliberately hidden anchor point in bin, clean and corrupted scores > 0, valid output frame",
                 "clean_positive": "positive clean anchor score in a valid output frame",
                 "flagged": "positive corrupted score <= threshold", "restored": "flagged point has positive score after bounded interpolation"},
              "counts": dict(totals), "thresholds": {}}
    for threshold in thresholds:
        tag = f"t{threshold:g}"
        tp, fp = totals[f"{tag}_affected_positive_flagged"], totals[f"{tag}_unaffected_positive_flagged"]
        result["thresholds"][str(threshold)] = {
            "affected_positive_detection_percent": percent(tp, totals["affected_positive"]),
            "unaffected_positive_false_flag_percent": percent(fp, totals["unaffected_positive"]),
            "clean_positive_flag_percent": percent(totals[f"{tag}_clean_flagged"], totals["clean_positive"]),
            "flagged_affected_restored_percent": percent(totals[f"{tag}_affected_positive_flagged_restored"], tp),
            "flagged_unaffected_restored_percent": percent(totals[f"{tag}_unaffected_positive_flagged_restored"], fp)}
    result["limitations"] = "Synthetic dropout; one fixed mask; bins are correlated and sampled from training calibration subjects. Affected does not imply erroneous output coordinates."
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sample[0])); writer.writeheader(); writer.writerows(per_sample)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
