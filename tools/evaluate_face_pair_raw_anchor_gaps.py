"""Paired evaluation of fixed max-5 repair under raw-frame anchor gaps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA libraries before importing torch.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress
from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask
from tools.evaluate_face_pair_anchor_gaps import ANCHORS, contiguous_hidden, interpolate_short_anchor_gaps
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights
from tools.evaluate_face_pair_precompression_repair import member_seed, model_type, select_members, tensor_hash

# Fixed before the evaluation. Start probabilities approximately equalize the
# fraction of missing original frames; overlapping starts can form longer gaps.
GAPS = ((2, 0.025), (5, 0.01), (10, 0.005))


class RawGaps(Dataset):
    def __init__(self, dataset, members, seed):
        self.dataset, self.members, self.seed = dataset, members, seed
        indices = {name: i for i, name in enumerate(dataset.members)}
        if len(indices) != len(dataset.members):
            raise ValueError("Duplicate dataset members")
        self.indices = [indices[name] for name in members]

    def __len__(self):
        return len(self.members)

    def compress(self, raw, frame):
        raw, mask = _confidence_weighted_temporal_compress(raw, frame, self.dataset.window_size)
        raw, mask = _temporal_crop_or_pad_sample_with_mask(
            raw, mask, self.dataset.window_size, random_start=False,
            deterministic_crop=self.dataset.deterministic_temporal_crop)
        raw[:2] *= raw[2:3] > 0
        return raw.masked_fill(~mask[None, :, None, None], 0).contiguous(), mask

    def __getitem__(self, index):
        raw, frame = self.dataset._read_raw_sample(self.indices[index])
        clean, mask = self.compress(raw, frame)
        anchor = raw[:, :, list(ANCHORS), :]
        valid = (anchor[2] > 0) & frame[:, None, None]
        damaged, counts = [], []
        for gap, probability in GAPS:
            generator = torch.Generator().manual_seed(member_seed(f"gap{gap}:{self.members[index]}", self.seed))
            draws = torch.rand((1, raw.shape[1], len(ANCHORS), raw.shape[3]), generator=generator)
            hidden = contiguous_hidden(draws, frame[None], gap, probability)[0] & valid
            altered = raw.clone()
            altered[:, :, list(ANCHORS), :] = anchor.masked_fill(hidden[None], 0)
            compressed, actual_mask = self.compress(altered, frame)
            if not torch.equal(mask, actual_mask):
                raise RuntimeError("Corruption changed the frame mask")
            damaged.append(compressed)
            counts.append(int(hidden.sum()))
        return clean, torch.stack(damaged), int(self.dataset.labels[self.indices[index]]), mask, torch.tensor(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("calibration", "validation"), required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if options.max_samples < 0 or options.batch_size < 1 or options.num_workers < 0:
        raise ValueError("Invalid batch size, worker count, or sample count")
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1 if options.device == "cpu" else 4)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args["temporal_compression"] != "confidence_weighted":
        raise ValueError("Expected confidence-weighted temporal compression")
    cls = model_type(args["model_variant"])
    model = cls(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(options.device).eval()
    weights = _soft_weights(model).to(options.device)
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)
    section = json.loads(options.manifest.read_text(encoding="utf-8"))[options.partition]
    full_members = section["members"]
    if hashlib.sha256("\n".join(full_members).encode()).hexdigest() != section["sha256"]:
        raise ValueError("Manifest digest mismatch")
    members, source = select_members(full_members, None, options.max_samples)
    dataset = RTMWZipDataset(
        options.archive, split="train" if options.partition == "calibration" else "val",
        split_protocol=args["split"], window_size=args["window_size"], num_joints=133,
        num_classes=args["num_classes"], layout="rtmw_133", max_persons=args["max_persons"],
        score_normalization=args.get("score_normalization", "auto"), feature_mode="raw", augment=False,
        temporal_compression="confidence_weighted", deterministic_temporal_crop=args.get("deterministic_temporal_crop", "center"))
    loader = DataLoader(RawGaps(dataset, members, options.seed), batch_size=options.batch_size,
                        shuffle=False, num_workers=options.num_workers)
    conditions = ["clean"] + [f"gap{gap}" for gap, _ in GAPS]
    hashes = {name: hashlib.sha256() for name in conditions + ["frame_mask"]}
    rows, hidden_totals = [], np.zeros(len(GAPS), dtype=np.int64)
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (clean, damaged, labels, mask, hidden) in enumerate(loader, 1):
            tensor_hash(hashes["clean"], clean)
            tensor_hash(hashes["frame_mask"], mask)
            for index, (gap, _) in enumerate(GAPS):
                tensor_hash(hashes[f"gap{gap}"], damaged[:, index])
            hidden_totals += hidden.sum(0).numpy()
            clean, damaged, mask = clean.to(options.device), damaged.to(options.device), mask.to(options.device)
            batch = [{"sample": members[len(rows) + index], "label": int(label)}
                     for index, label in enumerate(labels)]
            for index, condition in enumerate(conditions):
                original = clean if index == 0 else damaged[:, index - 1]
                repaired, _ = interpolate_short_anchor_gaps(original, mask, 5)
                for method, value in (("unmodified", original), ("max5", repaired)):
                    logits = _soft_logits(model(value, mask), weights)
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError(f"Nonfinite logits: {condition}/{method}")
                    for row, prediction in zip(batch, logits.argmax(-1).cpu().tolist()):
                        row[f"{condition}_{method}_prediction"] = prediction
            rows.extend(batch)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    metrics = {}
    for condition in conditions:
        base = np.array([row[f"{condition}_unmodified_prediction"] == row["label"] for row in rows])
        repaired = np.array([row[f"{condition}_max5_prediction"] == row["label"] for row in rows])
        metrics[condition] = {"unmodified_correct": int(base.sum()), "max5_correct": int(repaired.sum()),
                              "difference_pp": float(100 * (repaired.astype(int) - base.astype(int)).mean()),
                              "corrected": int((repaired & ~base).sum()), "new_errors": int((base & ~repaired).sum())}
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint": str(options.checkpoint), "checkpoint_sha256": hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
               "partition": options.partition, "samples": len(rows), "member_source": source,
               "members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
               "input_sha256": {name: digest.hexdigest() for name, digest in hashes.items()},
               "seed": options.seed, "gaps": dict(GAPS), "hidden_points": hidden_totals.tolist(),
               "repair": "fixed max5 on compressed score-zero anchors; no gap-length input", "metrics": metrics,
               "fusion": "frozen checkpoint training-subject gamma=1 calibration counts",
               "device": options.device, "seconds": time.perf_counter() - started,
               "caveat": "Synthetic gaps and reused calibration/validation partitions; not independent confirmation."}
    (options.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
