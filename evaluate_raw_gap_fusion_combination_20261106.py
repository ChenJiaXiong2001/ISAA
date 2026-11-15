"""Paired arithmetic/geometric fusion evaluation under fixed raw anchor gaps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/face_pair_hand_motion_20261106/code"))
import isaa.train  # Configure CUDA libraries before importing torch.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, _confidence_weighted_temporal_compress
from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask
from isaa.face_pair_training import compile_model, compile_scope
from experiments.geometric_consensus_20261106.evaluate_cuda import EXPECTED, fuse
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
    parser.add_argument("--reference", type=Path, required=True)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if options.max_samples < 0 or options.batch_size < 1 or options.num_workers < 0:
        raise ValueError("Invalid batch size, worker count, or sample count")
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1 if options.device == "cpu" else 4)
    checkpoint_sha = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != EXPECTED:
        raise ValueError("Unexpected checkpoint")
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
    if options.device != "cuda" or not args["compile"]:
        raise ValueError("Compiled CUDA inference required")
    network = compile_model(model, args["compile_mode"])
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)
    section = json.loads(options.manifest.read_text(encoding="utf-8"))[options.partition]
    full_members = section["members"]
    if hashlib.sha256("\n".join(full_members).encode()).hexdigest() != section["sha256"]:
        raise ValueError("Manifest digest mismatch")
    members, source = select_members(full_members, None, options.max_samples)
    with options.reference.open(newline="", encoding="utf-8") as stream:
        reference = {row["sample"]: row for row in csv.DictReader(stream)}
    if any(member not in reference for member in members):
        raise ValueError("Reference missing samples")
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
                    output = network(value, mask)
                    arithmetic, candidate = fuse(output["hand_face_logits"], output["body_face_logits"], weights)
                    legacy = _soft_logits(output, weights)
                    if not bool(torch.isfinite(arithmetic).all() and torch.isfinite(candidate).all()):
                        raise RuntimeError(f"Nonfinite logits: {condition}/{method}")
                    if not torch.allclose(arithmetic.exp().sum(-1), torch.ones(len(batch), device=options.device), atol=2e-6) or not torch.allclose(candidate.exp().sum(-1), torch.ones(len(batch), device=options.device), atol=2e-6):
                        raise RuntimeError("Unnormalized fusion")
                    if not torch.allclose(arithmetic, legacy, atol=2e-6, rtol=1e-6):
                        raise RuntimeError("Original fusion probability calculation differs")
                    for row, pa, pc in zip(batch, arithmetic.argmax(-1).cpu().tolist(), candidate.argmax(-1).cpu().tolist()):
                        row[f"{condition}_{method}_prediction"] = pa
                        row[f"{condition}_{method}_geometric_prediction"] = pc
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
        for method in ("unmodified", "max5"):
            baseline = np.array([row[f"{condition}_{method}_prediction"] == row["label"] for row in rows])
            candidate = np.array([row[f"{condition}_{method}_geometric_prediction"] == row["label"] for row in rows])
            changes = candidate.astype(np.int64) - baseline.astype(np.int64)
            groups = np.array([int(re.search(r"P(\d{3})", row["sample"]).group(1)) for row in rows])
            unique, inverse = np.unique(groups, return_inverse=True)
            sizes = np.bincount(inverse)
            net = np.bincount(inverse, weights=changes)
            draws = np.random.default_rng(20261106).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=20000)
            metrics[condition][f"{method}_geometric"] = {"arithmetic_correct": int(baseline.sum()), "geometric_correct": int(candidate.sum()), "difference_pp": float(100 * changes.mean()), "corrected": int((candidate & ~baseline).sum()), "new_errors": int((baseline & ~candidate).sum()), "subject_bootstrap_ci95_pp": np.percentile(100 * (draws @ net) / (draws @ sizes), [2.5, 97.5]).tolist(), "subjects": len(unique), "reference_prediction_disagreements": int(sum(row[f"{condition}_{method}_prediction"] != int(reference[row["sample"]][f"{condition}_{method}_prediction"]) for row in rows))}
            if condition != "clean":
                clean_changes = np.array([int(row[f"clean_{method}_geometric_prediction"] == row["label"]) - int(row[f"clean_{method}_prediction"] == row["label"]) for row in rows])
                relative = changes - clean_changes
                relative_net = np.bincount(inverse, weights=relative)
                metrics[condition][f"{method}_geometric"]["difference_vs_clean_pp"] = float(100 * relative.mean())
                metrics[condition][f"{method}_geometric"]["difference_vs_clean_subject_bootstrap_ci95_pp"] = np.percentile(100 * (draws @ relative_net) / (draws @ sizes), [2.5, 97.5]).tolist()
        a = np.array([int(row[f"{condition}_max5_prediction"] == row["label"]) - int(row[f"{condition}_unmodified_prediction"] == row["label"]) for row in rows])
        b = np.array([int(row[f"{condition}_max5_geometric_prediction"] == row["label"]) - int(row[f"{condition}_unmodified_geometric_prediction"] == row["label"]) for row in rows])
        effect = b - a
        net = np.bincount(inverse, weights=effect)
        metrics[condition]["repair_fusion_interaction"] = {"difference_pp": float(100 * effect.mean()), "subject_bootstrap_ci95_pp": np.percentile(100 * (draws @ net) / (draws @ sizes), [2.5, 97.5]).tolist()}
    input_digests = {name: digest.hexdigest() for name, digest in hashes.items()}
    reference_summary = json.loads(options.reference.with_name("summary.json").read_text(encoding="utf-8"))
    hash_match = None if options.max_samples else input_digests == reference_summary["input_sha256"]
    if hash_match is False:
        raise ValueError("Full input hashes differ from reference")
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint": str(options.checkpoint), "checkpoint_sha256": checkpoint_sha,
               "partition": options.partition, "samples": len(rows), "member_source": source,
               "members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
               "input_sha256": input_digests, "reference_full_input_hash_match": hash_match,
               "seed": options.seed, "gaps": dict(GAPS), "hidden_points": hidden_totals.tolist(),
               "repair": "fixed max5 on compressed score-zero anchors; no gap-length input", "metrics": metrics,
               "fusion": "frozen checkpoint training-subject gamma=1 calibration counts",
               "candidate_fusion": "50 percent original arithmetic and 50 percent normalized geometric",
               "compile": {**compile_scope(argparse.Namespace(**args)), "mode": args["compile_mode"]},
               "reference": str(options.reference),
               "device": options.device, "seconds": time.perf_counter() - started,
               "caveat": "Synthetic gaps and reused calibration/validation partitions; not independent confirmation."}
    (options.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
