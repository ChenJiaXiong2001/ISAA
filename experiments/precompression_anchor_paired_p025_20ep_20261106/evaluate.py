"""Frozen, shared-input comparison with anchor dropout before temporal compression."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "hand_motion_paired_p025_20ep_20261106"))
import isaa.train  # Configure CUDA wheel paths before importing torch.
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import load_checkpoint, validate_checkpoints
from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from tools.evaluate_face_pair_corruption import _soft_logits
from precompression_mask import PairedPrecompression


def make_dataset(archive, args):
    return RTMWZipDataset(archive, split="val", split_protocol=args["split"],
        window_size=args["window_size"], num_joints=133, num_classes=args["num_classes"],
        layout="rtmw_133", max_persons=args["max_persons"], feature_mode="raw", max_samples=0,
        score_normalization=args.get("score_normalization", "auto"), augment=False,
        temporal_compression="confidence_weighted",
        deterministic_temporal_crop=args.get("deterministic_temporal_crop", "center"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if options.batch_size <= 0 or options.num_workers < 0:
        raise ValueError("Invalid batch size or number of workers")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for compiled paired evaluation")
    device = torch.device("cuda")
    control, control_model, control_weights = load_checkpoint(options.control, device)
    candidate, candidate_model, candidate_weights = load_checkpoint(options.candidate, device)
    manifests = [json.loads((path.parent / "split_manifest.json").read_text(encoding="utf-8"))
                 for path in (options.control, options.candidate)]
    selected = validate_checkpoints(control, candidate, manifests)
    args = control["args"]
    if args["temporal_compression"] != "confidence_weighted":
        raise ValueError("Expected confidence-weighted temporal compression")
    for key, default in (("score_normalization", "auto"), ("deterministic_temporal_crop", "center")):
        if args.get(key, default) != candidate["args"].get(key, default):
            raise ValueError(f"Checkpoint input setting differs: {key}")
    dataset = make_dataset(options.archive, args)
    members = list(dataset.members)
    if len(members) != 16306 or len(set(members)) != len(members):
        raise ValueError("Expected 16,306 unique full-validation members")
    if members[:len(selected)] != selected:
        raise ValueError("Selection manifest is not a prefix of full validation")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = bool(args["tf32"])
    torch.backends.cuda.matmul.allow_tf32 = bool(args["tf32"])
    rows, results = [], []
    clean_digest, clean_mask_digest = None, None
    for probability in (0.05, 0.1):
        loader = DataLoader(PairedPrecompression(dataset, members, probability, options.seed),
            batch_size=options.batch_size, num_workers=options.num_workers, shuffle=False)
        digests = {name: hashlib.sha256() for name in ("clean", "corrupted", "frame_mask")}
        hits = {name: [[], []] for name in ("clean", "corrupted")}
        offset = 0
        with torch.inference_mode():
            for clean, corrupted, labels, frames, hidden, original_frames in loader:
                for name, value in (("clean", clean), ("corrupted", corrupted), ("frame_mask", frames)):
                    digests[name].update(value.contiguous().numpy().tobytes())
                labels_np = labels.numpy()
                frames = frames.to(device)
                for condition, features in (("clean", clean), ("corrupted", corrupted)):
                    if condition == "clean" and probability != 0.05:
                        continue
                    features = features.to(device)
                    logits = [_soft_logits(model(features, frames), weights)
                              for model, weights in ((control_model, control_weights), (candidate_model, candidate_weights))]
                    if not all(bool(torch.isfinite(value).all()) for value in logits):
                        raise RuntimeError("Nonfinite logits")
                    predictions = [value.argmax(-1).cpu().numpy() for value in logits]
                    correct = [value == labels_np for value in predictions]
                    for i in range(len(labels_np)):
                        rows.append({"condition": "clean" if condition == "clean" else f"anchor_pre_{probability:g}",
                            "sample": members[offset + i], "true_class_index": int(labels_np[i]),
                            "control_prediction": int(predictions[0][i]), "candidate_prediction": int(predictions[1][i]),
                            "control_correct": int(correct[0][i]), "candidate_correct": int(correct[1][i])})
                    for j in (0, 1):
                        hits[condition][j].extend(correct[j].tolist())
                offset += len(labels_np)
        if offset != len(members):
            raise RuntimeError("Incomplete validation traversal")
        mask_hash = digests["frame_mask"].hexdigest()
        if clean_digest is None:
            clean_digest = digests["clean"].hexdigest()
            clean_mask_digest = mask_hash
        elif mask_hash != clean_mask_digest:
            raise ValueError("Frame masks differ between probability levels")
        for condition in (("clean", "corrupted") if probability == 0.05 else ("corrupted",)):
            old, new = (np.asarray(x, dtype=np.int8) for x in hits[condition])
            result = {"condition": "clean" if condition == "clean" else f"anchor_pre_{probability:g}",
                      "samples": len(old), "control_top1_percent": float(old.mean() * 100),
                      "candidate_top1_percent": float(new.mean() * 100), "difference_pp": float((new-old).mean()*100),
                      "input_sha256": digests[condition].hexdigest(), "frame_mask_sha256": mask_hash}
            results.append(result)
            print(json.dumps(result), flush=True)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "paired_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"control": str(options.control), "candidate": str(options.candidate),
        "control_sha256": hashlib.sha256(options.control.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(options.candidate.read_bytes()).hexdigest(),
        "full_validation_members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
        "evaluated_samples_per_condition": len(members), "selection_samples": len(selected),
        "seed": options.seed, "mask_scheme": "independent valid shoulder/hip anchor-frame-person dropout before compression; member-specific SHA256 seed",
        "same_model_inputs": "one shared features and frame-mask tensor per batch",
        "fusion": "gamma=1 accuracy-proportional weights from each checkpoint's training-only calibration counts",
        "control_hf_weights": control_weights.cpu().tolist(), "candidate_hf_weights": candidate_weights.cpu().tolist(),
        "compile": "Inductor; inherited face LocalSTBlock eager boundary retained", "conditions": results}
    (options.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
