"""Compare two frozen face-pair models on identical corrupted validation inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA wheel paths before torch import.

import numpy as np
import torch
from torch.utils.data import DataLoader

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.face_pair_training import compile_model
from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_HAND_MOTION_VARIANT,
    FACE_PAIR_VARIANT, FacePairFusion, FacePairHandFaceGeometryFusion,
    FacePairHandMotionFusion,
)
from isaa.train import collate_rtmw
from isaa.utils.seed import seed_everything
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    models = {FACE_PAIR_VARIANT: FacePairFusion,
              FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT: FacePairHandFaceGeometryFusion,
              FACE_PAIR_HAND_MOTION_VARIANT: FacePairHandMotionFusion}
    model_type = models.get(args["model_variant"])
    if model_type is None:
        raise ValueError(f"Unsupported face-pair model variant: {args['model_variant']!r}")
    model = model_type(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    weights = _soft_weights(model).to(device)
    model = model.to(device).eval()
    return checkpoint, compile_model(model, args["compile_mode"]), weights


def paired_interval(differences, seed, repeats=20000):
    random = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=np.int8)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 200):
        count = min(200, repeats - start)
        indices = random.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1) * 100
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paired evaluation")
    device = torch.device("cuda")
    control, control_model, control_weights = load_checkpoint(options.control, device)
    candidate, candidate_model, candidate_weights = load_checkpoint(options.candidate, device)
    a, b = control["args"], candidate["args"]
    for key in ("split", "num_classes", "window_size", "max_persons", "temporal_compression", "max_samples"):
        if a[key] != b[key]:
            raise ValueError(f"Checkpoint data setting differs: {key}")
    manifests = [json.loads((path.parent / "split_manifest.json").read_text(encoding="utf-8"))
                 for path in (options.control, options.candidate)]
    for section in ("fit", "calibration", "validation"):
        if manifests[0][section]["members"] != manifests[1][section]["members"]:
            raise ValueError(f"Checkpoint {section} members differ")
    expected_members = manifests[0]["validation"]["members"]
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = bool(a["tf32"])
    torch.backends.cuda.matmul.allow_tf32 = bool(a["tf32"])
    rows, summaries, differences = [], [], {}
    for name, probability, seed in (("clean", 0.0, options.seed),
                                    ("point_0.05", 0.05, options.seed + 1),
                                    ("point_0.1", 0.10, options.seed + 2)):
        seed_everything(seed)
        dataset = RTMWZipDataset(options.archive, split="val", split_protocol=a["split"],
            window_size=a["window_size"], num_joints=133, num_classes=a["num_classes"],
            layout="rtmw_133", max_persons=a["max_persons"], feature_mode="raw",
            max_samples=a["max_samples"], augment=probability > 0,
            temporal_compression=a["temporal_compression"],
            augmentation_config={"random_temporal_crop": False, "joint_dropout_probability": probability})
        if list(dataset.members) != expected_members:
            raise ValueError("Validation member ordering differs from saved manifests")
        loader_args = {"num_workers": options.num_workers, "pin_memory": True,
                       "collate_fn": partial(collate_rtmw, feature_mode="raw")}
        if options.num_workers:
            loader_args.update(prefetch_factor=4, multiprocessing_context="spawn")
        loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False,
                            drop_last=False, **loader_args)
        control_hits, candidate_hits = [], []
        input_digest, mask_digest = hashlib.sha256(), hashlib.sha256()
        offset = 0
        with torch.inference_mode():
            for features, labels, frame_mask in loader:
                input_digest.update(features.contiguous().numpy().tobytes())
                mask_digest.update((features[:, 2] > 0).contiguous().numpy().tobytes())
                mask_digest.update(frame_mask.contiguous().numpy().tobytes())
                features, labels, frame_mask = (v.to(device, non_blocking=True)
                                                 for v in (features, labels, frame_mask))
                control_logits = _soft_logits(control_model(features, frame_mask), control_weights)
                candidate_logits = _soft_logits(candidate_model(features, frame_mask), candidate_weights)
                if not bool(torch.isfinite(control_logits).all() and torch.isfinite(candidate_logits).all()):
                    raise RuntimeError(f"Nonfinite logits in {name}")
                truth = labels.cpu().numpy()
                control_predictions = control_logits.argmax(-1).cpu().numpy()
                candidate_predictions = candidate_logits.argmax(-1).cpu().numpy()
                control_right = control_predictions == truth
                candidate_right = candidate_predictions == truth
                for index, (old, new) in enumerate(zip(control_right, candidate_right)):
                    rows.append({"condition": name, "sample": expected_members[offset + index],
                                 "true_class_index": int(truth[index]),
                                 "control_prediction": int(control_predictions[index]),
                                 "candidate_prediction": int(candidate_predictions[index]),
                                 "control_correct": int(old), "candidate_correct": int(new)})
                control_hits.extend(control_right.tolist())
                candidate_hits.extend(candidate_right.tolist())
                offset += len(labels)
        if offset != len(expected_members):
            raise RuntimeError("Incomplete validation traversal")
        old = np.asarray(control_hits, dtype=np.int8)
        new = np.asarray(candidate_hits, dtype=np.int8)
        diff = new - old
        differences[name] = diff
        summaries.append({"condition": name, "seed": seed, "samples": offset,
            "input_sha256": input_digest.hexdigest(), "validity_and_frame_mask_sha256": mask_digest.hexdigest(),
            "control_correct": int(old.sum()), "candidate_correct": int(new.sum()),
            "control_top1_percent": float(old.mean() * 100),
            "candidate_top1_percent": float(new.mean() * 100),
            "gain": int((diff == 1).sum()), "loss": int((diff == -1).sum()),
            "difference_pp": float(diff.mean() * 100),
            "bootstrap_95_percent_ci_pp": paired_interval(diff, seed=20261105)})
        print(json.dumps(summaries[-1]), flush=True)
    for result in summaries:
        result["control_drop_pp"] = summaries[0]["control_top1_percent"] - result["control_top1_percent"]
        result["candidate_drop_pp"] = summaries[0]["candidate_top1_percent"] - result["candidate_top1_percent"]
        drop_reduction = differences[result["condition"]] - differences["clean"]
        result["drop_reduction_pp"] = float(drop_reduction.mean() * 100)
        result["drop_reduction_bootstrap_95_percent_ci_pp"] = paired_interval(
            drop_reduction, seed=20261105)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "paired_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"control": str(options.control), "candidate": str(options.candidate),
        "control_sha256": hashlib.sha256(options.control.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(options.candidate.read_bytes()).hexdigest(),
        "control_epoch": control["epoch"], "candidate_epoch": candidate["epoch"],
        "validation_sha256": manifests[0]["validation"]["sha256"],
        "same_fit_calibration_validation_members": True,
        "same_model_inputs": "Both models are called with the same feature and frame-mask tensors in each batch",
        "fusion": "gamma=1 accuracy-proportional weights from each checkpoint's own calibration counts",
        "control_hf_weights": control_weights.cpu().tolist(),
        "candidate_hf_weights": candidate_weights.cpu().tolist(),
        "conditions": summaries, "bootstrap_replicates": 20000,
        "bootstrap_unit": "paired validation sample", "device": str(device),
        "compile": "inductor, existing face LocalSTBlock eager boundary retained"}
    (options.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
