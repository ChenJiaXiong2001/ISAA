"""Frozen, shared-input full-validation comparison of 20-epoch hand-motion runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import isaa.train  # Configure CUDA wheel paths before torch import.
import numpy as np
import torch
from torch.utils.data import DataLoader

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.face_pair_training import compile_model
from isaa.models.face_pair_fusion import FACE_PAIR_HAND_MOTION_VARIANT, FacePairHandMotionFusion
from isaa.train import collate_rtmw
from isaa.utils.seed import seed_everything
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights
from tools.check_face_pair_checkpoint import check_args, check_compile
VARIANT = FACE_PAIR_HAND_MOTION_VARIANT
CONDITIONS = (("clean", None, 0.0, 0),
              ("point_0.05", "joint_dropout_probability", 0.05, 1),
              ("point_0.1", "joint_dropout_probability", 0.1, 2),
              ("group_0.05", "grouped_joint_dropout_probability", 0.05, 101),
              ("group_0.1", "grouped_joint_dropout_probability", 0.1, 102))


def load_checkpoint(path, device, compile_for_inference=True):
    if path.name != "last.pt":
        raise ValueError("Both inputs must be the predeclared last.pt checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    check_args(args, "checkpoint.args", expected_variant=VARIANT)
    check_compile(checkpoint["compile"], args, "checkpoint.compile")
    variants = {VARIANT: FacePairHandMotionFusion}
    model_type = variants.get(args["model_variant"])
    if model_type is None:
        raise ValueError(f"Unsupported model variant: {args['model_variant']!r}")
    if checkpoint["architecture"] != model_type.ARCHITECTURE:
        raise ValueError("Checkpoint architecture does not match model variant")
    model = model_type(args["num_classes"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if not bool(args["compile"]) or args["init_checkpoint"] or args["resume_checkpoint"]:
        raise ValueError("Checkpoint does not document scratch-only mandatory compilation")
    counts = model.routing_class_counts
    if (bool((counts < 0).any()) or bool((model.routing_hand_face_correct < 0).any())
            or bool((model.routing_body_face_correct < 0).any())
            or bool((model.routing_hand_face_correct > counts).any())
            or bool((model.routing_body_face_correct > counts).any())):
        raise ValueError("Invalid frozen calibration counts")
    weights = _soft_weights(model).to(device)
    model = model.to(device).eval()
    return checkpoint, compile_model(model, args["compile_mode"]) if compile_for_inference else model, weights


def validate_checkpoints(control, candidate, manifests):
    left, right = control["args"], candidate["args"]
    if left["model_variant"] != VARIANT or right["model_variant"] != VARIANT:
        raise ValueError("Expected two hand-motion checkpoints")
    if control["epoch"] != 20 or candidate["epoch"] != 20:
        raise ValueError("Both comparisons require the predeclared epoch-20 checkpoint")
    expected = {"epochs": 20, "lr_scheduler": "warmup_cosine", "warmup_epochs": 1,
                "lr": 0.025, "warmup_start_lr": 0.001, "min_lr": 0.0001,
                "momentum": 0.9, "weight_decay": 0.0004, "grad_accum_steps": 1,
                "batch_size": 32, "max_samples": 4096, "seed": 17,
                "compile": True, "compile_mode": "default"}
    for key, value in expected.items():
        if left.get(key) != value or right.get(key) != value:
            raise ValueError(f"Checkpoint does not match predeclared protocol: {key}")
    ignored = {"save_dir", "log_interval", "precompression_anchor_dropout_probability",
               "precompression_anchor_clean_sample_probability"}
    for key in sorted((left.keys() | right.keys()) - ignored):
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Checkpoint setting differs: {key}")
    if (left.get("precompression_anchor_dropout_probability") != 0 or
            left.get("precompression_anchor_clean_sample_probability") != 0 or
            right.get("precompression_anchor_dropout_probability") != 0.05 or
            right.get("precompression_anchor_clean_sample_probability") != 0.75):
        raise ValueError("Expected clean control and predeclared p05/clean75 candidate")
    for section in ("fit", "calibration", "validation"):
        members = manifests[0][section]["members"]
        if members != manifests[1][section]["members"]:
            raise ValueError(f"Checkpoint {section} members differ")
        if len(members) != len(set(members)):
            raise ValueError(f"Duplicate {section} members")
        for manifest in manifests:
            part = manifest[section]
            if part["count"] != len(members) or part["sha256"] != hashlib.sha256("\n".join(members).encode()).hexdigest():
                raise ValueError(f"Invalid {section} manifest digest or count")
    if any(manifest.get("official_validation_used_for_routing") is not False for manifest in manifests):
        raise ValueError("Validation cannot be used for training calibration")
    sets = [set(manifests[0][section]["members"]) for section in ("fit", "calibration", "validation")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Fit, calibration and validation members must be disjoint")
    return manifests[0]["validation"]["members"]


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
    selection_members = validate_checkpoints(control, candidate, manifests)
    args = control["args"]
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = bool(args["tf32"])
    torch.backends.cuda.matmul.allow_tf32 = bool(args["tf32"])
    rows, results, full_members, clean_labels = [], [], None, None
    for condition, key, probability, offset_seed in CONDITIONS:
        seed = options.seed + offset_seed
        seed_everything(seed)
        augmentation = {"random_temporal_crop": False}
        if key is not None:
            augmentation[key] = probability
        dataset = RTMWZipDataset(options.archive, split="val", split_protocol=args["split"],
            window_size=args["window_size"], num_joints=133, num_classes=args["num_classes"],
            layout="rtmw_133", max_persons=args["max_persons"], feature_mode="raw", max_samples=0,
            augment=bool(key), temporal_compression=args["temporal_compression"], augmentation_config=augmentation)
        members = list(dataset.members)
        if len(members) != 16306 or len(set(members)) != len(members):
            raise ValueError("Expected 16,306 unique full-validation members")
        if members[:len(selection_members)] != selection_members:
            raise ValueError("Selection manifest is not a prefix of full validation")
        if full_members is not None and full_members != members:
            raise ValueError("Condition member ordering differs")
        full_members = members
        kwargs = {"num_workers": options.num_workers, "pin_memory": True,
                  "collate_fn": partial(collate_rtmw, feature_mode="raw")}
        if options.num_workers:
            kwargs.update(prefetch_factor=4, multiprocessing_context="spawn")
        loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, drop_last=False, **kwargs)
        input_digest, mask_digest = hashlib.sha256(), hashlib.sha256()
        control_hits, candidate_hits, current_labels = [], [], []
        offset = 0
        with torch.inference_mode():
            for features, labels, frame_mask in loader:
                input_digest.update(features.contiguous().numpy().tobytes())
                mask_digest.update((features[:, 2] > 0).contiguous().numpy().tobytes())
                mask_digest.update(frame_mask.contiguous().numpy().tobytes())
                features, labels, frame_mask = (v.to(device, non_blocking=True) for v in (features, labels, frame_mask))
                control_logits = _soft_logits(control_model(features, frame_mask), control_weights)
                candidate_logits = _soft_logits(candidate_model(features, frame_mask), candidate_weights)
                if not bool(torch.isfinite(control_logits).all() and torch.isfinite(candidate_logits).all()):
                    raise RuntimeError(f"Nonfinite logits: {condition}")
                truth = labels.cpu().numpy()
                current_labels.extend(truth.tolist())
                old_pred, new_pred = control_logits.argmax(-1).cpu().numpy(), candidate_logits.argmax(-1).cpu().numpy()
                old, new = old_pred == truth, new_pred == truth
                for i in range(len(truth)):
                    rows.append({"condition": condition, "sample": members[offset + i],
                        "true_class_index": int(truth[i]), "control_prediction": int(old_pred[i]),
                        "candidate_prediction": int(new_pred[i]), "control_correct": int(old[i]),
                        "candidate_correct": int(new[i])})
                control_hits.extend(old.tolist())
                candidate_hits.extend(new.tolist())
                offset += len(truth)
        if offset != len(members):
            raise RuntimeError("Incomplete validation traversal")
        if clean_labels is not None and current_labels != clean_labels:
            raise ValueError("Validation labels differ between conditions")
        clean_labels = current_labels
        old, new = np.asarray(control_hits, dtype=np.int8), np.asarray(candidate_hits, dtype=np.int8)
        results.append({"condition": condition, "seed": seed, "samples": offset,
            "input_sha256": input_digest.hexdigest(), "validity_and_frame_mask_sha256": mask_digest.hexdigest(),
            "control_top1_percent": float(old.mean() * 100), "candidate_top1_percent": float(new.mean() * 100),
            "difference_pp": float((new - old).mean() * 100)})
        print(json.dumps(results[-1]), flush=True)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    with (options.output_dir / "paired_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"control": str(options.control), "candidate": str(options.candidate),
        "control_sha256": hashlib.sha256(options.control.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(options.candidate.read_bytes()).hexdigest(),
        "selection_samples": len(selection_members), "evaluated_samples_per_condition": len(full_members),
        "full_validation_members_sha256": hashlib.sha256("\n".join(full_members).encode("utf-8")).hexdigest(),
        "same_fit_calibration_validation_members": True,
        "same_model_inputs": "one shared features and frame-mask tensor per batch",
        "fusion": "gamma=1 accuracy-proportional weights from each checkpoint's training-only calibration counts",
        "control_hf_weights": control_weights.cpu().tolist(), "candidate_hf_weights": candidate_weights.cpu().tolist(),
        "compile": "Inductor; inherited face LocalSTBlock eager boundary retained",
        "conditions": results}
    (options.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
