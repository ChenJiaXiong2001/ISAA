"""Evaluate an xsub60 checkpoint on an xset60 setup subset.

The checkpoint is loaded with its saved training configuration, while only the
dataset split is overridden to ``xset60``.  This keeps the model and all
preprocessing choices fixed.  The default population is the intersection with
the checkpoint's xsub60 validation set, which has already been used during
research and is not an independent validation set.  The optional full xset60
population includes fit/calibration samples and is diagnostic only.  The normal
path requires CUDA because the project's
compiled evaluation policy is mandatory; ``--smoke`` only scans the archive and
does not compile or run the network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _imports():
    # Importing isaa.train first configures the CUDA wheel paths used by this
    # repository.  Keep this lazy so --smoke can still validate metadata on a
    # machine without CUDA.
    import isaa.train  # noqa: F401
    import torch
    from torch.utils.data import DataLoader
    from isaa.face_pair_training import compile_model
    from isaa.train import collate_rtmw
    from isaa.data.rtmw_zip_dataset import RTMWZipDataset
    from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights
    return torch, DataLoader, compile_model, collate_rtmw, RTMWZipDataset, _soft_logits, _soft_weights


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_members(members: list[str]) -> str:
    return _sha_bytes("\n".join(members).encode("utf-8"))


def _sha_labels(labels) -> str:
    # Labels are serialized in their canonical little-endian int64 form so the
    # digest is independent of Python list formatting.
    import numpy as np

    arr = np.asarray(labels, dtype="<i8")
    return _sha_bytes(arr.tobytes())


def _model_class(checkpoint: dict):
    architecture = checkpoint.get("architecture")
    if architecture == "rtmw_best_wide_relative_handface_bodyface_derived_geometry_dropout_v1":
        from experiments.derived_geometry_dropout_20261107.model import DerivedGeometryDropoutFacePairFusion

        return DerivedGeometryDropoutFacePairFusion
    if architecture in {
        "best_face_pair",
        "face_pair_fusion",
        "rtmw_best_wide_relative_handface_bodyface_classwise_v1",
        None,
    }:
        from isaa.models.face_pair_fusion import FacePairFusion

        return FacePairFusion
    raise ValueError(f"Unsupported checkpoint architecture for xset evaluation: {architecture!r}")


def _fuse_geometric_mix(output, weights, torch):
    """50/50 normalized arithmetic and weighted-geometric probability pools."""
    import torch.nn.functional as F

    hand = output["hand_face_logits"].float()
    body = output["body_face_logits"].float()
    hlog = F.log_softmax(hand, dim=-1)
    blog = F.log_softmax(body, dim=-1)
    arithmetic = torch.logaddexp(weights.log() + hlog, torch.log1p(-weights) + blog)
    arithmetic = arithmetic - torch.logsumexp(arithmetic, dim=-1, keepdim=True)
    geometric = weights * hlog + (1.0 - weights) * blog
    geometric = geometric - torch.logsumexp(geometric, dim=-1, keepdim=True)
    mixed = torch.logaddexp(arithmetic, geometric) - math.log(2.0)
    mixed = mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)
    return mixed


def _build_dataset(args, archive, split, RTMWZipDataset):
    augmentation_config = {}
    return RTMWZipDataset(
        archive,
        split="val",
        split_protocol=split,
        window_size=int(args["window_size"]),
        num_joints=133,
        num_classes=int(args["num_classes"]),
        layout="rtmw_133",
        max_persons=int(args["max_persons"]),
        score_normalization=args.get("score_normalization", "auto"),
        feature_mode=args["feature_mode"],
        deterministic_temporal_crop=args.get("deterministic_temporal_crop", "center"),
        max_samples=0,
        augment=False,
        temporal_compression=args["temporal_compression"],
        augmentation_config=augmentation_config,
    )


def _metadata(dataset):
    members = list(dataset.members)
    labels = [int(x) for x in dataset.labels]
    if len(members) != len(labels) or not members:
        raise ValueError("Dataset members/labels are empty or have different lengths")
    for member, label in zip(members, labels):
        match = re.search(r"A(\d{3})", member, flags=re.IGNORECASE)
        if match is None or int(match.group(1)) != label + 1:
            raise ValueError(f"Member/action label mismatch: {member!r}, label={label}")
    return {
        "samples": len(members),
        "members_sha256": _sha_members(members),
        "labels_sha256": _sha_labels(labels),
        "first_member": members[0],
        "last_member": members[-1],
        "num_classes": len(set(labels)),
    }, members, np.asarray(labels, dtype=np.int64)


def _select_population(dataset, manifest, population):
    sections = {}
    for name in ("fit", "calibration", "validation"):
        item = manifest[name]
        members = item["members"]
        if len(members) != len(set(members)) or item.get("sha256") != _sha_members(members):
            raise ValueError(f"Invalid {name} manifest members or SHA256")
        if int(item.get("count", len(members))) != len(members):
            raise ValueError(f"Invalid {name} manifest count")
        sections[name] = set(members)
    if any(sections[a] & sections[b] for a, b in
           (("fit", "calibration"), ("fit", "validation"), ("calibration", "validation"))):
        raise ValueError("Checkpoint manifest partitions overlap")
    members = list(dataset.members)
    labels = list(dataset.labels)
    if len(members) != len(labels) or len(members) != len(set(members)):
        raise ValueError("Duplicate or mismatched xset dataset members")
    xset_members = set(members)
    overlap = {name: len(xset_members & values) for name, values in sections.items()}
    overlap["outside_manifest"] = len(xset_members - set.union(*sections.values()))
    if population == "validation-intersection":
        indices = [i for i, member in enumerate(members) if member in sections["validation"]]
        selected = {members[i] for i in indices}
        if selected & (sections["fit"] | sections["calibration"]):
            raise ValueError("Selected validation subset overlaps fit/calibration")
        interpretation = "Previously used xsub60 validation subset selected by xset60 setup; not independent"
    elif population == "full-diagnostic":
        indices = list(range(len(members)))
        interpretation = "Diagnostic only: full xset60 validation includes xsub60 fit/calibration samples"
    else:
        raise ValueError(f"Unknown population: {population}")
    if not indices:
        raise ValueError("Selected population is empty")
    return indices, overlap, interpretation


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", choices=("xset60",), default="xset60")
    parser.add_argument("--population", choices=("validation-intersection", "full-diagnostic"),
                        default="validation-intersection")
    parser.add_argument("--postprocess", choices=("arithmetic", "geometric_mix"), default="arithmetic")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Only scan dataset metadata; do not load/compile/run CUDA")
    return parser.parse_args()


def main() -> None:
    options = _parse_args()
    if options.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {options.output_dir}")
    if options.batch_size <= 0 or options.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be nonnegative")

    checkpoint_path = options.checkpoint.resolve()
    archive_path = options.archive.resolve()
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha = _sha_bytes(checkpoint_bytes)

    # Torch is imported lazily to make metadata-only smoke checks usable on a
    # CPU host.  The actual checkpoint read is still strict and read-only.
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    if saved.get("split") != "xsub60":
        raise ValueError(f"Expected an xsub60-trained checkpoint, got split={saved.get('split')!r}")
    if int(saved.get("max_samples", 0)) != 0:
        raise ValueError("Evaluation requires a full checkpoint configuration (max_samples=0)")
    compile_info = checkpoint.get("compile")
    compile_ok = bool(saved.get("compile")) and (
        compile_info is True
        or (isinstance(compile_info, dict)
            and bool(compile_info.get("enabled"))
            and bool(compile_info.get("required"))
            and compile_info.get("backend") == "inductor"
            and compile_info.get("suppress_errors") is False
            and compile_info.get("whole_model_eager_fallback") is False)
    )
    if not compile_ok:
        raise ValueError("Checkpoint was not trained with mandatory torch.compile")

    _, _, _, _, RTMWZipDataset, _, _ = _imports()
    dataset = _build_dataset(saved, archive_path, options.eval_split, RTMWZipDataset)
    full_metadata, _, _ = _metadata(dataset)
    manifest_path = checkpoint_path.parent / "split_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    indices, overlap, interpretation = _select_population(dataset, manifest, options.population)
    from types import SimpleNamespace
    selected = SimpleNamespace(members=[dataset.members[i] for i in indices],
                               labels=[dataset.labels[i] for i in indices])
    metadata, members, labels = _metadata(selected)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "architecture": checkpoint.get("architecture"),
        "training_split": saved.get("split"),
        "evaluation_split": options.eval_split,
        "population": options.population,
        "interpretation": interpretation,
        "manifest_sha256": _sha_bytes(manifest_bytes),
        "full_xset_dataset": full_metadata,
        "full_xset_overlap_with_checkpoint_manifest": overlap,
        "selected_overlap_with_fit": sum(member in set(manifest["fit"]["members"]) for member in members),
        "selected_overlap_with_calibration": sum(member in set(manifest["calibration"]["members"]) for member in members),
        "postprocess": options.postprocess,
        "dataset": metadata,
        "compile": {"required": True, "saved_compile": bool(saved.get("compile")), "checkpoint_compile": compile_info, "mode": saved.get("compile_mode")},
    }
    if options.smoke:
        options.output_dir.mkdir(parents=True, exist_ok=False)
        (options.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for compiled evaluation; no eager/CPU fallback")

    torch, DataLoader, compile_model, collate_rtmw, _, _soft_logits, _soft_weights = _imports()
    model_cls = _model_class(checkpoint)
    model = model_cls(int(saved["num_classes"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model = model.to(device).eval()
    weights = _soft_weights(model).to(device)
    compiled = compile_model(model, saved["compile_mode"])

    loader_kwargs = {
        "batch_size": options.batch_size,
        "shuffle": False,
        "num_workers": options.num_workers,
        "collate_fn": partial(collate_rtmw, feature_mode="raw"),
    }
    if options.num_workers:
        loader_kwargs.update(prefetch_factor=2, multiprocessing_context="spawn")
    from torch.utils.data import Subset
    loader = DataLoader(Subset(dataset, indices), **loader_kwargs)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    input_hash = hashlib.sha256()
    mask_hash = hashlib.sha256()
    predictions: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (features, batch_labels, frame_mask) in enumerate(loader, 1):
            start = len(predictions)
            if not np.array_equal(batch_labels.numpy(), labels[start:start + len(batch_labels)]):
                raise ValueError(f"Dataset label order mismatch at batch {step}")
            input_hash.update(features.contiguous().numpy().tobytes())
            mask_hash.update(frame_mask.contiguous().numpy().tobytes())
            features = features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            frame_mask = frame_mask.to(device, non_blocking=True)
            output = compiled(features, frame_mask)
            scores = _soft_logits(output, weights)
            if options.postprocess == "geometric_mix":
                scores = _fuse_geometric_mix(output, weights, torch)
            if not torch.isfinite(scores).all():
                raise ValueError(f"Nonfinite scores at batch {step}")
            predictions.extend(scores.argmax(dim=-1).cpu().tolist())
    if len(predictions) != len(labels):
        raise ValueError(f"Prediction count mismatch: {len(predictions)} != {len(labels)}")
    labels_np = labels
    predictions_np = np.asarray(predictions, dtype=np.int64)
    result.update({
        "input_sha256": input_hash.hexdigest(),
        "frame_mask_sha256": mask_hash.hexdigest(),
        "top1_percent": float(100.0 * np.mean(predictions_np == labels_np)),
        "correct": int(np.sum(predictions_np == labels_np)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    })
    with (options.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "member", "label", "prediction"])
        writer.writerows((i, member, int(label), int(prediction)) for i, (member, label, prediction) in enumerate(zip(members, labels, predictions_np)))
    (options.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
