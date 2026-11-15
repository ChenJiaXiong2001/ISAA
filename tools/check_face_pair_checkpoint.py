"""Validate a completed face-pair run and its saved inference/training state on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA-wheel library paths before importing torch.
import torch

from isaa.face_pair_training import routing_report
from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_HAND_MOTION_VARIANT,
    FACE_PAIR_RAW_XY_VARIANT, FACE_PAIR_VARIANT,
    FacePairFusion, FacePairHandFaceGeometryFusion, FacePairHandMotionFusion, FacePairRawXYFusion,
)
from isaa.utils.experiment import write_json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"Expected an object in {path}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_finite(value, location):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"Nonfinite tensor: {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{location}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            check_finite(item, f"{location}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite number: {location}")


def check_compile(policy, args, location):
    expected = {"enabled": True, "required": True, "backend": "inductor",
                "suppress_errors": False, "fail_on_recompile_limit_hit": True,
                "whole_model_eager_fallback": False, "triton_cudagraphs": False}
    for key, value in expected.items():
        require(policy.get(key) == value, f"Invalid {location}.{key}: expected {value!r}")
    require(policy.get("mode") == args.get("compile_mode"), f"Compile mode differs in {location}")


FACE_PAIR_MODELS = {
    FACE_PAIR_VARIANT: FacePairFusion,
    FACE_PAIR_RAW_XY_VARIANT: FacePairRawXYFusion,
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT: FacePairHandFaceGeometryFusion,
    FACE_PAIR_HAND_MOTION_VARIANT: FacePairHandMotionFusion,
}


def check_args(args, location, *, expected_variant=FACE_PAIR_VARIANT):
    require(args.get("model_variant") == expected_variant, f"Wrong model variant in {location}")
    require(args.get("compile") is True, f"Compilation disabled in {location}")
    require(args.get("init_checkpoint") is None and args.get("resume_checkpoint") is None,
            f"Checkpoint initialization/resumption found in {location}")
    require(args.get("feature_mode") == "raw" and args.get("npy_dir") is None
            and args.get("window_size") == 100 and args.get("temporal_compression") == "confidence_weighted",
            f"Baseline input protocol differs in {location}")
    require(args.get("lr_scheduler") == "warmup_cosine", f"Wrong scheduler in {location}")


def check_manifest(manifest, config):
    require(manifest.get("official_validation_used_for_routing") is False,
            "Manifest permits official validation in routing")
    calibration = config.get("calibration", {})
    require(calibration.get("official_validation_used_for_routing") is False,
            "Config permits official validation in routing")
    require(manifest.get("protocol") == config["args"]["split"], "Manifest protocol mismatch")
    member_sets, subject_sets, counts = {}, {}, {}
    for name in ("fit", "calibration", "validation"):
        part = manifest[name]
        members = part["members"]
        require(isinstance(members, list) and members, f"Empty {name} split")
        require(all(isinstance(member, str) for member in members), f"Invalid members in {name}")
        member_sets[name] = set(members)
        require(len(member_sets[name]) == len(members) == part["count"], f"Duplicate/count mismatch in {name}")
        digest = hashlib.sha256("\n".join(members).encode()).hexdigest()
        require(digest == part["sha256"], f"Manifest digest mismatch for {name}")
        require(config["data_split"][name] == {"count": len(members), "sha256": digest},
                f"Config data split differs for {name}")
        subjects = set()
        for member in members:
            match = re.search(r"P(\d{3})", member)
            require(match is not None, f"Missing subject in {member}")
            subjects.add(int(match.group(1)))
        subject_sets[name], counts[name] = subjects, len(members)
    for left, right in (("fit", "calibration"), ("fit", "validation"), ("calibration", "validation")):
        require(not member_sets[left] & member_sets[right], f"Member overlap: {left}/{right}")
    require(not subject_sets["fit"] & subject_sets["calibration"], "Fit/calibration subjects overlap")
    require(sorted(subject_sets["fit"]) == manifest["fit_subjects"], "Fit subject list mismatch")
    require(sorted(subject_sets["calibration"]) == manifest["calibration_subjects"] == calibration["subjects"],
            "Calibration subject list mismatch")
    if manifest["protocol"].startswith("xsub"):
        require(not (subject_sets["fit"] | subject_sets["calibration"]) & subject_sets["validation"],
                "Cross-subject validation subjects overlap training")
    return counts


def check_checkpoint(path, config, manifest, run_dir, checked_files):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    variant = checkpoint["args"]["model_variant"]
    model_type = FACE_PAIR_MODELS.get(variant)
    require(model_type is not None, f"Unsupported face-pair variant in {path.name}")
    require(checkpoint.get("architecture") == model_type.ARCHITECTURE, f"Wrong architecture in {path.name}")
    args = checkpoint["args"]
    check_args(args, f"{path.name}.args", expected_variant=variant)
    require(args == config["args"], f"Saved arguments differ in {path.name}")
    check_compile(checkpoint["compile"], args, f"{path.name}.compile")
    require(checkpoint["compile"] == config["compile"], f"Compile policy differs in {path.name}")
    model = model_type(num_classes=args["num_classes"])
    require(checkpoint["model_config"] == model.model_config == config["model_config"],
            f"Model config differs in {path.name}")
    check_finite(checkpoint["model"], f"{path.name}.model")
    model.load_state_dict(checkpoint["model"], strict=True)
    saved_weights = model.hand_face_weight.clone()
    model.set_routing_counts(model.routing_hand_face_correct, model.routing_body_face_correct,
                             model.routing_class_counts)
    torch.testing.assert_close(model.hand_face_weight, saved_weights, rtol=0, atol=0)
    require(float(model.hand_face_temperature) == float(model.body_face_temperature) == 1.,
            f"Unexpected probability temperatures in {path.name}")

    epoch, schedule = checkpoint["epoch"], checkpoint["lr_schedule"]
    require(isinstance(epoch, int) and 1 <= epoch <= args["epochs"], f"Invalid epoch in {path.name}")
    updates, batches = schedule["completed_updates"], schedule["batches_per_epoch"]
    require(updates > 0 and batches > 0 and updates == epoch * batches,
            f"Optimizer update/epoch count mismatch in {path.name}")
    require(schedule["name"] == "warmup_cosine" and schedule["total_updates"] == args["epochs"] * batches,
            f"Invalid saved scheduler in {path.name}")
    optimizer = checkpoint["optimizer"]
    require(optimizer.get("param_groups"), f"Missing optimizer parameters in {path.name}")
    parameter_ids = [index for group in optimizer["param_groups"] for index in group["params"]]
    require(len(parameter_ids) == len(set(parameter_ids)) == len(list(model.parameters())),
            f"Optimizer parameter coverage differs in {path.name}")
    if any(group.get("momentum", 0) > 0 for group in optimizer["param_groups"]):
        require(bool(optimizer.get("state")), f"Missing trained optimizer momentum in {path.name}")
    check_finite(optimizer, f"{path.name}.optimizer")
    check_finite(checkpoint["metrics"], f"{path.name}.metrics")

    expected_routing = routing_report(model, epoch)
    expected_routing["active"] = epoch >= args["pair_calibration_start_epoch"]
    require(checkpoint["routing"] == expected_routing, f"Routing buffers/report mismatch in {path.name}")
    routing_path = run_dir / f"routing_epoch_{epoch:03}.json"
    checked_files.add(routing_path)
    require(read_json(routing_path) == expected_routing, f"Saved routing JSON differs for {path.name}")
    total = int(model.routing_class_counts.sum())
    require(total == (manifest["calibration"]["count"] if expected_routing["active"] else 0),
            f"Routing sample count differs in {path.name}")
    choices = {name: sum(row["selected"] == name for row in expected_routing["classes"])
               for name in ("hand_face", "body_face", "average")}
    return {"epoch": epoch, "completed_updates": updates,
            "routing_active": expected_routing["active"], "routing_samples": total,
            "routing_selection": choices, "model_tensors": len(checkpoint["model"]),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer_state_entries": len(optimizer.get("state", {})),
            "strict_state_load": True, "finite_model_and_optimizer": True}, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    checked_files = {run_dir / name for name in
                     ("config.json", "status.json", "split_manifest.json", "routing_latest.json", "last.pt", "best.pt")}
    for path in checked_files:
        require(path.is_file(), f"Required saved artifact missing: {path}")
    config = read_json(run_dir / "config.json")
    status = read_json(run_dir / "status.json")
    manifest = read_json(run_dir / "split_manifest.json")
    require(status.get("status") == "completed", "Run has not completed successfully")
    variant = config.get("args", {}).get("model_variant")
    model_type = FACE_PAIR_MODELS.get(variant)
    require(model_type is not None, f"Unsupported face-pair model variant: {variant!r}")
    require(config.get("architecture") == model_type.ARCHITECTURE, "Config architecture mismatch")
    require(config.get("initialization") == "random_scratch_no_checkpoint", "Run was not initialized from scratch")
    check_args(config["args"], "config.args", expected_variant=variant)
    check_compile(config["compile"], config["args"], "config.compile")
    require(not any(str(flag).split("=", 1)[0] == "--no-compile" for flag in config.get("command", [])),
            "Saved command includes --no-compile")
    counts = check_manifest(manifest, config)
    summaries, checkpoints = {}, {}
    for name in ("last", "best"):
        summaries[name], checkpoints[name] = check_checkpoint(
            run_dir / f"{name}.pt", config, manifest, run_dir, checked_files)
    last, best = checkpoints["last"], checkpoints["best"]
    require(last["epoch"] == status["last_completed_epoch"] == config["args"]["epochs"],
            "Last checkpoint/status do not cover the completed epoch budget")
    require(best["epoch"] == status["best_epoch"] == last["metrics"]["best_epoch"],
            "Best checkpoint/status epoch mismatch")
    require(last["lr_schedule"] == config["lr_schedule"], "Latest scheduler differs from run config")
    require(status["best_val_acc"] == last["metrics"]["best_val_acc"] == best["metrics"]["best_val_acc"],
            "Best validation accuracy differs between saved records")
    require(read_json(run_dir / "routing_latest.json") == last["routing"], "Latest routing differs from last checkpoint")
    output = args.output.resolve()
    require(output not in checked_files, "Output would overwrite an input artifact")
    report = {"status": "passed", "run_dir": str(run_dir), "architecture": model_type.ARCHITECTURE,
              "torch": str(torch.__version__), "load_device": "cpu", "completed": True,
              "initialization": config["initialization"], "compile_guards_verified": True,
              "official_validation_used_for_routing": False, "disjoint_data_splits": True,
              "split_counts": counts, "checkpoints": summaries,
              "sha256": {path.relative_to(run_dir).as_posix(): sha256(path) for path in sorted(checked_files)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
