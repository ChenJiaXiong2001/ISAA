"""Cache compiled best/last HF/BF logits for read-only fusion experiments."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure the CUDA-wheel library paths before importing torch.
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Subset

from isaa.face_pair_training import _dataset, _loader, compile_model, routing_report
from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_HAND_MOTION_VARIANT,
    FACE_PAIR_RAW_XY_VARIANT, FACE_PAIR_VARIANT, FacePairFusion,
    FacePairHandFaceGeometryFusion, FacePairHandMotionFusion, FacePairRawXYFusion,
)
from isaa.utils.seed import seed_everything
from tools.check_face_pair_checkpoint import (
    check_args,
    check_compile,
    check_finite,
    check_manifest,
    require,
)
from tools.evaluate_face_pair import read_json


def member_digest(members):
    return hashlib.sha256("\n".join(members).encode()).hexdigest()


def model_type(variant):
    models = {FACE_PAIR_VARIANT: FacePairFusion, FACE_PAIR_RAW_XY_VARIANT: FacePairRawXYFusion,
              FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT: FacePairHandFaceGeometryFusion,
              FACE_PAIR_HAND_MOTION_VARIANT: FacePairHandMotionFusion}
    require(variant in models, f"Unsupported face-pair variant: {variant}")
    return models[variant]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def check_state_unchanged(model, initial, location):
    current = model.state_dict()
    require(current.keys() == initial.keys(), f"State names changed during {location}")
    for name, tensor in current.items():
        require(torch.equal(tensor.detach().cpu(), initial[name]),
                f"State tensor {name} changed during {location}")


def read_checkpoint(path, config, manifest):
    # Hash and deserialize one byte snapshot, even if the source gets replaced.
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    checkpoint = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    del contents
    args = checkpoint["args"]
    check_args(args, f"{path.name}.args", expected_variant=config["args"]["model_variant"])
    check_compile(checkpoint["compile"], args, f"{path.name}.compile")
    require(args == config["args"], f"{path.name} arguments differ from run config")
    require(checkpoint["compile"] == config["compile"],
            f"{path.name} compile policy differs from run config")
    require(checkpoint.get("architecture") == model_type(args["model_variant"]).ARCHITECTURE,
            f"Wrong architecture in {path.name}")
    check_finite(checkpoint["model"], f"{path.name}.model")
    check_finite(checkpoint["optimizer"], f"{path.name}.optimizer")
    check_finite(checkpoint["metrics"], f"{path.name}.metrics")
    check_finite(checkpoint["routing"], f"{path.name}.routing")
    require(isinstance(checkpoint["epoch"], int) and 1 <= checkpoint["epoch"] <= args["epochs"],
            f"Invalid saved epoch in {path.name}")
    require(checkpoint["lr_schedule"]["completed_updates"]
            == checkpoint["epoch"] * checkpoint["lr_schedule"]["batches_per_epoch"],
            f"Saved update count differs from epoch in {path.name}")
    return checkpoint, digest


def validate_loaded_state(model, checkpoint, config, run_dir, name):
    require(checkpoint["model_config"] == model.model_config == config["model_config"],
            f"{name} model configuration differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    routing = routing_report(model, checkpoint["epoch"])
    routing["active"] = checkpoint["epoch"] >= checkpoint["args"]["pair_calibration_start_epoch"]
    require(routing == checkpoint["routing"], f"{name} routing report differs from state")
    require(read_json(run_dir / f"routing_epoch_{checkpoint['epoch']:03}.json") == routing,
            f"{name} immutable epoch routing report differs")
    total = model.routing_class_counts.detach().cpu()
    hf = model.routing_hand_face_correct.detach().cpu()
    bf = model.routing_body_face_correct.detach().cpu()
    require(bool(((total >= 0) & (hf >= 0) & (bf >= 0) & (hf <= total) & (bf <= total)).all()),
            f"{name} invalid routing counts")
    weight = torch.where(hf > bf, 1.0, torch.where(hf < bf, 0.0, 0.5))
    require(torch.equal(weight, model.hand_face_weight.detach().cpu()),
            f"{name} routing selection disagrees with calibration counts")
    require(float(model.hand_face_temperature) == float(model.body_face_temperature) == 1.0,
            f"{name} temperatures differ from 1")
    expected_count = config["data_split"]["calibration"]["count"] if routing["active"] else 0
    require(int(total.sum()) == expected_count, f"{name} saved calibration sample count differs")
    return routing


def collect(compiled, loader, members, labels_in_order, args, device, checkpoint_sha,
            epoch, partition, destination):
    started = time.perf_counter()
    all_hf, all_bf, all_labels = [], [], []
    k = args.num_classes
    class_counts = torch.zeros(k, dtype=torch.long)
    hf_correct = torch.zeros_like(class_counts)
    bf_correct = torch.zeros_like(class_counts)
    samples = any_head_correct = 0
    hf_ce = bf_ce = 0.0
    maximum_normalization_error = {"HF": 0.0, "BF": 0.0}
    with torch.no_grad():
        for step, (features, labels, frame_mask) in enumerate(loader, 1):
            features, labels, frame_mask = (value.to(device, non_blocking=True)
                                            for value in (features, labels, frame_mask))
            output = compiled(features, frame_mask)
            hf = output["hand_face_logits"]
            bf = output["body_face_logits"]
            require(hf.shape == bf.shape == (labels.numel(), k),
                    f"Invalid output shape in {partition} batch {step}")
            for name, logits in (("HF", hf), ("BF", bf)):
                require(bool(torch.isfinite(logits).all()),
                        f"Nonfinite {name} logits in {partition} batch {step}")
                log_probability = F.log_softmax(logits.float(), dim=-1)
                probability = log_probability.exp()
                require(bool(torch.isfinite(log_probability).all() and torch.isfinite(probability).all()),
                        f"Nonfinite {name} probabilities in {partition} batch {step}")
                deviation = float((probability.sum(-1) - 1).abs().max())
                maximum_normalization_error[name] = max(maximum_normalization_error[name], deviation)
                require(deviation <= 1e-5,
                        f"Unnormalized {name} probabilities in {partition} batch {step}: {deviation}")
            labels_cpu = labels.detach().cpu()
            require(np.array_equal(labels_cpu.numpy(), labels_in_order[samples:samples + labels.numel()]),
                    f"Member/label ordering differs in {partition} batch {step}")
            hf_right = (hf.argmax(-1) == labels).detach().cpu()
            bf_right = (bf.argmax(-1) == labels).detach().cpu()
            class_counts += torch.bincount(labels_cpu, minlength=k)
            hf_correct += torch.bincount(labels_cpu[hf_right], minlength=k)
            bf_correct += torch.bincount(labels_cpu[bf_right], minlength=k)
            any_head_correct += int((hf_right | bf_right).sum())
            hf_ce += float(F.cross_entropy(hf, labels, reduction="sum"))
            bf_ce += float(F.cross_entropy(bf, labels, reduction="sum"))
            all_hf.append(hf.detach().float().cpu())
            all_bf.append(bf.detach().float().cpu())
            all_labels.append(labels_cpu)
            samples += labels.numel()
            if step == 1 or step % 50 == 0 or step == len(loader):
                print(f"epoch={epoch} partition={partition} progress={step}/{len(loader)} "
                      f"samples={samples} HF={int(hf_correct.sum()) / samples:.6f} "
                      f"BF={int(bf_correct.sum()) / samples:.6f} "
                      f"any_head={any_head_correct / samples:.6f} "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    require(samples == len(members) == len(labels_in_order) > 0,
            f"Incomplete {partition} traversal")
    require(torch.equal(class_counts, torch.from_numpy(np.bincount(labels_in_order, minlength=k))),
            f"Class counts differ for {partition}")
    digest = member_digest(members)
    cached = {"schema_version": 1, "hf": torch.cat(all_hf), "bf": torch.cat(all_bf),
              "labels": torch.cat(all_labels), "member_sha256": digest,
              "checkpoint_sha256": checkpoint_sha, "epoch": epoch, "partition": partition}
    require(cached["hf"].shape == cached["bf"].shape == (samples, k), "Cache tensor shape differs")
    require(cached["labels"].shape == (samples,), "Cache label shape differs")
    require(torch.equal(cached["labels"], torch.from_numpy(np.asarray(labels_in_order, dtype=np.int64))),
            "Cached labels differ from manifest ordering")
    check_finite(cached, f"cache.{partition}")
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(cached, temporary)
    temporary.replace(destination)
    result = {"file": str(destination), "file_sha256": sha256(destination),
              "partition": partition, "member_sha256": digest, "samples": samples,
              "class_counts": class_counts.tolist(), "HF_correct_by_class": hf_correct.tolist(),
              "BF_correct_by_class": bf_correct.tolist(),
              "HF_correct": int(hf_correct.sum()), "BF_correct": int(bf_correct.sum()),
              "HF_top1_percent": 100 * int(hf_correct.sum()) / samples,
              "BF_top1_percent": 100 * int(bf_correct.sum()) / samples,
              "HF_ce": hf_ce / samples, "BF_ce": bf_ce / samples,
              "any_head_correct": any_head_correct,
              "any_head_coverage_percent": 100 * any_head_correct / samples,
              "any_head_note": "Oracle coverage: either HF or BF top1 matches the true label; not an inference selector",
              "maximum_probability_normalization_error": maximum_normalization_error,
              "seconds": time.perf_counter() - started}
    del cached, all_hf, all_bf, all_labels
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    options = parser.parse_args()
    run_dir, output_dir = options.run_dir.resolve(), options.output_dir.resolve()
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    if options.batch_size is not None:
        require(options.batch_size > 0, "batch-size must be positive")
    if options.num_workers is not None:
        require(options.num_workers >= 0, "num-workers must be nonnegative")
    config = read_json(run_dir / "config.json")
    manifest = read_json(run_dir / "split_manifest.json")
    require(config.get("initialization") == "random_scratch_no_checkpoint",
            "Run was not initialized from scratch")
    model_class = model_type(config["args"]["model_variant"])
    check_args(config["args"], "config.args", expected_variant=config["args"]["model_variant"])
    check_compile(config["compile"], config["args"], "config.compile")
    split_counts = check_manifest(manifest, config)
    args = argparse.Namespace(**config["args"])
    if options.archive is not None:
        args.archive = str(options.archive.resolve())
    if options.batch_size is not None:
        args.test_batch_size = options.batch_size
    if options.num_workers is not None:
        args.num_workers = options.num_workers
    require(args.test_batch_size > 0 and args.num_workers >= 0, "Invalid loader configuration")
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device(options.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable; no fallback")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    train = _dataset(args, "train", False)
    val = _dataset(args, "val", False)
    require(len(set(train.members)) == len(train.members), "Duplicate train archive members")
    train_index = {member: index for index, member in enumerate(train.members)}
    cal_members = manifest["calibration"]["members"]
    require(set(cal_members).issubset(train_index), "Calibration members missing from training dataset")
    require(set(train.members) == set(manifest["fit"]["members"]) | set(cal_members),
            "Training members differ from saved fit/calibration union")
    calibration_indices = [train_index[member] for member in cal_members]
    calibration = Subset(train, calibration_indices)
    calibration_labels = train.labels[np.asarray(calibration_indices, dtype=np.int64)]
    require(member_digest(cal_members) == manifest["calibration"]["sha256"],
            "Calibration member hash differs")
    require([train.members[index] for index in calibration_indices] == cal_members,
            "Calibration member ordering differs")
    require(list(val.members) == manifest["validation"]["members"], "Validation member ordering differs")
    require(member_digest(val.members) == manifest["validation"]["sha256"],
            "Validation member hash differs")
    require(len(calibration) == split_counts["calibration"] and len(val) == split_counts["validation"],
            "Dataset sample counts differ from manifest")
    loaders = {"calibration": _loader(args, calibration, device, False),
               "val": _loader(args, val, device, False)}
    partitions = {"calibration": (cal_members, calibration_labels, "calibration.pt"),
                  "val": (val.members, val.labels, "val.pt")}

    output_dir.mkdir(parents=True, exist_ok=False)
    model = model_class(args.num_classes).to(device).eval()
    compiled = compile_model(model, args.compile_mode)
    started = time.perf_counter()
    reports = {}
    for name in ("best", "last"):
        checkpoint_path = run_dir / f"{name}.pt"
        checkpoint, checkpoint_sha = read_checkpoint(checkpoint_path, config, manifest)
        routing = validate_loaded_state(model, checkpoint, config, run_dir, name)
        initial_state = cpu_state(model)
        checkpoint_dir = output_dir / name
        checkpoint_dir.mkdir(exist_ok=False)
        print(json.dumps({"checkpoint": name, "epoch": checkpoint["epoch"],
                          "sha256": checkpoint_sha, "compile": config["compile"],
                          "calibration_samples": len(calibration), "validation_samples": len(val)},
                         ensure_ascii=False), flush=True)
        results = {}
        for partition, (members, labels, filename) in partitions.items():
            results[partition] = collect(compiled, loaders[partition], members, labels,
                                         args, device, checkpoint_sha, checkpoint["epoch"],
                                         partition, checkpoint_dir / filename)
            check_state_unchanged(model, initial_state, f"{name}/{partition}")

        cal_result = results["calibration"]
        hf_saved = initial_state["routing_hand_face_correct"].tolist()
        bf_saved = initial_state["routing_body_face_correct"].tolist()
        saved_counts = initial_state["routing_class_counts"].tolist()
        cal_comparison = {
            "count_difference_by_class": [actual - saved for actual, saved in
                                           zip(cal_result["class_counts"], saved_counts)],
            "HF_correct_difference_by_class": [actual - saved for actual, saved in
                                                zip(cal_result["HF_correct_by_class"], hf_saved)],
            "BF_correct_difference_by_class": [actual - saved for actual, saved in
                                                zip(cal_result["BF_correct_by_class"], bf_saved)],
            "saved_class_counts": saved_counts, "saved_HF_correct_by_class": hf_saved,
            "saved_BF_correct_by_class": bf_saved,
            "note": "Compiled CUDA/TF32 inference may have small prediction differences across runs; discrepancies are reported, not used to mutate checkpoint routing",
        }
        require(cal_result["class_counts"] == saved_counts,
                f"{name} calibration support differs from saved routing counts")
        val_result = results["val"]
        recorded = checkpoint["metrics"]["val"]
        require(recorded["samples"] == val_result["samples"],
                f"{name} recorded validation sample count differs")
        reports[name] = {
            "path": str(checkpoint_path), "sha256": checkpoint_sha,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
            "epoch": checkpoint["epoch"], "partitions": results,
            "routing_arrays": {"HF_correct": hf_saved, "BF_correct": bf_saved,
                               "class_counts": saved_counts,
                               "hard_HF_weight": initial_state["hand_face_weight"].tolist()},
            "args": checkpoint["args"], "model_config": checkpoint["model_config"],
            "compile": checkpoint["compile"],
            "calibration_routing_comparison": cal_comparison,
            "recorded_validation_comparison": {
                "HF_top1_difference_percentage_points": val_result["HF_top1_percent"]
                    - 100 * recorded["hand_face_acc"],
                "BF_top1_difference_percentage_points": val_result["BF_top1_percent"]
                    - 100 * recorded["body_face_acc"],
                "HF_ce_difference": val_result["HF_ce"] - recorded["hand_face_loss"],
                "BF_ce_difference": val_result["BF_ce"] - recorded["body_face_loss"],
            },
            "strict_state_load": True, "finite_saved_model_and_optimizer": True,
            "all_state_tensors_unchanged_during_inference": True,
            "saved_routing": routing,
        }
        del checkpoint, initial_state

    require(reports["best"]["epoch"] <= reports["last"]["epoch"],
            "Best epoch is newer than last epoch")
    report = {
        "schema_version": 1, "run_dir": str(run_dir), "output_dir": str(output_dir),
        "args": config["args"], "model_config": config["model_config"],
        "compile": config["compile"], "manifest": manifest,
        "cache_keys": {"hf": "CPU float32 [samples, classes] HF logits",
                       "bf": "CPU float32 [samples, classes] BF logits",
                       "labels": "CPU int64 [samples] zero-based action labels",
                       "member_sha256": "SHA256 of manifest members joined by newline in cache order",
                       "checkpoint_sha256": "SHA256 of exactly the checkpoint bytes loaded",
                       "epoch": "Saved checkpoint epoch", "partition": "calibration or val",
                       "schema_version": 1},
        "checkpoints": reports, "split_counts": split_counts,
        "selection_data": "training-subject calibration only; official val is evaluation only",
        "protocol": {"split": args.split, "window_size": args.window_size,
                     "max_persons": args.max_persons, "feature_mode": "raw",
                     "temporal_compression": "confidence_weighted", "augmentation": False,
                     "shuffle": False, "drop_last": False,
                     "batch_size": args.test_batch_size, "num_workers": args.num_workers,
                     "tf32": args.tf32, "compile": config["compile"]},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": str(device), "gpu": torch.cuda.get_device_name(device)
                        if device.type == "cuda" else None,
                    "seconds": time.perf_counter() - started},
    }
    check_finite(report, "cache_report")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "report": str(output_dir / "report.json"),
                      "epochs": {name: value["epoch"] for name, value in reports.items()},
                      "seconds": report["runtime"]["seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
