"""Read and validate durable face-pair checkpoints while training continues."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys


def find_code_dir(explicit):
    script = Path(__file__).resolve()
    candidates = ([explicit] if explicit else []) + [
        Path.cwd(), script.parent / "code", script.parent.parent / "code", script.parents[2]]
    for candidate in candidates:
        if candidate and (candidate / "tools/check_face_pair_checkpoint.py").is_file():
            return candidate.resolve()
    raise RuntimeError("Cannot locate source tree; supply --code-dir")


def signature(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report; otherwise print only")
    args = parser.parse_args()
    code_dir = find_code_dir(args.code_dir)
    sys.path.insert(0, str(code_dir))
    spec = importlib.util.spec_from_file_location(
        "face_pair_saved_state_check", code_dir / "tools/check_face_pair_checkpoint.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)  # The helper imports isaa.train before torch.
    check.torch.set_num_threads(min(4, check.torch.get_num_threads()))
    run_dir = args.run_dir.resolve(strict=True)
    config = check.read_json(run_dir / "config.json")
    status_before = check.read_json(run_dir / "status.json")
    manifest = check.read_json(run_dir / "split_manifest.json")
    check.require(status_before.get("status") in ("running", "completed"),
                  f"Training is not healthy: {status_before.get('status')}")
    check.require(config.get("architecture") == check.FacePairFusion.ARCHITECTURE,
                  "Config architecture mismatch")
    check.require(config.get("initialization") == "random_scratch_no_checkpoint",
                  "Training was not initialized from scratch")
    check.check_args(config["args"], "config.args")
    check.check_compile(config["compile"], config["args"], "config.compile")
    check.require(not any(str(flag).split("=", 1)[0] == "--no-compile"
                          for flag in config.get("command", [])), "Saved command disables compile")
    split_counts = check.check_manifest(manifest, config)
    checkpoint_paths = [run_dir / "last.pt", run_dir / "best.pt"]
    for path in checkpoint_paths:
        check.require(path.is_file(), f"No saved {path.name}; wait for a complete train/calibration/val epoch")

    # Training replaces checkpoint files atomically. A stable before/after
    # signature ensures hashes and both loaded checkpoints describe the same
    # saved files, even if an epoch finishes during this CPU-only inspection.
    for attempt in range(3):
        before = {path: signature(path) for path in checkpoint_paths}
        checked_files = {run_dir / "config.json", run_dir / "status.json", run_dir / "split_manifest.json",
                         run_dir / "routing_latest.json"}
        summaries, checkpoints, hashes = {}, {}, {}
        for path in checkpoint_paths:
            summaries[path.stem], checkpoints[path.stem] = check.check_checkpoint(
                path, config, manifest, run_dir, checked_files)
            hashes[path.name] = check.sha256(path)
        after = {path: signature(path) for path in checkpoint_paths}
        last, best = checkpoints["last"], checkpoints["best"]
        pair_matches = (best["epoch"] <= last["epoch"]
                        and best["epoch"] == last["metrics"]["best_epoch"]
                        and best["metrics"]["best_val_acc"] == last["metrics"]["best_val_acc"])
        if before == after and pair_matches:
            break
    else:
        raise RuntimeError("Could not read a stable, matching last/best pair; rerun the progress check")
    status_after = check.read_json(run_dir / "status.json")
    check.require(status_after.get("status") in ("running", "completed"),
                  f"Training changed to unhealthy state: {status_after.get('status')}")
    config_args = config["args"]
    report = {
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "code_dir": str(code_dir), "load_device": "cpu",
        "training_status_at_start": status_before, "training_status_at_end": status_after,
        "scope": "durable saved checkpoints; live batch progress may be newer",
        "configured_epochs": config_args["epochs"],
        "configuration": {key: config_args[key] for key in
                          ("model_variant", "device", "compile", "compile_mode", "batch_size",
                           "test_batch_size", "num_workers", "window_size", "temporal_compression",
                           "lr", "warmup_epochs", "pair_calibration_fraction", "pair_calibration_start_epoch")},
        "compile_guards_verified": True, "random_initialization_verified": True,
        "official_validation_used_for_routing": False, "disjoint_data_splits": True,
        "split_counts": split_counts, "checkpoints": summaries, "checkpoint_sha256": hashes,
        "routing_evidence": "each checkpoint matches its immutable routing_epoch JSON; buffers match accuracy counts",
        "read_attempts": attempt + 1,
    }
    if args.output:
        output = args.output.resolve()
        check.require(output not in checked_files | set(checkpoint_paths), "Output would overwrite training artifacts")
        output.parent.mkdir(parents=True, exist_ok=True)
        check.write_json(output, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
