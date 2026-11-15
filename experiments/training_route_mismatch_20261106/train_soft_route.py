"""Isolated gamma=1 training experiment; no existing training entry is changed."""
import sys
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train_best_face_pair_100 import training_arguments as original_arguments
from experiments.training_route_mismatch_20261106.soft_route import SoftRouteFacePair


def training_arguments(overrides):
    supplied = {x.split("=", 1)[0] for x in overrides if x.startswith("--")}
    if "--save-dir" not in supplied:
        raise ValueError("Provide a new experiment-specific --save-dir")
    return original_arguments(overrides)


def record_experiment_sources(save_dir, records):
    paths = [Path(__file__).resolve(), Path(__file__).resolve().with_name("soft_route.py"),
             Path(__file__).resolve().with_name("protocol.md")]
    hashes = {}
    with ZipFile(save_dir / "source.zip", "a", ZIP_DEFLATED) as archive:
        for path in paths:
            contents = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            if name in archive.namelist():
                raise RuntimeError(f"Experiment source already exists in archive: {name}")
            archive.writestr(name, contents)
            hashes[name] = hashlib.sha256(contents).hexdigest()
    records.update_config(experimental_training_fusion={
        "method": "previous_calibration_counts_gamma1_arithmetic",
        "inactive_or_both_zero": "equal_probability_average",
        "head_auxiliary_losses": "unchanged",
        "hard_routing_reports_and_buffer": "diagnostic_only_not_used_by_forward",
        "effective_weight_record": "effective_routing_epoch_NNN.json; applies to current validation and following training epoch",
        "source_sha256": hashes})


def save_effective_routing(path, report):
    rows = []
    for entry in report["classes"]:
        h, b = entry["hand_face_correct"], entry["body_face_correct"]
        rows.append({"class_index": entry["class_index"], "action_id": entry["action_id"],
                     "samples": entry["samples"], "hand_face_correct": h,
                     "body_face_correct": b, "hand_face_weight": h / (h + b) if h + b else 0.5})
    value = {"epoch": report["epoch"], "calibration_active": report["active"],
             "weight_rule": "gamma1_ratio_from_saved_counts",
             "applies_to": "current_epoch_validation_and_next_epoch_training",
             "prior_training_epoch_used": "previous_epoch_calibration_counts_or_initial_equal_weights",
             "hard_routing_report": "diagnostic_only_not_used_by_forward", "classes": rows}
    effective = path.with_name("effective_" + path.name)
    effective.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    sys.argv[1:] = training_arguments(sys.argv[1:])
    import isaa.face_pair_training as training
    from isaa.train import main as train_main

    original_runner = training.run_face_pair_training
    original_save_routing = training.save_routing

    def run_experiment(args, save_dir, records, device):
        record_experiment_sources(save_dir, records)
        original_runner(args, save_dir, records, device)

    def save_experiment_routing(path, report):
        original_save_routing(path, report)
        save_effective_routing(path, report)

    # Process-local substitution preserves the old class, buffers and runner.
    training.FacePairFusion = SoftRouteFacePair
    training.run_face_pair_training = run_experiment
    training.save_routing = save_experiment_routing
    train_main()


if __name__ == "__main__":
    main()
