"""Validate experimental metadata and reuse the strict face-pair run checker."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.training_route_mismatch_20261106.soft_route import SoftRouteFacePair
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT
from tools import check_face_pair_checkpoint as base


def check_experiment_metadata(run_dir):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metadata = config.get("experimental_training_fusion", {})
    if metadata.get("method") != "previous_calibration_counts_gamma1_arithmetic":
        raise ValueError("Missing gamma=1 experiment metadata")
    if config.get("architecture") != SoftRouteFacePair.ARCHITECTURE:
        raise ValueError("Wrong candidate architecture")
    expected = {"experiments/training_route_mismatch_20261106/" + p for p in
                ("train_soft_route.py", "soft_route.py", "protocol.md")}
    hashes = metadata.get("source_sha256", {})
    if set(hashes) != expected:
        raise ValueError("Unexpected experiment source list")
    with ZipFile(run_dir / "source.zip") as archive:
        for path in expected:
            if archive.namelist().count(path) != 1:
                raise ValueError(f"Missing or repeated source in archive: {path}")
            if hashlib.sha256(archive.read(path)).hexdigest() != hashes[path]:
                raise ValueError(f"Archived source digest mismatch: {path}")
    args = config["args"]
    expected_args = {"epochs": 20, "batch_size": 32, "max_samples": 4096,
                     "warmup_epochs": 1, "seed": 29, "lr": 0.025,
                     "pair_calibration_start_epoch": 1, "compile": True,
                     "compile_mode": "default"}
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise ValueError("Run differs from the fixed short-screen budget")
    for epoch in range(1, 21):
        route = json.loads((run_dir / f"routing_epoch_{epoch:03}.json").read_text(encoding="utf-8"))
        effective = json.loads((run_dir / f"effective_routing_epoch_{epoch:03}.json").read_text(encoding="utf-8"))
        if effective["epoch"] != epoch or len(effective["classes"]) != len(route["classes"]):
            raise ValueError("Effective route epoch or class count mismatch")
        for hard, soft in zip(route["classes"], effective["classes"]):
            h, b = hard["hand_face_correct"], hard["body_face_correct"]
            w = h / (h + b) if h + b else .5
            if soft["action_id"] != hard["action_id"] or abs(soft["hand_face_weight"] - w) > 1e-12:
                raise ValueError("Effective route does not match saved calibration counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    check_experiment_metadata(options.run_dir)
    base.FACE_PAIR_MODELS[FACE_PAIR_VARIANT] = SoftRouteFacePair
    base.main()


if __name__ == "__main__":
    main()
