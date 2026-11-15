"""Check the actual experiment arguments against the existing two controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import isaa.train as training
from train import training_arguments, validate_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-config", type=Path, required=True)
    parser.add_argument("--fixed-config", type=Path, required=True)
    options = parser.parse_args()
    configs = [json.loads(p.read_text()) for p in (options.clean_config, options.fixed_config)]
    sys.argv = ["train.py", *training_arguments([
        "--archive", "/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip",
        "--device", "cuda", "--compile", "--compile-mode", "default",
        "--epochs", "20", "--warmup-epochs", "1", "--num-workers", "8",
        "--seed", "17", "--max-samples", "4096", "--log-interval", "20",
        "--pair-calibration-start-epoch", "1", "--precompression-anchor-dropout-probability", "0.025",
        "--precompression-anchor-clean-sample-probability", "0.5",
        "--save-dir", "outputs/anchor_clean_anneal_20ep4096_20261106"])]
    actual = vars(validate_protocol(training.parse_args()))
    ignored = {"save_dir", "precompression_anchor_dropout_probability",
               "precompression_anchor_clean_sample_probability", "anchor_clean_curriculum"}
    for config in configs:
        expected = config["args"]
        diffs = {key: (actual.get(key), expected.get(key)) for key in actual.keys() | expected.keys()
                 if key not in ignored and actual.get(key) != expected.get(key)}
        if diffs:
            raise ValueError(f"Training settings differ: {diffs}")
        schedule = config["lr_schedule"]
        if (config["initialization"] != "random_scratch_no_checkpoint" or
                schedule["batches_per_epoch"] != 117 or schedule["total_updates"] != 2340 or
                schedule["warmup_updates"] != 117 or schedule["completed_updates"] != 2340):
            raise ValueError("Control initialization or learning-rate schedule differs")
    print("Both controls match the candidate protocol; 117 batches/epoch and 2340 updates")


if __name__ == "__main__":
    main()
