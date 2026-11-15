"""Summarize frozen score-power compression predictions.

The screen writes one row per (power, validation member).  This post-process
keeps the comparison paired at the sample and subject levels and does not
select a power from a held-out split.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError("empty predictions file")
    powers = sorted({float(row["power"]) for row in rows})
    by_power: dict[float, list[dict[str, str]]] = {p: [] for p in powers}
    for row in rows:
        by_power[float(row["power"])].append(row)
    n = len(by_power[powers[0]])
    if any(len(v) != n for v in by_power.values()):
        raise ValueError("power blocks have different sample counts")
    samples = [row["sample"] for row in by_power[powers[0]]]
    labels = np.asarray([int(row["label"]) for row in by_power[powers[0]]])
    subjects = np.asarray([int(sample.split("P", 1)[1][:3]) for sample in samples])
    lengths: list[int] = []
    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        for sample in samples:
            name = sample if sample in names else next(x for x in names if x.endswith(sample))
            with archive.open(name) as stream:
                lengths.append(int(np.load(io.BytesIO(stream.read()), allow_pickle=False)["keypoints"].shape[0]))
    lengths_np = np.asarray(lengths)
    unique_subjects = np.unique(subjects)
    subject_indices = [np.flatnonzero(subjects == subject_id) for subject_id in unique_subjects]
    rng = np.random.default_rng(20261107)
    draws = rng.integers(0, len(unique_subjects), size=(args.bootstrap, len(unique_subjects)))

    predictions: dict[float, np.ndarray] = {}
    for power in powers:
        block = by_power[power]
        if [row["sample"] for row in block] != samples:
            raise ValueError(f"sample order differs for power={power}")
        predictions[power] = np.asarray([int(row["soft_prediction"]) for row in block])
    baseline_power = 1.0 if 1.0 in predictions else powers[0]
    baseline = predictions[baseline_power] == labels
    result: dict[str, object] = {
        "predictions": str(args.predictions),
        "samples": n,
        "powers": powers,
        "baseline_power": baseline_power,
        "length_gt100": int((lengths_np > 100).sum()),
        "conditions": {},
    }
    for power in powers:
        correct = predictions[power] == labels
        subject_gains = np.asarray([
            float((correct[index] & ~baseline[index]).sum() - (~correct[index] & baseline[index]).sum())
            / len(index)
            for index in subject_indices
        ])
        boots = subject_gains[draws].mean(axis=1)
        result["conditions"][str(power)] = {
            "correct": int(correct.sum()),
            "accuracy_percent": float(100.0 * correct.mean()),
            "gain_vs_baseline_pp": float(100.0 * (correct.mean() - baseline.mean())),
            "bootstrap95_gain_vs_baseline_pp": [
                float(100.0 * np.quantile(boots, 0.025)),
                float(100.0 * np.quantile(boots, 0.975)),
            ],
            "corrected": int((correct & ~baseline).sum()),
            "regressed": int((~correct & baseline).sum()),
            "subject_positive": int((subject_gains > 0).sum()),
            "subject_negative": int((subject_gains < 0).sum()),
            "length_bins": {
                "le100": int((correct[lengths_np <= 100]).sum()),
                "gt100": int((correct[lengths_np > 100]).sum()),
                "le100_n": int((lengths_np <= 100).sum()),
                "gt100_n": int((lengths_np > 100).sum()),
            },
        }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
