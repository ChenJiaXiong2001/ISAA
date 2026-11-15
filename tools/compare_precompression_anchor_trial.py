"""Paired subject-bootstrap comparison for clean vs pre-compression anchor dropout trials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def _load_evaluation(path: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with (path / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["sample"] for row in rows]
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    if not rows or len(names) != len(set(names)):
        raise ValueError(f"Missing or duplicate predictions in {path}")
    if digest != summary.get("members_sha256") or len(rows) != summary.get("samples"):
        raise ValueError(f"Prediction membership does not match summary in {path}")
    return summary, rows


def _hits(rows: list[dict], condition: str) -> np.ndarray:
    return np.array(
        [int(row[f"{condition}_unmodified_prediction"]) == int(row["label"]) for row in rows],
        dtype=np.int8,
    )


def _subject_bootstrap(values: np.ndarray, groups: np.ndarray, counts: np.ndarray,
                       draws: np.ndarray) -> list[float]:
    subject_sums = np.bincount(groups, weights=values, minlength=len(counts))
    sampled_counts = draws @ counts
    sampled_values = 100.0 * (draws @ subject_sums) / sampled_counts
    return [float(x) for x in np.percentile(sampled_values, [2.5, 97.5])]


def compare(candidate_paths: dict[str, Path], control_paths: dict[str, Path],
            repeats: int, seed: int) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    loaded_candidate = {name: _load_evaluation(path) for name, path in candidate_paths.items()}
    loaded_control = {name: _load_evaluation(path) for name, path in control_paths.items()}
    reference_rows = loaded_candidate["p05"][1]
    names = [row["sample"] for row in reference_rows]
    labels = [row["label"] for row in reference_rows]
    membership_sha = loaded_candidate["p05"][0]["members_sha256"]

    for probability in ("p05", "p10"):
        for label, loaded in (("candidate", loaded_candidate), ("control", loaded_control)):
            summary, rows = loaded[probability]
            if summary.get("members_sha256") != membership_sha:
                raise ValueError(f"{label} {probability} uses a different validation member set")
            if [row["sample"] for row in rows] != names or [row["label"] for row in rows] != labels:
                raise ValueError(f"{label} {probability} prediction order/labels differ")
        candidate_summary = loaded_candidate[probability][0]
        control_summary = loaded_control[probability][0]
        candidate_hash = candidate_summary.get("input_sha256", {}).get("corrupted")
        control_hash = control_summary.get("input_sha256", {}).get("corrupted")
        if candidate_hash != control_hash:
            raise ValueError(f"Candidate/control {probability} corruption masks differ")
        candidate_clean_hash = candidate_summary.get("input_sha256", {}).get("clean")
        control_clean_hash = control_summary.get("input_sha256", {}).get("clean")
        if candidate_clean_hash != control_clean_hash:
            raise ValueError(f"Candidate/control {probability} clean inputs differ")

    subjects = []
    for name in names:
        match = re.search(r"P(\d{3})R\d{3}A\d{3}", name)
        if match is None:
            raise ValueError(f"Could not parse subject from sample {name}")
        subjects.append(match.group(1))
    subject_names, groups = np.unique(subjects, return_inverse=True)
    counts = np.bincount(groups)
    if len(subject_names) < 2:
        raise ValueError("Need at least two subjects for clustered bootstrap")
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(subject_names), np.full(len(subject_names), 1 / len(subject_names)),
                            size=repeats)

    candidate_hits = {p: {c: _hits(rows, c) for c in ("clean", "pre_dropout")}
                       for p, (_, rows) in loaded_candidate.items()}
    control_hits = {p: {c: _hits(rows, c) for c in ("clean", "pre_dropout")}
                    for p, (_, rows) in loaded_control.items()}
    result = {
        "samples": len(names), "subjects": len(subject_names), "repeats": repeats,
        "seed": seed, "bootstrap_unit": "complete subject; sample-weighted accuracy",
        "members_sha256": membership_sha,
        "checkpoint_sha256": {
            "candidate": {p: loaded_candidate[p][0].get("checkpoint_sha256") for p in ("p05", "p10")},
            "control": {p: loaded_control[p][0].get("checkpoint_sha256") for p in ("p05", "p10")},
        },
        "input_sha256": {p: loaded_candidate[p][0].get("input_sha256") for p in ("p05", "p10")},
        "conditions": {},
    }
    clean_diff = candidate_hits["p05"]["clean"] - control_hits["p05"]["clean"]
    result["conditions"]["clean"] = {
        "candidate_accuracy_percent": float(candidate_hits["p05"]["clean"].mean() * 100),
        "control_accuracy_percent": float(control_hits["p05"]["clean"].mean() * 100),
        "candidate_minus_control_pp": float(clean_diff.mean() * 100),
        "candidate_minus_control_subject_bootstrap_95_percent_ci_pp":
            _subject_bootstrap(clean_diff, groups, counts, draws),
    }
    for probability in ("p05", "p10"):
        candidate_clean = candidate_hits[probability]["clean"]
        control_clean = control_hits[probability]["clean"]
        candidate_corrupt = candidate_hits[probability]["pre_dropout"]
        control_corrupt = control_hits[probability]["pre_dropout"]
        diff = candidate_corrupt - control_corrupt
        difference_in_degradation = (candidate_corrupt - candidate_clean) - (control_corrupt - control_clean)
        result["conditions"][probability] = {
            "candidate_corrupted_accuracy_percent": float(candidate_corrupt.mean() * 100),
            "control_corrupted_accuracy_percent": float(control_corrupt.mean() * 100),
            "candidate_minus_control_corrupted_pp": float(diff.mean() * 100),
            "candidate_minus_control_corrupted_subject_bootstrap_95_percent_ci_pp":
                _subject_bootstrap(diff, groups, counts, draws),
            "candidate_change_from_clean_pp": float((candidate_corrupt - candidate_clean).mean() * 100),
            "control_change_from_clean_pp": float((control_corrupt - control_clean).mean() * 100),
            "robustness_gain_difference_in_degradation_pp":
                float(difference_in_degradation.mean() * 100),
            "robustness_gain_subject_bootstrap_95_percent_ci_pp":
                _subject_bootstrap(difference_in_degradation, groups, counts, draws),
        }
    result["caveat"] = (
        "Short-run, single-seed scratch-training screen on a fixed validation split; "
        "subject bootstrap measures validation sampling uncertainty, not seed or domain uncertainty."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-p05", type=Path, required=True)
    parser.add_argument("--candidate-p10", type=Path, required=True)
    parser.add_argument("--control-p05", type=Path, required=True)
    parser.add_argument("--control-p10", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(
        {"p05": args.candidate_p05, "p10": args.candidate_p10},
        {"p05": args.control_p05, "p10": args.control_p10},
        args.repeats,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
