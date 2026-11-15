"""Fail-closed identity and arithmetic guard for the frozen temporal policy.

The original remote post-processing script joined every temporal prediction CSV
to the canonical labels by row index.  This helper makes that join explicit and
refuses to report metrics when a candidate CSV has a different member or label
sequence.  It performs no model inference and never writes into a production
run directory.

Typical remote invocation::

    python temporal_policy_identity_guard.py \
      --canonical /home/xiong/Desktop/ISAA/outputs/canonical_p10_pair_full_20261106/predictions.csv \
      --cw-dir /home/xiong/Desktop/ISAA/outputs/temporal_corruption_cw_singlepass_full_20261106 \
      --none-dir /home/xiong/Desktop/ISAA/outputs/temporal_corruption_none_center_full_20261106 \
      --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
      --expected-summary /home/xiong/Desktop/ISAA/outputs/temporal_policy_subject_bootstrap_20261107.json \
      --output /home/xiong/Desktop/ISAA/outputs/temporal_policy_identity_guard_20261107.json

Pass ``--manifest`` when the run's split_manifest.json is available.  The
manifest check is deliberately strict; a CSV-derived member hash alone does
not establish that the rows came from the formal validation partition.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CONDITIONS = ("clean", "p05", "p10")
EXPECTED_COUNT = 16_306
SAMPLE_RE = re.compile(r"P(?P<subject>\d{3}).*A(?P<action>\d{3})", re.IGNORECASE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def length_sha256(samples: List[str], lengths: List[int]) -> str:
    payload = (f"{sample}\t{length}" for sample, length in zip(samples, lengths))
    return line_sha256(payload)


def read_rows(path: Path, required: Tuple[str, ...]) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return rows


def parse_int(value: str, field: str, path: Path, row_index: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: row {row_index} has non-integer {field}: {value!r}") from exc


def validate_canonical(rows: List[Dict[str, str]], path: Path) -> Tuple[List[str], List[int]]:
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"{path}: expected {EXPECTED_COUNT} rows, found {len(rows)}")
    samples = [row["sample"] for row in rows]
    if len(set(samples)) != len(samples):
        raise ValueError(f"{path}: duplicate sample member")
    labels: List[int] = []
    for i, row in enumerate(rows, 2):
        label = parse_int(row["label"], "label", path, i)
        if not 0 <= label < 60:
            raise ValueError(f"{path}: row {i} label outside [0, 59]: {label}")
        match = SAMPLE_RE.search(row["sample"])
        if match is None:
            raise ValueError(f"{path}: row {i} sample has no NTU subject/action: {row['sample']!r}")
        action = int(match.group("action")) - 1
        if action != label:
            raise ValueError(
                f"{path}: row {i} label/sample mismatch: label={label}, action={action}, sample={row['sample']}"
            )
        labels.append(label)
    return samples, labels


def validate_candidate(
    path: Path,
    canonical_samples: List[str],
    canonical_labels: List[int],
) -> Tuple[List[int], Dict[str, object]]:
    # evaluate_face_pair_corruption.py intentionally writes sample_index rather
    # than repeating the long member path.  If a future producer includes a
    # sample column too, it is checked as an additional invariant below.
    rows = read_rows(path, ("sample_index", "label", "prediction"))
    if len(rows) != len(canonical_samples):
        raise ValueError(f"{path}: expected {len(canonical_samples)} rows, found {len(rows)}")
    indices: List[int] = []
    labels: List[int] = []
    predictions: List[int] = []
    for i, row in enumerate(rows, 2):
        sample_index = parse_int(row["sample_index"], "sample_index", path, i)
        label = parse_int(row["label"], "label", path, i)
        prediction = parse_int(row["prediction"], "prediction", path, i)
        indices.append(sample_index)
        if not 0 <= label < 60 or not 0 <= prediction < 60:
            raise ValueError(f"{path}: row {i} label/prediction outside [0, 59]")
        labels.append(label)
        predictions.append(prediction)
    expected_indices = list(range(len(canonical_samples)))
    if indices != expected_indices:
        first = next(i for i, (a, b) in enumerate(zip(indices, expected_indices)) if a != b)
        raise ValueError(
            f"{path}: sample_index order differs at row {first + 2}: candidate={indices[first]}, "
            f"expected={expected_indices[first]}"
        )
    if "sample" in rows[0]:
        samples = [row["sample"] for row in rows]
        if samples != canonical_samples:
            first = next(i for i, (a, b) in enumerate(zip(samples, canonical_samples)) if a != b)
            raise ValueError(
                f"{path}: sample order differs at row {first + 2}: candidate={samples[first]!r}, "
                f"canonical={canonical_samples[first]!r}"
            )
    if labels != canonical_labels:
        first = next(i for i, (a, b) in enumerate(zip(labels, canonical_labels)) if a != b)
        raise ValueError(
            f"{path}: label order differs at row {first + 2}: candidate={labels[first]}, "
            f"canonical={canonical_labels[first]}"
        )
    return predictions, {
        "path": str(path),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "sample_index_sha256": line_sha256([str(value) for value in indices]),
        "label_sha256": line_sha256([str(value) for value in labels]),
        "sample_index_and_label_match_canonical": True,
        "embedded_sample_column_checked": "sample" in rows[0],
    }


def read_lengths_csv(path: Path, canonical_samples: List[str]) -> List[int]:
    rows = read_rows(path, ("sample", "raw_frames"))
    if len(rows) != len(canonical_samples):
        raise ValueError(f"{path}: expected {len(canonical_samples)} rows, found {len(rows)}")
    samples = [row["sample"] for row in rows]
    if samples != canonical_samples:
        raise ValueError(f"{path}: sample order differs from canonical CSV")
    lengths = [parse_int(row["raw_frames"], "raw_frames", path, i) for i, row in enumerate(rows, 2)]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"{path}: raw_frames must be positive")
    return lengths


def read_lengths_archive(path: Path, canonical_samples: List[str]) -> List[int]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - the project runtime includes numpy
        raise RuntimeError("numpy is required for --archive length verification") from exc
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        lengths: List[int] = []
        for sample in canonical_samples:
            if sample in names:
                name = sample
            else:
                matches = [candidate for candidate in names if candidate.endswith(sample)]
                if len(matches) != 1:
                    raise ValueError(f"{path}: expected one archive member ending with {sample!r}, found {len(matches)}")
                name = matches[0]
            with archive.open(name) as stream:
                data = np.load(io.BytesIO(stream.read()), allow_pickle=False)
                if "keypoints" not in data:
                    raise ValueError(f"{path}: {name} has no keypoints array")
                length = int(data["keypoints"].shape[0])
            if length <= 0:
                raise ValueError(f"{path}: {name} has non-positive frame length {length}")
            lengths.append(length)
    return lengths


def manifest_members(path: Path) -> Tuple[List[str], str, str, bool]:
    value = json.loads(path.read_text(encoding="utf-8"))
    section = value.get("validation", value)
    members = section.get("members")
    expected = section.get("sha256")
    if not isinstance(members, list) or not isinstance(expected, str):
        raise ValueError(f"{path}: expected validation.members and validation.sha256")
    actual = line_sha256([str(member) for member in members])
    # Keep both values in the output so provenance can be reviewed without
    # trusting a recorded digest blindly.  The normal case is self-consistent;
    # a stale digest becomes an explicit warning while member-list equality is
    # still enforced by the caller.
    return [str(member) for member in members], expected, actual, actual == expected


def expected_metric_check(result: Dict[str, object], path: Path) -> Dict[str, object]:
    expected = json.loads(path.read_text(encoding="utf-8"))
    checks: Dict[str, object] = {}
    metrics = result["metrics"]
    for condition in CONDITIONS:
        observed = metrics[condition]
        recorded = expected.get(condition)
        if not isinstance(recorded, dict):
            raise ValueError(f"{path}: missing condition {condition!r}")
        for key in ("n", "control_acc", "policy_acc", "gain_pp"):
            if key == "n":
                actual_value = observed["n"]
                expected_value = recorded.get("n")
                if actual_value != expected_value:
                    raise ValueError(f"{path}: {condition}.{key} differs: {actual_value} != {expected_value}")
            else:
                actual_value = float(observed[key])
                expected_value = float(recorded[key])
                if abs(actual_value - expected_value) > 1e-9:
                    raise ValueError(
                        f"{path}: {condition}.{key} differs: {actual_value:.15g} != {expected_value:.15g}"
                    )
        checks[condition] = {"match": True, "recorded": recorded}
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--cw-dir", type=Path, required=True)
    parser.add_argument("--none-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--lengths-csv", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-member-sha")
    parser.add_argument("--expected-summary", type=Path)
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.archive and args.lengths_csv:
        raise ValueError("pass only one of --archive and --lengths-csv")
    if not args.archive and not args.lengths_csv:
        raise ValueError("one of --archive or --lengths-csv is required for length identity")
    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative")
    if args.output.exists():
        raise FileExistsError(args.output)

    canonical_rows = read_rows(args.canonical, ("sample", "label"))
    canonical_samples, canonical_labels = validate_canonical(canonical_rows, args.canonical)
    warnings = []
    if args.manifest:
        members, manifest_sha, manifest_derived_sha, manifest_self_hash_consistent = manifest_members(args.manifest)
        if members != canonical_samples:
            raise ValueError("canonical sample order differs from formal validation manifest")
        if not manifest_self_hash_consistent:
            warnings.append(
                "manifest recorded sha256 differs from the SHA derived from its own members list"
            )
    else:
        manifest_sha = manifest_derived_sha = None
        manifest_self_hash_consistent = None
    canonical_sample_sha = line_sha256(canonical_samples)
    if args.expected_member_sha and canonical_sample_sha != args.expected_member_sha:
        raise ValueError(
            f"canonical member SHA mismatch: actual={canonical_sample_sha}, expected={args.expected_member_sha}"
        )

    if args.lengths_csv:
        lengths = read_lengths_csv(args.lengths_csv, canonical_samples)
        length_source = str(args.lengths_csv)
    else:
        lengths = read_lengths_archive(args.archive, canonical_samples)
        length_source = str(args.archive)
    length_sha = length_sha256(canonical_samples, lengths)

    predictions: Dict[str, Dict[str, List[int]]] = {}
    files: Dict[str, Dict[str, object]] = {}
    for mode, directory in (("cw", args.cw_dir), ("none", args.none_dir)):
        predictions[mode] = {}
        files[mode] = {}
        for condition in CONDITIONS:
            suffix = {"clean": "clean", "p05": "point_0.05", "p10": "point_0.1"}[condition]
            path = directory / f"predictions_{suffix}.csv"
            values, metadata = validate_candidate(path, canonical_samples, canonical_labels)
            predictions[mode][condition] = values
            files[mode][condition] = metadata

    subject_ids = []
    for sample in canonical_samples:
        match = SAMPLE_RE.search(sample)
        if match is None:
            raise ValueError(f"sample has no subject: {sample!r}")
        subject_ids.append(int(match.group("subject")))

    metrics: Dict[str, Dict[str, object]] = {}
    for condition in CONDITIONS:
        control = predictions["none"][condition]
        candidate = predictions["cw"][condition]
        control_correct = [prediction == label for prediction, label in zip(control, canonical_labels)]
        candidate_correct = [prediction == label for prediction, label in zip(candidate, canonical_labels)]
        policy_correct = [
            candidate_ok if length > args.threshold else control_ok
            for length, candidate_ok, control_ok in zip(lengths, candidate_correct, control_correct)
        ]
        subject_delta: Dict[str, float] = {}
        for subject in sorted(set(subject_ids)):
            selected = [i for i, value in enumerate(subject_ids) if value == subject]
            subject_delta[str(subject)] = 100.0 * (
                sum(policy_correct[i] for i in selected) - sum(control_correct[i] for i in selected)
            ) / len(selected)
        metrics[condition] = {
            "n": len(canonical_labels),
            "control_correct": sum(control_correct),
            "policy_correct": sum(policy_correct),
            "control_acc": sum(control_correct) / len(canonical_labels),
            "policy_acc": sum(policy_correct) / len(canonical_labels),
            "gain_pp": 100.0 * (sum(policy_correct) - sum(control_correct)) / len(canonical_labels),
            "corrected": sum(policy_ok and not control_ok for policy_ok, control_ok in zip(policy_correct, control_correct)),
            "regressed": sum(control_ok and not policy_ok for policy_ok, control_ok in zip(policy_correct, control_correct)),
            "subject_gain_pp": subject_delta,
            "subjects_positive": sum(value > 0 for value in subject_delta.values()),
            "subjects_negative": sum(value < 0 for value in subject_delta.values()),
            "subjects_zero": sum(value == 0 for value in subject_delta.values()),
        }

    short_diffs = {
        condition: sum(
            candidate != control and length <= args.threshold
            for candidate, control, length in zip(predictions["cw"][condition], predictions["none"][condition], lengths)
        )
        for condition in CONDITIONS
    }
    result: Dict[str, object] = {
        "status": "PASS_WITH_WARNINGS" if warnings else "PASS",
        "canonical": {
            "path": str(args.canonical),
            "sha256": file_sha256(args.canonical),
            "rows": len(canonical_samples),
            "sample_sha256": canonical_sample_sha,
            "label_sha256": line_sha256([str(value) for value in canonical_labels]),
            "unique_samples": len(set(canonical_samples)),
            "sample_label_action_consistency": True,
        },
        "manifest": {
            "path": str(args.manifest) if args.manifest else None,
            "recorded_sha256": manifest_sha,
            "derived_sha256": manifest_derived_sha,
            "self_hash_consistent": manifest_self_hash_consistent,
            "checked": args.manifest is not None,
        },
        "lengths": {
            "source": length_source,
            "rows": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "length_gt_threshold": sum(length > args.threshold for length in lengths),
            "length_sha256": length_sha,
            "hash_definition": "SHA256 of ordered '<sample>\\t<raw_frames>' lines",
        },
        "files": files,
        "threshold": args.threshold,
        "short_prediction_diffs": short_diffs,
        "metrics": metrics,
        "warnings": warnings,
    }
    if args.expected_summary:
        result["expected_summary"] = expected_metric_check(result, args.expected_summary)
        result["expected_summary_path"] = str(args.expected_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
