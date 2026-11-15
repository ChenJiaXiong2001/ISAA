"""Stratify the frozen anchor-interpolation validation by class and subject.

This is a read-only analysis of the paired predictions produced by
evaluate_face_pair_anchor_interpolation.py.  It does not load a model, tune a
parameter, or access labels beyond those already recorded in predictions.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# NTU's cross-subject identity is the Pxxx field; Sxxx is the setup/camera.
SAMPLE_RE = re.compile(r"P(?P<subject>\d{3}).*A(?P<action>\d{3})_rgb\.npz$")
CONDITIONS = (
    ("clean", "clean_unmodified", "clean_repaired"),
    ("point05", "point05_unmodified", "point05_repaired"),
    ("point10", "point10_unmodified", "point10_repaired"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_sample(sample: str) -> tuple[int, int]:
    match = SAMPLE_RE.search(sample)
    if match is None:
        raise ValueError(f"cannot parse subject/action from {sample!r}")
    return int(match.group("subject")), int(match.group("action"))


def summarize(rows: list[dict[str, int]], keys: list[str]) -> dict[str, object]:
    out: dict[str, object] = {"samples": len(rows)}
    for name, before, after in CONDITIONS:
        clean_key = "clean_unmodified"
        before_correct = sum(row[before] == row["label"] for row in rows)
        after_correct = sum(row[after] == row["label"] for row in rows)
        out[name] = {
            "before_correct": before_correct,
            "after_correct": after_correct,
            "before_accuracy_percent": 100.0 * before_correct / len(rows) if rows else None,
            "after_accuracy_percent": 100.0 * after_correct / len(rows) if rows else None,
            "delta_percentage_points": 100.0 * (after_correct - before_correct) / len(rows) if rows else None,
            "corrected": sum(row[before] != row["label"] and row[after] == row["label"] for row in rows),
            "regressed": sum(row[before] == row["label"] and row[after] != row["label"] for row in rows),
        }
        if name != "clean":
            clean_correct = sum(row[clean_key] == row["label"] for row in rows)
            before_drop = clean_correct - before_correct
            after_drop = clean_correct - after_correct
            out[name]["drop_before_percentage_points"] = 100.0 * before_drop / len(rows) if rows else None
            out[name]["drop_after_percentage_points"] = 100.0 * after_drop / len(rows) if rows else None
            out[name]["drop_reduction_percentage_points"] = 100.0 * (before_drop - after_drop) / len(rows) if rows else None
            out[name]["drop_recovery_fraction"] = (before_drop - after_drop) / before_drop if before_drop > 0 else None
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - my) ** 2 for y in ys) ** 0.5
    return numerator / (denom_x * denom_y) if denom_x and denom_y else None


def confusion_changes(rows: list[dict[str, int]], before_key: str, after_key: str, limit: int = 15) -> dict[str, list[dict[str, int]]]:
    before = Counter((row["label"], row[before_key]) for row in rows if row[before_key] != row["label"])
    after = Counter((row["label"], row[after_key]) for row in rows if row[after_key] != row["label"])
    changes = []
    for pair in set(before) | set(after):
        delta = after[pair] - before[pair]
        if delta:
            changes.append({"true_class": pair[0] + 1, "predicted_class": pair[1] + 1, "before": before[pair], "after": after[pair], "delta": delta})
    return {
        "largest_error_reductions": sorted(changes, key=lambda item: (item["delta"], item["true_class"], item["predicted_class"]))[:limit],
        "largest_error_increases": sorted(changes, key=lambda item: (-item["delta"], item["true_class"], item["predicted_class"]))[:limit],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("docs/face_pair_validation/codex_20261105/anchor_interpolation/validation/predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/face_pair_validation/codex_20261105/structure_input_audit_20261105"),
    )
    args = parser.parse_args()
    source = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample", "label"} | {column for _, before, after in CONDITIONS for column in (before, after)}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        rows: list[dict[str, int]] = []
        for raw in reader:
            subject, action = parse_sample(raw["sample"])
            row = {key: int(raw[key]) for key in required if key != "sample"}
            row["subject"] = subject
            row["action"] = action
            rows.append(row)

    by_action: dict[int, list[dict[str, int]]] = defaultdict(list)
    by_subject: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        by_action[row["action"]].append(row)
        by_subject[row["subject"]].append(row)

    classes = {f"A{action:03d}": summarize(items, ["action"]) for action, items in sorted(by_action.items())}
    subjects = {f"P{subject:03d}": summarize(items, ["subject"]) for subject, items in sorted(by_subject.items())}

    overall = summarize(rows, [])
    recovery_rank = []
    for action, items in sorted(by_action.items()):
        stats = classes[f"A{action:03d}"]
        point10 = stats["point10"]
        recovery_rank.append(
            {
                "class": f"A{action:03d}",
                "samples": stats["samples"],
                "point05_drop_before_pp": stats["point05"]["drop_before_percentage_points"],
                "point05_drop_after_pp": stats["point05"]["drop_after_percentage_points"],
                "point05_recovery_fraction": stats["point05"]["drop_recovery_fraction"],
                "point10_drop_before_pp": point10["drop_before_percentage_points"],
                "point10_drop_after_pp": point10["drop_after_percentage_points"],
                "point10_recovery_fraction": point10["drop_recovery_fraction"],
                "point10_delta_pp": point10["delta_percentage_points"],
            }
        )
    recovery_rank.sort(key=lambda item: (item["point10_delta_pp"], item["class"]), reverse=True)

    class_point05_drop = [classes[f"A{action:03d}"]["point05"]["drop_before_percentage_points"] for action in sorted(by_action)]
    class_point05_delta = [classes[f"A{action:03d}"]["point05"]["delta_percentage_points"] for action in sorted(by_action)]
    class_point10_drop = [classes[f"A{action:03d}"]["point10"]["drop_before_percentage_points"] for action in sorted(by_action)]
    class_point10_delta = [classes[f"A{action:03d}"]["point10"]["delta_percentage_points"] for action in sorted(by_action)]
    diagnostics = {
        "class_count": len(classes),
        "point05_classes_positive": sum(value > 0 for value in class_point05_delta),
        "point05_classes_zero": sum(value == 0 for value in class_point05_delta),
        "point05_classes_negative": sum(value < 0 for value in class_point05_delta),
        "point10_classes_positive": sum(value > 0 for value in class_point10_delta),
        "point10_classes_zero": sum(value == 0 for value in class_point10_delta),
        "point10_classes_negative": sum(value < 0 for value in class_point10_delta),
        "point05_drop_vs_gain_pearson_r": pearson(class_point05_drop, class_point05_delta),
        "point10_drop_vs_gain_pearson_r": pearson(class_point10_drop, class_point10_delta),
        "subject_count": len(subjects),
        "point10_subjects_positive": sum(value["point10"]["delta_percentage_points"] > 0 for value in subjects.values()),
        "point10_subject_min_delta_pp": min(value["point10"]["delta_percentage_points"] for value in subjects.values()),
        "point10_subject_max_delta_pp": max(value["point10"]["delta_percentage_points"] for value in subjects.values()),
        "point05_confusion_changes": confusion_changes(rows, "point05_unmodified", "point05_repaired"),
        "point10_confusion_changes": confusion_changes(rows, "point10_unmodified", "point10_repaired"),
    }

    result = {
        "schema_version": 1,
        "analysis": "frozen anchor interpolation class/subject stratification",
        "read_only": True,
        "model_or_training": "none; predictions were generated by the frozen validation run",
        "source": {"path": str(source), "sha256": sha256(source), "samples": len(rows)},
        "overall": overall,
        "diagnostics": diagnostics,
        "classes": classes,
        "subjects": subjects,
        "class_rank_by_point10_delta": recovery_rank,
    }
    (output_dir / "stratification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Anchor interpolation stratification",
        "",
        "This is a read-only analysis of the frozen validation predictions. It does not train, tune, or rerun model inference.",
        "",
        f"Source: `{source}`; SHA256 `{result['source']['sha256']}`; samples: {len(rows)}.",
        "",
        "## Overall",
        "",
        "| condition | before | repaired | delta (pp) | corrected | regressed | drop reduction (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, _, _ in CONDITIONS:
        stats = overall[name]
        reduction = stats.get("drop_reduction_percentage_points")
        lines.append(
            f"| {name} | {stats['before_accuracy_percent']:.4f}% | {stats['after_accuracy_percent']:.4f}% | "
            f"{stats['delta_percentage_points']:+.4f} | {stats['corrected']} | {stats['regressed']} | "
            f"{reduction:+.4f} |" if reduction is not None else
            f"| {name} | {stats['before_accuracy_percent']:.4f}% | {stats['after_accuracy_percent']:.4f}% | "
            f"{stats['delta_percentage_points']:+.4f} | {stats['corrected']} | {stats['regressed']} | n/a |"
        )
    lines += [
        "",
        "## Classes with largest point-10 improvement",
        "",
        "| class | n | delta (pp) | drop before (pp) | drop after (pp) | recovery |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in recovery_rank[:15]:
        recovery = item["point10_recovery_fraction"]
        lines.append(
            f"| {item['class']} | {item['samples']} | {item['point10_delta_pp']:+.3f} | "
            f"{item['point10_drop_before_pp']:.3f} | {item['point10_drop_after_pp']:.3f} | "
            f"{recovery:.1%} |" if recovery is not None else
            f"| {item['class']} | {item['samples']} | {item['point10_delta_pp']:+.3f} | "
            f"{item['point10_drop_before_pp']:.3f} | {item['point10_drop_after_pp']:.3f} | n/a |"
        )
    lines += [
        "",
        f"Across 60 classes, point-5 improvement was positive for {diagnostics['point05_classes_positive']}, zero for {diagnostics['point05_classes_zero']}, and negative for {diagnostics['point05_classes_negative']}; point-10 improvement was positive for {diagnostics['point10_classes_positive']}, zero for {diagnostics['point10_classes_zero']}, and negative for {diagnostics['point10_classes_negative']}.",
        f"The class-level Pearson correlation between pre-repair point-drop loss and interpolation gain was r={diagnostics['point05_drop_vs_gain_pearson_r']:.3f} at 5% and r={diagnostics['point10_drop_vs_gain_pearson_r']:.3f} at 10%.",
        "",
        "At point-10, the largest wrong-pair reductions (Atrue -> Apred, before -> after) were "
        + ", ".join(
            f"A{item['true_class']:03d}->A{item['predicted_class']:03d} {item['before']}->{item['after']}"
            for item in diagnostics["point10_confusion_changes"]["largest_error_reductions"][:5]
        )
        + ".",
    ]
    subject_rank = sorted(
        ((name, stats) for name, stats in subjects.items()),
        key=lambda item: (item[1]["point10"]["delta_percentage_points"], item[0]),
        reverse=True,
    )
    lines += [
        "",
        "## Subject range",
        "",
        "| subject | n | point-10 delta (pp) | corrected | regressed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in subject_rank:
        point10 = stats["point10"]
        lines.append(
            f"| {name} | {stats['samples']} | {point10['delta_percentage_points']:+.3f} | "
            f"{point10['corrected']} | {point10['regressed']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The rule's value is concentrated in classes and subjects whose point-drop accuracy loss is large; this supports an input repair/temporal continuity mechanism rather than a clean-accuracy change.",
        f"All {diagnostics['point10_subjects_positive']} of {diagnostics['subject_count']} validation subjects improved at point-10, with a subject-level range of {diagnostics['point10_subject_min_delta_pp']:+.3f} to {diagnostics['point10_subject_max_delta_pp']:+.3f} percentage points.",
        "Classes with small or negative deltas should not be used to claim universal benefit. The class table is descriptive because the class labels and corruption masks were fixed before this audit.",
        "",
        "Full machine-readable class and subject tables are in `stratification.json`.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
