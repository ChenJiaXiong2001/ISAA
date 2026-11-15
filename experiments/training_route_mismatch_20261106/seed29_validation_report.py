import hashlib
import json
from pathlib import Path

root = Path("experiments/training_route_mismatch_20261106")
vp = root / "paired_validation_seed29_20261107_first"
cp = root / "paired_calibration_seed29_20261107"
sm = root / "paired_smoke_seed29_20261107_first"
s = json.loads((vp / "summary.json").read_text())
rows = []
for condition in s["conditions"]:
    for weights in ("own_weights", "common_control_weights"):
        x = condition[weights]
        rows.append({
            "condition": condition["condition"],
            "weights": weights,
            "control_top1_percent": x["control_top1_percent"],
            "candidate_top1_percent": x["candidate_top1_percent"],
            "difference_pp": x["difference_pp"],
            "subject_ci95_pp": x["subject_ci95_pp"],
            "corrected": x["corrected"],
            "regressed": x["regressed"],
            "equal_subject_difference_pp": x["equal_subject_difference_pp"],
        })
own = {row["condition"]: row for row in rows if row["weights"] == "own_weights"}
gate_clean = own["clean"]["difference_pp"] >= 0.25
gate_corruption = all(
    own[name]["difference_pp"] >= -0.25 for name in own if name != "clean"
)
artifacts = {}
for directory in (sm, cp, vp):
    artifacts[str(directory)] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir() if path.is_file()
    }
result = {
    "experiment": "soft_route_candidate_seed29_20261107",
    "checkpoint_control": "/home/xiong/Desktop/ISAA/outputs/soft_route_matched_control_20ep4096_seed29_20261107/20261107-002410-557ef497/last.pt",
    "checkpoint_candidate": "/home/xiong/Desktop/ISAA/outputs/soft_route_candidate_20ep4096_seed29_20261107/20261107-003848-597510d0/last.pt",
    "validation_summary": str(vp / "summary.json"),
    "calibration_summary": str(cp / "summary.json"),
    "smoke_summary": str(sm / "summary.json"),
    "rows": rows,
    "preregistered_gate": {
        "clean_at_least_0_25pp": gate_clean,
        "all_corruption_at_least_minus_0_25pp": gate_corruption,
        "passed": gate_clean and gate_corruption,
    },
    "artifacts_sha256": artifacts,
}
(root / "seed29_validation_summary_20261107.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n"
)
markdown = [
    "# Seed-29 soft-route paired validation (2026-11-07)",
    "",
    "Validation used epoch-20 `last.pt` checkpoints with identical input batches, frame masks, split manifest, and gamma=1 fusion reconstruction. The candidate was scratch-trained for 20 epochs with `torch.compile` mandatory/default on CUDA.",
    "",
    "## Own training weights",
    "",
    "| condition | control % | candidate % | delta pp | subject bootstrap CI95 pp | corrected/regressed | equal-subject delta pp |",
    "|---|---:|---:|---:|---|---:|---:|",
]
for condition in s["conditions"]:
    x = condition["own_weights"]
    markdown.append(
        f"| {condition['condition']} | {x['control_top1_percent']:.4f} | {x['candidate_top1_percent']:.4f} | {x['difference_pp']:+.4f} | [{x['subject_ci95_pp'][0]:+.4f}, {x['subject_ci95_pp'][1]:+.4f}] | {x['corrected']}/{x['regressed']} | {x['equal_subject_difference_pp']:+.4f} |"
    )
markdown += [
    "",
    "## Fixed-control weight sensitivity",
    "",
    "| condition | delta pp | subject bootstrap CI95 pp | corrected/regressed |",
    "|---|---:|---|---:|",
]
for condition in s["conditions"]:
    x = condition["common_control_weights"]
    markdown.append(
        f"| {condition['condition']} | {x['difference_pp']:+.4f} | [{x['subject_ci95_pp'][0]:+.4f}, {x['subject_ci95_pp'][1]:+.4f}] | {x['corrected']}/{x['regressed']} |"
    )
markdown += [
    "",
    "## Pre-registered gate",
    "",
    f"- clean own-weight gate (>= +0.25 pp): **{'PASS' if gate_clean else 'FAIL'}** ({own['clean']['difference_pp']:+.4f} pp)",
    f"- all listed corruption own-weight gate (>= -0.25 pp): **{'PASS' if gate_corruption else 'FAIL'}** (group 5% is {own['group_0.05']['difference_pp']:+.4f} pp)",
    f"- overall candidate decision: **{'PASS' if gate_clean and gate_corruption else 'FAIL; do not promote'}**",
    "",
    "Fixed-control rows are transfer-weight sensitivity only, not independent efficacy evidence.",
]
(root / "seed29_validation_summary_20261107.md").write_text("\n".join(markdown) + "\n")
