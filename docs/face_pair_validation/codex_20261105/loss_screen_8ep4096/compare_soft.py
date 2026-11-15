"""Pair fixed gamma=1 predictions from two frozen checkpoint evaluations."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["sample"]: row for row in csv.DictReader(stream)}


def main():
    summaries = [load_json(ROOT / f"face_pair_loss_soft_eval{v}_8ep4096_20261105" / "summary.json")
                 for v in (15, 10)]
    manifests = [load_json(ROOT / f / "split_manifest.json")
                 for f in ("fusion_h050_f150", "control_h050_f100")]
    members = manifests[0]["validation"]["members"]
    assert members == manifests[1]["validation"]["members"]
    assert len(members) == len(set(members)) == 4096
    assert summaries[0]["validation_members_sha256"] == summaries[1]["validation_members_sha256"]
    assert manifests[0]["calibration"] == manifests[1]["calibration"]
    errors = [load_rows(ROOT / f"face_pair_loss_soft_eval{v}_8ep4096_20261105" / "errors.csv")
              for v in (15, 10)]
    for records in errors:
        assert set(records).issubset(set(members))
    rows = []
    differences = []
    for member in members:
        # The evaluator omits a sample only when HF, BF and soft are all correct.
        correct = [records[member]["soft_correct"] == "True" if member in records else True
                   for records in errors]
        differences.append(int(correct[0]) - int(correct[1]))
        rows.append({"sample": member, "fusion15_soft_correct": int(correct[0]),
                     "fusion10_soft_correct": int(correct[1]), "difference": differences[-1]})
    diff = np.array(differences, dtype=np.int8)
    assert sum(row["fusion15_soft_correct"] for row in rows) == summaries[0]["methods"]["soft"]["correct"]
    assert sum(row["fusion10_soft_correct"] for row in rows) == summaries[1]["methods"]["soft"]["correct"]
    rng = np.random.default_rng(20261105)
    # Bootstrap the empirical {-1, 0, +1} distribution for paired samples.
    counts = np.array([(diff == -1).sum(), (diff == 0).sum(), (diff == 1).sum()])
    draws = rng.multinomial(len(diff), counts / len(diff), size=20000)
    bootstrap = (draws[:, 2] - draws[:, 0]) * 100.0 / len(diff)
    weights15 = np.array(summaries[0]["soft_routing"]["hf_weights"])
    weights10 = np.array(summaries[1]["soft_routing"]["hf_weights"])
    result = {"validation_members_sha256": summaries[0]["validation_members_sha256"],
              "same_ordered_validation_and_calibration_members": True,
              "samples": len(diff), "gain": int(counts[2]), "loss": int(counts[0]),
              "unchanged_correctness": int(counts[1]), "net_gain": int(diff.sum()),
              "difference_pp": float(diff.mean() * 100), "bootstrap_replicates": 20000,
              "bootstrap_seed": 20261105,
              "bootstrap_95_percent_ci_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
              "soft_correct_fusion15": summaries[0]["methods"]["soft"]["correct"],
              "soft_correct_fusion10": summaries[1]["methods"]["soft"]["correct"],
              "different_class_hf_weights": int(np.count_nonzero(np.abs(weights15 - weights10) > 1e-6)),
              "hf_weight_mean_abs_difference": float(np.abs(weights15 - weights10).mean()),
              "hf_weight_max_abs_difference": float(np.abs(weights15 - weights10).max()),
              "weights_note": "Each checkpoint uses its own frozen training-subject calibration counts; same formula and calibration members, not identical weights."}
    with (ROOT / "soft_paired_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "soft_paired_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
