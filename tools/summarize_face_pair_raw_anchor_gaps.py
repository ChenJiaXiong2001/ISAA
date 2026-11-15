"""Summarize fixed max5 anchor-gap repair by subject."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.summarize_precompression_subject_bootstrap import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    with (args.evaluation_dir / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    digest = hashlib.sha256("\n".join(row["sample"] for row in rows).encode()).hexdigest()
    if len(rows) != summary["samples"] or digest != summary["members_sha256"]:
        raise ValueError("Predictions do not match evaluation membership")
    result = {"samples": len(rows), "members_sha256": digest, "input_sha256": summary["input_sha256"],
              "conditions": {}, "caveat": "Fixed synthetic gaps; reused partitions and cluster bootstrap do not establish independent generalization."}
    for condition in ("clean", "gap2", "gap5", "gap10"):
        mapped = [{**row, "pre_dropout_unmodified_prediction": row[f"{condition}_unmodified_prediction"],
                   "pre_dropout_max5_prediction": row[f"{condition}_max5_prediction"]} for row in rows]
        calculation = summarize(mapped, args.repeats, args.seed)
        result["conditions"][condition] = calculation["conditions"]["pre_dropout"]
        result["subjects"] = calculation["subjects"]
        result["repeats"] = calculation["repeats"]
        result["bootstrap_unit"] = calculation["bootstrap_unit"]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
