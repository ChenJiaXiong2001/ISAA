"""Durable run records and the original CTR-GCN epoch learning-rate schedule."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def ctrgcn_learning_rate(epoch_index, base_lr, warmup_epochs, milestones, decay):
    """Use the official zero-based epoch convention, including warmup."""
    if epoch_index < 0:
        raise ValueError("epoch_index must be nonnegative")
    if epoch_index < warmup_epochs:
        return base_lr * (epoch_index + 1) / warmup_epochs
    return base_lr * decay ** sum(epoch_index >= step for step in milestones)


def create_run_directory(base: Path) -> Path:
    path = base / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class RunRecords:
    """Flush completed batches immediately; atomically update run status."""

    def __init__(self, directory: Path, args: dict, root: Path):
        self.directory = directory
        self.started = time.perf_counter()
        self.state = {"status": "running", "last_completed_epoch": 0, "best_epoch": None,
                      "best_val_acc": None, "last_batch": None}
        self.config = {"args": args, "command": sys.argv, "python": platform.python_version(),
                       "platform": platform.platform(), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "accuracy_units": "fraction", "timing": "host wall time, includes metric recording",
                       "reference": {
                           "repository": "https://github.com/Uason-Chen/CTR-GCN",
                           "commit": "67d8710578b842a5d6384cd8293d627f03c6ddc1",
                           "config": ("config/nturgbd120-cross-set/default.yaml" if args.get("split") == "xset120"
                                      else "config/nturgbd120-cross-subject/default.yaml"),
                           "schedule": "main.py:adjust_learning_rate (zero-based milestones)",
                           "adaptation": "RTMW relative xy/score, masks, 32 joints; existing crop/pad preprocessing"}}
        self.update_config()
        # Archive the actual source used by this run, excluding data and weights.
        with ZipFile(directory / "source.zip", "w", ZIP_DEFLATED) as archive:
            paths = [root / "main.py", root / "requirements.txt", *sorted((root / "isaa").rglob("*.py"))]
            for path in paths:
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        self.batch_file = (directory / "batches.jsonl").open("w", encoding="utf-8", buffering=1)
        self.epoch_file = (directory / "epochs.csv").open("w", encoding="utf-8", newline="", buffering=1)
        self.epoch_writer = None
        self.finish("running")

    def update_config(self, **values):
        self.config.update(values)
        write_json(self.directory / "config.json", self.config)

    def batch(self, row):
        row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **row}
        self.batch_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.state["last_batch"] = {key: row[key] for key in ("epoch", "phase", "step")}

    def epoch(self, row):
        if self.epoch_writer is None:
            self.epoch_writer = csv.DictWriter(self.epoch_file, fieldnames=list(row))
            self.epoch_writer.writeheader()
        self.epoch_writer.writerow(row)
        self.epoch_file.flush()
        self.state.update(last_completed_epoch=row["epoch"], best_epoch=row["best_epoch"],
                          best_val_acc=row["best_val_acc"])
        self.finish("running")

    def finish(self, status, error=None):
        self.state.update(status=status, elapsed_seconds=time.perf_counter() - self.started,
                          updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if error is not None:
            self.state["error"] = error
        write_json(self.directory / "status.json", self.state)

    def close(self):
        self.batch_file.close()
        self.epoch_file.close()
