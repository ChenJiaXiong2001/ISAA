"""Capture live training evidence with Python's standard library only."""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_info(path):
    if not path.is_file():
        return {"exists": False}
    info = path.stat()
    return {"exists": True, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "modified_at_utc": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat()}


def read_json(path, errors):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def json_safe(value):
    """Keep failure captures serializable even when a recorded value is NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


class NumericEvidence:
    def __init__(self):
        self.fields = {}

    def inspect(self, row, location, errors, *, csv_values=False):
        for name, original in row.items():
            if not (name == "loss" or name.endswith("_loss") or name == "gradient_norm"
                    or name.endswith("_gradient_norm")):
                continue
            field = self.fields.setdefault(name, {"recorded": 0, "finite": 0, "invalid": 0,
                                                   "minimum": None, "maximum": None, "latest": None})
            field["recorded"] += 1
            value = original
            if csv_values:
                try:
                    value = float(original)
                except (TypeError, ValueError):
                    value = None
            if not finite_number(value):
                field["invalid"] += 1
                if len(errors) < 30:
                    errors.append(f"Nonfinite/non-numeric {location}.{name}: {original!r}")
                continue
            field["finite"] += 1
            field["minimum"] = value if field["minimum"] is None else min(field["minimum"], value)
            field["maximum"] = value if field["maximum"] is None else max(field["maximum"], value)
            field["latest"] = value

    def report(self):
        count = sum(field["recorded"] for field in self.fields.values())
        invalid = sum(field["invalid"] for field in self.fields.values())
        return {"status": "failed" if invalid else "passed" if count else "pending",
                "all_recorded_values_finite": not invalid if count else None,
                "recorded_values": count, "invalid_values": invalid, "fields": self.fields}


def capture_batches(path, errors):
    evidence = NumericEvidence()
    train_evidence = NumericEvidence()
    recent = deque(maxlen=40)
    latest = latest_train = None
    rows = train_rows = 0
    previous_updates = None
    malformed = incomplete_tail = 0
    digest = hashlib.sha256()
    captured_bytes = 0
    if path.is_file():
        with path.open("rb") as stream:
            # Stop at the initial size so a growing log remains quick to read.
            stream.seek(0, 2)
            cutoff = stream.tell()
            stream.seek(0)
            line_number = 0
            while stream.tell() < cutoff:
                line = stream.readline(cutoff - stream.tell())
                if not line:
                    break
                captured_bytes += len(line)
                digest.update(line)
                line_number += 1
                if not line.endswith(b"\n"):
                    incomplete_tail = 1
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected JSON object")
                except (ValueError, UnicodeDecodeError) as exc:
                    malformed += 1
                    if len(errors) < 30:
                        errors.append(f"batches.jsonl line {line_number}: {exc}")
                    continue
                rows += 1
                latest = row
                evidence.inspect(row, f"batch[{line_number}]", errors)
                if row.get("phase") == "train":
                    train_rows += 1
                    latest_train = row
                    train_evidence.inspect(row, f"train_batch[{line_number}]", [])
                    updates = row.get("completed_updates")
                    if updates is not None:
                        valid = isinstance(updates, int) and not isinstance(updates, bool) and updates > 0
                        if not valid or (previous_updates is not None and updates <= previous_updates):
                            if len(errors) < 30:
                                errors.append(f"Invalid/non-increasing completed_updates at batch line {line_number}")
                        if valid:
                            previous_updates = updates
                    if finite_number(row.get("loss")):
                        recent.append({key: row.get(key) for key in
                                       ("epoch", "step", "completed_updates", "loss", "gradient_norm")})
    values = [row["loss"] for row in recent]
    earlier, later = values[:-20], values[-20:]
    basis = "previous_20_vs_latest_20_train_batches"
    if not earlier and len(values) >= 2:
        middle = len(values) // 2
        earlier, later = values[:middle], values[middle:]
        basis = "earlier_vs_later_halves_of_available_recent_train_batches"
    old_mean = sum(earlier) / len(earlier) if earlier else None
    new_mean = sum(later) / len(later) if later else None
    delta = new_mean - old_mean if old_mean is not None and new_mean is not None else None
    trend = {"basis": basis, "earlier_count": len(earlier), "recent_count": len(later),
             "earlier_mean": old_mean, "recent_mean": new_mean, "mean_change": delta,
             "direction": "pending" if delta is None else "decreased" if delta < 0 else "increased" if delta > 0 else "unchanged",
             "recent_train_batches": list(recent)}
    return {"records": rows, "train_records": train_rows, "latest_actual_batch": latest,
            "latest_train_batch": latest_train, "latest_recorded_train_updates": previous_updates,
            "numeric_validation": evidence.report(), "train_loss_recent_change": trend,
            "required_train_fields_pending": [name for name in ("loss", "gradient_norm")
                                              if not train_evidence.fields.get(name, {}).get("recorded")],
            "malformed_complete_lines": malformed, "incomplete_trailing_line_pending": bool(incomplete_tail),
            "captured_bytes": captured_bytes, "captured_prefix_sha256": digest.hexdigest()}


def capture_epochs(path, errors):
    evidence = NumericEvidence()
    rows = []
    pending_tail = False
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines.pop()
            pending_tail = True
        try:
            for index, row in enumerate(csv.DictReader(io.StringIO("".join(lines))), 1):
                if None in row or any(value is None for value in row.values()):
                    errors.append(f"Malformed epochs.csv row {index}")
                    continue
                evidence.inspect(row, f"epoch_csv[{index}]", errors, csv_values=True)
                rows.append(row)
        except csv.Error as exc:
            errors.append(f"epochs.csv: {exc}")
    return {"records": len(rows), "latest_epoch": rows[-1] if rows else None,
            "numeric_validation": evidence.report(), "incomplete_trailing_row_pending": pending_tail}


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def capture_process(pid):
    result = run_command(["ps", "-p", str(pid), "-o", "pid=,ppid=,stat=,etime=,%cpu=,%mem=,args="])
    columns = result["stdout"].split(None, 6)
    process = dict(zip(("pid", "ppid", "stat", "elapsed", "cpu_percent", "memory_percent", "command"), columns))
    running = result["returncode"] == 0 and len(columns) == 7
    zombie = process.get("stat", "").startswith("Z")
    correct_command = "train_best_face_pair_100" in process.get("command", "")
    return {"requested_pid": pid, "present": running, "not_zombie": running and not zombie,
            "training_command_matches": correct_command,
            "healthy": running and not zombie and correct_command, "process": process, "ps": result}


def capture_gpu():
    fields = ["index", "name", "uuid", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu", "power.draw"]
    result = run_command(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    rows = [dict(zip(fields, row)) for row in csv.reader(result["stdout"].splitlines(), skipinitialspace=True)
            if len(row) == len(fields)]
    return {"status": "passed" if result["returncode"] == 0 and rows else "pending",
            "units": {"memory.used": "MiB", "memory.total": "MiB", "utilization.gpu": "%",
                      "temperature.gpu": "C", "power.draw": "W"}, "rows": rows, "csv_query": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, help="Optional output JSON; always print the capture")
    args = parser.parse_args()
    if args.pid < 1:
        parser.error("--pid must be positive")
    run_dir = args.run_dir.resolve(strict=True)
    errors = []
    started = utc_now()
    config = read_json(run_dir / "config.json", errors)
    status = read_json(run_dir / "status.json", errors)
    batches = capture_batches(run_dir / "batches.jsonl", errors)
    epochs = capture_epochs(run_dir / "epochs.csv", errors)
    process, gpu = capture_process(args.pid), capture_gpu()
    if not process["healthy"]:
        errors.append("Requested process is absent, zombie, or not the face-pair training command")
    if status and status.get("status") in ("failed", "interrupted"):
        errors.append(f"Training status is {status['status']}: {status.get('error')}")
    pending = (not config or not status or not batches["train_records"]
               or bool(batches["required_train_fields_pending"]) or gpu["status"] == "pending")
    report = {"schema_version": 1, "status": "failed" if errors else "pending" if pending else "passed",
              "captured_at_start_utc": started, "captured_at_end_utc": utc_now(), "run_dir": str(run_dir),
              "collector": "standard_library_only_no_torch_import_no_training_changes",
              "config": config, "training_status": status, "batches": batches, "epochs": epochs,
              "checkpoints": {name: file_info(run_dir / name) for name in ("last.pt", "best.pt")},
              "files": {name: file_info(run_dir / name) for name in
                        ("config.json", "status.json", "epochs.csv", "batches.jsonl")},
              "process": process, "gpu": gpu, "errors": errors,
              "interpretation": "Latest logged batch can precede the current in-flight batch; missing epoch checkpoints remain pending."}
    payload = json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        output = args.output.resolve()
        if output.parent == run_dir and output.name in (
                "config.json", "status.json", "epochs.csv", "batches.jsonl", "last.pt", "best.pt"):
            parser.error("--output would overwrite a training artifact")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    print(payload, flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
