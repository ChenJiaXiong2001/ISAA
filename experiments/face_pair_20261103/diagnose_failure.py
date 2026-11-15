"""Read-only CPU diagnosis of face-pair checkpoints and recorded batches.

Run as a saved file so isaa.train can re-exec with its bundled CUDA library
environment before torch is imported. No model forward or optimizer step runs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


def clean_json(value):
    """Keep strict JSON even when an input contains nonfinite diagnostics."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code_dir = args.code_dir.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    output = args.output.resolve()
    inputs = {path.resolve() for path in run_dir.iterdir() if path.is_file()}
    if output in inputs:
        raise ValueError("Output must not overwrite an existing run artifact")
    sys.path.insert(0, str(code_dir))
    import isaa.train  # Must precede torch; any re-exec reruns this saved file.
    import torch
    from isaa.models.face_pair_fusion import FacePairFusion

    torch.set_num_threads(min(4, torch.get_num_threads()))
    errors = []

    def read_json(name):
        path = run_dir / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            return None

    def tensor_info(name, value):
        tensor = value.detach().cpu()
        finite = torch.isfinite(tensor)
        nonfinite_count = int((~finite).sum())
        finite_values = tensor[finite].double()
        return {"name": name, "prefix": name.split(".", 1)[0],
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "numel": tensor.numel(), "finite": nonfinite_count == 0,
                "nonfinite_count": nonfinite_count,
                "nan_count": int(torch.isnan(tensor).sum()) if tensor.is_floating_point() else 0,
                "positive_inf_count": int(torch.isposinf(tensor).sum()) if tensor.is_floating_point() else 0,
                "negative_inf_count": int(torch.isneginf(tensor).sum()) if tensor.is_floating_point() else 0,
                "max_abs": float(finite_values.abs().max()) if finite_values.numel() else None,
                "min": float(finite_values.min()) if finite_values.numel() else None,
                "max": float(finite_values.max()) if finite_values.numel() else None,
                "mean": float(finite_values.mean()) if finite_values.numel() else None}

    def summarize(rows):
        ordered = sorted(rows, key=lambda row: row["max_abs"] if row["max_abs"] is not None else -1,
                         reverse=True)
        prefixes = {}
        for row in rows:
            group = prefixes.setdefault(row["prefix"], {"tensors": 0, "numel": 0,
                    "nonfinite_count": 0, "max_abs": None, "largest_tensor": None,
                    "top15_max_abs": []})
            group["tensors"] += 1
            group["numel"] += row["numel"]
            group["nonfinite_count"] += row["nonfinite_count"]
            if row["max_abs"] is not None and (group["max_abs"] is None or row["max_abs"] > group["max_abs"]):
                group["max_abs"] = row["max_abs"]
                group["largest_tensor"] = row["name"]
        for row in ordered:
            group = prefixes[row["prefix"]]
            if len(group["top15_max_abs"]) < 15:
                group["top15_max_abs"].append(row)
        return {"tensor_count": len(rows), "numel": sum(row["numel"] for row in rows),
                "all_finite": all(row["finite"] for row in rows),
                "nonfinite_tensors": [row for row in rows if not row["finite"]],
                "by_module_prefix": prefixes, "top15_max_abs": ordered[:15]}

    checkpoint_path = run_dir / "last.pt"
    checkpoint_report = {"path": str(checkpoint_path), "load_device": "cpu",
                         "scope": "durable last completed epoch; failed live epoch state is not saved here"}
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_state = checkpoint["model"]
        num_classes = int(checkpoint.get("args", {}).get("num_classes", 60))
        model = FacePairFusion(num_classes=num_classes)
        parameter_names = [name for name, _ in model.named_parameters()]
        parameter_set = set(parameter_names)
        model_rows, parameter_rows, bn_rows, buffer_rows = [], [], [], []
        for name, value in model_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            row = tensor_info(name, value)
            model_rows.append(row)
            if name in parameter_set:
                parameter_rows.append(row)
            elif name.endswith(("running_mean", "running_var", "num_batches_tracked")):
                bn_rows.append(row)
            else:
                buffer_rows.append(row)
        optimizer = checkpoint.get("optimizer", {})
        saved_ids = [item for group in optimizer.get("param_groups", []) for item in group.get("params", [])]
        id_to_name = dict(zip(saved_ids, parameter_names))
        exact_coverage = len(saved_ids) == len(parameter_names) == len(set(saved_ids))
        if not exact_coverage:
            errors.append("Optimizer parameter ids do not match current model parameter count; name mapping may be incomplete")
        optimizer_rows, momentum_rows, nonfinite_scalars = [], [], []

        def walk_optimizer(value, location, parameter_name=None):
            if isinstance(value, torch.Tensor):
                name = f"{parameter_name}.{location}" if parameter_name else location
                row = tensor_info(name, value)
                optimizer_rows.append(row)
                if location.endswith("momentum_buffer"):
                    momentum_rows.append(row)
            elif isinstance(value, dict):
                for key, item in value.items():
                    walk_optimizer(item, f"{location}.{key}" if location else str(key), parameter_name)
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    walk_optimizer(item, f"{location}[{index}]", parameter_name)
            elif isinstance(value, float) and not math.isfinite(value):
                nonfinite_scalars.append({"location": location, "value": str(value)})

        for saved_id, state in optimizer.get("state", {}).items():
            walk_optimizer(state, "optimizer", id_to_name.get(saved_id, f"unknown_parameter_{saved_id}"))
        walk_optimizer(optimizer.get("param_groups", []), "param_groups")
        checkpoint_report.update(epoch=checkpoint.get("epoch"), architecture=checkpoint.get("architecture"),
            metrics=checkpoint.get("metrics"), lr_schedule=checkpoint.get("lr_schedule"),
            compile=checkpoint.get("compile"),
            model=summarize(model_rows), parameters=summarize(parameter_rows),
            batch_norm_statistics=summarize(bn_rows), other_buffers=summarize(buffer_rows),
            optimizer={**summarize(optimizer_rows),
                       "all_finite": not nonfinite_scalars and all(row["finite"] for row in optimizer_rows),
                       "nonfinite_scalars": nonfinite_scalars,
                       "parameter_name_mapping_complete": exact_coverage,
                       "param_groups": [{key: value for key, value in group.items() if key != "params"}
                                        for group in optimizer.get("param_groups", [])]},
            momentum=summarize(momentum_rows))
    except Exception as exc:
        errors.append(f"last.pt diagnosis: {type(exc).__name__}: {exc}")
        checkpoint_report["error"] = f"{type(exc).__name__}: {exc}"

    rows = []
    try:
        with (run_dir / "batches.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    row["record_line"] = line_number
                    rows.append(row)
                except (ValueError, TypeError) as exc:
                    errors.append(f"batches.jsonl:{line_number}: {exc}")
    except OSError as exc:
        errors.append(f"batches.jsonl: {exc}")

    def numeric(row, key):
        value = row.get(key)
        return float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) else None

    def largest(key):
        values = [row for row in rows if numeric(row, key) is not None]
        return sorted(values, key=lambda row: numeric(row, key) if math.isfinite(numeric(row, key)) else math.inf,
                      reverse=True)[:20]

    outliers = [row for row in rows if (numeric(row, "loss") is not None and
                (not math.isfinite(numeric(row, "loss")) or numeric(row, "loss") > 50))
                or (numeric(row, "gradient_norm") is not None and
                    (not math.isfinite(numeric(row, "gradient_norm")) or numeric(row, "gradient_norm") > 100))]
    grouped = {}
    for row in rows:
        key = f"epoch={row.get('epoch')} phase={row.get('phase')}"
        group = grouped.setdefault(key, {"epoch": row.get("epoch"), "phase": row.get("phase"),
            "records": 0, "first_step": row.get("step"), "last_step": None,
            "loss_gt50": 0, "gradient_gt100": 0, "nonfinite_loss": 0,
            "nonfinite_gradient": 0, "loss_max": None, "gradient_max": None,
            "outlier_steps": [], "weighted_loss_sum": 0., "samples": 0})
        group["records"] += 1
        group["last_step"] = row.get("step")
        loss, gradient = numeric(row, "loss"), numeric(row, "gradient_norm")
        if loss is not None:
            group["nonfinite_loss"] += int(not math.isfinite(loss))
            group["loss_gt50"] += int(loss > 50)
            group["loss_max"] = loss if group["loss_max"] is None else max(group["loss_max"], loss)
            samples = int(row.get("samples", 0))
            group["samples"] += samples
            group["weighted_loss_sum"] += loss * samples
        if gradient is not None:
            group["nonfinite_gradient"] += int(not math.isfinite(gradient))
            group["gradient_gt100"] += int(gradient > 100)
            group["gradient_max"] = gradient if group["gradient_max"] is None else max(group["gradient_max"], gradient)
        if ((loss is not None and (not math.isfinite(loss) or loss > 50)) or
                (gradient is not None and (not math.isfinite(gradient) or gradient > 100))):
            group["outlier_steps"].append(row.get("step"))
    for group in grouped.values():
        group["recorded_mean_loss"] = group["weighted_loss_sum"] / group["samples"] if group["samples"] else None
    epochs = []
    try:
        with (run_dir / "epochs.csv").open(encoding="utf-8", newline="") as stream:
            epochs = list(csv.DictReader(stream))
    except OSError as exc:
        errors.append(f"epochs.csv: {exc}")
    status = read_json("status.json")
    config = read_json("config.json")
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "code_dir": str(code_dir), "torch": str(torch.__version__),
        "read_only": True, "load_device": "cpu", "status": status,
        "configured_args": config.get("args") if isinstance(config, dict) else None,
        "checkpoint": checkpoint_report, "epochs": epochs,
        "batches": {"records": len(rows), "first_record": rows[0] if rows else None,
            "last_record": rows[-1] if rows else None, "top20_loss": largest("loss"),
            "top20_gradient_norm": largest("gradient_norm"),
            "outlier_definition": "loss > 50 or gradient_norm > 100 or either is nonfinite",
            "outlier_count": len(outliers), "first_outlier": outliers[0] if outliers else None,
            "last_outlier": outliers[-1] if outliers else None,
            "outlier_records": outliers, "per_epoch_phase": list(grouped.values())},
        "errors": errors,
        "limitation": "Nonfinite failure batch is raised before batch logging; durable last.pt may precede the failure epoch."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_json(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean_json({"report": str(output), "checkpoint_epoch": checkpoint_report.get("epoch"),
        "checkpoint_model_finite": checkpoint_report.get("model", {}).get("all_finite"),
        "checkpoint_optimizer_finite": checkpoint_report.get("optimizer", {}).get("all_finite"),
        "batch_records": len(rows), "outlier_count": len(outliers),
        "first_outlier": outliers[0] if outliers else None, "errors": errors}), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
