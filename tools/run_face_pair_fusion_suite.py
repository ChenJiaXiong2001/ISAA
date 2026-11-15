"""Compare frozen HF/BF inference strategies; select only on internal calibration."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA-wheel libraries before importing torch.
import numpy as np
import torch

from tools.check_face_pair_checkpoint import check_finite, require
from tools.evaluate_best import NTU60_NAMES
from tools.evaluate_face_pair import write_csv
from tools.face_pair_fusion_sweep import (
    CONFIDENCE_NAMES, LogitFusion, class_weights_from_counts,
    fit_temperature, temperature_nll, validate_strategy_tensors,
)

GAMMAS = (0., .25, .5, 1., 2., 4., 8.)
BETAS = (.5, 1., 2.)
FUSION_BATCH = 512


def write_json(path, value):
    check_finite(value, str(path))
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cache(cache_dir, report, checkpoint, partition):
    meta = report["checkpoints"][checkpoint]
    file = cache_dir / checkpoint / f"{partition}.pt"
    contents = file.read_bytes()
    require(hashlib.sha256(contents).hexdigest() == meta["partitions"][partition]["file_sha256"],
            f"Cache digest differs: {file}")
    cached = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    require(cached["checkpoint_sha256"] == meta["sha256"], "Checkpoint digest differs")
    require(cached["epoch"] == meta["epoch"] and cached["partition"] == partition, "Cache identity differs")
    expected = report["manifest"]["calibration" if partition == "calibration" else "validation"]
    require(cached["member_sha256"] == expected["sha256"], "Cache member digest differs")
    n, k = expected["count"], report["args"]["num_classes"]
    require(cached["hf"].shape == cached["bf"].shape == (n, k), "Cached logits shape differs")
    require(cached["labels"].shape == (n,), "Cached labels shape differs")
    require(cached["labels"].dtype == torch.long, "Cached labels are not int64")
    require(bool(((cached["labels"] >= 0) & (cached["labels"] < k)).all()), "Invalid cached labels")
    check_finite(cached, "cached")
    return cached


def build_candidates(meta, temperatures):
    arrays = meta["routing_arrays"]
    configs, priors = [], []

    def add(name, family, mode, beta, gamma, weight, temperature_name, temperature):
        configs.append({"id": f"{temperature_name}/{name}", "family": family,
                        "temperature": temperature_name, "T_HF": temperature[0], "T_BF": temperature[1],
                        "confidence_mode": CONFIDENCE_NAMES[mode], "confidence_mode_id": mode,
                        "beta": beta, "gamma": gamma})
        priors.append(torch.as_tensor(weight, dtype=torch.float32))

    k = len(arrays["class_counts"])
    weights = class_weights_from_counts(arrays["HF_correct"], arrays["BF_correct"], arrays["class_counts"], GAMMAS)
    for temperature_name, temperature in (("raw", [1., 1.]), ("calibrated", temperatures)):
        for name, weight in (("HF", torch.ones(k)), ("BF", torch.zeros(k))):
            add(name, "single_head", 0, 0., None, weight, temperature_name, temperature)
        for gamma, weight in zip(GAMMAS, weights):
            add(f"static_gamma_{gamma:g}", "static", 0, 0., gamma, weight, temperature_name, temperature)
        add("hard", "hard", 0, 0., None, arrays["hard_HF_weight"], temperature_name, temperature)
        for mode in (1, 2, 3):
            for gamma in (0., 1.):
                weight = weights[GAMMAS.index(gamma)]
                for beta in BETAS:
                    add(f"{CONFIDENCE_NAMES[mode]}_gamma_{gamma:g}_beta_{beta:g}", "dynamic", mode,
                        beta, gamma, weight, temperature_name, temperature)
    require(len({config["id"] for config in configs}) == len(configs) == 56, "Candidate registry differs")
    return configs, torch.stack(priors)


def evaluate(compiled, cached, configs, priors, device):
    n, k = cached["hf"].shape
    s = len(configs)
    temperatures = torch.tensor([[c["T_HF"], c["T_BF"]] for c in configs], device=device)
    betas = torch.tensor([c["beta"] for c in configs], device=device)
    modes = torch.tensor([c["confidence_mode_id"] for c in configs], device=device, dtype=torch.long)
    priors = priors.to(device)
    validate_strategy_tensors(priors, temperatures, betas, modes)
    predictions = np.empty((s, n), dtype=np.int64)
    matrices = np.zeros((s, k, k), dtype=np.int64)
    top5 = np.zeros(s, dtype=np.int64)
    ce = np.zeros(s)
    brier = np.zeros(s)
    ece_count, ece_right, ece_conf = (np.zeros((s, 15)) for _ in range(3))
    max_normalization_error = 0.
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, n, FUSION_BATCH):
            length = min(FUSION_BATCH, n - offset)
            hf = cached["hf"][offset:offset + length].to(device)
            bf = cached["bf"][offset:offset + length].to(device)
            if length < FUSION_BATCH:
                hf = torch.nn.functional.pad(hf, (0, 0, 0, FUSION_BATCH - length))
                bf = torch.nn.functional.pad(bf, (0, 0, 0, FUSION_BATCH - length))
            result = compiled(hf, bf, priors, temperatures, betas, modes)[:, :length]
            require(bool(torch.isfinite(result).all()), "Fusion produced nonfinite log probabilities")
            probability = result.exp()
            deviation = float((probability.sum(-1) - 1).abs().max())
            require(deviation < 1e-5, f"Fusion probability normalization failed: {deviation}")
            max_normalization_error = max(max_normalization_error, deviation)
            labels = cached["labels"][offset:offset + length].to(device)
            index = labels[None, :, None].expand(s, -1, 1)
            true_logp = result.gather(-1, index)[..., 0]
            confidence, predicted = probability.max(-1)
            ce += (-true_logp.double().sum(-1)).cpu().numpy()
            brier += (probability.double().square().sum(-1) - 2 * true_logp.double().exp() + 1).sum(-1).cpu().numpy()
            top5 += (result.topk(min(5, k), dim=-1).indices == labels[None, :, None]).any(-1).sum(-1).cpu().numpy()
            pred = predicted.cpu().numpy()
            conf = confidence.cpu().numpy()
            truth = labels.cpu().numpy()
            predictions[:, offset:offset + length] = pred
            for i in range(s):
                np.add.at(matrices[i], (truth, pred[i]), 1)
                bins = np.minimum((conf[i] * 15).astype(int), 14)
                np.add.at(ece_count[i], bins, 1)
                np.add.at(ece_right[i], bins, pred[i] == truth)
                np.add.at(ece_conf[i], bins, conf[i])
    rows = []
    support = np.bincount(cached["labels"].numpy(), minlength=k)
    for i, config in enumerate(configs):
        matrix = matrices[i]
        require(np.array_equal(matrix.sum(1), support), "Confusion supports differ")
        correct = int(matrix.trace())
        require(correct == int((predictions[i] == cached["labels"].numpy()).sum()), "Prediction counts differ")
        recall = np.divide(matrix.diagonal(), support, out=np.zeros(k), where=support > 0)
        precision = np.divide(matrix.diagonal(), matrix.sum(0), out=np.zeros(k), where=matrix.sum(0) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=precision + recall > 0)
        rows.append({**config, "samples": n, "correct": correct, "top1_percent": 100 * correct / n,
                     "top5_percent": 100 * int(top5[i]) / n, "nll": float(ce[i] / n),
                     "brier": float(brier[i] / n),
                     "ece15_percent": float(100 * np.abs(ece_right[i] - ece_conf[i]).sum() / n),
                     "macro_accuracy_percent": float(100 * recall[support > 0].mean()),
                     "macro_f1_percent": float(100 * f1[support > 0].mean())})
    return rows, predictions, matrices, {"max_normalization_error": max_normalization_error,
                                         "seconds": time.perf_counter() - started}


def choose(rows):
    # Exclude single-head diagnostics from the fusion strategy selection.
    eligible = [row for row in rows if row["family"] != "single_head"]
    return min(eligible, key=lambda row: (-row["correct"], row["nll"], row["id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    cache_dir, output_dir = options.cache_dir.resolve(), options.output_dir.resolve()
    require(not output_dir.exists(), "Output directory already exists")
    require(torch.cuda.is_available(), "This suite requires CUDA; no fallback")
    torch.set_num_threads(4)
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    report = json.loads((cache_dir / "report.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    compiled = torch.compile(LogitFusion().to(device), backend="inductor", fullgraph=True,
                             options={"triton.cudagraphs": False})
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = {"gammas": GAMMAS, "confidence_betas": BETAS, "confidence_modes": ["maxprob", "margin", "normalized_entropy"],
            "dynamic_priors_gamma": [0, 1], "temperature_range": [.1, 10.],
            "selection": "internal calibration correct count descending, NLL ascending, candidate ID ascending",
            "checkpoint_candidates": {n: c["epoch"] for n, c in report["checkpoints"].items()},
            "historical_selection": "best.pt epoch was originally selected using official val hard-routing accuracy; last.pt is final epoch",
            "calibration_provenance": "training-subject holdout previously used for routing; an internal development set",
            "no_backbone_training": True, "official_val_used_for_new_parameter_selection": False}
    write_json(output_dir / "plan.json", plan)
    meta_by_checkpoint, cal_rows, configs_by_checkpoint, weights_by_checkpoint = {}, {}, {}, {}
    calibration_records = []
    # Complete all parameter fitting and selection before loading val cache files.
    for name in ("best", "last"):
        meta = report["checkpoints"][name]
        cached = load_cache(cache_dir, report, name, "calibration")
        fits = {}
        for head in ("hf", "bf"):
            temperature = fit_temperature(cached[head], cached["labels"])
            before = temperature_nll(cached[head], cached["labels"], 1.)
            after = temperature_nll(cached[head], cached["labels"], temperature)
            require(after <= before + 1e-10, "Temperature fitting worsened calibration NLL")
            fits[head] = {"temperature": temperature, "nll_before": before, "nll_after": after,
                          "boundary_hit": abs(temperature - .1) < 1e-8 or abs(temperature - 10.) < 1e-8}
        temperatures = [fits["hf"]["temperature"], fits["bf"]["temperature"]]
        configs, weights = build_candidates(meta, temperatures)
        rows, _, _, evidence = evaluate(compiled, cached, configs, weights, device)
        winner = choose(rows)
        meta_by_checkpoint[name] = {"epoch": meta["epoch"], "checkpoint_sha256": meta["sha256"],
                                     "temperature_fits": fits, "calibration_selection": winner,
                                     "calibration_evidence": evidence}
        configs_by_checkpoint[name], weights_by_checkpoint[name], cal_rows[name] = configs, weights, rows
        calibration_records.extend({"checkpoint": name, "epoch": meta["epoch"], **row} for row in rows)
        print(json.dumps({"phase": "calibration_complete", "checkpoint": name, "temperatures": fits,
                          "selected": winner}, ensure_ascii=False), flush=True)
    selected_record = min((dict(checkpoint=name, **data["calibration_selection"]) for name, data in meta_by_checkpoint.items()),
                          key=lambda row: (-row["correct"], row["nll"], row["checkpoint"], row["id"]))
    write_csv(output_dir / "calibration_results.csv", calibration_records, list(calibration_records[0]))
    selection = {"selected": selected_record, "checkpoints": meta_by_checkpoint,
                 "selection_uses_official_val": False, "cache_report_sha256": digest_file(cache_dir / "report.json"),
                 "plan_sha256": digest_file(output_dir / "plan.json")}
    write_json(output_dir / "selection_before_val.json", selection)
    selection_hash = digest_file(output_dir / "selection_before_val.json")
    print(json.dumps({"phase": "selection_frozen_before_val", "sha256": selection_hash, "selected": selected_record}), flush=True)

    val_records, val_outputs = [], {}
    selected_metrics = None
    for name in ("best", "last"):
        cached = load_cache(cache_dir, report, name, "val")
        configs, weights = configs_by_checkpoint[name], weights_by_checkpoint[name]
        rows, predictions, matrices, evidence = evaluate(compiled, cached, configs, weights, device)
        rows_by_id = {row["id"]: row for row in rows}
        cal_by_id = {row["id"]: row for row in cal_rows[name]}
        winner_id = meta_by_checkpoint[name]["calibration_selection"]["id"]
        checkpoint_dir = output_dir / name
        checkpoint_dir.mkdir()
        ids = [config["id"] for config in configs]
        baseline_i = ids.index("raw/static_gamma_1")
        truth = cached["labels"].numpy()
        baseline_correct = predictions[baseline_i] == truth
        for i, row in enumerate(rows):
            right = predictions[i] == truth
            row.update(calibration_selected=row["id"] == winner_id,
                       calibration_top1_percent=cal_by_id[row["id"]]["top1_percent"],
                       corrected_vs_raw_gamma1=int((right & ~baseline_correct).sum()),
                       harmed_vs_raw_gamma1=int((~right & baseline_correct).sum()))
            val_records.append({"checkpoint": name, "epoch": meta_by_checkpoint[name]["epoch"], **row})
        # Detailed exports for the calibrated choice, plus predictions for all candidates.
        chosen_i = ids.index(winner_id)
        matrix = matrices[chosen_i]
        write_csv(checkpoint_dir / "selected_confusion.csv",
                  [{"true_action": f"A{i+1:03}", **{f"A{j+1:03}": int(matrix[i,j]) for j in range(matrix.shape[1])}}
                   for i in range(matrix.shape[0])], ["true_action", *[f"A{i+1:03}" for i in range(matrix.shape[0])]])
        class_rows = []
        for c in range(matrix.shape[0]):
            support = int(matrix[c].sum())
            class_rows.append({"action_id": f"A{c+1:03}", "name": NTU60_NAMES[c], "samples": support,
                               "correct": int(matrix[c,c]), "top1_percent": 100 * int(matrix[c,c]) / support if support else None,
                               "HF_prior": float(weights[chosen_i,c]), "BF_prior": 1-float(weights[chosen_i,c])})
        write_csv(checkpoint_dir / "selected_per_class.csv", class_rows, list(class_rows[0]))
        np.savez_compressed(checkpoint_dir / "predictions.npz", predictions=predictions, labels=truth,
                            ids=np.asarray(ids, dtype=str), confusion_matrices=matrices)
        selected = rows_by_id[winner_id]
        val_outputs[name] = {**meta_by_checkpoint[name], "calibration_selected_val": selected,
                             "val_evidence": evidence,
                             "highest_observed_val": max(rows, key=lambda row: (row["correct"], -row["nll"])),
                             "candidate_count": len(rows)}
        if selected_record["checkpoint"] == name:
            selected_metrics = selected
        print(json.dumps({"phase": "val_complete", "checkpoint": name,
                          "calibration_selected": selected,
                          "highest_observed_val": val_outputs[name]["highest_observed_val"]}, ensure_ascii=False), flush=True)
    require(digest_file(output_dir / "selection_before_val.json") == selection_hash, "Selection changed after val")
    write_csv(output_dir / "validation_results.csv", val_records, list(val_records[0]))
    summary = {"plan": plan, "checkpoints": val_outputs, "candidate_count_total": len(val_records),
               "calibration_selected": selected_record, "calibration_selected_validation_metrics": selected_metrics,
               "selection_frozen_before_val_sha256": selection_hash,
               "selection_unchanged_after_val": True,
               "highest_observed_val_is_exploratory": True,
               "cache_report_sha256": digest_file(cache_dir / "report.json"),
               "compile": {"backend": "inductor", "fullgraph": True, "required": True,
                           "scope": "all new static and dynamic fusion computations on cached compiled-backbone logits",
                           "suppress_errors": False, "fail_on_recompile_limit_hit": True,
                           "whole_model_eager_fallback": False},
               "runtime": {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(device)}}
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"status": "completed", "candidate_count_total": len(val_records),
                      "calibration_selected": selected_record,
                      "validation_metrics": selected_metrics}), flush=True)


if __name__ == "__main__":
    main()
