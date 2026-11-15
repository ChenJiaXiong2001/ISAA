"""Screen fixed dynamic-reliability modes using leave-one-subject-out logits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.face_pair_fusion_sweep import (
    CONFIDENCE_MAXPROB, CONFIDENCE_MARGIN, CONFIDENCE_NORMALIZED_ENTROPY,
    CONFIDENCE_UNIFORM, LogitFusion, class_weights_from_counts,
)


MODES = {"uniform": CONFIDENCE_UNIFORM, "maxprob": CONFIDENCE_MAXPROB,
         "margin": CONFIDENCE_MARGIN, "normalized_entropy": CONFIDENCE_NORMALIZED_ENTROPY}
BETA = (0.5, 1.0, 2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261105)
    parser.add_argument("--replicates", type=int, default=20000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists: {args.output_dir}")
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    report = json.loads(args.cache_report.read_text(encoding="utf-8"))
    members = report["manifest"]["calibration"]["members"]
    member_hash = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if member_hash != cache["member_sha256"]:
        raise RuntimeError("Cached logits and calibration manifest differ")
    subject_ids = []
    for member in members:
        match = re.search(r"P(\d{3})", member)
        if not match:
            raise RuntimeError(f"No subject identifier in {member}")
        subject_ids.append(int(match.group(1)))
    subjects = torch.tensor(subject_ids)
    hf, bf, labels = cache["hf"].float(), cache["bf"].float(), cache["labels"].long()
    if hf.shape != bf.shape or hf.shape[0] != len(members) or labels.shape != (len(members),):
        raise RuntimeError("Cache tensor shapes differ")
    unique = sorted(set(subject_ids))
    if len(unique) != 2:
        raise RuntimeError("Expected the two-subject calibration cache")

    # All confidence modes and exponents are fixed in advance; no candidate is
    # selected from either held-out subject's labels.
    strategy_names = ["static_gamma1"] + [f"{mode}_beta_{beta:g}" for mode in MODES if mode != "uniform"
                                           for beta in BETA]
    model = LogitFusion().eval()
    rows, folds = [], []
    rng = np.random.default_rng(args.bootstrap_seed)
    with torch.no_grad():
        for held_out in unique:
            fit, test = subjects != held_out, subjects == held_out
            train_y, train_hf, train_bf = labels[fit], hf[fit], bf[fit]
            total = torch.bincount(train_y, minlength=hf.shape[-1])
            hf_correct = torch.bincount(train_y[train_hf.argmax(-1) == train_y], minlength=hf.shape[-1])
            bf_correct = torch.bincount(train_y[train_bf.argmax(-1) == train_y], minlength=hf.shape[-1])
            weight = class_weights_from_counts(hf_correct, bf_correct, total, [1.0])[0]
            test_y, test_hf, test_bf = labels[test], hf[test], bf[test]
            matrices = [("static_gamma1", CONFIDENCE_UNIFORM, 0.0)]
            matrices.extend((f"{mode}_beta_{beta:g}", mode_id, beta)
                            for mode, mode_id in MODES.items() if mode != "uniform" for beta in BETA)
            confidence = torch.tensor([item[2] for item in matrices], dtype=torch.float32)
            mode_tensor = torch.tensor([item[1] for item in matrices], dtype=torch.long)
            logits = model(test_hf, test_bf, weight.expand(len(matrices), -1),
                           torch.ones((len(matrices), 2)), confidence, mode_tensor)
            if not torch.isfinite(logits).all() or not torch.allclose(
                    logits.exp().sum(-1), torch.ones_like(logits[:, :, 0]), atol=1e-5):
                raise RuntimeError("Invalid candidate probability outputs")
            predictions = logits.argmax(-1)
            test_members = [member for member, keep in zip(members, test.tolist()) if keep]
            fold_result = {"fit_subject": next(s for s in unique if s != held_out),
                           "heldout_subject": held_out, "fit_samples": int(fit.sum()),
                           "test_samples": int(test.sum()), "candidates": {}}
            for index, (name, _, _) in enumerate(matrices):
                correct = predictions[index] == test_y
                baseline = predictions[0] == test_y
                gain = int((~baseline & correct).sum())
                loss = int((baseline & ~correct).sum())
                n = len(test_y)
                draw = rng.multinomial(n, [gain / n, loss / n, (n - gain - loss) / n],
                                       size=args.replicates)
                ci = np.percentile(100 * (draw[:, 0] - draw[:, 1]) / n, [2.5, 97.5]).tolist()
                fold_result["candidates"][name] = {
                    "correct": int(correct.sum()), "top1_percent": 100 * float(correct.float().mean()),
                    "nll": float(F.nll_loss(logits[index], test_y)), "gain": gain, "loss": loss,
                    "delta_pp": 100 * (gain - loss) / n, "sample_bootstrap_95_ci_pp": ci}
                for member, label, base_pred, pred in zip(test_members, test_y.tolist(),
                        predictions[0].tolist(), predictions[index].tolist()):
                    rows.append({"sample": member, "fit_subject": fold_result["fit_subject"],
                                 "heldout_subject": held_out, "label": label,
                                 "candidate": name, "baseline_pred": base_pred, "prediction": pred})
            folds.append(fold_result)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "checkpoint_sha256": cache["checkpoint_sha256"], "cache_epoch": cache["epoch"],
        "cache_partition": cache["partition"], "calibration_members_sha256": member_hash,
        "subjects": unique, "method": "LOSO classwise gamma1 accuracy weights fit only on other subject; fixed dynamic modes/betas; no heldout selection",
        "fixed_betas": list(BETA), "no_official_validation_used": True,
        "bootstrap_seed": args.bootstrap_seed, "bootstrap_replicates": args.replicates,
        "folds": folds}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
