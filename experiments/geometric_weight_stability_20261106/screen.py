"""Fixed alpha=1 weight screen and calibration-only stability audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F

EXPECTED = "aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7"


def weights(hf, bf, y, alpha=0):
    h = torch.bincount(y[hf.argmax(-1) == y], minlength=60).float()
    b = torch.bincount(y[bf.argmax(-1) == y], minlength=60).float()
    return torch.where(h + b > 0, (h + alpha) / (h + b + 2 * alpha), 0.5)


def mixture(hf, bf, w):
    h, b = hf.log_softmax(-1), bf.log_softmax(-1)
    a = w * h.exp() + (1 - w) * b.exp()
    a = a / a.sum(-1, keepdim=True)
    g = (w * h + (1 - w) * b).softmax(-1)
    return (a + g) * 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opt = parser.parse_args()
    if opt.output.exists():
        raise FileExistsError(opt.output)
    torch.set_num_threads(1)
    cache = torch.load(opt.cache, map_location="cpu", weights_only=True)
    members = json.loads(opt.report.read_text())["manifest"]["calibration"]["members"]
    digest = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if cache["partition"] != "calibration" or cache["checkpoint_sha256"] != EXPECTED or cache["member_sha256"] != digest:
        raise ValueError("Unexpected checkpoint, partition or manifest")
    hf, bf, labels = cache["hf"].float(), cache["bf"].float(), cache["labels"].long()
    if hf.shape != (3191, 60) or hf.shape != bf.shape or len(members) != len(set(members)):
        raise ValueError("Unexpected samples")
    parsed = [re.search(r"S(\d{3})C\d{3}P(\d{3})R\d{3}A(\d{3})", m) for m in members]
    if any(not m for m in parsed) or any(int(m.group(3)) != int(y) + 1 for m, y in zip(parsed, labels)):
        raise ValueError("Invalid labels")
    setups = torch.tensor([int(m.group(1)) for m in parsed])
    subjects = torch.tensor([int(m.group(2)) for m in parsed])
    if sorted(subjects.unique().tolist()) != [9, 18]:
        raise ValueError("Unexpected subjects")
    folds, by_subject = [], {}
    for s in (9, 18):
        keep = subjects == s
        by_subject[s] = weights(hf[keep], bf[keep], labels[keep])
        fit, test = ~keep, keep
        base = mixture(hf[test], bf[test], weights(hf[fit], bf[fit], labels[fit]))
        candidate = mixture(hf[test], bf[test], weights(hf[fit], bf[fit], labels[fit], 1))
        y = labels[test]
        before, after = base.argmax(-1).eq(y), candidate.argmax(-1).eq(y)
        folds.append({"heldout_subject": s, "samples": len(y), "baseline_correct": int(before.sum()),
                      "candidate_correct": int(after.sum()), "gain": int((~before & after).sum()),
                      "loss": int((before & ~after).sum()),
                      "baseline_nll": float(F.nll_loss(base.log(), y)),
                      "candidate_nll": float(F.nll_loss(candidate.log(), y))})
    delta = (by_subject[9] - by_subject[18]).abs()
    classes = []
    for c in range(60):
        classes.append({"action": c + 1, "p009_samples": int(((subjects == 9) & (labels == c)).sum()),
                        "p018_samples": int(((subjects == 18) & (labels == c)).sum()),
                        "p009_hf_weight": float(by_subject[9][c]), "p018_hf_weight": float(by_subject[18][c]),
                        "abs_difference": float(delta[c])})
    setup_groups = []
    for s in (9, 18):
        for setup in setups[subjects == s].unique(sorted=True).tolist():
            keep = (subjects == s) & (setups == setup)
            setup_groups.append({"subject": s, "setup": setup, "samples": int(keep.sum()),
                                 "weight_mean_absolute_difference_from_subject":
                                     float((weights(hf[keep], bf[keep], labels[keep]) - by_subject[s]).abs().mean())})
    result = {"checkpoint_sha256": EXPECTED, "cache_sha256": hashlib.sha256(opt.cache.read_bytes()).hexdigest(),
              "member_sha256": digest, "folds": folds, "classes": classes, "setups": setup_groups,
              "subject_weight_abs_diff_mean": float(delta.mean()),
              "subject_weight_abs_diff_median": float(delta.median()),
              "subject_weight_abs_diff_max": float(delta.max()),
              "limitations": "Exploratory 2-subject calibration data; setup is not chronological time; no validation labels."}
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("classes", "setups")}, indent=2))


if __name__ == "__main__":
    main()
