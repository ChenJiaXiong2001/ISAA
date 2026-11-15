"""Test one pre-specified confusion correction using calibration-only LOSO."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    section = json.loads(args.manifest.read_text(encoding="utf-8"))["calibration"]
    members = section["members"]
    digest = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if cache["partition"] != "calibration" or cache["member_sha256"] != digest or section["sha256"] != digest:
        raise ValueError("Calibration manifest and cache must match")
    hf, bf, labels = cache["hf"].float(), cache["bf"].float(), cache["labels"].long()
    if hf.shape != bf.shape or hf.ndim != 2 or len(labels) != len(members) or len(hf) != len(labels):
        raise ValueError("Logits, labels and members do not align")
    if not torch.isfinite(hf).all() or not torch.isfinite(bf).all() or len(set(members)) != len(members):
        raise ValueError("Non-finite logits or duplicate members")
    subjects = []
    for member, label in zip(members, labels.tolist()):
        subject = re.search(r"P(\d{3})", member)
        action = re.search(r"A(\d{3})", member)
        if not subject or not action or int(action.group(1)) != label + 1:
            raise ValueError("Invalid member subject or action label")
        subjects.append(int(subject.group(1)))
    if len(set(subjects)) != 2:
        raise ValueError("Expected exactly two calibration subjects")
    subjects = torch.tensor(subjects)
    print("cache_sha256", hashlib.sha256(args.cache.read_bytes()).hexdigest(),
          "checkpoint_sha256", cache["checkpoint_sha256"], "members_sha256", digest)
    print("candidate: p_corrected = 0.75*p + 0.25*M*p; M has 10 identity pseudo-observations per class")
    print("M[true,pred] = sum of calibration p[pred] for samples of true class, column-normalized")
    total_gain = total_loss = total_n = 0
    for heldout in sorted(set(subjects.tolist())):
        fit = subjects != heldout
        test = ~fit
        fit_labels, test_labels = labels[fit], labels[test]
        classes = hf.shape[1]
        h = torch.bincount(fit_labels[hf[fit].argmax(-1) == fit_labels], minlength=classes).float()
        b = torch.bincount(fit_labels[bf[fit].argmax(-1) == fit_labels], minlength=classes).float()
        n = h + b
        w = torch.where(n > 0, h / n.clamp_min(1), 0.5)
        def fused(mask):
            p = w * hf[mask].softmax(-1) + (1 - w) * bf[mask].softmax(-1)
            return p / p.sum(-1, keepdim=True)
        fit_p, test_p = fused(fit), fused(test)
        matrix = torch.eye(classes) * 10.0
        matrix.index_add_(0, fit_labels, fit_p)
        matrix /= matrix.sum(0, keepdim=True)
        candidate = 0.75 * test_p + 0.25 * (test_p @ matrix.T)
        if not torch.allclose(candidate.sum(-1), torch.ones(len(candidate)), atol=1e-6):
            raise RuntimeError("Candidate does not normalize")
        old = test_p.argmax(-1) == test_labels
        new = candidate.argmax(-1) == test_labels
        gain, loss = int((new & ~old).sum()), int((old & ~new).sum())
        total_gain += gain
        total_loss += loss
        total_n += len(test_labels)
        print("heldout", heldout, "fit_n", int(fit.sum()), "test_n", int(test.sum()),
              "baseline", int(old.sum()), "candidate", int(new.sum()), "gain", gain, "loss", loss,
              "delta_pp", round(100 * (gain - loss) / len(test_labels), 4),
              "nll", round(float(F.nll_loss(test_p.log(), test_labels)), 6),
              "candidate_nll", round(float(F.nll_loss(candidate.log(), test_labels)), 6))
    print("pooled gain", total_gain, "loss", total_loss, "n", total_n,
          "delta_pp", round(100 * (total_gain - total_loss) / total_n, 4))
    print("No official validation labels used; training-time routing used both calibration subjects.")


if __name__ == "__main__":
    main()
