"""Explore smoothed classwise HF/BF weights on a calibration-only logit cache."""

import argparse
import hashlib
import importlib.util
import re

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache")
    parser.add_argument("--reference-module", help="Optional path to face_pair_fusion_sweep.py for parity checks")
    args = parser.parse_args()
    reference = None
    if args.reference_module:
        spec = importlib.util.spec_from_file_location("face_pair_fusion_reference", args.reference_module)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    torch.set_num_threads(1)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    if cache.get("partition") != "calibration":
        raise ValueError("Expected training calibration cache")
    members = cache["members"]
    if hashlib.sha256("\n".join(members).encode()).hexdigest() != cache["member_sha256"]:
        raise ValueError("Member hash mismatch")
    subjects = []
    for member in members:
        match = re.search(r"P(\d{3})", member)
        if not match:
            raise ValueError("Cannot determine subject")
        subjects.append(int(match.group(1)))
    subject = torch.tensor(subjects)
    hf, bf, y = cache["hf"].float(), cache["bf"].float(), cache["labels"].long()
    if len(set(subjects)) != 2 or len(y) != len(members) or hf.shape != bf.shape or hf.shape[0] != len(y):
        raise ValueError("Expected aligned logits from two subjects")
    if not torch.isfinite(hf).all() or not torch.isfinite(bf).all():
        raise ValueError("Non-finite logits")
    if len(set(members)) != len(members) or (y < 0).any() or (y >= hf.shape[1]).any():
        raise ValueError("Duplicate members or invalid class labels")
    for member, label in zip(members, y.tolist()):
        action = re.search(r"A(\d{3})", member)
        if not action or int(action.group(1)) != label + 1:
            raise ValueError("Member action and label do not match")
    print("checkpoint_sha256", cache["checkpoint_sha256"])
    print("partition", cache["partition"], "subject_counts", [(s, subjects.count(s)) for s in sorted(set(subjects))])
    hf_pred, bf_pred = hf.argmax(-1), bf.argmax(-1)
    oracle = (hf_pred == y) | (bf_pred == y)
    print("head_disagreements", int((hf_pred != bf_pred).sum()),
          "hf_correct", int((hf_pred == y).sum()),
          "bf_correct", int((bf_pred == y).sum()), "oracle_correct", int(oracle.sum()))
    pooled = {a: [0, 0, 0] for a in (0, 1, 3, 10)}
    for hold in sorted(set(subjects)):
        fit, test = subject != hold, subject == hold
        yf, yt = y[fit], y[test]
        c = hf.shape[1]
        support = torch.bincount(yt, minlength=c)
        print("subject_class_support", hold, "classes", int((support > 0).sum()),
              "min", int(support.min()), "max", int(support.max()),
              "counts", support.tolist())
        h = torch.bincount(yf[hf[fit].argmax(-1) == yf], minlength=c).float()
        b = torch.bincount(yf[bf[fit].argmax(-1) == yf], minlength=c).float()
        fit_support = torch.bincount(yf, minlength=c)
        ph, pb = hf[test].softmax(-1), bf[test].softmax(-1)
        results = []
        baseline = None
        for alpha in pooled:
            total = h + b + 2 * alpha
            weights = torch.where(total > 0, (h + alpha) / total.clamp_min(1), 0.5)
            score = weights * ph + (1 - weights) * pb
            score = score / score.sum(-1, keepdim=True)
            if reference is not None:
                if alpha == 0 and not torch.allclose(weights, reference.class_weights_from_counts(h, b, fit_support, [1.0])[0]):
                    raise RuntimeError("Baseline class weights differ from reference")
                reference_scores = reference.LogitFusion()(hf[test], bf[test], weights[None],
                    torch.ones((1, 2)), torch.zeros(1), torch.zeros(1, dtype=torch.long))[0].exp()
                if not torch.allclose(score, reference_scores, atol=1e-6, rtol=1e-5):
                    raise RuntimeError("Fused probabilities differ from reference")
            correct = score.argmax(-1) == yt
            if baseline is None:
                baseline = correct
            gain = int((correct & ~baseline).sum())
            loss = int((~correct & baseline).sum())
            pooled[alpha][0] += gain
            pooled[alpha][1] += loss
            pooled[alpha][2] += len(yt)
            results.append((alpha, int(correct.sum()), gain, loss,
                            round(100 * (gain - loss) / len(yt), 4),
                            round(float(F.nll_loss(score.log(), yt)), 6)))
        print("heldout", hold, "fit_n", int(fit.sum()), "test_n", int(test.sum()),
              "zero_correct_classes", int(((h + b) == 0).sum()))
        print("alpha,correct,gain,loss,delta_pp,nll", results)
    print("pooled alpha,gain,loss,n,delta_pp", [(a, g, l, n, round(100 * (g-l) / n, 4))
                                                 for a, (g, l, n) in pooled.items()])
    print("exploratory: calibration subjects contributed routing during model training;")
    print("fold separation applies only to post-hoc weights; official validation unused here")
    if reference is not None:
        print("reference weights and fused probabilities: PASS")


if __name__ == "__main__":
    main()
