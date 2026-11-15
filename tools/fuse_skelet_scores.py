"""Fuse four SkeleT prediction streams by sample identity (j,b,jm,bm)."""
import argparse
import json
from pathlib import Path

import numpy as np


def fuse(paths, weights=(3, 3, 2, 2)):
    if len(paths) != 4 or len(weights) != 4 or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Require j,b,jm,bm files and four nonnegative weights with positive sum")
    names = labels = total = None
    for path, weight in zip(paths, weights):
        with np.load(path, allow_pickle=False) as data:
            current_names = data["sample_names"].astype(str)
            current_labels = data["labels"]
            probabilities = data["probabilities"]
        if len(set(current_names.tolist())) != len(current_names):
            raise ValueError(f"Duplicate sample identity in {path}")
        if probabilities.ndim != 2 or probabilities.shape[0] != len(current_names) or current_labels.shape != (len(current_names),):
            raise ValueError(f"Invalid prediction shapes in {path}")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError(f"Invalid probabilities in {path}")
        np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-5)
        if names is None:
            names, labels = current_names, current_labels
            total = np.zeros_like(probabilities, dtype=np.float64)
        elif set(names.tolist()) != set(current_names.tolist()):
            raise ValueError(f"Sample identities differ in {path}")
        index = {name: i for i, name in enumerate(current_names)}
        order = np.asarray([index[name] for name in names])
        if probabilities.shape != total.shape or not np.array_equal(current_labels[order], labels):
            raise ValueError(f"Class count or sample labels differ in {path}")
        total += weight * probabilities[order]
    total /= sum(weights)
    top1 = float(np.mean(total.argmax(-1) == labels))
    top5 = float(np.mean((np.argsort(total, axis=-1)[:, -5:] == labels[:, None]).any(-1)))
    return total, labels, names, dict(top1=top1, top5=top5, samples=len(names), weights=list(weights))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs=4, help="j.npz b.npz jm.npz bm.npz, in this order")
    parser.add_argument("--weights", type=float, nargs=4, default=[3, 3, 2, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores, labels, names, metrics = fuse(args.scores, args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, probabilities=scores, labels=labels, sample_names=names)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))
