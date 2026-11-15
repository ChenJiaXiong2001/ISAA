"""Independently check the fixed calibration ensemble using float64 NumPy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def normalized_log_prob(x):
    x = x.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def scores(cache_path, model_path):
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    state = torch.load(model_path, map_location="cpu", weights_only=False)["model"]
    h = state["routing_hand_face_correct"].numpy().astype(np.float64)
    b = state["routing_body_face_correct"].numpy().astype(np.float64)
    w = np.divide(h, h + b, out=np.full_like(h, 0.5), where=(h + b) != 0)
    hp, bp = normalized_log_prob(cache["hf"].numpy()), normalized_log_prob(cache["bf"].numpy())
    arithmetic = w * np.exp(hp) + (1 - w) * np.exp(bp)
    arithmetic /= arithmetic.sum(axis=1, keepdims=True)
    geometric = w * hp + (1 - w) * bp
    geometric = np.exp(normalized_log_prob(geometric))
    return cache["labels"].numpy(), arithmetic, (arithmetic + geometric) * 0.5


def main():
    p = argparse.ArgumentParser()
    for name in ("original-cache", "original-checkpoint", "raw-cache", "raw-checkpoint", "summary"):
        p.add_argument("--" + name, type=Path, required=True)
    o = p.parse_args()
    torch.set_num_threads(1)
    labels, _, ref = scores(o.original_cache, o.original_checkpoint)
    raw_labels, raw, _ = scores(o.raw_cache, o.raw_checkpoint)
    if not np.array_equal(labels, raw_labels):
        raise ValueError("Label mismatch")
    cand = .75 * ref + .25 * raw
    a, b = ref.argmax(axis=1) == labels, cand.argmax(axis=1) == labels
    values = {"samples": len(labels), "reference_correct": int(a.sum()),
              "candidate_correct": int(b.sum()), "corrected": int((~a & b).sum()),
              "regressed": int((a & ~b).sum())}
    expected = json.loads(o.summary.read_text(encoding="utf-8"))["overall"]
    if any(values[k] != expected[k] for k in values):
        raise AssertionError((values, expected))
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
