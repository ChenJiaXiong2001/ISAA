"""Inspect frozen HF/BF score margins for paired prediction disagreements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["validation"]
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != audit["checkpoint_sha256"] or checkpoint_sha != cache["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    members = manifest["members"]
    digest = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if digest != manifest["sha256"] or digest != cache["member_sha256"] or digest != audit["members_sha256"]:
        raise ValueError("Manifest mismatch")
    state = checkpoint["model"]
    counts = state["routing_class_counts"].float()
    hf = state["routing_hand_face_correct"].float()
    bf = state["routing_body_face_correct"].float()
    ha = torch.where(counts > 0, hf / counts.clamp_min(1), 0)
    ba = torch.where(counts > 0, bf / counts.clamp_min(1), 0)
    weight = torch.where(ha + ba > 0, ha / (ha + ba), 0.5)
    indices = {name: index for index, name in enumerate(members)}
    rows = []
    for change in audit["clean_prediction_disagreements"]:
        i = indices[change["sample"]]
        if int(cache["labels"][i]) != change["label"]:
            raise ValueError("Cached label mismatch")
        hand = F.log_softmax(cache["hf"][i].float(), -1)
        body = F.log_softmax(cache["bf"][i].float(), -1)
        score = torch.logaddexp(weight.log() + hand, torch.log1p(-weight) + body)
        probability = torch.softmax(score, -1)
        values, labels = probability.topk(2)
        if int(labels[0]) != change["eager_prediction"]:
            raise ValueError("Cached prediction differs from recorded eager prediction")
        other = change["compiled_prediction"]
        rows.append({**change, "sample_index": i, "batch_start": i - i % 32,
                     "cached_top2_classes": labels.tolist(), "cached_top2_probabilities": values.tolist(),
                     "cached_top2_margin": float(values[0] - values[1]),
                     "cached_compiled_class_probability": float(probability[other]),
                     "cached_winner_minus_compiled_class_probability": float(values[0] - probability[other])})
    result = {"checkpoint_sha256": checkpoint_sha, "members_sha256": digest,
              "cache_sha256": sha256(args.cache), "rows": rows,
              "caveat": "Cached scores come from a prior compiled run; they do not isolate the cause of later run differences."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
