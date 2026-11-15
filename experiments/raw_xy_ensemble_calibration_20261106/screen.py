"""CPU-only fixed ensemble on paired calibration caches and frozen weights."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(cache_path, checkpoint_path):
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if cache["checkpoint_sha256"] != sha(checkpoint_path):
        raise ValueError("Cache and checkpoint digest disagree")
    state = checkpoint["model"]
    h = state["routing_hand_face_correct"].float()
    b = state["routing_body_face_correct"].float()
    n = state["routing_class_counts"].float()
    if not bool(((h >= 0) & (b >= 0) & (h <= n) & (b <= n)).all()):
        raise ValueError("Invalid checkpoint calibration counts")
    w = torch.where(h + b > 0, h / (h + b), 0.5)
    if len(cache["labels"]) != int(n.sum()):
        raise ValueError("Checkpoint and cache calibration sizes differ")
    if float(state["hand_face_temperature"]) != 1 or float(state["body_face_temperature"]) != 1:
        raise ValueError("Expected unit temperatures")
    hp, bp = F.log_softmax(cache["hf"].float(), -1), F.log_softmax(cache["bf"].float(), -1)
    arithmetic = torch.logaddexp(w.log() + hp, torch.log1p(-w) + bp)
    arithmetic -= torch.logsumexp(arithmetic, -1, keepdim=True)
    geometric = w * hp + (1 - w) * bp
    geometric -= torch.logsumexp(geometric, -1, keepdim=True)
    return cache, checkpoint, arithmetic.exp(), 0.5 * (arithmetic.exp() + geometric.exp())


def summarize(ref, alt, labels):
    base, new = ref.argmax(-1), alt.argmax(-1)
    r, c = base == labels, new == labels
    return {"samples": len(labels), "reference_correct": int(r.sum()),
            "candidate_correct": int(c.sum()), "reference_percent": float(100 * r.float().mean()),
            "candidate_percent": float(100 * c.float().mean()),
            "delta_pp": float(100 * (c.float() - r.float()).mean()),
            "corrected": int((~r & c).sum()), "regressed": int((r & ~c).sum()),
            "changed_predictions": int((base != new).sum())}


def main():
    p = argparse.ArgumentParser()
    for name in ("original-cache", "original-checkpoint", "raw-cache", "raw-checkpoint", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    o = p.parse_args()
    if o.output.exists():
        raise FileExistsError(o.output)
    torch.set_num_threads(1)
    if sha(o.original_checkpoint) != "aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7":
        raise ValueError("Wrong original checkpoint")
    a, ac, _, reference = load(o.original_cache, o.original_checkpoint)
    b, bc, raw_arithmetic, _ = load(o.raw_cache, o.raw_checkpoint)
    if ac["args"]["model_variant"] != "best-face-pair" or bc["args"]["model_variant"] != "best-face-pair-raw-xy":
        raise ValueError("Wrong model variant")
    if a["partition"] != "calibration" or b["partition"] != "calibration":
        raise ValueError("Only calibration caches are allowed")
    if a["member_sha256"] != b["member_sha256"] or not torch.equal(a["labels"], b["labels"]):
        raise ValueError("Caches are not paired")
    manifest = json.loads(o.manifest.read_text(encoding="utf-8"))
    members = manifest["calibration"]["members"]
    if hashlib.sha256("\n".join(members).encode()).hexdigest() != a["member_sha256"]:
        raise ValueError("Manifest and cache differ")
    subjects = np.array([int(re.search(r"P(\d{3})", m).group(1)) for m in members])
    labels = a["labels"]
    candidate = 0.75 * reference + 0.25 * raw_arithmetic
    report = {"checkpoint_original_sha256": sha(o.original_checkpoint),
              "checkpoint_raw_sha256": sha(o.raw_checkpoint),
              "checkpoint_epochs": [ac["epoch"], bc["epoch"]],
              "cache_original_sha256": sha(o.original_cache), "cache_raw_sha256": sha(o.raw_cache),
              "manifest_sha256": sha(o.manifest), "ordered_members_sha256": a["member_sha256"],
              "baseline": "original frozen gamma=1 50/50 arithmetic/geometric",
              "candidate": "75% baseline plus 25% raw-XY frozen gamma=1 arithmetic",
              "overall": summarize(reference, candidate, labels),
              "raw_xy_alone": summarize(reference, raw_arithmetic, labels), "subjects": {}}
    for person in np.unique(subjects):
        idx = torch.from_numpy(subjects == person)
        report["subjects"][f"P{person:03d}"] = summarize(reference[idx], candidate[idx], labels[idx])
    o.output.parent.mkdir(parents=True, exist_ok=True)
    with o.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
