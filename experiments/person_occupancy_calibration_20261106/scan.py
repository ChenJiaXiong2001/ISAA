"""Describe person-slot occupancy and fixed model errors on calibration data."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import FacePairFusion


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows):
    count = len(rows)
    return {"samples": count, "correct": sum(row["correct"] for row in rows),
            "accuracy_percent": 100 * sum(row["correct"] for row in rows) / count if count else None}


def main():
    p = argparse.ArgumentParser()
    for key in ("checkpoint", "manifest", "archive", "cache", "output-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    o = p.parse_args()
    if o.output_dir.exists():
        raise FileExistsError(o.output_dir)
    torch.set_num_threads(1)
    if digest(o.checkpoint) != "aacf3902d0157c9c33075c828029c876220290710eeed0e879a4ad94782aeee7":
        raise ValueError("Wrong epoch-83 checkpoint")
    checkpoint = torch.load(o.checkpoint, map_location="cpu", weights_only=False)
    cache = torch.load(o.cache, map_location="cpu", weights_only=True)
    manifest = json.loads(o.manifest.read_text(encoding="utf-8"))["calibration"]
    members = manifest["members"]
    mdigest = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if mdigest != cache["member_sha256"] or mdigest != manifest["sha256"] or cache["checkpoint_sha256"] != digest(o.checkpoint):
        raise ValueError("Checkpoint, manifest or cached member mismatch")
    state = checkpoint["model"]
    h, b = state["routing_hand_face_correct"].float(), state["routing_body_face_correct"].float()
    weight = torch.where(h + b > 0, h / (h + b), 0.5)
    hp, bp = F.log_softmax(cache["hf"].float(), -1), F.log_softmax(cache["bf"].float(), -1)
    arithmetic = torch.logaddexp(weight.log() + hp, torch.log1p(-weight) + bp)
    arithmetic -= torch.logsumexp(arithmetic, -1, keepdim=True)
    geometric = weight * hp + (1 - weight) * bp
    geometric -= torch.logsumexp(geometric, -1, keepdim=True)
    pred = (arithmetic.exp() + geometric.exp()).argmax(-1)
    args = checkpoint["args"]
    data = RTMWZipDataset(o.archive, split="train", split_protocol=args["split"],
        window_size=args["window_size"], num_joints=133, num_classes=args["num_classes"],
        layout="rtmw_133", max_persons=args["max_persons"],
        score_normalization=args.get("score_normalization", "auto"), feature_mode="raw",
        augment=False, temporal_compression="confidence_weighted",
        deterministic_temporal_crop=args.get("deterministic_temporal_crop", "center"))
    index = {name: i for i, name in enumerate(data.members)}
    if len(index) != len(data.members) or not set(members).issubset(index):
        raise ValueError("Dataset members mismatch")
    body = list(FacePairFusion.BODY_INDICES)
    clean_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    rows = []
    for i, member in enumerate(members):
        raw, label, mask = data[index[member]]
        if label.item() != cache["labels"][i].item():
            raise ValueError("Dataset label mismatch")
        clean_hash.update(raw.contiguous().numpy().tobytes())
        mask_hash.update(mask.contiguous().numpy().tobytes())
        occupancy = (raw[2, :, body, 1] > 0).any(-1) & mask
        fraction = occupancy.sum().item() / max(mask.sum().item(), 1)
        band = "zero" if fraction == 0 else "below_half" if fraction < .5 else "half_or_more"
        rows.append({"sample": member, "subject": int(re.search(r"P(\d{3})", member).group(1)),
                     "action": label.item() + 1, "valid_frames": int(mask.sum()),
                     "slot1_visible_frames": int(occupancy.sum()), "slot1_fraction": fraction,
                     "occupancy_group": band, "prediction": int(pred[i]) + 1,
                     "correct": int(pred[i] == label)})
        if (i + 1) % 400 == 0:
            print(f"samples={i + 1}/{len(members)}", flush=True)
    groups = ("zero", "below_half", "half_or_more")
    report = {"samples": len(rows), "members_sha256": mdigest, "checkpoint_sha256": digest(o.checkpoint),
              "cache_sha256": digest(o.cache), "clean_input_sha256": clean_hash.hexdigest(),
              "frame_mask_sha256": mask_hash.hexdigest(), "body_indices": body,
              "overall": summary(rows), "groups": {}, "subjects": {}, "actions": {},
              "limitations": "Calibration only; observational; archive slots are not stable identities; interactions may need both people."}
    for g in groups:
        report["groups"][g] = summary([r for r in rows if r["occupancy_group"] == g])
    for subject in sorted({r["subject"] for r in rows}):
        report["subjects"][f"P{subject:03d}"] = {g: summary([r for r in rows if r["subject"] == subject and r["occupancy_group"] == g]) for g in groups}
    for action in range(1, 61):
        report["actions"][f"A{action:03d}"] = {g: summary([r for r in rows if r["action"] == action and r["occupancy_group"] == g]) for g in groups}
    report["interaction_indicator"] = {f"{a}_{b}": {g: summary([r for r in rows if a <= r["action"] <= b and r["occupancy_group"] == g]) for g in groups} for a, b in ((1, 49), (50, 60))}
    o.output_dir.mkdir(parents=True, exist_ok=False)
    with (o.output_dir / "samples.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (o.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("samples", "members_sha256", "clean_input_sha256", "frame_mask_sha256", "groups", "subjects", "interaction_indicator")}, indent=2))


if __name__ == "__main__":
    main()
