"""Compare frozen canonical and p10 models on paired synthetic missingness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaa.train  # Set CUDA library paths before importing torch.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset, RTMW_133_FINGER_GROUPS, _confidence_weighted_temporal_compress
from isaa.data.skeleton_dataset import _temporal_crop_or_pad_sample_with_mask
from isaa.face_pair_training import compile_model
from isaa.layouts.rtmw_133 import RTMW_133_FACE_GROUPS
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion

EXPECTED = {
    "baseline": "674643b36e5fe77888f315c5b735259e5ba0f4109321c06d0c343d4b0748113f",
    "p10": "d7cd23babf9bdbbf5f1650f5722d004b064bf9b6f1d8675f28aa4207307cbba3",
}
ANCHORS = (5, 6, 11, 12)
CONDITIONS = ("clean", "anchor_pre_05", "anchor_pre_10", "point_05", "point_10", "group_05", "group_10")


def seeded_generator(sample, seed, kind):
    key = sample if kind == "anchor" else f"{kind}:{sample}"
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))


class PairedInputs(Dataset):
    def __init__(self, dataset, members, seed):
        self.dataset, self.members, self.seed = dataset, members, seed
        if list(dataset.members[:len(members)]) != members:
            raise ValueError("Dataset and manifest order differ")

    def __len__(self):
        return len(self.members)

    def compress(self, raw, frame):
        result, mask = _confidence_weighted_temporal_compress(raw, frame, self.dataset.window_size)
        result, mask = _temporal_crop_or_pad_sample_with_mask(
            result, mask, self.dataset.window_size, random_start=False,
            deterministic_crop=self.dataset.deterministic_temporal_crop)
        result[:2] *= result[2:3] > 0
        return result.masked_fill(~mask[None, :, None, None], 0).contiguous(), mask

    def __getitem__(self, index):
        raw, frame = self.dataset._read_raw_sample(index)
        clean, mask = self.compress(raw, frame)
        name = self.members[index]
        anchor = raw[:, :, list(ANCHORS), :]
        draws = torch.rand(anchor[2].shape, generator=seeded_generator(name, self.seed, "anchor"))
        damaged = []
        for p in (0.05, 0.10):
            hidden = (draws < p) & (anchor[2] > 0) & frame[:, None, None]
            value = raw.clone()
            value[:, :, list(ANCHORS), :] = anchor.masked_fill(hidden[None], 0)
            value, actual_mask = self.compress(value, frame)
            if not torch.equal(mask, actual_mask):
                raise RuntimeError("Frame mask changed")
            damaged.append(value)
        point_draws = torch.rand(clean[2, :, :23, :].shape, generator=seeded_generator(name, self.seed, "point"))
        group_draws = torch.rand((len(RTMW_133_FINGER_GROUPS + RTMW_133_FACE_GROUPS), clean.shape[1]), generator=seeded_generator(name, self.seed, "group"))
        for p in (0.05, 0.10):
            value = clean.clone()
            value[:, :, :23, :] *= (point_draws >= p)[None]
            damaged.append(value)
        for p in (0.05, 0.10):
            value = clean.clone()
            for group, group_draw in zip(RTMW_133_FINGER_GROUPS + RTMW_133_FACE_GROUPS, group_draws):
                value[:, :, group, :] *= (group_draw >= p)[None, :, None, None]
            damaged.append(value)
        return torch.stack([clean, *damaged]), int(self.dataset.labels[index]), mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "p10", "archive", "manifest", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    o = parser.parse_args()
    if o.output_dir.exists():
        raise FileExistsError(o.output_dir)
    if o.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if o.max_samples < 0 or o.batch_size < 1 or o.num_workers < 0:
        raise ValueError("Invalid evaluation parameters")
    models, checkpoints = {}, {}
    common_keys = ("model_variant", "split", "window_size", "num_classes", "max_persons", "feature_mode", "temporal_compression", "max_samples", "tf32")
    for name in ("baseline", "p10"):
        path = getattr(o, name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED[name]:
            raise ValueError(f"Wrong {name} checkpoint: {digest}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        if args["model_variant"] != "body-local-hand-ctr-wide-relative" or not args["compile"]:
            raise ValueError("Expected compiled canonical baseline")
        model = BodyLocalHandCTRWideRelativeFusion(num_classes=args["num_classes"])
        if model.ARCHITECTURE != checkpoint["architecture"]:
            raise ValueError("Architecture mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        models[name], checkpoints[name] = model.to(o.device).eval(), checkpoint
    a, b = (checkpoints[name]["args"] for name in ("baseline", "p10"))
    if any(a.get(k) != b.get(k) for k in common_keys):
        raise ValueError("Incompatible model input configuration")
    if (b.get("joint_dropout_probability"), b.get("grouped_joint_dropout_probability")) != (0.1, 0.1):
        raise ValueError("Unexpected p10 training augmentation")
    torch.set_num_threads(1 if o.device == "cpu" else 4)
    torch.backends.cudnn.allow_tf32 = a.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = a.get("tf32", True)
    manifest = json.loads(o.manifest.read_text(encoding="utf-8"))["validation"]
    full_members = manifest["members"]
    if hashlib.sha256("\n".join(full_members).encode()).hexdigest() != manifest["sha256"]:
        raise ValueError("Manifest hash mismatch")
    members = full_members[:o.max_samples] if o.max_samples else full_members
    dataset = RTMWZipDataset(o.archive, split="val", split_protocol=a["split"], window_size=a["window_size"], num_joints=133, num_classes=a["num_classes"], layout="rtmw_133", max_persons=a["max_persons"], feature_mode="raw", augment=False, temporal_compression="confidence_weighted")
    if list(dataset.members) != full_members:
        raise ValueError("Full ordered dataset members differ from manifest")
    paired = PairedInputs(dataset, members, o.seed)
    for i in range(min(4, len(paired))):
        variants, label, mask = paired[i]
        reference, reference_label, reference_mask = dataset[i]
        if label != int(reference_label) or not torch.equal(mask, reference_mask) or not torch.equal(variants[0], reference):
            raise ValueError("Clean reconstruction differs from dataset")
    networks = {name: compile_model(model, "default") if o.device == "cuda" else model for name, model in models.items()}
    loader = DataLoader(paired, batch_size=o.batch_size, shuffle=False, num_workers=o.num_workers)
    hashes = {name: hashlib.sha256() for name in (*CONDITIONS, "frame_mask")}
    rows, start = [], time.perf_counter()
    with torch.inference_mode():
        for step, (inputs, labels, mask) in enumerate(loader, 1):
            for i, condition in enumerate(CONDITIONS):
                hashes[condition].update(inputs[:, i].contiguous().numpy().tobytes())
            hashes["frame_mask"].update(mask.contiguous().numpy().tobytes())
            batch = [{"sample": members[len(rows) + i], "label": int(label)} for i, label in enumerate(labels)]
            mask = mask.to(o.device)
            for i, condition in enumerate(CONDITIONS):
                value = inputs[:, i].to(o.device)
                for name, network in networks.items():
                    logits = network(value, mask)
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite logits")
                    for row, prediction in zip(batch, logits.argmax(-1).cpu().tolist()):
                        row[f"{condition}_{name}"] = prediction
            rows.extend(batch)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(rows)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    subjects = np.array([int(re.search(r"P(\d{3})", row["sample"]).group(1)) for row in rows])
    unique, inverse = np.unique(subjects, return_inverse=True)
    sizes = np.bincount(inverse)
    draws = np.random.default_rng(20261106).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=20000)
    metrics, clean_delta = {}, None
    for condition in CONDITIONS:
        old = np.array([row[f"{condition}_baseline"] == row["label"] for row in rows])
        new = np.array([row[f"{condition}_p10"] == row["label"] for row in rows])
        delta = new.astype(int) - old.astype(int)
        if clean_delta is None:
            clean_delta = delta
        net = np.bincount(inverse, weights=delta)
        relative = np.bincount(inverse, weights=delta-clean_delta)
        metrics[condition] = {"baseline_correct": int(old.sum()), "p10_correct": int(new.sum()), "difference_pp": float(100*delta.mean()), "corrected": int((new & ~old).sum()), "new_errors": int((old & ~new).sum()), "subject_bootstrap_ci95_pp": np.percentile(100*(draws @ net)/(draws @ sizes), [2.5,97.5]).tolist(), "difference_vs_clean_pp": float(100*(delta-clean_delta).mean()), "difference_vs_clean_subject_ci95_pp": np.percentile(100*(draws @ relative)/(draws @ sizes), [2.5,97.5]).tolist()}
    o.output_dir.mkdir(parents=True, exist_ok=False)
    with (o.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"checkpoints_sha256": EXPECTED, "checkpoint_epochs": {name: checkpoints[name]["epoch"] for name in checkpoints}, "samples": len(rows), "subjects": len(unique), "members_sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(), "input_sha256": {name: value.hexdigest() for name,value in hashes.items()}, "seed": o.seed, "metrics": metrics, "device": o.device, "compile": "inductor default, inherited face LocalSTBlock eager boundary" if o.device == "cuda" else "CPU eager smoke only", "mask_scheme": "per-member deterministic; p05 and p10 share random draws; anchor before compression, point and groups after compression", "caveat": "Repeatedly used validation set; synthetic missingness; checkpoint selection and training differences preclude general causal claims", "seconds": time.perf_counter()-start}
    (o.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
