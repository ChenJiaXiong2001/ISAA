"""Same-input evaluation of hard-route training versus lagged soft-route training."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "precompression_anchor_paired_p025_20ep_20261106"))
import isaa.train  # Configure CUDA wheel paths before torch import.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.training_route_mismatch_20261106.soft_route import SoftRouteFacePair
from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.face_pair_training import compile_model
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.train import collate_rtmw
from isaa.utils.seed import seed_everything
from tools.check_face_pair_checkpoint import check_args, check_compile
from tools.evaluate_face_pair_corruption import _soft_logits, _soft_weights
from precompression_mask import PairedPrecompression

STANDARD = (("clean", None, 0., 0), ("point_0.05", "joint_dropout_probability", .05, 1),
            ("point_0.1", "joint_dropout_probability", .1, 2),
            ("group_0.05", "grouped_joint_dropout_probability", .05, 101),
            ("group_0.1", "grouped_joint_dropout_probability", .1, 102))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def members_digest(members):
    return digest("\n".join(members).encode())


def validate_checkpoints(control, candidate, manifests):
    if control.get("architecture") != FacePairFusion.ARCHITECTURE or candidate.get("architecture") != SoftRouteFacePair.ARCHITECTURE:
        raise ValueError("Expected original hard control and experimental soft candidate")
    expected = {"epochs": 20, "lr_scheduler": "warmup_cosine", "warmup_epochs": 1,
                "lr": .025, "warmup_start_lr": .001, "min_lr": .0001, "momentum": .9,
                "weight_decay": .0004, "grad_accum_steps": 1, "batch_size": 32,
                "max_samples": 4096, "seed": 29, "pair_calibration_start_epoch": 1,
                "compile": True, "compile_mode": "default"}
    left, right = control["args"], candidate["args"]
    for checkpoint, args in ((control, left), (candidate, right)):
        if checkpoint.get("epoch") != 20:
            raise ValueError("Expected epoch-20 last.pt")
        check_args(args, "checkpoint.args", expected_variant=FACE_PAIR_VARIANT)
        check_compile(checkpoint["compile"], args, "checkpoint.compile")
        for key, value in expected.items():
            if args.get(key) != value:
                raise ValueError(f"Short-screen protocol mismatch: {key}")
    ignored = {"save_dir", "log_interval"}
    for key in sorted((left.keys() | right.keys()) - ignored):
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Checkpoint argument mismatch: {key}")
    sets = []
    for section in ("fit", "calibration", "validation"):
        parts = [m[section] for m in manifests]
        members = parts[0]["members"]
        if not members or len(set(members)) != len(members) or members != parts[1]["members"]:
            raise ValueError(f"Invalid or mismatched {section} members")
        for part in parts:
            if part["count"] != len(members) or part["sha256"] != members_digest(members):
                raise ValueError(f"Invalid {section} count or digest")
        sets.append(set(members))
    if any(m.get("official_validation_used_for_routing") is not False for m in manifests):
        raise ValueError("Official validation used for routing")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Overlapping member partitions")
    return manifests[0]


def verify_sources(left, right):
    with ZipFile(left) as a, ZipFile(right) as b:
        names = [n for n in a.namelist() if n.startswith("isaa/") and n.endswith(".py")]
        names += ["train_best_face_pair_100.py", "train_best_weighted_100_cosine.py"]
        if len(names) != len(set(names)) or not names:
            raise ValueError("Invalid control source archive")
        for name in names:
            if a.namelist().count(name) != 1 or b.namelist().count(name) != 1 or a.read(name) != b.read(name):
                raise ValueError(f"Training source mismatch: {name}")
        other = {n for n in b.namelist() if n.startswith("isaa/") and n.endswith(".py")}
        if other != {n for n in names if n.startswith("isaa/")}:
            raise ValueError("Training source file list mismatch")
    return {"files": len(names), "control_sha256": digest(left.read_bytes()), "candidate_sha256": digest(right.read_bytes())}


def load_checkpoint(path, device, model_class):
    if path.name != "last.pt":
        raise ValueError("Expected last.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != model_class.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model = model_class(checkpoint["args"]["num_classes"])
    if checkpoint.get("model_config") != model.model_config:
        raise ValueError("Checkpoint model configuration mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    n, h, b = model.routing_class_counts, model.routing_hand_face_correct, model.routing_body_face_correct
    if bool(((n < 0) | (h < 0) | (b < 0) | (h > n) | (b > n)).any()):
        raise ValueError("Invalid calibration counts")
    if float(model.hand_face_temperature) != 1. or float(model.body_face_temperature) != 1.:
        raise ValueError("Expected fixed unit temperatures")
    weights = _soft_weights(model).to(device)
    return checkpoint, compile_model(model.to(device).eval(), checkpoint["args"]["compile_mode"]), weights


def paired_predictions(outputs, weights):
    own = [_soft_logits(outputs[i], weights[i]) for i in range(2)]
    common = [_soft_logits(outputs[i], weights[0]) for i in range(2)]
    if not all(bool(torch.isfinite(x).all()) for x in own + common):
        raise RuntimeError("Nonfinite fused probabilities")
    if not torch.allclose(own[1], outputs[1]["logits"].float(), atol=2e-5, rtol=2e-5):
        raise RuntimeError("Candidate fused output does not reconstruct from branch logits")
    return [x.argmax(-1) for x in own], [x.argmax(-1) for x in common]


def validate_order(members, labels, expected_members, expected_labels):
    if list(members) != list(expected_members) or list(labels) != list(expected_labels):
        raise ValueError("Sample or label order mismatch")


class NamedSamples(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        return self.dataset.members[source_index], self.dataset[source_index]


def named_standard_collate(batch):
    names, values = zip(*batch)
    return list(names), collate_rtmw(list(values), feature_mode="raw")


def verify_cached_predictions(arrays, rows, weights):
    if len(arrays) != 2 or arrays[0].shape != arrays[1].shape or arrays[0].shape[:2] != (len(rows), 2):
        raise ValueError("Cached logits do not match prediction rows")
    for index, key in ((0, "control_prediction"), (1, "candidate_prediction")):
        raw = torch.from_numpy(arrays[index])
        out = {"hand_face_logits": raw[:, 0], "body_face_logits": raw[:, 1]}
        rebuilt = _soft_logits(out, weights[index].detach().cpu()).argmax(-1).tolist()
        if rebuilt != [r[key] for r in rows]:
            raise ValueError(f"Cached {key} does not reconstruct")
    raw = torch.from_numpy(arrays[1])
    shared = _soft_logits({"hand_face_logits": raw[:, 0], "body_face_logits": raw[:, 1]}, weights[0].detach().cpu()).argmax(-1).tolist()
    if shared != [r["shared_candidate_prediction"] for r in rows]:
        raise ValueError("Cached shared-weight predictions do not reconstruct")


def condition_stats(rows, prediction_key):
    subjects = [re.search(r"P(\d{3})R\d{3}A\d{3}", r["sample"]) for r in rows]
    if any(x is None for x in subjects):
        raise ValueError("Cannot identify subject")
    names, groups = np.unique([x.group(1) for x in subjects], return_inverse=True)
    truth = np.array([r["true_class_index"] for r in rows])
    c = np.array([r["control_prediction"] for r in rows]) == truth
    n = np.array([r[prediction_key] for r in rows]) == truth
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=n.astype(int) - c.astype(int), minlength=len(names))
    ci = None
    if len(names) >= 2:
        draws = np.random.default_rng(20261109).multinomial(len(names), np.full(len(names), 1 / len(names)), size=20000)
        ci = np.percentile(100 * (draws @ sums) / (draws @ counts), [2.5, 97.5]).tolist()
    return {"control_top1_percent": float(c.mean()*100), "candidate_top1_percent": float(n.mean()*100),
            "difference_pp": float((n.astype(int)-c.astype(int)).mean()*100), "subject_ci95_pp": ci,
            "corrected": int((~c & n).sum()), "regressed": int((c & ~n).sum()),
            "equal_subject_difference_pp": float(np.mean(sums/counts)*100),
            "subjects": {str(name): {"samples": int(count), "difference_pp": float(total/count*100)}
                         for name, count, total in zip(names, counts, sums)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("control", "candidate", "archive", "output-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--partition", choices=("calibration", "validation"), required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--smoke-samples", type=int, default=0,
                   help="Only for a CUDA integration smoke check (1-8 samples); never full-validation evidence")
    o = p.parse_args()
    if o.output_dir.exists():
        raise FileExistsError(o.output_dir)
    if o.batch_size <= 0 or o.num_workers < 0 or not 0 <= o.smoke_samples <= 8 or not torch.cuda.is_available():
        raise RuntimeError("Valid loader configuration and CUDA are required")
    device = torch.device("cuda")
    control, cm, cw = load_checkpoint(o.control, device, FacePairFusion)
    candidate, nm, nw = load_checkpoint(o.candidate, device, SoftRouteFacePair)
    manifests = [json.loads((path.parent / "split_manifest.json").read_text(encoding="utf-8")) for path in (o.control, o.candidate)]
    manifest = validate_checkpoints(control, candidate, manifests)
    for checkpoint in (control, candidate):
        counts = checkpoint["model"]["routing_class_counts"]
        if int(counts.sum()) != manifest["calibration"]["count"]:
            raise ValueError("Checkpoint routing counts do not cover calibration members")
    source = verify_sources(o.control.parent / "source.zip", o.candidate.parent / "source.zip")
    args = control["args"]
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = bool(args["tf32"])
    split = "train" if o.partition == "calibration" else "val"

    def dataset(augment=None):
        return RTMWZipDataset(o.archive, split=split, split_protocol=args["split"], window_size=args["window_size"],
            num_joints=133, num_classes=args["num_classes"], layout="rtmw_133", max_persons=args["max_persons"],
            feature_mode="raw", max_samples=0, score_normalization=args.get("score_normalization", "auto"),
            augment=bool(augment), temporal_compression=args["temporal_compression"],
            deterministic_temporal_crop=args.get("deterministic_temporal_crop", "center"),
            augmentation_config=augment or {"random_temporal_crop": False})

    reference = dataset()
    members = manifest["calibration"]["members"] if split == "train" else list(reference.members)
    if len(set(members)) != len(members) or (split == "val" and (len(members) != 16306 or members[:len(manifest["validation"]["members"])] != manifest["validation"]["members"])):
        raise ValueError("Unexpected full validation members")
    if o.smoke_samples:
        members = members[:o.smoke_samples]
        print(f"INTEGRATION_SMOKE samples={len(members)}: results are not performance evidence", flush=True)
    index = {name: i for i, name in enumerate(reference.members)}
    if len(index) != len(reference.members) or any(name not in index for name in members):
        raise ValueError("Member selection mismatch")
    indices = [index[name] for name in members]
    labels_expected = [int(reference.labels[i]) for i in indices]
    all_rows, arrays, results = [], [], []

    def process(name, loader, kind):
        rows, logits_cache = [], [[], []]
        inputs, masks = hashlib.sha256(), hashlib.sha256()
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                if kind == "anchor":
                    names, values = batch
                    _, features, labels, frame_mask, _, _ = values
                else:
                    names, values = batch
                    features, labels, frame_mask = values
                inputs.update(features.contiguous().numpy().tobytes())
                masks.update((features[:, 2] > 0).contiguous().numpy().tobytes())
                masks.update(frame_mask.contiguous().numpy().tobytes())
                count = len(labels)
                validate_order(names, labels.tolist(), members[offset:offset+count], labels_expected[offset:offset+count])
                f, mask = features.to(device), frame_mask.to(device)
                out = [cm(f, mask), nm(f, mask)]
                own, common = paired_predictions(out, (cw, nw))
                for i in range(2):
                    logits_cache[i].append(torch.stack((out[i]["hand_face_logits"], out[i]["body_face_logits"]), dim=1).float().cpu().numpy())
                old, new, shared_old, shared_new = [x.cpu().tolist() for x in (own[0], own[1], common[0], common[1])]
                if old != shared_old:
                    raise RuntimeError("Control own and shared weights disagree")
                for i, label in enumerate(labels.tolist()):
                    rows.append({"condition": name, "sample": members[offset+i], "true_class_index": label,
                                 "control_prediction": old[i], "candidate_prediction": new[i], "shared_candidate_prediction": shared_new[i]})
                offset += count
        if offset != len(members):
            raise ValueError("Incomplete partition")
        validate_order([r["sample"] for r in rows], [r["true_class_index"] for r in rows], members, labels_expected)
        cached = [np.concatenate(x) for x in logits_cache]
        verify_cached_predictions(cached, rows, (cw, nw))
        result = {"condition": name, "samples": offset, "input_sha256": inputs.hexdigest(), "mask_sha256": masks.hexdigest(),
                  "own_weights": condition_stats(rows, "candidate_prediction"), "common_control_weights": condition_stats(rows, "shared_candidate_prediction")}
        print(json.dumps(result), flush=True)
        all_rows.extend(rows)
        arrays.append(cached)
        results.append(result)

    for name, key, probability, seed_offset in STANDARD:
        seed_everything(o.seed + seed_offset)
        ds = dataset({"random_temporal_crop": False, key: probability} if key else None)
        validate_order(ds.members, ds.labels, reference.members, reference.labels)
        loader = DataLoader(NamedSamples(ds, indices), batch_size=o.batch_size, shuffle=False, num_workers=o.num_workers,
                            collate_fn=named_standard_collate, pin_memory=True)
        process(name, loader, "standard")
    for probability in (.05, .1):
        ds = PairedPrecompression(reference, members, probability, o.seed)
        loader = DataLoader(NamedSamples(ds, range(len(members))), batch_size=o.batch_size, shuffle=False, num_workers=o.num_workers)
        process(f"anchor_pre_{probability:g}", loader, "anchor")
    o.output_dir.mkdir(parents=True, exist_ok=False)
    with (o.output_dir / "paired_samples.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(o.output_dir / "paired_branch_logits.npz", control=np.concatenate([x[0] for x in arrays]), candidate=np.concatenate([x[1] for x in arrays]))
    weight_arrays = [x.detach().cpu().numpy() for x in (cw, nw)]
    summary = {"run_kind": "integration_smoke_not_performance_evidence" if o.smoke_samples else "full_partition",
               "smoke_samples_requested": o.smoke_samples,
               "control_sha256": digest(o.control.read_bytes()), "candidate_sha256": digest(o.candidate.read_bytes()),
               "source_archives": source, "partition": o.partition, "members_sha256": members_digest(members),
               "seed": o.seed, "num_workers": o.num_workers, "batch_size": o.batch_size,
               "selection_manifest_sha256": [digest((path.parent / "split_manifest.json").read_bytes()) for path in (o.control, o.candidate)],
               "weight_values": [x.tolist() for x in weight_arrays], "weight_sha256": [digest(x.tobytes()) for x in weight_arrays],
               "fusion": "gamma=1 from each training calibration; common uses control weights", "conditions": results,
               "same_inputs": "identical tensor and frame mask within each paired batch",
               "bootstrap": "20000 sample-weighted subject resamples when at least 2 subjects; exploratory, does not correct selection bias"}
    (o.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
