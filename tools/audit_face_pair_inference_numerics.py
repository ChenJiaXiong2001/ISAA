"""Compare four fixed batches under controlled inference settings."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA libraries before importing torch.
import torch
from torch.nn import functional as F

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.face_pair_training import compile_model
from isaa.models.face_pair_fusion import FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_corruption import _soft_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    audit = json.loads(options.audit.read_text(encoding="utf-8"))
    checkpoint_sha = hashlib.sha256(options.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != audit["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    manifest = json.loads(options.manifest.read_text(encoding="utf-8"))
    members = manifest["validation"]["members"]
    digest = hashlib.sha256("\n".join(members).encode()).hexdigest()
    if digest != manifest["validation"]["sha256"] or digest != audit["members_sha256"]:
        raise ValueError("Validation manifest mismatch")
    def dataset(split):
        return RTMWZipDataset(
            options.archive, split=split, split_protocol=args["split"], window_size=args["window_size"],
            num_joints=133, num_classes=args["num_classes"], layout="rtmw_133", max_persons=args["max_persons"],
            score_normalization=args.get("score_normalization", "auto"), feature_mode="raw", augment=False,
            temporal_compression="confidence_weighted", deterministic_temporal_crop=args.get("deterministic_temporal_crop", "center"))
    val, train = dataset("val"), dataset("train")
    if list(val.members) != members:
        raise ValueError("Validation order mismatch")
    train_indices = {name: index for index, name in enumerate(train.members)}
    calibration = manifest["calibration"]["members"]
    if hashlib.sha256("\n".join(calibration).encode()).hexdigest() != manifest["calibration"]["sha256"]:
        raise ValueError("Calibration manifest mismatch")
    def batch(data, indices):
        features, labels, mask = collate_rtmw([data[index] for index in indices], feature_mode="raw")
        return features.to("cuda"), labels, mask.to("cuda")
    positions = {name: i for i, name in enumerate(members)}
    cases = audit["clean_prediction_disagreements"]
    starts = sorted({positions[case["sample"]] // 32 * 32 for case in cases})
    batches = {start: batch(val, range(start, start + 32)) for start in starts}
    warm = batch(train, [train_indices[name] for name in calibration[:32]])
    model = FacePairFusion(args["num_classes"])
    if checkpoint["architecture"] != model.ARCHITECTURE:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to("cuda").eval()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = args.get("tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = args.get("tf32", True)
    weights = _soft_weights(model).to("cuda")
    torch.backends.cudnn.benchmark = False
    compiled = compile_model(model, args["compile_mode"])
    results, metadata = {}, []
    started = time.perf_counter()
    def forward(name, module, benchmark):
        torch.backends.cudnn.benchmark = benchmark
        for start, (features, labels, mask) in batches.items():
            output = module(features, mask)
            hand, body = output["hand_face_logits"].float(), output["body_face_logits"].float()
            logits = torch.logaddexp(weights.log() + F.log_softmax(hand, -1),
                                     torch.log1p(-weights) + F.log_softmax(body, -1))
            probabilities = F.softmax(logits, -1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Nonfinite probabilities")
            results[f"{name}/{start}"] = {"hf": hand.cpu(), "bf": body.cpu(),
                                          "probabilities": probabilities.cpu(), "labels": labels}
        torch.cuda.synchronize()
        metadata.append({"name": name, "cudnn_benchmark": benchmark,
                         "seconds_since_start": time.perf_counter() - started})
        print(json.dumps(metadata[-1]), flush=True)
    with torch.inference_mode():
        forward("eager_benchmark_off", model, False)
        forward("compiled_benchmark_off", compiled, False)
        forward("compiled_benchmark_on_before_calibration", compiled, True)
        compiled(warm[0], warm[2])
        torch.cuda.synchronize()
        forward("compiled_benchmark_on_after_calibration", compiled, True)
        forward("eager_benchmark_on_after_calibration", model, True)
    details = []
    for case in cases:
        index = positions[case["sample"]]
        start, offset = index // 32 * 32, index % 32
        variants = {}
        for event in metadata:
            current = results[f"{event['name']}/{start}"]["probabilities"][offset]
            top = torch.topk(current, 2)
            variants[event["name"]] = {"top2_classes": top.indices.tolist(),
                                       "top2_probabilities": top.values.tolist(),
                                       "top2_margin": float(top.values[0] - top.values[1])}
        details.append({**case, "index": index, "batch_start": start, "variants": variants})
    hashes = {str(start): {"input_sha256": hashlib.sha256(features.cpu().numpy().tobytes()).hexdigest(),
                           "mask_sha256": hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()}
              for start, (features, _, mask) in batches.items()}
    options.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(results, options.output_dir / "logits_and_probabilities.pt")
    summary = {"checkpoint_sha256": checkpoint_sha, "members_sha256": digest, "batch_hashes": hashes,
               "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "tf32": args.get("tf32", True), "compile_mode": args["compile_mode"],
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "events": metadata, "cases": details,
               "caveat": "Same-process cuDNN autotune caches may persist across phases; warm-up order is not isolated from earlier phases. Small fixed sample, inference only."}
    (options.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
