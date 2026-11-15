"""Paired CPU screen for time-reversal TTA on a frozen face-pair checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import isaa.train  # Configure CUDA libraries before torch.
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import FACE_PAIR_VARIANT, FacePairFusion
from isaa.train import collate_rtmw
from tools.evaluate_face_pair_corruption import _soft_weights
from tools.evaluate_face_pair_precompression_repair import PairedPrecompression


def fused(output, weights):
    hf = F.log_softmax(output["hand_face_logits"].float(), -1)
    bf = F.log_softmax(output["body_face_logits"].float(), -1)
    w = weights.to(hf.device).view(1, -1)
    score = torch.logaddexp(torch.log(w) + hf, torch.log1p(-w) + bf)
    return score - torch.logsumexp(score, -1, keepdim=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--probability", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = ck["args"]
    if args["model_variant"] != FACE_PAIR_VARIANT:
        raise ValueError("Expected seed2 best-face-pair checkpoint")
    model = FacePairFusion(args["num_classes"])
    if ck["architecture"] != model.ARCHITECTURE:
        raise ValueError("Architecture mismatch")
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(a.device)
    weights = _soft_weights(model)
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))["validation"]
    members = manifest["members"][:a.max_samples]
    if hashlib.sha256("\n".join(manifest["members"]).encode()).hexdigest() != manifest["sha256"]:
        raise ValueError("Manifest digest mismatch")
    ds = RTMWZipDataset(a.archive, split="val", split_protocol=args["split"], window_size=args["window_size"],
        num_joints=133, num_classes=args["num_classes"], layout="rtmw_133", max_persons=args["max_persons"],
        score_normalization=args.get("score_normalization", "auto"), feature_mode="raw", augment=False,
        temporal_compression="confidence_weighted", deterministic_temporal_crop="center")
    paired = PairedPrecompression(ds, members, a.probability, a.seed)
    loader = DataLoader(paired, batch_size=a.batch_size, shuffle=False, num_workers=0)
    rows, input_hash = [], hashlib.sha256()
    totals = {condition: {method: 0 for method in ("base", "reverse_tta")} for condition in ("clean", "pre_dropout")}
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (clean, corrupted, labels, frames, hidden, nframes) in enumerate(loader, 1):
            input_hash.update(clean.contiguous().numpy().tobytes()); input_hash.update(corrupted.contiguous().numpy().tobytes())
            clean, corrupted, labels, frames = [x.to(a.device) for x in (clean, corrupted, labels, frames)]
            batch_rows = [{"sample": members[len(rows) + i], "label": int(y), "hidden_original_points": int(h)}
                          for i, (y, h) in enumerate(zip(labels.cpu(), hidden))]
            for condition, x in (("clean", clean), ("pre_dropout", corrupted)):
                forward = fused(model(x, frames), weights)
                reverse = fused(model(x.flip(1), frames.flip(1)), weights)
                # Average normalized log-probabilities then renormalize.
                tta = (forward + reverse) * 0.5
                tta = tta - torch.logsumexp(tta, -1, keepdim=True)
                for method, logits in (("base", forward), ("reverse_tta", tta)):
                    pred = logits.argmax(-1).cpu().tolist()
                    totals[condition][method] += sum(int(p == int(y)) for p, y in zip(pred, labels.cpu()))
                    for row, p in zip(batch_rows, pred): row[f"{condition}_{method}"] = p
            rows.extend(batch_rows)
            if step == 1 or step % 8 == 0 or step == len(loader):
                print(f"batch={step}/{len(loader)} samples={len(rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    for condition in totals:
        for method in totals[condition]:
            totals[condition][method] = {"correct": totals[condition][method], "top1_percent": 100*totals[condition][method]/len(rows)}
        totals[condition]["difference_pp"] = totals[condition]["reverse_tta"]["top1_percent"]-totals[condition]["base"]["top1_percent"]
    a.output_dir.mkdir(parents=True)
    with (a.output_dir/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"checkpoint":str(a.checkpoint),"checkpoint_sha256":hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
      "members_sha256":hashlib.sha256("\n".join(members).encode()).hexdigest(),"samples":len(rows),"seed":a.seed,
      "probability":a.probability,"input_sha256":input_hash.hexdigest(),"device":a.device,
      "method":"mean forward/reversed-temporal normalized HF/BF gamma=1 log probabilities, renormalized",
      "conditions":totals,"seconds":time.perf_counter()-started,
      "caveat":"fixed first 512 validation samples; synthetic anchor point dropout; no training or validation recalibration"}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__": main()
