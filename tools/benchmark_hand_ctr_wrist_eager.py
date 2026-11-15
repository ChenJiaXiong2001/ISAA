"""Benchmark the hand CTR wrist model without torch.compile.

This is intentionally an isolated performance probe.  It does not change the
required compile settings of any training entry point and does not write a
checkpoint.  The loader and model protocol match the 100-frame training run.
"""
import argparse
import time
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.hand_ctr_wrist_fusion import HandCTRWristFusion
from isaa.train import collate_rtmw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/ntu60_skeletons_rtmw.zip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-persons", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measure-batches", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark is intended for a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    archive = Path(args.archive).resolve()
    dataset = RTMWZipDataset(
        archive,
        split="train",
        split_protocol="xsub60",
        window_size=100,
        num_joints=133,
        num_classes=60,
        layout="rtmw_133",
        max_persons=args.max_persons,
        feature_mode="raw",
        max_samples=args.max_samples,
        temporal_compression="confidence_weighted",
        augment=True,
        augmentation_config={"random_temporal_crop": False},
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=lambda samples: collate_rtmw(samples, feature_mode="raw"),
    )
    if len(loader) < args.warmup_batches + args.measure_batches:
        raise RuntimeError(
            f"not enough batches: {len(loader)} < "
            f"{args.warmup_batches + args.measure_batches}"
        )

    model = HandCTRWristFusion(num_classes=60).to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0004
    )
    criterion = nn.CrossEntropyLoss()
    print(
        f"eager benchmark: device={device} batch={args.batch_size} "
        f"window=100 persons={args.max_persons} workers={args.num_workers} "
        f"batches={args.measure_batches}",
        flush=True,
    )
    print(f"archive={archive} samples={len(dataset)}", flush=True)

    iterator = iter(loader)
    for _ in range(args.warmup_batches):
        features, labels, frame_mask = next(iterator)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        frame_mask = frame_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features, frame_mask), labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)

    fetch_times: List[float] = []
    step_times: List[float] = []
    losses: List[float] = []
    for index in range(args.measure_batches):
        fetch_start = time.perf_counter()
        features, labels, frame_mask = next(iterator)
        fetch_times.append(time.perf_counter() - fetch_start)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        frame_mask = frame_mask.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features, frame_mask), labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        losses.append(float(loss.detach()))
        print(
            f"batch={index + 1}/{args.measure_batches} "
            f"fetch={fetch_times[-1]:.3f}s step={step_times[-1]:.3f}s "
            f"total={fetch_times[-1] + step_times[-1]:.3f}s loss={losses[-1]:.5f}",
            flush=True,
        )

    mean_fetch = sum(fetch_times) / len(fetch_times)
    mean_step = sum(step_times) / len(step_times)
    print(
        f"summary: fetch_mean={mean_fetch:.4f}s step_mean={mean_step:.4f}s "
        f"total_mean={mean_fetch + mean_step:.4f}s "
        f"throughput={args.batch_size / (mean_fetch + mean_step):.3f} samples/s "
        f"max_memory={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
