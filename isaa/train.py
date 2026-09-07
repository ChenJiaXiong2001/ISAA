"""Train fixed-local / main-node CTR on the existing NTU120 RTMW ZIP dataset."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.rtmw_local_ctr import RTMWLocalCTR
from isaa.utils.console_logging import (
    estimate_eta,
    finish_progress_lines,
    format_duration,
    format_timestamped_lines,
    format_tqdm_progress,
    write_progress_lines,
)
from isaa.utils.seed import seed_everything
from isaa.layouts import register_skeleton_presets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/ntu120_skeletons_rtmw.zip")
    parser.add_argument("--split", choices=("xsub120", "xset120"), default="xsub120")
    parser.add_argument("--num-classes", type=int, default=120)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fine-start-epoch", type=int, default=0,
                        help="Compatibility option; full-node CTR is enabled from epoch 1 by default")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Default: up to 8 on CUDA, 0 on CPU; 0 disables workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Queued batches per worker")
    parser.add_argument("--log-interval", type=float, default=0.5, help="Progress refresh interval in seconds")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow CUDA TF32 matmul/convolution; --no-tf32 uses full FP32 precision")
    parser.add_argument("--max-samples", type=int, default=0, help="Per split; 0 uses all samples")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Synthetic forward check; no ZIP needed")
    args = parser.parse_args()
    if min(args.num_classes, args.window_size, args.batch_size, args.epochs) < 1:
        parser.error("num-classes, window-size, batch-size and epochs must be positive")
    if args.fine_start_epoch < 0:
        parser.error("fine-start-epoch must be >= 0")
    if ((args.num_workers is not None and args.num_workers < 0)
            or args.max_samples < 0 or not 0 < args.lr < float("inf")):
        parser.error("num-workers/max-samples must be nonnegative and lr positive and finite")
    if args.prefetch_factor < 1 or not 0 < args.log_interval < float("inf"):
        parser.error("prefetch-factor and log-interval must be positive and finite")
    return args


def collate_rtmw(samples):
    """Select x/y/score in the worker, before DataLoader pins the batch."""
    features, labels, frame_mask = default_collate(samples)
    return features[:, [0, 1, 4]].contiguous(), labels, frame_mask


def run_epoch(
    model, loader, device, optimizer=None, *, progress_context=None, log_interval=0.5
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    total = 0
    progress_context = progress_context or {}
    total_steps = len(loader)
    stage_started_at = time.perf_counter()
    run_started_at = progress_context.get("run_started_at", stage_started_at)
    total_units = max(progress_context.get("total_units", total_steps), 1)
    completed_before = progress_context.get("completed_before", 0)
    last_refresh = stage_started_at
    batch_fetch_started = stage_started_at
    data_wait_seconds = 0.0
    try:
        with torch.set_grad_enabled(training):
            for step, (x, labels, frame_mask) in enumerate(loader, start=1):
                data_wait_seconds += time.perf_counter() - batch_fetch_started
                non_blocking = device.type == "cuda"
                x = x.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                frame_mask = frame_mask.to(device, non_blocking=non_blocking)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(x, frame_mask)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at batch {step}")
                if training:
                    loss.backward()
                    optimizer.step()
                count = labels.size(0)
                # Keep metrics on device until epoch end. Preserve the fail-fast loss check above.
                with torch.no_grad():
                    correct = (logits.argmax(dim=1) == labels).sum()
                    top5 = logits.topk(min(5, logits.size(1)), dim=1).indices
                    correct_top5 = (top5 == labels[:, None]).any(dim=1).sum()
                    totals += torch.stack((loss.detach() * count, correct, correct_top5)).to(totals.dtype)
                total += count
                now = time.perf_counter()
                if step == 1 or now - last_refresh >= log_interval:
                    total_completed = min(completed_before + step, total_units)
                    write_progress_lines((
                        format_tqdm_progress(step, total_steps, now - stage_started_at),
                        "total_progress: " + format_tqdm_progress(total_completed, total_units, now - run_started_at),
                    ))
                    last_refresh = now
                batch_fetch_started = time.perf_counter()
        if total == 0:
            raise ValueError("Cannot run an epoch with an empty DataLoader")
        loss_sum, correct, correct_top5 = totals.cpu().tolist()
        now = time.perf_counter()
        write_progress_lines((
            format_tqdm_progress(total_steps, total_steps, now - stage_started_at),
            "total_progress: " + format_tqdm_progress(
                min(completed_before + total_steps, total_units), total_units, now - run_started_at
            ),
        ))
    finally:
        finish_progress_lines()
    return {
        "loss": loss_sum / total, "top1": correct / total, "top5": correct_top5 / total,
        "samples_per_second": total / max(now - stage_started_at, 1e-9),
        "data_wait_seconds": data_wait_seconds,
    }


def main() -> None:
    args = parse_args()
    register_skeleton_presets()
    seed_everything(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    elif device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if args.num_workers is None:
        args.num_workers = min(8, max(1, (os.cpu_count() or 1) // 2)) if device.type == "cuda" else 0
    model = RTMWLocalCTR(num_classes=args.num_classes).to(device)
    print(f"ISAA RTMWLocalCTR ctr_gcn=full_133 channelwise_topology device={device} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    if device.type == "cuda":
        print(f"runtime: gpu={torch.cuda.get_device_name(device)} tf32={args.tf32} cudnn_benchmark=True",
              flush=True)
    print(f"runtime: batch_size={args.batch_size} num_workers={args.num_workers} "
          f"pin_memory={device.type == 'cuda'} persistent_workers={args.num_workers > 0} "
          f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else None}", flush=True)
    if args.dry_run:
        x = torch.randn(2, 3, args.window_size, 133, 2, device=device)
        x[:, 2] = 1
        model.eval()
        with torch.no_grad():
            logits = model(x)
        print(f"input={tuple(x.shape)} logits={tuple(logits.shape)} finite="
              f"{torch.isfinite(logits).all().item()}")
        return

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = PROJECT_ROOT / archive
    loaders = {}
    for split in ("train", "val"):
        dataset = RTMWZipDataset(
            archive, split=split, split_protocol=args.split,
            window_size=args.window_size, num_joints=133, num_classes=args.num_classes,
            layout="rtmw_133", max_persons=2, max_samples=args.max_samples,
            augment=split == "train", augmentation_config={"random_temporal_crop": True},
        )
        loaders[split] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == "train",
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            collate_fn=collate_rtmw,
        )
        print(f"{split}: {len(dataset)} samples ({args.split})", flush=True)

    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "outputs" / args.split
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_accuracy = -1.0
    training_started_at = time.perf_counter()
    train_steps = len(loaders["train"])
    steps_per_epoch = train_steps + len(loaders["val"])
    total_run_steps = steps_per_epoch * args.epochs
    for epoch in range(1, args.epochs + 1):
        model.set_fine_enabled(True)
        completed_before = (epoch - 1) * steps_per_epoch
        progress_context = {
            "run_started_at": training_started_at,
            "total_units": total_run_steps,
            "completed_before": completed_before,
        }
        finish_progress_lines()
        stage = "full_ctr"
        print(f"Training epoch: {epoch}/{args.epochs} stage={stage} "
              f"lr={optimizer.param_groups[0]['lr']:.8g}", flush=True)
        train_metrics = run_epoch(
            model, loaders["train"], device, optimizer, progress_context=progress_context,
            log_interval=args.log_interval,
        )
        print(f"Eval epoch: {epoch}/{args.epochs}", flush=True)
        progress_context["completed_before"] = completed_before + train_steps
        val_metrics = run_epoch(
            model, loaders["val"], device, progress_context=progress_context, log_interval=args.log_interval
        )
        total_elapsed = time.perf_counter() - training_started_at
        completed_steps = epoch * steps_per_epoch
        remaining_seconds = estimate_eta(total_elapsed, completed_steps, total_run_steps - completed_steps)
        print(format_timestamped_lines(
            f"epoch={epoch}",
            f"train_acc: {train_metrics['top1'] * 100.0:.2f}%",
            f"Top1: {val_metrics['top1'] * 100.0:.2f}%",
            f"Top5: {val_metrics['top5'] * 100.0:.2f}%",
            f"loss: {val_metrics['loss']:.4f}",
            "total_progress: " + format_tqdm_progress(completed_steps, total_run_steps, total_elapsed),
            f"remaining_total: {format_duration(remaining_seconds)}",
            f"throughput: train={train_metrics['samples_per_second']:.1f} samples/s "
            f"val={val_metrics['samples_per_second']:.1f} samples/s",
            f"data_wait: train={train_metrics['data_wait_seconds']:.2f}s "
            f"val={val_metrics['data_wait_seconds']:.2f}s",
        ), flush=True)
        val_loss, val_accuracy = val_metrics["loss"], val_metrics["top1"]
        is_best = val_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, val_accuracy)
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "args": vars(args), "val_loss": val_loss, "val_accuracy": val_accuracy,
            "best_accuracy": best_accuracy, "architecture": "rtmw_local_ctr",
            "stage": stage,
        }
        torch.save(checkpoint, save_dir / "last.pt")
        status = "finished" if epoch == args.epochs else "epoch_end"
        print(f"checkpoint: saved {save_dir / 'last.pt'} status={status} epoch={epoch} "
              f"step={train_steps} next_epoch={epoch + 1} global_step={epoch * train_steps}", flush=True)
        if is_best:
            torch.save(checkpoint, save_dir / "best.pt")
    print(f"best_val_acc={best_accuracy:.2%} checkpoints={save_dir}")


if __name__ == "__main__":
    main()
