"""Train RTMW main-node CTR-GCN with the official NTU120 optimizer schedule."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.data.rtmw_npy_dataset import RTMWNpyDataset
from isaa.models.rtmw_local_ctr import RTMWLocalCTR
from isaa.utils.console_logging import (
    estimate_eta,
    finish_progress_lines,
    format_duration,
    format_timestamped_lines,
    format_tqdm_progress,
    write_progress_lines,
    mirror_console_to_file,
)
from isaa.utils.experiment import RunRecords, create_run_directory, ctrgcn_learning_rate
from isaa.utils.seed import seed_everything
from isaa.layouts import register_skeleton_presets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/ntu120_skeletons_rtmw.zip")
    parser.add_argument("--split", choices=("xsub60", "xset60", "xsub120", "xset120"), default="xsub120")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--npy-dir", default=None,
                        help="Preprocessed root containing train/ and val/ data.npy + labels.npy")
    parser.add_argument("--input-bn1d", action="store_true",
                        help="Use one ordinary nn.BatchNorm1d across 3x32 input channels")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--main-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Only 32 main nodes; --no-main-only restores the auxiliary experiment")
    parser.add_argument("--aux-start-epoch", "--fine-start-epoch", dest="fine_start_epoch", type=int, default=0,
                        help="Enable the 133-node auxiliary branch after this epoch; 0 enables it immediately")
    parser.add_argument("--auxiliary-channels", type=int, default=16,
                        help="Width of the two fixed-graph auxiliary layers")
    parser.add_argument("--backbone-width", choices=tuple(RTMWLocalCTR.CHANNEL_PRESETS), default="standard",
                        help="compact: 48/96/192 channels; standard: original 64/128/256")
    parser.add_argument("--native-bn", action=argparse.BooleanOptionalAction, default=None,
                        help="Use native BatchNorm2d; default on for main-only, off for masked auxiliary runs")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="Use torch.compile on CUDA; default on for CUDA and off for CPU")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
                        default="reduce-overhead")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay", type=float, default=0.0004)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-steps", type=int, nargs="+", default=[35, 55],
                        help="Zero-based decay epochs, matching the official CTR-GCN code")
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop an incomplete training batch, as in official CTR-GCN")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Default: up to 8 on CUDA, 0 on CPU; 0 disables workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Queued batches per worker")
    parser.add_argument("--log-interval", type=float, default=0.5, help="Progress refresh interval in seconds")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow CUDA TF32 matmul/convolution; --no-tf32 uses full FP32 precision")
    parser.add_argument("--cudnn", action=argparse.BooleanOptionalAction, default=True,
                        help="Use cuDNN for CUDA convolutions; --no-cudnn uses PyTorch native CUDA kernels")
    parser.add_argument("--max-samples", type=int, default=0, help="Per split; 0 uses all samples")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--save-dir", default=None, help="Parent directory; each run creates a unique subdirectory")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Synthetic forward check; no ZIP needed")
    args = parser.parse_args()
    if args.num_classes is None:
        args.num_classes = 60 if args.split in {"xsub60", "xset60"} else 120
    if min(args.num_classes, args.window_size, args.batch_size, args.test_batch_size, args.epochs, args.auxiliary_channels) < 1:
        parser.error("num-classes, window-size, batch-size, epochs and auxiliary-channels must be positive")
    if args.fine_start_epoch < 0:
        parser.error("fine-start-epoch must be >= 0")
    if not 0 <= args.momentum < 1 or (args.nesterov and args.momentum == 0):
        parser.error("momentum must be in [0,1); Nesterov requires positive momentum")
    if not 0 <= args.weight_decay < float("inf") or not 0 < args.lr_decay <= 1:
        parser.error("weight-decay must be finite and nonnegative; lr-decay must be in (0,1]")
    if args.warmup_epochs < 0 or any(step < 0 for step in args.lr_steps):
        parser.error("warmup-epochs and lr-steps must be nonnegative")
    if args.lr_steps != sorted(set(args.lr_steps)):
        parser.error("lr-steps must be strictly increasing")
    if ((args.num_workers is not None and args.num_workers < 0)
            or args.max_samples < 0 or not 0 < args.lr < float("inf")):
        parser.error("num-workers/max-samples must be nonnegative and lr positive and finite")
    if args.prefetch_factor < 1 or not 0 < args.log_interval < float("inf"):
        parser.error("prefetch-factor and log-interval must be positive and finite")
    return args


def collate_rtmw(samples, *, main_indices=None):
    """Select x/y/score in the worker, before DataLoader pins the batch."""
    features, labels, frame_mask = default_collate(samples)
    features = features[:, [0, 1, 4]]
    if main_indices is not None:
        features = features.index_select(3, main_indices)
    return features.contiguous(), labels, frame_mask


def run_epoch(
    model, loader, device, optimizer=None, *, progress_context=None, log_interval=0.5,
    records=None, epoch=0, learning_rate=0.0,
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
    record_values = (torch.empty((total_steps, 3), device=device, dtype=torch.float32)
                     if records is not None else None)
    record_wait = []
    record_host_seconds = []
    try:
        with torch.set_grad_enabled(training):
            for step, (x, labels, frame_mask) in enumerate(loader, start=1):
                batch_started = time.perf_counter()
                batch_wait = batch_started - batch_fetch_started
                data_wait_seconds += batch_wait
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
                if records is not None:
                    record_values[step - 1].copy_(torch.stack(
                        (loss.detach(), correct.float(), correct_top5.float())))
                    record_wait.append(batch_wait)
                    record_host_seconds.append(time.perf_counter() - batch_started)
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
        if records is not None:
            values = record_values[:total].cpu().tolist()
            phase = "train" if training else "val"
            for index, (batch_loss, batch_correct, batch_top5) in enumerate(values):
                count = int(loader.batch_size or 1)
                if index == total_steps - 1 and total_steps * count != total:
                    count = total - index * count
                records.batch({
                    "epoch": epoch, "phase": phase, "step": index + 1,
                    "global_step": (epoch - 1) * total_steps + index + 1,
                    "samples": count, "loss": batch_loss, "top1": batch_correct / count,
                    "top5": batch_top5 / count, "lr": learning_rate,
                    "data_wait_seconds": record_wait[index],
                    "host_batch_seconds": record_host_seconds[index],
                })
    finally:
        finish_progress_lines()
    return {
        "loss": loss_sum / total, "top1": correct / total, "top5": correct_top5 / total,
        "samples_per_second": total / max(now - stage_started_at, 1e-9),
        "data_wait_seconds": data_wait_seconds,
        "elapsed_seconds": now - stage_started_at, "samples": total, "steps": total_steps,
    }


def main() -> None:
    args = parse_args()
    variant = "rtmw_ctr32_only_v5" if args.main_only else RTMWLocalCTR.ARCHITECTURE
    if args.input_bn1d:
        variant += "_inputbn1d"
    base = (Path(args.save_dir) if args.save_dir else
            PROJECT_ROOT / "outputs" / f"{variant}_{args.backbone_width}_sgd" / args.split)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    save_dir = create_run_directory(base)
    with mirror_console_to_file(save_dir / "console.log"):
        records = RunRecords(save_dir, vars(args), PROJECT_ROOT)
        try:
            _run(args, save_dir, records)
        except BaseException as exc:
            finish_progress_lines()
            records.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", repr(exc))
            traceback.print_exc()
            raise
        else:
            records.finish("dry_run_completed" if args.dry_run else "completed")
        finally:
            records.close()


def _run(args, save_dir, records) -> None:
    register_skeleton_presets()
    seed_everything(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    cudnn_version = None
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    elif device.type == "cuda":
        if args.cudnn:
            try:
                cudnn_version = torch.backends.cudnn.version()
            except RuntimeError as exc:
                print(f"warning: cuDNN initialization failed; falling back to native CUDA kernels: {exc}",
                      flush=True)
                args.cudnn = False
        torch.backends.cudnn.enabled = args.cudnn
        torch.backends.cudnn.benchmark = args.cudnn
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if args.num_workers is None:
        args.num_workers = min(8, max(1, (os.cpu_count() or 1) // 2)) if device.type == "cuda" else 0
    model = RTMWLocalCTR(num_classes=args.num_classes, auxiliary_channels=args.auxiliary_channels,
                         backbone_width=args.backbone_width, main_only=args.main_only,
                         native_bn=args.native_bn,
                         input_norm="bn1d" if args.input_bn1d else "point_bn2d").to(device)
    model.set_fine_enabled(not args.main_only and args.fine_start_epoch == 0)
    model_config = {"backbone_width": model.backbone_width, "channels": model.channels,
                    "auxiliary_channels": args.auxiliary_channels, "num_classes": args.num_classes,
                    "main_only": args.main_only, "native_bn": model.native_bn,
                    "input_norm": model.input_norm_type}
    records.update_config(args=vars(args), model_config=model_config, architecture=model.ARCHITECTURE,
                          optimizer={"name": "SGD", "base_lr": args.lr, "momentum": args.momentum,
                                     "nesterov": args.nesterov, "weight_decay": args.weight_decay},
                          main_joint_indices=model.main_joint_indices.cpu().tolist(),
                          runtime={"torch": torch.__version__, "cuda": torch.version.cuda,
                                   "device": str(device), "gpu": torch.cuda.get_device_name(device)
                                   if device.type == "cuda" else None,
                                   "cudnn": cudnn_version,
                                   "cudnn_enabled": torch.backends.cudnn.enabled},
                          preprocessing={"channels": ["relative_x", "relative_y", "score"],
                                         "crop": ("precomputed fixed window" if args.npy_dir else
                                                  "random fixed window / center validation, pad short clips"),
                                         "augmentation": {"random_temporal_crop": not bool(args.npy_dir)}})
    print(f"ISAA {model.experiment_name} backbone=ctr_gcn_32 auxiliary={'off' if args.main_only else 'fixed_71'} "
          f"channels={model.channels} "
          f"auxiliary_channels={args.auxiliary_channels} bn={'native' if model.native_bn else 'masked'} "
          f"input_norm={model.input_norm_type} device={device} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    print(f"optimizer: SGD lr={args.lr} momentum={args.momentum} nesterov={args.nesterov} "
          f"weight_decay={args.weight_decay} warmup={args.warmup_epochs} "
          f"steps_zero_based={args.lr_steps} decay={args.lr_decay} epochs={args.epochs}", flush=True)
    print(f"records: {save_dir}", flush=True)
    if device.type == "cuda":
        print(f"runtime: gpu={torch.cuda.get_device_name(device)} tf32={args.tf32} "
              f"cudnn={args.cudnn} cudnn_benchmark={args.cudnn}",
              flush=True)
    print(f"runtime: batch_size={args.batch_size} num_workers={args.num_workers} "
          f"pin_memory={device.type == 'cuda'} persistent_workers={args.num_workers > 0} "
          f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else None}", flush=True)
    if args.dry_run:
        x = torch.randn(2, 3, args.window_size, 133, 2, device=device)
        x[:, 2] = 1
        if args.main_only:
            x = x.index_select(3, model.main_joint_indices)
        model.eval()
        with torch.no_grad():
            result = model(x, return_node_features=True)
        print(f"input={tuple(x.shape)} logits={tuple(result['logits'].shape)} "
              f"main_features={tuple(result['node_features'].shape)} "
              f"auxiliary_features={None if result['auxiliary_node_features'] is None else tuple(result['auxiliary_node_features'].shape)} "
              f"finite={torch.isfinite(result['logits']).all().item()}")
        return

    compile_enabled = args.compile if args.compile is not None else device.type == "cuda"
    train_model = model
    if compile_enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requested, but this PyTorch has no torch.compile")
        print(f"compile: torch.compile mode={args.compile_mode} (first batch will compile)", flush=True)
        train_model = torch.compile(model, mode=args.compile_mode)
    records.update_config(compile={"enabled": compile_enabled, "mode": args.compile_mode})

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = PROJECT_ROOT / archive
    loaders = {}
    dataset_info = {}
    collate = (default_collate if args.npy_dir else
               partial(collate_rtmw, main_indices=model.main_joint_indices.cpu() if args.main_only else None))
    for split in ("train", "val"):
        if args.npy_dir:
            dataset = RTMWNpyDataset(Path(args.npy_dir) / split, max_samples=args.max_samples)
            if dataset.data.shape[2] != args.window_size:
                raise ValueError(
                    f"预处理窗口 T={dataset.data.shape[2]} 与 --window-size={args.window_size} 不一致"
                )
        else:
            dataset = RTMWZipDataset(
                archive, split=split, split_protocol=args.split,
                window_size=args.window_size, num_joints=133, num_classes=args.num_classes,
                layout="rtmw_133", max_persons=2, max_samples=args.max_samples,
                augment=split == "train", augmentation_config={"random_temporal_crop": True},
            )
        loaders[split] = DataLoader(
            dataset, batch_size=args.batch_size if split == "train" else args.test_batch_size,
            shuffle=split == "train", drop_last=args.drop_last and split == "train",
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            collate_fn=collate,
        )
        dataset_info[split] = {"samples": len(dataset), "batches": len(loaders[split])}
        if len(loaders[split]) == 0:
            raise ValueError("Empty loader: lower --batch-size or use --no-drop-last for a small smoke test")
        print(f"{split}: {len(dataset)} samples ({args.split})", flush=True)

    if args.npy_dir:
        records.update_config(dataset=dataset_info, dataset_format="npy_memmap", npy_dir=str(Path(args.npy_dir).resolve()))
    else:
        records.update_config(dataset=dataset_info, archive=str(archive.resolve()),
                              archive_bytes=archive.stat().st_size, archive_mtime_ns=archive.stat().st_mtime_ns)
    optimizer = torch.optim.SGD(train_model.parameters(), lr=args.lr, momentum=args.momentum,
                                nesterov=args.nesterov, weight_decay=args.weight_decay)
    best_accuracy = -1.0
    best_epoch = 0
    training_started_at = time.perf_counter()
    train_steps = len(loaders["train"])
    steps_per_epoch = train_steps + len(loaders["val"])
    total_run_steps = steps_per_epoch * args.epochs
    for epoch in range(1, args.epochs + 1):
        model.set_fine_enabled(not args.main_only and epoch > args.fine_start_epoch)
        learning_rate = ctrgcn_learning_rate(epoch - 1, args.lr, args.warmup_epochs,
                                            args.lr_steps, args.lr_decay)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        completed_before = (epoch - 1) * steps_per_epoch
        progress_context = {
            "run_started_at": training_started_at,
            "total_units": total_run_steps,
            "completed_before": completed_before,
        }
        finish_progress_lines()
        stage = "ctr32+aux71" if model.fine_enabled else "ctr32_only"
        print(f"Training epoch: {epoch}/{args.epochs} stage={stage} "
              f"lr={optimizer.param_groups[0]['lr']:.8g}", flush=True)
        train_metrics = run_epoch(
            train_model, loaders["train"], device, optimizer, progress_context=progress_context,
            log_interval=args.log_interval,
            records=records, epoch=epoch, learning_rate=learning_rate,
        )
        print(f"Eval epoch: {epoch}/{args.epochs}", flush=True)
        progress_context["completed_before"] = completed_before + train_steps
        val_metrics = run_epoch(
            train_model, loaders["val"], device, progress_context=progress_context, log_interval=args.log_interval,
            records=records, epoch=epoch, learning_rate=learning_rate,
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
        if is_best:
            best_epoch = epoch
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "args": vars(args), "val_loss": val_loss, "val_accuracy": val_accuracy,
            "best_accuracy": best_accuracy, "architecture": model.ARCHITECTURE,
            "model_config": model_config, "best_epoch": best_epoch,
            "lr_schedule": {"last_epoch_index": epoch - 1, "warmup_epochs": args.warmup_epochs,
                            "milestones": args.lr_steps, "decay": args.lr_decay, "base_lr": args.lr},
            "train_metrics": train_metrics, "val_metrics": val_metrics,
            "stage": stage,
        }
        temporary = save_dir / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(save_dir / "last.pt")
        status = "finished" if epoch == args.epochs else "epoch_end"
        print(f"checkpoint: saved {save_dir / 'last.pt'} status={status} epoch={epoch} "
              f"step={train_steps} next_epoch={epoch + 1} global_step={epoch * train_steps}", flush=True)
        if is_best:
            temporary = save_dir / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(save_dir / "best.pt")
        records.epoch({"epoch": epoch, "stage": stage, "lr": learning_rate,
                       **{f"train_{key}": value for key, value in train_metrics.items()},
                       **{f"val_{key}": value for key, value in val_metrics.items()},
                       "best_val_acc": best_accuracy, "best_epoch": best_epoch,
                       "total_elapsed_seconds": time.perf_counter() - training_started_at})
    print(f"best_val_acc={best_accuracy:.2%} checkpoints={save_dir}")


if __name__ == "__main__":
    main()
