"""Train the BodyLocalFusion RTMW-133 action-recognition baseline."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from contextlib import nullcontext
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _prefer_bundled_cuda_libraries() -> None:
    """Make the PyTorch wheel's CUDA libraries win over system CUDA paths.

    The server also has a system CUDA/cuDNN installation under ``/usr/local``.
    PyTorch wheels ship a matching runtime beside the installed ``torch``
    package, which may be in a user site directory rather than ``sys.prefix``.
    Resolve that package location before importing torch so every train launch,
    including nohup/queued launches, uses the compatible libraries.
    """
    if not sys.platform.startswith("linux"):
        return
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_roots = [Path(sys.prefix) / "lib" / py_tag / "site-packages" / "nvidia"]
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None and torch_spec.origin:
        torch_site = Path(torch_spec.origin).resolve().parent.parent
        nvidia_roots.insert(0, torch_site / "nvidia")
    bundled = []
    for nvidia_root in nvidia_roots:
        for package in ("cudnn", "cublas", "cuda_runtime", "cu12", "cu13"):
            path = nvidia_root / package / "lib"
            if path.is_dir() and path not in bundled:
                bundled.append(path)
    bundled = [path for path in bundled if path.is_dir()]
    if not bundled:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    # Drop inherited CUDA/cuDNN directories.  PyTorch's wheel supplies the
    # matching runtime; retaining /usr/local/cuda/lib64 can reintroduce 9.3.
    filtered = [
        path for path in existing
        if path
        and "/usr/local/cuda" not in path
        and "/usr/local/lib64" not in path
        and "/nvidia/" not in path.lower()
    ]
    desired = ":".join([str(path) for path in bundled] + filtered)
    # The dynamic loader reads LD_LIBRARY_PATH before Python starts.  Re-exec
    # once so changing it here also affects libraries loaded by torch later.
    if os.environ.get("ISAA_CUDNN_ENV_FIXED") != "1":
        os.environ["LD_LIBRARY_PATH"] = desired
        os.environ["ISAA_CUDNN_ENV_FIXED"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    os.environ["LD_LIBRARY_PATH"] = desired


_prefer_bundled_cuda_libraries()

import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.data.rtmw_npy_dataset import RTMWNpyDataset
from isaa.data.ntu_preprocessed_dataset import NTUPreprocessedDataset
from isaa.models.rtmw_local_ctr import RTMWLocalCTR
from isaa.models.body_local_fusion import (
    BodyLocalFusion, BodyLocalDropoutFusion, BodyLocalFullFusion, BodyLocalRelativeFusion,
    BodyLocalHandCTRRelativeFusion, BodyLocalHandCTRWideRelativeFusion,
    BodyLocalHandCTRWideRelativeRoutedFusion,
    BodyLocalRelativeSplitFusion,
    TorsoCenteredCrossBranchFusion,
    OfficialTorsoCenteredCrossBranchFusion,
)
from isaa.models.original_ctrgcn import (
    OriginalCTRGCN,
    build_official_ntu_adjacency,
    build_rtmw_adjacency_for_nodes,
    build_rtmw32_auxiliary_adjacency,
)
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
from isaa.layouts.rtmw_133 import RTMW_32_NODE_INDICES


# Only stages whose model/data contract is already implemented are exposed by
# the entry point.  Keeping this list deliberately small prevents a stage name
# from silently enabling several unfinished ISAA changes at once.
_IMPLEMENTED_ABLATION_STAGES = {
    "s00_original25": {"node_count": 25, "label": "original CTR-GCN, 25 RTMW nodes", "variant": "original", "feature": "raw", "local_graph": False},
    "s01_original32": {"node_count": 32, "label": "original CTR-GCN, 32 RTMW nodes", "variant": "original", "feature": "raw", "local_graph": False},
    "s02_localgraph32": {"node_count": 32, "label": "CTR-GCN with learnable topology constrained to RTMW32 graph edges", "variant": "original", "feature": "raw", "local_graph": True},
    "s03_original32_aux": {"node_count": 32, "label": "original CTR-GCN, 32 learnable anchors plus anchor-only auxiliary nodes", "variant": "original_aux", "feature": "raw", "local_graph": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/ntu60_skeletons_rtmw.zip")
    parser.add_argument("--split", choices=("xsub60", "xset60", "xsub120", "xset120"), default="xsub60")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--npy-dir", default=None,
                        help="Preprocessed dataset root: official NTU train.npz/test.npz or legacy train/val NPY")
    parser.add_argument("--input-bn1d", action="store_true",
                        help="Use one ordinary nn.BatchNorm1d across 3x32 input channels")
    parser.add_argument("--ablation-stage", choices=tuple(_IMPLEMENTED_ABLATION_STAGES), default=None,
                        help=("Run one implemented single-change baseline stage. "
                              "s00_original25 and s01_original32 differ only in node count; "
                              "later stages are intentionally unavailable until implemented."))
    parser.add_argument("--model-variant", choices=("isaa", "original", "body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug", "body-local-full", "torso-cross-attn", "torso-cross-attn-official"), default=None,
                        help="Default: BodyLocalFusion; isaa/original are legacy explicit comparison models")
    parser.add_argument("--feature-mode", choices=("isaa", "raw"), default=None,
                        help="Input features: ISAA relative xy/score or raw x/y/score")
    parser.add_argument("--max-persons", type=int, default=2,
                        help="Maximum person tracks kept by the RTMW loader")
    parser.add_argument("--node-count", type=int, choices=(25, 32, 50, 71, 133), default=None,
                        help="Progressive RTMW input size: 25 semantic, 32 centers, 50/71 expanded, or all 133")
    parser.add_argument("--num-main-nodes", type=int, default=None,
                        help="Backward-compatible alias for --node-count in ISAA main-only runs")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps; keeps effective batch size when full RTMW baseline is memory-bound")
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--main-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Legacy ISAA switch; BodyLocalFusion always uses the full RTMW-133 input")
    parser.add_argument("--aux-start-epoch", "--fine-start-epoch", dest="fine_start_epoch", type=int, default=0,
                        help="Enable the 133-node auxiliary branch after this epoch; 0 enables it immediately")
    parser.add_argument("--auxiliary-channels", type=int, default=16,
                        help="Width of the two fixed-graph auxiliary layers")
    parser.add_argument("--backbone-width", choices=tuple(RTMWLocalCTR.CHANNEL_PRESETS), default="standard",
                        help="compact: 48/96/192 channels; standard: original 64/128/256")
    parser.add_argument("--native-bn", action=argparse.BooleanOptionalAction, default=None,
                        help="Use native BatchNorm2d for the legacy ISAA main-only model")
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
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Default: 8 workers for the BodyLocalFusion training protocol; 0 disables workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Queued batches per worker")
    parser.add_argument("--log-interval", type=float, default=0.5, help="Progress refresh interval in seconds")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow CUDA TF32 matmul/convolution; --no-tf32 uses full FP32 precision")
    parser.add_argument("--max-samples", type=int, default=0, help="Per split; 0 uses all samples")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--save-dir", default=None, help="Parent directory; each run creates a unique subdirectory")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Synthetic forward check; no ZIP needed")
    parser.add_argument("--console-only", action="store_true",
                        help="Keep training output on the terminal without creating console.log")
    args = parser.parse_args()
    if args.npy_dir:
        print("warning: --npy-dir is deprecated; all training now streams raw samples from --archive in memory", file=sys.stderr)
        args.npy_dir = None
    if args.ablation_stage is not None:
        stage = _IMPLEMENTED_ABLATION_STAGES[args.ablation_stage]
        expected_nodes = int(stage["node_count"])
        # Stage selection is an auditable preset.  A caller may still choose
        # runtime/training settings (batch size, seed, device, ...), but may
        # not combine the stage with a different model, feature path, or node
        # mapping.  In particular, a conflicting explicit --feature-mode is
        # rejected instead of being silently rewritten.
        required_variant = stage["variant"]
        required_feature = stage["feature"]
        if args.model_variant not in (None, required_variant, "original" if required_variant == "original_aux" else required_variant):
            parser.error(f"{args.ablation_stage} requires --model-variant {required_variant}")
        if args.feature_mode not in (None, required_feature):
            parser.error(f"{args.ablation_stage} requires --feature-mode {required_feature}")
        if args.node_count not in (None, expected_nodes):
            parser.error(
                f"{args.ablation_stage} fixes --node-count {expected_nodes}; "
                f"received {args.node_count}"
            )
        if args.num_main_nodes not in (None, expected_nodes):
            parser.error(
                f"{args.ablation_stage} fixes the node mapping at {expected_nodes}; "
                f"--num-main-nodes {args.num_main_nodes} is incompatible"
            )
        args.model_variant = "original" if required_variant == "original_aux" else required_variant
        args.feature_mode = required_feature
        args.node_count = expected_nodes
        args.main_only = required_variant == "isaa"
    elif args.model_variant is None:
        args.model_variant = "body-local"
    if args.num_classes is None:
        args.num_classes = 60 if args.split in {"xsub60", "xset60"} else 120
    if args.node_count is None:
        args.node_count = args.num_main_nodes
    if args.model_variant in {"body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug", "body-local-full", "torso-cross-attn", "torso-cross-attn-official"}:
        args.main_only = False
        args.node_count = 133
        args.feature_mode = "raw" if args.feature_mode is None else args.feature_mode
        if args.feature_mode != "raw":
            parser.error("body-local variants require raw x/y/score features")
    elif args.model_variant == "original":
        # The official model consumes all 133 RTMW joints and raw x/y/score.
        args.main_only = False
        args.feature_mode = "raw" if args.feature_mode is None else args.feature_mode
        if args.feature_mode != "raw":
            parser.error("--model-variant original requires --feature-mode raw")
        args.node_count = 133 if args.node_count is None else args.node_count
    else:
        if args.main_only is None:
            # The legacy ISAA variant remains main-only unless the caller
            # explicitly requests its auxiliary branch.
            args.main_only = True
        if args.feature_mode is None:
            args.feature_mode = "isaa"
        args.node_count = 32 if args.node_count is None else args.node_count
        if args.node_count == 133 and args.main_only:
            parser.error("ISAA main-only 模式的 node-count 只能是 25/32/50/71；133 请使用 --no-main-only")
        if not args.main_only and args.node_count not in {32, 133}:
            parser.error("ISAA 辅助分支当前只支持 32 节点主干；25/50/71 阶段请保持 --main-only")
    args.num_main_nodes = args.node_count if args.node_count != 133 else 32
    if min(args.num_classes, args.window_size, args.batch_size, args.test_batch_size,
           args.epochs, args.auxiliary_channels, args.max_persons,
           args.grad_accum_steps) < 1:
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


def collate_rtmw(samples, *, main_indices=None, feature_mode="isaa"):
    """Select model channels and optional main nodes in the worker."""
    features, labels, frame_mask = default_collate(samples)
    if feature_mode == "isaa":
        features = features[:, [0, 1, 4]]
    elif feature_mode != "raw":
        raise ValueError(f"Unsupported feature_mode: {feature_mode!r}")
    if main_indices is not None:
        features = features.index_select(3, main_indices)
    return features.contiguous(), labels, frame_mask


def collate_preprocessed(samples, *, main_indices=None):
    """Collate fixed RTMW/NTU NPY features and optionally select nodes."""
    features, labels, frame_mask = default_collate(samples)
    if main_indices is not None and int(main_indices.max().item()) < features.shape[3]:
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
    grad_accum_steps = max(1, int(progress_context.get("grad_accum_steps", 1)))
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
                if training and (step - 1) % grad_accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(x, frame_mask)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at batch {step}")
                if training:
                    group_size = min(grad_accum_steps,
                                     total_steps - ((step - 1) // grad_accum_steps) * grad_accum_steps)
                    (loss / group_size).backward()
                    if step % grad_accum_steps == 0 or step == total_steps:
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
    if args.model_variant == "body-local":
        variant = BodyLocalFusion.ARCHITECTURE
    elif args.model_variant == "body-local-time-aug":
        variant = BodyLocalFusion.ARCHITECTURE + "_time_aug"
    elif args.model_variant == "body-local-coord-aug":
        variant = BodyLocalFusion.ARCHITECTURE + "_coord_aug"
    elif args.model_variant == "body-local-dropout":
        variant = BodyLocalDropoutFusion.ARCHITECTURE
    elif args.model_variant == "body-local-relative":
        variant = BodyLocalRelativeFusion.ARCHITECTURE
    elif args.model_variant == "body-local-relative-split":
        variant = BodyLocalRelativeSplitFusion.ARCHITECTURE
    elif args.model_variant == "body-local-hand-ctr-relative":
        variant = BodyLocalHandCTRRelativeFusion.ARCHITECTURE
    elif args.model_variant == "body-local-hand-ctr-wide-relative":
        variant = BodyLocalHandCTRWideRelativeFusion.ARCHITECTURE
    elif args.model_variant == "body-local-hand-ctr-wide-relative-routed":
        variant = BodyLocalHandCTRWideRelativeRoutedFusion.ARCHITECTURE
    elif args.model_variant == "body-local-full":
        variant = BodyLocalFullFusion.ARCHITECTURE
    elif args.model_variant == "torso-cross-attn":
        variant = TorsoCenteredCrossBranchFusion.ARCHITECTURE
    elif args.model_variant == "torso-cross-attn-official":
        variant = OfficialTorsoCenteredCrossBranchFusion.ARCHITECTURE
    elif args.model_variant == "original":
        variant = f"{OriginalCTRGCN.ARCHITECTURE}_n{args.node_count}"
        if args.ablation_stage == "s03_original32_aux":
            variant += "_anchor_aux"
    else:
        variant = f"rtmw_ctr{args.num_main_nodes}_only_v5" if args.main_only else RTMWLocalCTR.ARCHITECTURE
    if args.input_bn1d:
        variant += "_inputbn1d"
    base = (Path(args.save_dir) if args.save_dir else
            PROJECT_ROOT / "outputs" / f"{variant}_{args.backbone_width}_sgd" / args.split)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    save_dir = create_run_directory(base)
    log_context = nullcontext() if args.console_only else mirror_console_to_file(save_dir / "console.log")
    with log_context:
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
    if hasattr(torch.backends, "cudnn"):
        try:
            cudnn_version = torch.backends.cudnn.version()
        except RuntimeError as exc:
            raise RuntimeError("cuDNN initialization failed; cuDNN is mandatory in this project") from exc
        # cuDNN is a required backend. There is intentionally no command-line
        # switch or fallback path that disables it.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if args.num_workers is None:
        args.num_workers = min(8, max(1, (os.cpu_count() or 1) // 2)) if device.type == "cuda" else 0
    if args.model_variant in {"body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug"}:
        model_class = (BodyLocalRelativeSplitFusion if args.model_variant == "body-local-relative-split"
                       else BodyLocalHandCTRWideRelativeRoutedFusion if args.model_variant == "body-local-hand-ctr-wide-relative-routed"
                       else BodyLocalHandCTRWideRelativeFusion if args.model_variant == "body-local-hand-ctr-wide-relative"
                       else BodyLocalHandCTRRelativeFusion if args.model_variant == "body-local-hand-ctr-relative"
                       else BodyLocalDropoutFusion if args.model_variant == "body-local-dropout"
                       else BodyLocalRelativeFusion if args.model_variant == "body-local-relative"
                       else BodyLocalFusion)
        model = model_class(num_classes=args.num_classes).to(device)
        model_config = {"variant": args.model_variant, "num_classes": args.num_classes,
                        "body_indices": list(model.BODY_INDICES), "hand_indices": list(model.HAND_INDICES),
                        "face_tokens": 6, "local_nodes": 48,
                        "body_channels": model.CHANNELS, "local_channels": model.LOCAL_CHANNELS,
                        "fusion_dropout": float(getattr(model, "DROPOUT", 0.0)),
                        "local_coordinate_mode": getattr(model, "LOCAL_COORDINATE_MODE", "raw"),
                        "hand_channels": getattr(model, "HAND_CHANNELS", None),
                        "hand_input_channels": getattr(model, "HAND_INPUT_CHANNELS", 3),
                        "local_branch_mode": ("shared_hand_ctr_wide_face_st_routed" if args.model_variant == "body-local-hand-ctr-wide-relative-routed"
                                              else "shared_hand_ctr_wide_face_st" if args.model_variant == "body-local-hand-ctr-wide-relative"
                                              else "shared_hand_ctr_face_st" if args.model_variant == "body-local-hand-ctr-relative"
                                              else "independent_hand_face" if args.model_variant == "body-local-relative-split"
                                              else "joint_hand_face")}
    elif args.model_variant == "body-local-full":
        model = BodyLocalFullFusion(num_classes=args.num_classes).to(device)
        model_config = {"variant": "body-local-full", "num_classes": args.num_classes,
                        "body_indices": list(model.BODY_INDICES), "hand_indices": list(model.HAND_INDICES),
                        "face_tokens": 6, "local_nodes": 48,
                        "body_channels": model.CHANNELS, "local_channels": model.LOCAL_CHANNELS}
    elif args.model_variant == "torso-cross-attn":
        model = TorsoCenteredCrossBranchFusion(num_classes=args.num_classes).to(device)
        model_config = {"variant": "torso-cross-attn", "num_classes": args.num_classes,
                        "body_indices": list(model.BODY_INDICES), "hand_indices": list(model.HAND_INDICES),
                        "face_tokens": 6, "body_channels": model.CHANNELS,
                        "local_channels": model.LOCAL_CHANNELS,
                        "fusion": "per-frame torso-query; independent hand/face sigmoid gates"}
    elif args.model_variant == "torso-cross-attn-official":
        model = OfficialTorsoCenteredCrossBranchFusion(num_classes=args.num_classes).to(device)
        model_config = {"variant": "torso-cross-attn-official", "num_classes": args.num_classes,
                        "body_indices": list(model.BODY_INDICES), "hand_indices": list(model.HAND_INDICES),
                        "face_tokens": 6,
                        "body_backbone": "official CTR-GCN ten-block feature extractor",
                        "body_channels": model.BODY_CHANNELS,
                        "local_backbone": "official ST-GCN feature extractor with edge importance",
                        "local_channels": model.LOCAL_CHANNELS,
                        "local_strides": model.LOCAL_STRIDES,
                        "fusion": "per-frame torso-query; independent hand/face sigmoid gates"}
    elif args.model_variant == "original":
        aux_stage = args.ablation_stage == "s03_original32_aux"
        data_root = Path(args.npy_dir) if args.npy_dir else None
        official_preprocessed = bool(
            data_root is not None and (
                (data_root / "train_data.npy").is_file()
                or (data_root / "train.npz").is_file()
            )
        )
        if aux_stage:
            original_graph = build_rtmw32_auxiliary_adjacency()
            original_indices = tuple(range(133))
            model_nodes = 133
            local_graph = True
            learnable_graph_nodes = tuple(RTMW_32_NODE_INDICES)
        elif official_preprocessed and args.node_count == 25:
            original_graph = build_official_ntu_adjacency()
            original_indices = tuple(range(25))
            model_nodes = 25
            local_graph = False
            learnable_graph_nodes = None
        else:
            original_graph, original_indices = build_rtmw_adjacency_for_nodes(args.node_count)
            model_nodes = args.node_count
            local_graph = bool(_IMPLEMENTED_ABLATION_STAGES.get(args.ablation_stage or "", {}).get("local_graph", False))
            learnable_graph_nodes = None
        model = OriginalCTRGCN(
            num_classes=args.num_classes,
            num_point=model_nodes,
            num_person=args.max_persons,
            graph=original_graph,
            local_graph=local_graph,
            learnable_graph_nodes=learnable_graph_nodes,
        ).to(device)
        model_config = {"variant": "original", "num_classes": args.num_classes,
                        "num_point": model_nodes, "num_person": args.max_persons,
                        "input_channels": 3, "graph": "official_rtmw_regional_induced",
                        "node_indices": list(original_indices),
                        "ablation_stage": args.ablation_stage}
    else:
        model = RTMWLocalCTR(num_classes=args.num_classes, auxiliary_channels=args.auxiliary_channels,
                             backbone_width=args.backbone_width, main_only=args.main_only,
                             native_bn=args.native_bn,
                             input_norm="bn1d" if args.input_bn1d else "point_bn2d",
                             main_node_count=args.num_main_nodes).to(device)
        model.set_fine_enabled(not args.main_only and args.fine_start_epoch == 0)
        model_config = {"variant": "isaa", "backbone_width": model.backbone_width,
                        "channels": model.channels, "auxiliary_channels": args.auxiliary_channels,
                        "num_classes": args.num_classes, "main_only": args.main_only,
                        "native_bn": model.native_bn, "input_norm": model.input_norm_type,
                        "main_node_count": model.main_node_count,
                        "node_indices": model.main_joint_indices.cpu().tolist()}
    temporal_augmented = args.model_variant == "body-local-time-aug"
    coordinate_augmented = args.model_variant == "body-local-coord-aug"
    temporal_augmentation_config = {
        "crop_min_ratio": 0.9375,
        "max_shift": 2,
        "jitter_probability": 0.0,
        "temporal_enabled": temporal_augmented,
        "coordinate_jitter_std": 0.003 if coordinate_augmented else 0.0,
    }
    reference = {
        "repository": "https://github.com/Uason-Chen/CTR-GCN",
        "commit": "67d8710578b842a5d6384cd8293d627f03c6ddc1",
        "config": ("config/rtmw133/default.yaml" if args.model_variant == "original" else
                   ("config/nturgbd120-cross-set/default.yaml" if args.split == "xset120"
                    else "config/nturgbd120-cross-subject/default.yaml")),
        "schedule": "main.py:adjust_learning_rate (zero-based milestones)",
        "adaptation": ("BodyLocalFusion: body22 CTR-GCN + hand42/face6 ST-GCN, raw x/y/score, masked pooling and pre-classifier fusion"
                       if args.model_variant in {"body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug", "body-local-full", "torso-cross-attn", "torso-cross-attn-official"} else
                       (f"official RTMW induced {args.node_count}-node graph, raw x/y/score, ordinary BN, no masks"
                        if args.model_variant == "original" else
                        "RTMW relative xy/score, masks, 32 joints; existing crop/pad preprocessing")),
    }
    records.update_config(args=vars(args), model_config=model_config, architecture=model.ARCHITECTURE,
                          optimizer={"name": "SGD", "base_lr": args.lr, "momentum": args.momentum,
                                     "nesterov": args.nesterov, "weight_decay": args.weight_decay,
                                     "grad_accum_steps": args.grad_accum_steps,
                                     "effective_batch_size": args.batch_size * args.grad_accum_steps},
                          main_joint_indices=(model.main_joint_indices.cpu().tolist()
                                              if hasattr(model, "main_joint_indices") else None),
                          runtime={"torch": torch.__version__, "cuda": torch.version.cuda,
                                   "device": str(device), "gpu": torch.cuda.get_device_name(device)
                                   if device.type == "cuda" else None,
                                   "cudnn": cudnn_version,
                                   "cudnn_enabled": torch.backends.cudnn.enabled},
                          reference=reference,
                          preprocessing={"channels": (["raw_x", "raw_y", "score", "torso_relative_x", "torso_relative_y"]
                                                      if args.model_variant == "body-local-relative-split" else
                                                      (["raw_x", "raw_y", "score", "torso_relative_x", "torso_relative_y", "cross_hand_distance", "cross_hand_direction_x", "cross_hand_direction_y"]
                                                       if args.model_variant in {"body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed"} else
                                                      (["x", "y", "score"] if args.feature_mode == "raw"
                                                       else ["relative_x", "relative_y", "score"]))),
                                         "crop": ("precomputed fixed window" if args.npy_dir else
                                                  "random fixed window / center validation, pad short clips"),
                                         "augmentation": ({
                                             "enabled": True,
                                             "random_temporal_crop": temporal_augmented,
                                             **temporal_augmentation_config,
                                         } if (temporal_augmented or coordinate_augmented) else {
                                             "random_temporal_crop": not bool(args.npy_dir),
                                         })})
    if args.model_variant == "original":
        print(f"Original CTR-GCN RTMW{args.node_count} points={args.max_persons} device={device} "
              f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
        if args.ablation_stage is not None:
            stage_label = _IMPLEMENTED_ABLATION_STAGES.get(args.ablation_stage or "", {}).get("label")
            print(f"ablation_stage: {args.ablation_stage}"
                  + (f" ({stage_label})" if stage_label else ""), flush=True)
    elif args.model_variant in {"body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug"}:
        label = ("BodyLocalHandCTRWideRelativeRoutedFusion" if args.model_variant == "body-local-hand-ctr-wide-relative-routed"
                 else "BodyLocalHandCTRWideRelativeFusion" if args.model_variant == "body-local-hand-ctr-wide-relative"
                 else "BodyLocalHandCTRRelativeFusion" if args.model_variant == "body-local-hand-ctr-relative"
                 else "BodyLocalRelativeSplitFusion" if args.model_variant == "body-local-relative-split"
                 else "BodyLocalRelativeFusion" if args.model_variant == "body-local-relative"
                 else "BodyLocalDropoutFusion" if args.model_variant == "body-local-dropout"
                 else "BodyLocalFusionTimeAug" if args.model_variant == "body-local-time-aug"
                 else "BodyLocalFusionCoordAug" if args.model_variant == "body-local-coord-aug"
                 else "BodyLocalFusion")
        print(f"{label} RTMW body22+hand42+face6 device={device} "
              f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    elif args.model_variant == "body-local-full":
        print(f"BodyLocalFullFusion RTMW body22+hand42+face6 device={device} "
              f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    elif args.model_variant == "torso-cross-attn":
        print(f"TorsoCenteredCrossBranchFusion RTMW body22+hand42+face6 device={device} "
              f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    elif args.model_variant == "torso-cross-attn-official":
        print(f"OfficialTorsoCenteredCrossBranchFusion RTMW body22+hand42+face6 device={device} "
              f"body_channels={model.BODY_CHANNELS} local_channels={model.LOCAL_CHANNELS} "
              f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    else:
        print(f"ISAA {model.experiment_name} backbone=ctr_gcn_{args.num_main_nodes} auxiliary={'off' if args.main_only else 'fixed_71'} "
              f"channels={model.channels} "
              f"auxiliary_channels={args.auxiliary_channels} bn={'native' if model.native_bn else 'masked'} "
              f"input_norm={model.input_norm_type} device={device} "
              f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"optimizer: SGD lr={args.lr} momentum={args.momentum} nesterov={args.nesterov} "
          f"weight_decay={args.weight_decay} warmup={args.warmup_epochs} "
          f"steps_zero_based={args.lr_steps} decay={args.lr_decay} epochs={args.epochs}", flush=True)
    print(f"records: {save_dir}", flush=True)
    if device.type == "cuda":
        print(f"runtime: gpu={torch.cuda.get_device_name(device)} tf32={args.tf32} "
              f"cudnn=True cudnn_benchmark=True",
              flush=True)
    print(f"runtime: batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps} "
          f"effective_batch_size={args.batch_size * args.grad_accum_steps} num_workers={args.num_workers} "
          f"pin_memory={device.type == 'cuda'} persistent_workers={args.num_workers > 0} "
          f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else None}", flush=True)
    if args.dry_run:
        people = args.max_persons if args.model_variant in {"original", "body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug", "body-local-full", "torso-cross-attn", "torso-cross-attn-official"} else 2
        synthetic_nodes = (133 if args.ablation_stage == "s03_original32_aux" else args.node_count) if args.model_variant in {"original", "body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug", "body-local-full", "torso-cross-attn", "torso-cross-attn-official"} else (
            args.node_count if args.main_only else 133
        )
        synthetic_channels = (8 if args.model_variant in {"body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed"}
                              else 5 if args.model_variant == "body-local-relative-split" else 3)
        x = torch.randn(2, synthetic_channels, args.window_size, synthetic_nodes, people, device=device)
        x[:, 2] = 1
        if args.main_only and synthetic_nodes == 133:
            x = x.index_select(3, model.main_joint_indices)
        model.eval()
        with torch.no_grad():
            if args.model_variant == "original":
                logits = model(x)
                print(f"input={tuple(x.shape)} logits={tuple(logits.shape)} "
                      f"finite={torch.isfinite(logits).all().item()}")
            elif args.model_variant in {"body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug", "body-local-full", "torso-cross-attn", "torso-cross-attn-official"}:
                logits = model(x)
                print(f"input={tuple(x.shape)} logits={tuple(logits.shape)} "
                      f"finite={torch.isfinite(logits).all().item()}")
            else:
                result = model(x, return_node_features=True)
                print(f"input={tuple(x.shape)} logits={tuple(result['logits'].shape)} "
                      f"main_features={tuple(result['node_features'].shape)} "
                      f"auxiliary_features={None if result['auxiliary_node_features'] is None else tuple(result['auxiliary_node_features'].shape)} "
                      f"finite={torch.isfinite(result['logits']).all().item()}")
        return

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is required in this project, but this PyTorch has no torch.compile")
    compile_enabled = True
    print(f"compile: torch.compile mode={args.compile_mode} (first batch will compile)", flush=True)
    train_model = torch.compile(model, mode=args.compile_mode)
    records.update_config(compile={"enabled": compile_enabled, "mode": args.compile_mode})

    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = PROJECT_ROOT / archive
    loaders = {}
    dataset_info = {}
    if args.npy_dir:
        if args.model_variant in {"original", "body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug", "body-local-full", "torso-cross-attn", "torso-cross-attn-official"}:
            npy_indices = None if args.ablation_stage == "s03_original32_aux" or args.node_count == 133 else torch.tensor(original_indices, dtype=torch.long)
        elif args.main_only:
            npy_indices = model.main_joint_indices.cpu()
        else:
            npy_indices = None
        collate = partial(collate_preprocessed, main_indices=npy_indices)
    else:
        if args.model_variant in {"original", "body-local", "body-local-dropout", "body-local-relative", "body-local-relative-split", "body-local-hand-ctr-relative", "body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed", "body-local-time-aug", "body-local-coord-aug", "body-local-full", "torso-cross-attn", "torso-cross-attn-official"}:
            main_indices = None if args.ablation_stage == "s03_original32_aux" or args.node_count == 133 else torch.tensor(original_indices, dtype=torch.long)
        else:
            main_indices = model.main_joint_indices.cpu() if args.main_only else None
        collate = partial(collate_rtmw, main_indices=main_indices,
                          feature_mode=args.feature_mode)
    for split in ("train", "val"):
        if args.npy_dir:
            root = Path(args.npy_dir)
            dataset_nodes = 133 if args.ablation_stage == "s03_original32_aux" else args.node_count
            official_path = root / ("train.npz" if split == "train" else "test.npz")
            ctrgcn_path = root / ("train_data.npy" if split == "train" else "val_data.npy")
            if official_path.is_file():
                dataset = NTUPreprocessedDataset(
                    official_path, max_samples=args.max_samples, expected_nodes=dataset_nodes,
                    window_size=args.window_size,
                    augment=(temporal_augmented or coordinate_augmented) and split == "train",
                    augmentation_config=temporal_augmentation_config,
                )
            elif ctrgcn_path.is_file():
                dataset = NTUPreprocessedDataset(
                    ctrgcn_path, max_samples=args.max_samples, expected_nodes=dataset_nodes,
                    window_size=args.window_size,
                    augment=(temporal_augmented or coordinate_augmented) and split == "train",
                    augmentation_config=temporal_augmentation_config,
                )
            else:
                # Legacy ISAA NPY cache remains available for non-official RTMW runs.
                dataset = RTMWNpyDataset(
                    root / split, max_samples=args.max_samples,
                    augment=(temporal_augmented or coordinate_augmented) and split == "train",
                    augmentation_config=temporal_augmentation_config,
                )
                if (args.model_variant in {"body-local-hand-ctr-wide-relative", "body-local-hand-ctr-wide-relative-routed"}
                        and dataset.data.shape[1] != 8):
                    raise ValueError(
                        "body-local-hand-ctr-wide-relative now requires the eight-channel "
                        "cross-hand-distance-direction cache. Run tools/preprocess_torso_relative_npy.py "
                        "with --cross-hand-distance into a new output directory."
                    )
                if dataset.data.shape[2] != args.window_size:
                    raise ValueError(
                        f"预处理窗口 T={dataset.data.shape[2]} 与 --window-size={args.window_size} 不一致"
                    )
        else:
            dataset = RTMWZipDataset(
                archive, split=split, split_protocol=args.split,
                window_size=args.window_size, num_joints=133, num_classes=args.num_classes,
                layout="rtmw_133", max_persons=args.max_persons,
                feature_mode=args.feature_mode, max_samples=args.max_samples,
                augment=split == "train", augmentation_config={
                    "random_temporal_crop": True,
                    **temporal_augmentation_config,
                },
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
        has_official_npz = any((Path(args.npy_dir) / name).is_file() for name in ("train.npz", "test.npz"))
        records.update_config(
            dataset=dataset_info,
            dataset_format="official_ntu_npz" if has_official_npz else "npy_memmap",
            npy_dir=str(Path(args.npy_dir).resolve()),
        )
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
        if args.model_variant == "isaa":
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
            "grad_accum_steps": args.grad_accum_steps,
        }
        finish_progress_lines()
        if args.model_variant == "original":
            stage = args.ablation_stage or "original_ctrgcn"
        elif args.model_variant == "body-local":
            stage = "body22_ctr+hands42_face6_stgcn"
        elif args.model_variant == "body-local-time-aug":
            stage = "body22_ctr+hands42_face6_stgcn_temporal_aug"
        elif args.model_variant == "body-local-coord-aug":
            stage = "body22_ctr+hands42_face6_stgcn_coordinate_aug"
        elif args.model_variant == "body-local-dropout":
            stage = "body22_ctr+hands42_face6_stgcn_dropout_fusion"
        elif args.model_variant == "body-local-relative":
            stage = "body22_ctr+hands42_face6_stgcn_torso_relative"
        elif args.model_variant == "body-local-relative-split":
            stage = "body22_ctr+hand_stgcn_face_stgcn_torso_relative"
        elif args.model_variant == "body-local-hand-ctr-relative":
            stage = "body22_ctr+hand_ctr21_face_stgcn_torso_relative"
        elif args.model_variant == "body-local-hand-ctr-wide-relative":
            stage = "body22_ctr+hand_ctr21_wide_face_stgcn_torso_relative"
        elif args.model_variant == "body-local-hand-ctr-wide-relative-routed":
            stage = "body22_ctr+hand_ctr21_wide_face_stgcn_torso_relative_stage_quality_gate"
        elif args.model_variant == "body-local-full":
            stage = "body22_ctr+hands42_face6_stgcn_full"
        elif args.model_variant == "torso-cross-attn":
            stage = "body22_ctr+hand_face_torso_centered_cross_attn"
        elif args.model_variant == "torso-cross-attn-official":
            stage = "official_ctr_body22+official_stgcn_hand_face_torso_centered_cross_attn"
        else:
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
