"""Independent two-stage trainer; the historical training pipeline is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from isaa.data.input_joint_dataset import InputJointDataset
from isaa.models.input_joint_selection import (
    INPUT_SELECTION_VARIANT, InputJointSelectionFusion, validate_spec,
)
from isaa.utils.batch_cosine import BatchWarmupCosine
from isaa.utils.console_logging import mirror_console_to_file
from isaa.utils.experiment import RunRecords, create_run_directory, write_json
from isaa.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
COMPILE_SCOPE = {
    "enabled": True, "required": True, "suppress_errors": False,
    "fail_on_recompile_limit_hit": True,
    "scope": "global stochastic gates, raw geometry, body/hand neural encoders, pooling, fusion, classifier and sparsity penalty",
    "eager_control": "preserved original face LocalSTBlock boundary",
    "whole_model_eager_fallback": False,
}


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--model-variant", choices=(INPUT_SELECTION_VARIANT,), default=INPUT_SELECTION_VARIANT)
    p.add_argument("--stage", choices=("select", "compact"), default="select")
    p.add_argument("--subset", type=Path, help="Manifest only, never search weights; required for compact training")
    p.add_argument("--selection-config", type=Path, default=ROOT / "configs/input_joint_selection.json")
    p.add_argument("--export-selection", type=Path, help="Export a trained checkpoint's fixed input subset; no training")
    p.add_argument("--selection-output", type=Path, help="Required with --export-selection")
    p.add_argument("--export-threshold", type=float)
    p.add_argument("--evaluate", type=Path, help="Evaluate a matching checkpoint with compiled forward")
    p.add_argument("--archive", type=Path, default=ROOT / "data/ntu60_skeletons_rtmw.zip")
    p.add_argument("--split", choices=("xsub60", "xset60", "xsub120", "xset120"), default="xsub60")
    p.add_argument("--num-classes", type=int)
    p.add_argument("--feature-mode", choices=("raw",), default="raw")
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--temporal-compression", choices=("confidence_weighted",), default="confidence_weighted")
    p.add_argument("--max-persons", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--test-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lr", type=float, default=.1)
    p.add_argument("--lr-scheduler", choices=("warmup_cosine",), default="warmup_cosine")
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--warmup-start-lr", type=float, default=.001)
    p.add_argument("--min-lr", type=float, default=.0001)
    p.add_argument("--momentum", type=float, default=.9)
    p.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--weight-decay", type=float, default=.0004)
    p.add_argument("--max-grad-norm", type=float, default=0.)
    p.add_argument("--gate-sigma", type=float)
    p.add_argument("--gate-initial-mu", type=float)
    p.add_argument("--sparsity-weight", type=float)
    p.add_argument("--gate-warmup-epochs", type=int)
    p.add_argument("--gate-ramp-epochs", type=int)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--log-interval", type=float, default=10.)
    p.add_argument("--save-dir", type=Path)
    p.add_argument("--dry-run", action="store_true", help="Actual compiled synthetic forward and backward")
    args = p.parse_args(argv)
    if "--no-compile" in argv or not args.compile:
        p.error("Input joint selection requires torch.compile; --no-compile is not allowed")
    cfg = json.loads(args.selection_config.read_text(encoding="utf-8"))
    if cfg.pop("version", None) != 1:
        p.error("Unsupported selection config version")
    expected = {"gate_sigma", "gate_initial_mu", "sparsity_weight", "gate_warmup_epochs", "gate_ramp_epochs", "export_threshold"}
    if set(cfg) != expected:
        p.error("Selection config has unknown or missing settings")
    for key in expected:
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
    if args.num_classes is None:
        args.num_classes = 60 if args.split.endswith("60") else 120
    if args.export_selection and args.evaluate:
        p.error("Export and evaluation are separate operations")
    if bool(args.export_selection) != bool(args.selection_output):
        p.error("--export-selection and --selection-output must be used together")
    if not (args.export_selection or args.evaluate) and ((args.stage == "compact") != (args.subset is not None)):
        p.error("Compact training requires --subset; selection training does not accept --subset")
    if args.subset is not None:
        validate_spec(json.loads(args.subset.read_text(encoding="utf-8")))
    if min(args.window_size, args.max_persons, args.epochs, args.batch_size, args.test_batch_size, args.num_classes, args.prefetch_factor) < 1:
        p.error("Frame, person, epoch, class and batch counts must be positive")
    if args.num_workers < 0 or args.max_samples < 0:
        p.error("Workers/max-samples cannot be negative")
    if not isinstance(args.gate_warmup_epochs, int) or args.gate_warmup_epochs < 0 or not isinstance(args.gate_ramp_epochs, int) or args.gate_ramp_epochs < 1:
        p.error("Gate warmup must be nonnegative and ramp must be positive integers")
    numeric = (args.gate_sigma, args.gate_initial_mu, args.sparsity_weight, args.export_threshold,
               args.lr, args.warmup_start_lr, args.min_lr, args.weight_decay, args.momentum,
               args.max_grad_norm, args.log_interval)
    if not all(math.isfinite(v) for v in numeric) or args.gate_sigma <= 0 or args.sparsity_weight < 0 or not 0 <= args.export_threshold < 1:
        p.error("Invalid gate/loss/threshold settings; all numeric settings must be finite")
    if not 0 <= args.warmup_epochs < args.epochs or not 0 <= args.min_lr <= args.lr or not 0 < args.warmup_start_lr <= args.lr:
        p.error("Invalid cosine learning-rate schedule")
    if args.weight_decay < 0 or args.max_grad_norm < 0 or args.momentum < 0 or args.log_interval <= 0 or (args.nesterov and args.momentum <= 0):
        p.error("Invalid optimizer or logging settings")
    if args.save_dir is None:
        args.save_dir = ROOT / f"outputs/input_joint_{args.stage}_100_ntu60_xsub"
    args.compile = True
    return args


def build_model(args):
    subset = json.loads(args.subset.read_text(encoding="utf-8")) if args.subset else None
    return InputJointSelectionFusion(args.num_classes, stage=args.stage, subset=subset,
                                    gate_sigma=args.gate_sigma, gate_initial_mu=args.gate_initial_mu)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != InputJointSelectionFusion.ARCHITECTURE:
        raise ValueError("Use an input-joint-selection checkpoint, not original best weights")
    model = InputJointSelectionFusion(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def export_checkpoint(path, output, threshold):
    model, checkpoint = load_checkpoint(path)
    spec = model.export_selection(threshold)
    spec.update(source_checkpoint=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_epoch=checkpoint["epoch"], export_threshold=threshold,
                weights_transferred=False, selection_basis="training_loss_only; threshold fixed by CLI/config")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing selection: {output}")
    write_json(output, spec)
    print(f"Exported {spec['raw_input_count']} raw inputs ({len(spec['face_indices'])} face points retained): {output}", flush=True)
    return spec


def compile_model(model, mode):
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is mandatory but unavailable")
    torch._dynamo.config.suppress_errors = False
    if not hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        raise RuntimeError("Required-compile selection needs a PyTorch runtime supporting fail_on_recompile_limit_hit")
    # suppress_errors alone does not prevent eager fallback at the Dynamo
    # recompile limit. Fail explicitly rather than silently bypassing compile.
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    if hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
    print(f"compile required: mode={mode}; {COMPILE_SCOPE['scope']}; {COMPILE_SCOPE['eager_control']}", flush=True)
    return torch.compile(model, mode=mode)


def make_loader(args, model, phase):
    dataset = InputJointDataset(
        args.archive, input_indices=model.selection["input_indices"], split=phase,
        split_protocol=args.split, window_size=args.window_size, num_classes=args.num_classes,
        layout="rtmw_133", max_persons=args.max_persons, augment=False,
        temporal_compression=args.temporal_compression, max_samples=args.max_samples,
    )
    return DataLoader(dataset, batch_size=args.batch_size if phase == "train" else args.test_batch_size,
                      shuffle=phase == "train", drop_last=args.drop_last if phase == "train" else False,
                      num_workers=args.num_workers, pin_memory=str(args.device).startswith("cuda"),
                      persistent_workers=args.num_workers > 0,
                      prefetch_factor=args.prefetch_factor if args.num_workers else None)


def gate_weight(epoch, args):
    return args.sparsity_weight * min(1., max(0., (epoch - args.gate_warmup_epochs) / args.gate_ramp_epochs))


def run_epoch(compiled, loader, device, *, optimizer=None, scheduler=None, weight=0.,
              epoch=0, records=None, max_grad_norm=0., log_interval=10.):
    training = optimizer is not None
    compiled.train(training)
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    samples = 0
    start = last_log = time.monotonic()
    with torch.set_grad_enabled(training):
        for step, (x, labels, frame_mask) in enumerate(loader, 1):
            x, labels, frame_mask = (v.to(device, non_blocking=True) for v in (x, labels, frame_mask))
            if training:
                lr = scheduler.lr_at(scheduler.completed_updates)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
            logits, penalty = compiled(x, frame_mask, return_selection=True)
            ce = nn.functional.cross_entropy(logits, labels)
            loss = ce + weight * penalty if training else ce
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, step {step}")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(compiled.parameters(), max_grad_norm or float("inf"), error_if_nonfinite=True)
                optimizer.step()
                scheduler.completed_updates += 1
            count = labels.size(0)
            with torch.no_grad():
                top = logits.topk(min(5, logits.size(1)), dim=1).indices
                values = torch.stack((loss.detach(), ce.detach(), penalty.detach(),
                                      (top[:, 0] == labels).float().mean(),
                                      (top == labels[:, None]).any(1).float().mean()))
                totals += values * count
            samples += count
            now = time.monotonic()
            if now - last_log >= log_interval or step == len(loader):
                metrics = (totals / samples).tolist()
                row = dict(epoch=epoch, phase="train" if training else "val", step=step,
                           loss=metrics[0], classification_loss=metrics[1], sparsity_penalty=metrics[2],
                           top1=metrics[3], top5=metrics[4], elapsed_seconds=now - start,
                           lr=optimizer.param_groups[0]["lr"] if training else 0.)
                if records:
                    records.batch(row)
                print(f"{row['phase']} epoch={epoch} step={step}/{len(loader)} loss={row['loss']:.4f} top1={row['top1']:.4f}", flush=True)
                last_log = now
    if not samples:
        raise ValueError("Empty loader; reduce batch size or disable drop-last for a smoke run")
    loss, ce, penalty, top1, top5 = (totals / samples).tolist()
    return dict(loss=loss, classification_loss=ce, sparsity_penalty=penalty, top1=top1, top5=top5,
                samples=samples, samples_per_second=samples / (time.monotonic() - start))


def run(args):
    if args.export_selection:
        export_checkpoint(args.export_selection, args.selection_output, args.export_threshold)
        return
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no fallback")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if args.evaluate:
        model, checkpoint = load_checkpoint(args.evaluate)
        for key in ("archive", "split", "window_size", "temporal_compression", "max_persons", "num_classes"):
            # An explicit archive override is useful when evaluating on another host.
            if key != "archive" or not any(v == "--archive" or v.startswith("--archive=") for v in sys.argv[1:]):
                setattr(args, key, checkpoint["args"][key])
        model.to(device)
        compiled = compile_model(model, args.compile_mode)
        print(json.dumps(run_epoch(compiled, make_loader(args, model, "val"), device,
                                   log_interval=args.log_interval), indent=2), flush=True)
        return
    model = build_model(args).to(device)
    model_config = dict(num_classes=args.num_classes, stage=args.stage,
                        subset=model.selection if args.stage == "compact" else None,
                        gate_sigma=args.gate_sigma, gate_initial_mu=args.gate_initial_mu)
    directory = create_run_directory(args.save_dir)
    serialized_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    records = RunRecords(directory, serialized_args, ROOT)
    records.update_config(model_config=model_config, architecture=model.ARCHITECTURE,
                          initialization="random_from_scratch; subset manifest transfers indices only",
                          input_selection=model.selection, compile={**COMPILE_SCOPE, "mode": args.compile_mode},
                          reference={"baseline": "body-local-hand-ctr-wide-relative; original pooled 100-frame/100-epoch",
                                     "selection_method": "https://proceedings.mlr.press/v119/yamada20a.html",
                                     "adaptation": "global grouped Gaussian gates; mirrored hand pairs; face fixed; new skeleton adaptation"},
                          runtime={"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
                          preprocessing={"slice_before": "score normalization, frame validity, temporal compression and geometry",
                                         "face_raw_inputs": 68, "score_normalization": "auto over selected inputs only"})
    print(f"run={directory} stage={args.stage} raw_inputs={len(model.selection['input_indices'])}", flush=True)
    try:
        with mirror_console_to_file(directory / "console.log"):
            compiled = compile_model(model, args.compile_mode)
            if args.dry_run:
                x = torch.randn(2, 3, args.window_size, len(model.selection["input_indices"]), args.max_persons, device=device) * .1
                x[:, 2] = 1.
                frame_mask = torch.ones(2, args.window_size, device=device, dtype=torch.bool)
                logits, penalty = compiled(x, frame_mask, return_selection=True)
                (logits.square().mean() + args.sparsity_weight * penalty).backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise RuntimeError("Nonfinite compiled backward gradients")
                records.update_config(dry_run_result={"compiled_forward_backward": True, "shape": list(logits.shape)})
                records.finish("dry_run_passed")
                print("Compiled forward/backward passed", flush=True)
                return
            train, val = make_loader(args, model, "train"), make_loader(args, model, "val")
            scheduler = BatchWarmupCosine(epochs=args.epochs, batches_per_epoch=len(train),
                                         warmup_epochs=args.warmup_epochs, base_lr=args.lr,
                                         warmup_start_lr=args.warmup_start_lr, min_lr=args.min_lr)
            gate_params = list(model.gates.parameters()) if args.stage == "select" else []
            gate_ids = {id(p) for p in gate_params}
            groups = [{"params": [p for p in model.parameters() if id(p) not in gate_ids], "weight_decay": args.weight_decay}]
            if gate_params:
                groups.append({"params": gate_params, "weight_decay": 0.})
            optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
            records.update_config(dataset={"train": len(train.dataset), "val": len(val.dataset)},
                                  optimizer={"name": "SGD", "gate_weight_decay": 0., "backbone_weight_decay": args.weight_decay},
                                  lr_schedule=scheduler.state_dict())
            best, best_epoch = -1., 0
            for epoch in range(1, args.epochs + 1):
                if args.stage == "select":
                    model.gates.noise_scale.fill_(float(epoch > args.gate_warmup_epochs))
                weight = gate_weight(epoch, args) if args.stage == "select" else 0.
                tr = run_epoch(compiled, train, device, optimizer=optimizer, scheduler=scheduler,
                               weight=weight, epoch=epoch, records=records,
                               max_grad_norm=args.max_grad_norm, log_interval=args.log_interval)
                va = run_epoch(compiled, val, device, epoch=epoch, records=records, log_interval=args.log_interval)
                subset = model.export_selection(args.export_threshold)
                selection_row = {"epoch": epoch, "threshold": args.export_threshold, "raw_input_count": subset["raw_input_count"],
                                 "body_indices": subset["body_indices"], "hand_local_indices": subset["hand_local_indices"]}
                if args.stage == "select":
                    selection_row.update(gate_values=model.gates.deterministic().cpu().tolist(),
                                         expected_optional_count=float(model.gates.expected_optional_count().detach().cpu()))
                with (directory / "selection_history.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(selection_row) + "\n")
                is_best = va["top1"] > best
                if is_best:
                    best, best_epoch = va["top1"], epoch
                checkpoint = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  args=serialized_args, architecture=model.ARCHITECTURE, model_config=model_config,
                                  val_accuracy=va["top1"], best_accuracy=best, best_epoch=best_epoch,
                                  selection=subset, lr_schedule=scheduler.state_dict(),
                                  train_metrics=tr, val_metrics=va)
                for name in (["last"] + (["best"] if is_best else [])):
                    tmp = directory / f"{name}.pt.tmp"
                    torch.save(checkpoint, tmp)
                    tmp.replace(directory / f"{name}.pt")
                    write_json(directory / f"{name}_selection.json", subset)
                records.epoch({"epoch": epoch, "stage": args.stage, "sparsity_weight": weight,
                               "raw_input_count": subset["raw_input_count"],
                               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()},
                               "best_epoch": best_epoch, "best_val_acc": best})
                print(f"epoch={epoch} val={va['top1']:.4f} exported_inputs={subset['raw_input_count']} (includes face68)", flush=True)
            records.finish("finished")
    except BaseException as exc:
        records.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        records.close()
