"""Four-stage adaptive-width experiment, without official-val model selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

# Preserve the existing entry point's Linux CUDA/cuDNN bootstrap before torch.
from isaa.train import collate_rtmw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.body_local_fusion import BodyLocalHandCTRWideRelativeFusion
from isaa.models.body_local_full_gcn import BodyLocalFullGCNFusion
from isaa.models.class_adaptive_gcn import ClassAdaptiveGCN, ADAPTIVE_GCN_VARIANT, relative_cost
from isaa.utils.adaptive_width import (split_width_subjects, distillation_loss, calibrate_profile_bn,
                                       greedy_width_search, select_search_examples)
from isaa.utils.console_logging import mirror_console_to_file
from isaa.utils.experiment import RunRecords, create_run_directory, write_json, ctrgcn_learning_rate
from isaa.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = "outputs/body_local_hand_ctr_wide_crosshand_direction_zip/20261008-220235-1777a79b/best.pt"


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path, default=ROOT / "configs/class_adaptive_gcn.json")
    selected, _ = selector.parse_known_args(argv)
    defaults = json.loads(selected.config.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=selected.config)
    parser.add_argument("--stage", choices=("all", "supernet", "search", "route"), default="all")
    parser.add_argument("--experiment", choices=("class", "shared"), default="class")
    parser.add_argument("--archive", default="data/ntu60_skeletons_rtmw.zip")
    parser.add_argument("--split", choices=("xsub60", "xset60", "xsub120", "xset120"), default="xsub60")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER)
    parser.add_argument("--no-teacher", action="store_true", help="Explicit ablation: disable pretrained teacher")
    parser.add_argument("--init-checkpoint", help="Full GCN initialization or matching adaptive weights; no optimizer resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--max-persons", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-dir", default="outputs/class_adaptive_gcn_ntu60_xsub")
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"))
    parser.add_argument("--compile-backend", default="inductor", help="eager is useful for compile API checks on CPU")
    parser.add_argument("--compile-limit", type=int, default=64)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0004)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--console-only", action="store_true")
    parser.add_argument("--skip-official-evaluation", action="store_true", help="Save a calibrated model without opening official val")
    parser.add_argument("--stage-maxima", nargs=3, type=int, default=defaults["stage_maxima"])
    parser.add_argument("--group-size", type=int, default=defaults["group_size"])
    parser.add_argument("--router-hidden", type=int, default=defaults["router_hidden"])
    parser.add_argument("--topk", type=int, default=defaults["topk"])
    int_keys = ("supernet_epochs", "gate_epochs", "route_epochs", "search_steps", "search_positives",
                "search_negatives", "shared_samples_per_class", "min_class_samples", "bn_calibration_batches")
    float_keys = ("lr", "gate_lr", "route_lr", "distill_weight", "distill_temperature", "width_penalty",
                  "search_cost_weight", "search_accuracy_tolerance", "search_loss_tolerance",
                  "routing_accuracy_tolerance", "calibration_fraction", "confidence_threshold", "fallback_threshold")
    for key in int_keys + float_keys:
        parser.add_argument("--" + key.replace("_", "-"), type=int if key in int_keys else float, default=defaults[key])
    args = parser.parse_args(argv)
    args.config = str(args.config.resolve())
    args.num_classes = args.num_classes or (60 if args.split.endswith("60") else 120)
    args.model_variant, args.feature_mode = ADAPTIVE_GCN_VARIANT, "raw"
    for key in ("num_classes", "window_size", "max_persons", "batch_size", "test_batch_size", "grad_accum_steps",
                "prefetch_factor", "compile_limit", "search_positives", "search_negatives", "shared_samples_per_class",
                "min_class_samples", "bn_calibration_batches"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("supernet_epochs", "gate_epochs", "route_epochs", "search_steps", "num_workers", "max_samples"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    for key in float_keys + ("momentum", "weight_decay"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    if args.distill_temperature <= 0 or not 0 < args.calibration_fraction < 1 or not 0 < args.momentum < 1:
        parser.error("Invalid temperature, calibration fraction or momentum")
    if not 0 <= args.fallback_threshold <= args.confidence_threshold <= 1:
        parser.error("Routing thresholds require 0 <= fallback <= confidence <= 1")
    if any(getattr(args, k) > 1 for k in ("search_accuracy_tolerance", "routing_accuracy_tolerance")):
        parser.error("Accuracy tolerances must be fractions between zero and one")
    if args.stage in {"search", "route"} and not args.init_checkpoint and not args.dry_run:
        parser.error(f"--stage {args.stage} requires --init-checkpoint")
    if not args.dry_run and args.stage in {"all", "supernet"} and args.supernet_epochs < 1:
        parser.error("Supernet training requires at least one epoch")
    return args


def path_from_root(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def checkpoint_model(checkpoint):
    if checkpoint.get("architecture") != ClassAdaptiveGCN.ARCHITECTURE:
        raise ValueError("Expected a class-adaptive full-GCN checkpoint")
    config = dict(checkpoint["model_config"])
    config.pop("variant", None)
    model = ClassAdaptiveGCN(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.normalization_stats = checkpoint.get("normalization_stats", {})
    model.calibration_metadata = checkpoint.get("calibration_metadata", {})
    return model


def load_adaptive_checkpoint(path, device="cpu", *, compile=True, backend="inductor"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint_model(checkpoint).to(device).eval()
    model.enable_compile(compile, backend=backend)
    return model


def save_checkpoint(path, model, args, epoch, *, phase, metrics=None, **extra):
    values = {"epoch": epoch, "architecture": model.ARCHITECTURE, "model": model.state_dict(),
              "model_config": model.model_config(), "args": vars(args), "phase": phase,
              "normalization_stats": model.normalization_stats,
              "selection_metric": ("training_internal_label_guided_proposal_CE_plus_router_CE" if phase == "gates"
                                   else "training_internal_subject_selection_CE_plus_router_CE"),
              "selection_metrics": metrics, "val_accuracy": None,
              "calibration_metadata": getattr(model, "calibration_metadata", {}), **extra}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    temporary.replace(path)


def make_datasets(args):
    options = dict(split_protocol=args.split, window_size=args.window_size, num_joints=133,
                   num_classes=args.num_classes, layout="rtmw_133", max_persons=args.max_persons,
                   feature_mode="raw", max_samples=args.max_samples)
    archive = path_from_root(args.archive)
    fit_data = RTMWZipDataset(archive, split="train", augment=True,
                             augmentation_config={"random_temporal_crop": True}, **options)
    clean_data = RTMWZipDataset(archive, split="train", augment=False, **options)
    fit, search, selection, subjects = split_width_subjects(clean_data.members, args.calibration_fraction, args.seed)
    return fit_data, clean_data, fit, search, selection, subjects, options


def make_loader(data, args, device, *, training=False):
    from functools import partial
    return DataLoader(data, batch_size=args.batch_size if training else args.test_batch_size,
                      shuffle=training, drop_last=training and args.drop_last,
                      num_workers=args.num_workers, pin_memory=device.type == "cuda",
                      persistent_workers=args.num_workers > 0,
                      prefetch_factor=args.prefetch_factor if args.num_workers else None,
                      collate_fn=partial(collate_rtmw, feature_mode="raw"))


@torch.no_grad()
def collect_fixed(model, batches, device, profile, *, compiled=False):
    values, labels = [], []
    for x, y, mask in batches:
        logits = model.execute(x.to(device), mask.to(device), profile, compiled=compiled)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite fixed-width prediction")
        values.append(logits.cpu())
        labels.append(y.cpu())
    return torch.cat(values), torch.cat(labels)


@torch.no_grad()
def evaluate(model, loader, device, *, mode="adaptive"):
    model.eval()
    confusion = torch.zeros(model.num_classes, model.num_classes, dtype=torch.long)
    total = correct5 = fallback = router_correct = 0
    loss_sum = cost_sum = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for x, labels, mask in loader:
        x, mask, labels_device = x.to(device), mask.to(device), labels.to(device)
        output = model(x, mask, return_routing=True, mode=mode)
        logits = output["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite adaptive prediction")
        prediction = logits.argmax(1).cpu()
        confusion += torch.bincount(labels * model.num_classes + prediction,
                                    minlength=model.num_classes ** 2).reshape_as(confusion)
        total += len(labels)
        correct5 += int((logits.topk(min(5, model.num_classes), dim=1).indices == labels_device[:, None]).any(1).sum())
        loss_sum += float(F.cross_entropy(logits, labels_device, reduction="sum"))
        if mode == "adaptive":
            fallback += int(output["fallback"].sum())
            router_correct += int((output["router_logits"].argmax(1) == labels_device).sum())
            cost_sum += sum(relative_cost(row, model.stage_maxima) for row in output["widths"].cpu().tolist())
        else:
            profile = model.max_profile if mode == "max" else model.shared_widths.tolist()
            cost_sum += len(labels) * relative_cost(profile, model.stage_maxima)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not total:
        raise ValueError("Empty evaluation loader")
    support, correct = confusion.sum(1), confusion.diagonal()
    recall = correct.double() / support.clamp_min(1)
    precision = correct.double() / confusion.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"mode": mode, "samples": total, "correct": int(correct.sum()), "top1": float(correct.sum()) / total,
            "top5": correct5 / total, "loss": loss_sum / total,
            "macro_recall": float(recall[support > 0].mean()), "macro_f1": float(f1[support > 0].mean()),
            "fallback_rate": fallback / total, "router_top1": router_correct / total if mode == "adaptive" else None,
            "mean_conv_cost_proxy": cost_sum / total,
            "seconds_including_compile_and_loading": elapsed, "samples_per_second_including_compile_and_loading": total / elapsed,
            "confusion": confusion.tolist()}


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()


def backward_checked(loss, accumulation):
    if not bool(torch.isfinite(loss.detach())):
        raise RuntimeError("Nonfinite training loss before backward")
    (loss / accumulation).backward()


def prepare_banks(model, fit_batches, device, profiles):
    model.eval()
    model.normalization_stats = {}
    for profile in sorted(set(tuple(p) for p in profiles)):
        model.normalization_stats[model.profile_key(profile)] = calibrate_profile_bn(model, fit_batches, profile, device)


def train_phase(model, teacher, args, loader, selection_loader, fit_batches, records, save_dir, device,
                phase, epochs, base_lr, epoch_offset):
    if not epochs:
        return epoch_offset
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("width_logits", "router.")) if phase == "gates"
                                 else not name.startswith("width_logits"))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=base_lr,
                                momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    best = math.inf
    best_path = save_dir / (phase + "_best.pt")
    randomizer = random.Random(args.seed + epoch_offset)
    for local_epoch in range(epochs):
        print(f"{phase}: starting epoch {local_epoch + 1}/{epochs}; waiting for first data batch", flush=True)
        epoch = epoch_offset + local_epoch + 1
        lr = ctrgcn_learning_rate(local_epoch, base_lr, min(5, epochs),
                                 sorted(set((max(1, int(epochs * .6)), max(1, int(epochs * .85))))), .1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        if phase != "supernet":
            freeze_bn(model)
        total = correct = 0
        loss_sum = 0.0
        for step, (x, labels, mask) in enumerate(loader, 1):
            verbose = step == 1
            if verbose:
                print(f"{phase}: first batch loaded; teacher forward begins", flush=True)
            x, labels, mask = x.to(device), labels.to(device), mask.to(device)
            if (step - 1) % args.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            accumulation = min(args.grad_accum_steps, len(loader) - ((step - 1) // args.grad_accum_steps) * args.grad_accum_steps)
            with torch.no_grad():
                target = teacher(x, mask) if teacher is not None else None
            losses = []
            if phase == "supernet":
                random_profile = tuple(randomizer.randrange(1, cap // args.group_size + 1) * args.group_size
                                       for cap in model.stage_maxima * 3)
                detached_maximum = None
                for profile, compiled, update in ((model.max_profile, True, True),
                                                  (model.min_profile, True, False), (random_profile, False, False)):
                    if verbose:
                        print(f"{phase}: first batch forward widths={profile} compiled={args.compile and compiled}", flush=True)
                    logits = model.execute(x, mask, profile, compiled=compiled, update_bn=update)
                    if detached_maximum is None:
                        detached_maximum = logits.detach()
                        anchor = logits.detach()
                    reference = target if target is not None else detached_maximum
                    loss = distillation_loss(logits, labels, reference, args.distill_weight, args.distill_temperature) / 3
                    if verbose:
                        print(f"{phase}: first batch backward begins widths={profile}", flush=True)
                    backward_checked(loss, accumulation)
                    if verbose:
                        print(f"{phase}: first batch backward finished widths={profile}", flush=True)
                    losses.append(float(loss.detach()))
            elif phase == "gates":
                logits = model.execute(x, mask, model.max_profile, labels=labels)
                loss = distillation_loss(logits, labels, target, args.distill_weight, args.distill_temperature)
                loss = loss + args.width_penalty * model.expected_width_fraction(labels)
                backward_checked(loss, accumulation)
                anchor = logits.detach()
                losses.append(float(loss.detach()))
            else:
                output = model(x, mask, return_routing=True, mode="shared" if args.experiment == "shared" else "adaptive")
                logits = output["logits"]
                loss = distillation_loss(logits, labels, target, args.distill_weight, args.distill_temperature)
                backward_checked(loss, accumulation)
                anchor = logits.detach()
                losses.append(float(loss.detach()))
            route_loss = F.cross_entropy(model.route_logits(x, mask), labels)
            backward_checked(route_loss, accumulation)
            losses.append(float(route_loss.detach()))
            if not all(math.isfinite(value) for value in losses):
                raise RuntimeError(f"Nonfinite {phase} loss, epoch={epoch}, step={step}")
            if step % args.grad_accum_steps == 0 or step == len(loader):
                optimizer.step()
            count = len(labels)
            total += count
            batch_correct = int((anchor.argmax(1) == labels).sum())
            correct += batch_correct
            loss_sum += sum(losses) * count
            records.batch({"epoch": epoch, "phase": phase, "step": step, "loss": sum(losses),
                           "top1": batch_correct / count, "samples": count, "lr": lr})
            if step == 1 or step % 100 == 0 or step == len(loader):
                print(f"{phase} epoch={local_epoch + 1}/{epochs} step={step}/{len(loader)} lr={lr:.6g} loss={loss_sum / total:.4f}", flush=True)
        model.eval()
        if phase == "route":
            prepare_banks(model, fit_batches, device, [*model.class_widths.tolist(), model.shared_widths.tolist(), model.max_profile])
            mode = "shared" if args.experiment == "shared" else "adaptive"
        else:
            mode = "max"  # Label-guided soft-gate accuracy is deliberately not a selection metric.
            model.normalization_stats = {}
        metrics = evaluate(model, selection_loader, device, mode=mode)
        with torch.no_grad():
            route_total = route_loss_sum = 0
            for x, y, mask in selection_loader:
                route_loss_sum += float(F.cross_entropy(model.route_logits(x.to(device), mask.to(device)), y.to(device), reduction="sum"))
                route_total += len(y)
        selection_loss = metrics["loss"] + route_loss_sum / route_total
        if phase == "gates":
            # Label-guided CE selects proposals only, never deployment accuracy.
            with torch.no_grad():
                proposal_loss = 0.0
                for x, y, mask in selection_loader:
                    y = y.to(device)
                    logits = model.execute(x.to(device), mask.to(device), model.max_profile, labels=y)
                    proposal_loss += float(F.cross_entropy(logits, y, reduction="sum"))
            selection_loss = proposal_loss / route_total + route_loss_sum / route_total
            metrics["label_guided_proposal_loss"] = proposal_loss / route_total
        if selection_loss < best:
            best = selection_loss
            best_epoch = epoch
            save_checkpoint(best_path, model, args, epoch, phase=phase, metrics=metrics,
                            selection_loss=selection_loss, best_epoch=epoch)
        save_checkpoint(save_dir / "last.pt", model, args, epoch, phase=phase, metrics=metrics, best_epoch=best_epoch)
        records.epoch({"epoch": epoch, "phase": phase, "lr": lr, "train_loss": loss_sum / total,
                       "train_top1": correct / total, "selection_loss": selection_loss,
                       "selection_top1": metrics["top1"], "best_val_acc": None, "best_epoch": best_epoch})
        print(f"{phase} epoch={epoch} internal_selection_top1={metrics['top1']:.4%} selection_loss={selection_loss:.4f}", flush=True)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.normalization_stats = checkpoint.get("normalization_stats", {})
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return epoch_offset + epochs


def cache_examples(dataset, indices, args, device):
    # Bound memory per search, rather than caching the entire archive.
    return list(make_loader(Subset(dataset, indices), args, device))


def search_profiles(model, args, clean, search_indices, fit_batches, device, save_dir):
    model.eval()
    model.normalization_stats = {}
    maximum_bank = calibrate_profile_bn(model, fit_batches, model.max_profile, device)
    model.load_bn(maximum_bank)
    search_loader = make_loader(Subset(clean, search_indices), args, device)
    maximum_logits, labels = collect_fixed(model, search_loader, device, model.max_profile)
    shared_indices = []
    for c in range(args.num_classes):
        positions = (labels == c).nonzero().flatten().tolist()
        random.Random(args.seed + c).shuffle(positions)
        shared_indices.extend(search_indices[i] for i in positions[:args.shared_samples_per_class])
    proposals = model.proposed_profiles().cpu().tolist()
    def run(indices, target, proposed):
        batches = cache_examples(clean, indices, args, device)
        def predict(profile):
            bank = maximum_bank if profile == model.max_profile else calibrate_profile_bn(model, fit_batches, profile, device)
            model.load_bn(bank)
            result = collect_fixed(model, batches, device, profile)
            model.load_bn(maximum_bank)
            return result
        return greedy_width_search(predict, model.max_profile, target=target, proposed=proposed,
                                   group_size=args.group_size, max_steps=args.search_steps,
                                   cost_weight=args.search_cost_weight, accuracy_tolerance=args.search_accuracy_tolerance,
                                   loss_tolerance=args.search_loss_tolerance, maxima=model.stage_maxima,
                                   progress=lambda step, profile, metrics, tested: print(
                                       f"search class={target} step={step} widths={profile} loss={metrics['loss']:.4f} cost_proxy={metrics['cost_proxy']:.3f} tested={tested}", flush=True))
    print("Searching a shared-width control on training-internal subjects", flush=True)
    shared = run(sorted(shared_indices), None, None)
    write_json(save_dir / "shared_search.json", shared)
    rows = []
    for c in range(args.num_classes):
        print(f"Searching class={c + 1}/{args.num_classes}", flush=True)
        count = int((labels == c).sum())
        if args.experiment == "shared" or count < args.min_class_samples:
            result = {"profile": shared["profile"], "search": "shared_control" if args.experiment == "shared" else "insufficient_positive_examples"}
        else:
            positions = select_search_examples(labels, maximum_logits, c, positives=args.search_positives,
                                               negatives=args.search_negatives, seed=args.seed)
            result = run([search_indices[i] for i in positions], c, proposals[c])
        rows.append({"class_index": c, "action_id": f"A{c + 1:03}", "available_search_positives": count, **result})
        write_json(save_dir / "class_width_search.json", {"partition": "train_internal_search_subjects",
                   "official_validation_used": False, "classes": rows, "complete": c == args.num_classes - 1})
    model.set_profiles([row["profile"] for row in rows], shared["profile"])
    prepare_banks(model, fit_batches, device, [*model.class_widths.tolist(), model.shared_widths.tolist(), model.max_profile])
    export_width_table(model, save_dir)
    return rows


def export_width_table(model, save_dir):
    rows = []
    for c, profile in enumerate(model.class_widths.cpu().tolist()):
        rows.append({"class_index": c, "action_id": f"A{c + 1:03}",
                     "body": profile[:3], "hand": profile[3:6], "face": profile[6:],
                     "conv_cost_proxy": relative_cost(profile, model.stage_maxima)})
    write_json(save_dir / "class_widths.json", {"stage_layers": [4, 3, 3], "group_size": model.group_size,
               "stage_maxima": list(model.stage_maxima), "shared_widths": model.shared_widths.tolist(), "classes": rows})
    with (save_dir / "class_widths.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["action_id", *[f"{branch}_stage{s}" for branch in ("body", "hand", "face") for s in (1, 2, 3)], "conv_cost_proxy"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"action_id": row["action_id"], "conv_cost_proxy": row["conv_cost_proxy"],
                             **{f"{branch}_stage{s + 1}": row[branch][s] for branch in ("body", "hand", "face") for s in range(3)}})


@torch.no_grad()
def calibrate_routing(model, loader, args, device, save_dir):
    model.eval()
    values, targets = [], []
    for x, labels, mask in loader:
        values.append(model.route_logits(x.to(device), mask.to(device)).cpu())
        targets.append(labels)
    logits, labels = torch.cat(values), torch.cat(targets)
    temperatures = (.5, .75, 1., 1.5, 2., 3.)
    temperature = min(temperatures, key=lambda t: float(F.cross_entropy(logits / t, labels)))
    model.router_temperature.fill_(temperature)
    shared = evaluate(model, loader, device, mode="shared")
    candidates = []
    choices = [(args.confidence_threshold, args.fallback_threshold), (.7, .3), (.8, .45), (.9, .6), (1., 1.)]
    for confidence, fallback in sorted(set(choices)):
        model.confidence_threshold, model.fallback_threshold = confidence, fallback
        metrics = evaluate(model, loader, device)
        candidates.append({"confidence_threshold": confidence, "fallback_threshold": fallback, **metrics})
    feasible = [row for row in candidates if row["top1"] >= shared["top1"] - args.routing_accuracy_tolerance]
    winner = min(feasible or candidates, key=lambda r: (r["loss"] + args.search_cost_weight * r["mean_conv_cost_proxy"], -r["top1"]))
    model.confidence_threshold, model.fallback_threshold = winner["confidence_threshold"], winner["fallback_threshold"]
    write_json(save_dir / "routing_calibration.json", {"partition": "train_internal_selection_subjects",
               "temperature": temperature, "shared": shared, "candidates": candidates, "selected": winner,
               "official_validation_used": False})


def export_branches(model, save_dir):
    directory = save_dir / "branches"
    directory.mkdir(exist_ok=True)
    for branch, encoder in (("body", model.body_encoder), ("hand", model.hand_encoder), ("face", model.face_encoder)):
        offset = {"body": 0, "hand": 3, "face": 6}[branch]
        torch.save({"kind": "st" if branch == "face" else "ctr", "weights": encoder.state_dict(),
                    "branch": branch, "in_channels": encoder.in_channels, "strides": encoder.strides,
                    "stage_maxima": model.stage_maxima, "stage_layers": (4, 3, 3),
                    "class_stage_widths": model.class_widths[:, offset:offset + 3].cpu(),
                    "shared_stage_widths": model.shared_widths[offset:offset + 3].cpu(),
                    "normalization_stats": {model.profile_key(tuple(map(int, profile.split(',')))[offset:offset + 3]): {name.removeprefix(branch + "_encoder."): values
                        for name, values in bank.items() if name.startswith(branch + "_encoder.")}
                        for profile, bank in model.normalization_stats.items()}}, directory / (branch + "_gcn.pt"))


def run(args, directory, records, device):
    seed_everything(args.seed)
    model = ClassAdaptiveGCN(args.num_classes, stage_maxima=args.stage_maxima, group_size=args.group_size,
                             router_hidden=args.router_hidden, confidence_threshold=args.confidence_threshold,
                             fallback_threshold=args.fallback_threshold, topk=args.topk)
    initialization = None
    if args.init_checkpoint:
        initialization = torch.load(path_from_root(args.init_checkpoint), map_location="cpu", weights_only=False)
        if initialization.get("architecture") == ClassAdaptiveGCN.ARCHITECTURE:
            model = checkpoint_model(initialization)
            if model.num_classes != args.num_classes or model.stage_maxima != tuple(args.stage_maxima) or model.group_size != args.group_size:
                raise ValueError("Checkpoint class count/maxima/group size must match the selected config")
        elif initialization.get("architecture") == BodyLocalFullGCNFusion.ARCHITECTURE:
            base_keys = BodyLocalFullGCNFusion(args.num_classes).state_dict().keys()
            if set(initialization["model"]) != set(base_keys):
                raise ValueError("Full-GCN initialization contains unexpected tensors")
            result = model.load_state_dict(initialization["model"], strict=False)
            if result.unexpected_keys or any(not k.startswith(("router.", "width_logits", "class_widths", "shared_widths", "profiles_ready", "router_temperature")) for k in result.missing_keys):
                raise ValueError(f"Invalid full-GCN initialization: {result}")
        else:
            raise ValueError("Initialization needs full-GCN or adaptive-GCN weights; the best model is a teacher")
    model.to(device)
    model.enable_compile(args.compile, mode=args.compile_mode, backend=args.compile_backend, limit=args.compile_limit)
    records.update_config(architecture=model.ARCHITECTURE, model_config=model.model_config(),
                          runtime={"torch": torch.__version__, "device": str(device), "cuda": torch.version.cuda},
                          compile={"enabled": args.compile, "backend": args.compile_backend, "mode": args.compile_mode,
                                   "scope": "individual GCN blocks; preprocessing, gates, routing and fusion eager",
                                   "cudagraphs": False,
                                   "random_training_and_search": "eager", "profile_cache_limit": args.compile_limit})
    print(f"compile={args.compile} backend={args.compile_backend} mode={args.compile_mode} cudagraphs=False; "
          "fixed GCN blocks compile, exploratory widths/search are eager", flush=True)
    if args.dry_run:
        x = torch.randn(2, 3, min(args.window_size, 9), 133, args.max_persons, device=device) * .01
        x[:, 2] = 1
        mask = torch.ones(2, x.shape[2], device=device, dtype=torch.bool)
        labels = torch.tensor([0, args.num_classes - 1], device=device)
        model.eval()
        full = model.execute(x, mask, model.max_profile)
        narrow = model.execute(x, mask, model.min_profile)
        gated = model.execute(x, mask, model.max_profile, labels=labels)
        loss = F.cross_entropy(gated, labels) + F.cross_entropy(model.route_logits(x, mask), labels)
        loss.backward()
        if not torch.isfinite(loss) or model.width_logits.grad is None:
            raise RuntimeError("Dry-run gate/route gradients failed")
        model.set_profiles([model.min_profile] * args.num_classes, model.max_profile)
        output = model(x, mask, return_routing=True)
        save_checkpoint(directory / "dry_run.pt", model, args, 0, phase="dry_run")
        restored = checkpoint_model(torch.load(directory / "dry_run.pt", map_location="cpu", weights_only=False)).to(device).eval()
        torch.testing.assert_close(restored(x, mask), output["logits"])
        print(f"dry-run full={tuple(full.shape)} narrow={tuple(narrow.shape)} gate_grad={float(model.width_logits.grad.abs().sum()):.6g} roundtrip=passed", flush=True)
        return
    teacher = None
    print("setup: loading teacher and datasets", flush=True)
    if not args.no_teacher and args.stage != "search":
        teacher_path = path_from_root(args.teacher_checkpoint)
        if not teacher_path.is_file():
            raise FileNotFoundError(f"Teacher missing: {teacher_path}. Set --teacher-checkpoint to the 93.08% best.pt; --no-teacher is an explicit ablation.")
        checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != BodyLocalHandCTRWideRelativeFusion.ARCHITECTURE:
            raise ValueError("Teacher must be the original best wide-relative model")
        if checkpoint["args"]["num_classes"] != args.num_classes or checkpoint["args"]["split"] != args.split:
            raise ValueError("Teacher class count and split protocol must match the student")
        teacher = BodyLocalHandCTRWideRelativeFusion(args.num_classes).to(device).eval()
        teacher.load_state_dict(checkpoint["model"], strict=True)
        teacher.requires_grad_(False)
    fit_data, clean, fit, search, selection, subjects, dataset_options = make_datasets(args)
    print(f"setup: datasets ready fit={len(fit)} search={len(search)} selection={len(selection)}; caching BN fitting batches", flush=True)
    fit_loader = make_loader(Subset(fit_data, fit), args, device, training=True)
    selection_loader = make_loader(Subset(clean, selection), args, device)
    if not len(fit_loader):
        raise ValueError("Empty fitting loader; lower batch size or pass --no-drop-last")
    bn_indices = list(fit)
    random.Random(args.seed + 50).shuffle(bn_indices)
    bn_indices = bn_indices[:args.bn_calibration_batches * args.test_batch_size]
    fit_batches = cache_examples(clean, bn_indices, args, device)
    print(f"setup: BN fitting cache ready batches={len(fit_batches)}", flush=True)
    metadata = {"partition": "train_internal_three_way_subject_split", **subjects,
                "fit_samples": len(fit), "search_samples": len(search), "selection_samples": len(selection),
                "official_validation_used_for_selection": False,
                "teacher_prior_holdout_exposure": "not_excluded_pretrained_on_full_training" if teacher is not None else "none",
                "student_initial_checkpoint_prior_exposure": "not_verified" if initialization is not None else "none_from_scratch",
                "teacher_checkpoint": args.teacher_checkpoint if teacher is not None else None,
                "fit_members": [clean.members[i] for i in fit], "search_members": [clean.members[i] for i in search],
                "selection_members": [clean.members[i] for i in selection]}
    if initialization and initialization.get("architecture") == ClassAdaptiveGCN.ARCHITECTURE:
        old = initialization.get("calibration_metadata", {})
        for key in ("fit_members", "search_members", "selection_members"):
            if old.get(key) != metadata[key]:
                raise ValueError("Adaptive checkpoint must retain exactly the same fitting/search/selection partition")
        metadata["student_initial_checkpoint_prior_exposure"] = old.get("student_initial_checkpoint_prior_exposure")
        metadata["teacher_prior_holdout_exposure"] = old.get("teacher_prior_holdout_exposure", metadata["teacher_prior_holdout_exposure"])
        metadata["teacher_checkpoint"] = old.get("teacher_checkpoint", metadata["teacher_checkpoint"])
    model.calibration_metadata = metadata
    write_json(directory / "width_partition.json", metadata)
    records.update_config(width_partition={k: v for k, v in metadata.items() if not k.endswith("members")})
    offset = 0
    if args.stage in {"all", "supernet"}:
        offset = train_phase(model, teacher, args, fit_loader, selection_loader, fit_batches, records, directory,
                             device, "supernet", args.supernet_epochs, args.lr, offset)
        offset = train_phase(model, teacher, args, fit_loader, selection_loader, fit_batches, records, directory,
                             device, "gates", args.gate_epochs, args.gate_lr, offset)
        if args.stage == "supernet":
            save_checkpoint(directory / "supernet.pt", model, args, offset, phase="supernet_finished")
            return
    if args.stage in {"all", "search"}:
        search_profiles(model, args, clean, search, fit_batches, device, directory)
        save_checkpoint(directory / "searched.pt", model, args, offset, phase="search")
        if args.stage == "search":
            return
    if not bool(model.profiles_ready):
        raise ValueError("Route training requires searched.pt with a complete class width table")
    offset = train_phase(model, teacher, args, fit_loader, selection_loader, fit_batches, records, directory,
                         device, "route", args.route_epochs, args.route_lr, offset)
    model.eval()
    prepare_banks(model, fit_batches, device, [*model.class_widths.tolist(), model.shared_widths.tolist(), model.max_profile])
    calibrate_routing(model, selection_loader, args, device, directory)
    if args.experiment == "shared":
        model.confidence_threshold = model.fallback_threshold = 1.0
    export_width_table(model, directory)
    export_branches(model, directory)
    save_checkpoint(directory / "best.pt", model, args, offset, phase="final")
    if not args.skip_official_evaluation:
        # Construct the official partition only after weights/configs/thresholds are fixed.
        val = RTMWZipDataset(path_from_root(args.archive), split="val", augment=False, **dataset_options)
        loader = make_loader(val, args, device)
        metrics = {mode: evaluate(model, loader, device, mode=mode) for mode in ("max", "shared", "adaptive")}
        write_json(directory / "evaluation.json", metrics)
        save_checkpoint(directory / "best.pt", model, args, offset, phase="final",
                        val_accuracy=metrics["adaptive"]["top1"], val_metrics=metrics)
        print(f"official Top1: max={metrics['max']['top1']:.4%} shared={metrics['shared']['top1']:.4%} adaptive={metrics['adaptive']['top1']:.4%}", flush=True)
    print(f"Saved calibrated model, class width table and independent branches: {directory}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use the Ubuntu CUDA environment or --device cpu for checks")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    directory = create_run_directory(path_from_root(args.save_dir))
    context = nullcontext() if args.console_only else mirror_console_to_file(directory / "console.log")
    with context:
        records = RunRecords(directory, vars(args), ROOT)
        # The shared recorder predates this standalone entry point.
        from zipfile import ZipFile
        with ZipFile(directory / "source.zip", "a") as archive:
            archive.write(ROOT / "train_class_adaptive_gcn.py", "train_class_adaptive_gcn.py")
            archive.write(args.config, "configs/class_adaptive_gcn.json")
        try:
            run(args, directory, records, device)
        except BaseException as exc:
            records.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", repr(exc))
            raise
        else:
            records.finish("dry_run_completed" if args.dry_run else "completed")
        finally:
            records.close()


if __name__ == "__main__":
    main()
