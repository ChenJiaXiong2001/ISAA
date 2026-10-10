"""Train auxiliary exits, estimate hand requirements, then evaluate hard routing."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.class_hand_routed import BodyLocalClassHandRoutedFusion
from isaa.utils.experiment import ctrgcn_learning_rate, write_json
from isaa.utils.hand_requirements import build_hand_requirements, split_routing_calibration


def auxiliary_loss(output, labels, *, body_weight=1.0, no_hand_weight=1.0,
                   distill_weight=0.5, temperature=2.0):
    full = output["logits"]
    loss = F.cross_entropy(full, labels)
    teacher = (full.detach() / temperature).softmax(-1)
    for key, weight in (("body_logits", body_weight), ("no_hand_logits", no_hand_weight)):
        logits = output[key]
        loss = loss + weight * F.cross_entropy(logits, labels)
        if distill_weight:
            loss = loss + distill_weight * temperature ** 2 * F.kl_div(
                (logits / temperature).log_softmax(-1), teacher, reduction="batchmean")
    return loss


def collect_predictions(model, loader, device):
    model.eval()
    values = {key: [] for key in ("logits", "body_logits", "no_hand_logits", "labels")}
    with torch.inference_mode():
        for x, labels, mask in loader:
            output = model(x.to(device), mask.to(device), hand_mode="all", return_auxiliary=True)
            for key in ("logits", "body_logits", "no_hand_logits"):
                if not torch.isfinite(output[key]).all():
                    raise RuntimeError(f"Nonfinite calibration {key}")
                values[key].append(output[key].cpu())
            values["labels"].append(labels.cpu())
    return {key: torch.cat(items) for key, items in values.items()}


def evaluate_routing(model, loader, device):
    model.eval()
    total = correct = correct5 = called = 0
    loss_sum = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for x, labels, mask in loader:
            labels = labels.to(device)
            output = model(x.to(device), mask.to(device), return_routing=True)
            logits = output["logits"]
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite adaptive logits")
            total += len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            correct5 += int((logits.topk(min(5, logits.shape[1]), dim=1).indices == labels[:, None]).any(1).sum())
            called += int(output["hand_called"].sum())
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
    elapsed = time.perf_counter() - started
    return {"loss": loss_sum / total, "top1": correct / total, "top5": correct5 / total,
            "samples": total, "correct": correct, "hand_called": called,
            "hand_call_rate": called / total, "seconds": elapsed,
            "samples_per_second": total / max(elapsed, 1e-9)}


def _save_checkpoint(path, model, args, epoch, **extra):
    checkpoint = {"epoch": epoch, "model": model.state_dict(), "args": vars(args),
                  "architecture": model.ARCHITECTURE,
                  "model_config": {"variant": args.model_variant, "num_classes": args.num_classes,
                                   "hand_threshold": model.hand_threshold,
                                   "confidence_threshold": model.confidence_threshold,
                                   "body_temperature": model.body_temperature}, **extra}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def run_class_hand_training(args, save_dir, records, device):
    from isaa.train import PROJECT_ROOT, collate_rtmw
    model = BodyLocalClassHandRoutedFusion(
        args.num_classes, args.hand_route_threshold,
        args.body_confidence_threshold, args.body_probability_temperature)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.initialize_from_baseline(checkpoint)
    if args.class_hand_freeze_backbone and not args.init_checkpoint and not args.dry_run:
        raise ValueError("Frozen-backbone training requires --init-checkpoint; use --no-class-hand-freeze-backbone to train from scratch")
    model.freeze_backbone(args.class_hand_freeze_backbone)
    model.to(device)
    records.update_config(
        architecture=model.ARCHITECTURE,
        model_config={"variant": args.model_variant, "num_classes": args.num_classes,
                      "hand_threshold": args.hand_route_threshold,
                      "confidence_threshold": args.body_confidence_threshold,
                      "body_temperature": args.body_probability_temperature},
        runtime={"torch": torch.__version__, "cuda": torch.version.cuda,
                 "device": str(device), "cudnn": torch.backends.cudnn.version()},
        compile={"enabled": False, "reason": "Per-sample hard branch dispatch uses eager execution"},
        parameters={"total": sum(p.numel() for p in model.parameters()),
                    "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    print("class-hand routing: eager execution; auxiliary heads receive direct classification supervision", flush=True)
    if args.dry_run:
        model.eval()
        x = torch.randn(2, 3, 8, 133, args.max_persons, device=device) * 0.01
        x[:, 2] = 1
        with torch.inference_mode():
            result = model(x, return_routing=True)
        print(f"input={tuple(x.shape)} logits={tuple(result['logits'].shape)} "
              f"hand_called={int(result['hand_called'].sum())}/2 "
              f"finite={bool(torch.isfinite(result['logits']).all())}", flush=True)
        return
    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = PROJECT_ROOT / archive
    dataset_options = dict(split_protocol=args.split, window_size=args.window_size,
                           num_joints=133, num_classes=args.num_classes, layout="rtmw_133",
                           max_persons=args.max_persons, feature_mode="raw", max_samples=args.max_samples)
    train_data = RTMWZipDataset(archive, split="train", augment=True,
                               augmentation_config={"random_temporal_crop": True}, **dataset_options)
    clean_train = RTMWZipDataset(archive, split="train", augment=False, **dataset_options)
    fit, calibration, subjects = split_routing_calibration(
        train_data.members, args.routing_calibration_fraction, args.seed)
    fit_data, calibration_data = Subset(train_data, fit), Subset(clean_train, calibration)
    collate = partial(collate_rtmw, feature_mode="raw")
    def loader(dataset, training=False):
        return DataLoader(dataset, batch_size=args.batch_size if training else args.test_batch_size,
                          shuffle=training, drop_last=training and args.drop_last,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.num_workers > 0,
                          prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                          collate_fn=collate)
    fit_loader, calibration_loader = loader(fit_data, True), loader(calibration_data)
    if not len(fit_loader):
        raise ValueError("Empty fitting loader; lower batch size or use --no-drop-last")
    metadata = {
        "partition": "train_internal_subject_holdout", "split_protocol": args.split,
        "calibration_subjects": subjects, "fit_samples": len(fit), "calibration_samples": len(calibration),
        "official_validation_used_for_requirements": False,
        "initial_checkpoint": args.init_checkpoint,
        "initial_backbone_calibration_exposure": "not_verified" if args.init_checkpoint else "excluded",
        "estimate_scope": "auxiliary_heads_holdout" if args.init_checkpoint else "all_paths_holdout",
        "window_size": args.window_size, "max_persons": args.max_persons,
        "archive": str(archive.resolve()), "archive_bytes": archive.stat().st_size,
    }
    records.update_config(routing_calibration=metadata, backbone_frozen=args.class_hand_freeze_backbone,
                          archive=str(archive.resolve()), dataset={
                              "train": {"samples": len(fit), "batches": len(fit_loader)},
                              "calibration": {"samples": len(calibration), "batches": len(calibration_loader)}})
    write_json(save_dir / "routing_partition.json", {**metadata,
               "fit_members_sha256": hashlib.sha256("\n".join(train_data.members[i] for i in fit).encode()).hexdigest(),
               "calibration_members": [clean_train.members[i] for i in calibration]})
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=args.lr, momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = 0
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        lr = ctrgcn_learning_rate(epoch - 1, args.lr, args.warmup_epochs, args.lr_steps, args.lr_decay)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for step, (x, labels, mask) in enumerate(fit_loader, 1):
            x, labels, mask = x.to(device), labels.to(device), mask.to(device)
            if (step - 1) % args.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            output = model(x, mask, hand_mode="all", return_auxiliary=True)
            loss = auxiliary_loss(output, labels, body_weight=args.body_loss_weight,
                                  no_hand_weight=args.no_hand_loss_weight,
                                  distill_weight=args.routing_distill_weight,
                                  temperature=args.routing_distill_temperature)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, step {step}")
            group_size = min(args.grad_accum_steps, len(fit_loader) - (step - 1) // args.grad_accum_steps * args.grad_accum_steps)
            (loss / group_size).backward()
            if step % args.grad_accum_steps == 0 or step == len(fit_loader):
                optimizer.step()
            count = len(labels)
            batch_correct = int((output["logits"].argmax(1) == labels).sum())
            total += count
            correct += batch_correct
            loss_sum += float(loss.detach()) * count
            records.batch({"epoch": epoch, "phase": "train", "step": step,
                           "loss": float(loss.detach()), "top1": batch_correct / count,
                           "samples": count, "lr": lr})
            if step == 1 or step % 100 == 0 or step == len(fit_loader):
                print(f"heads epoch={epoch}/{args.epochs} step={step}/{len(fit_loader)} lr={lr:.6g} loss={loss_sum/total:.4f}", flush=True)
        predictions = collect_predictions(model, calibration_loader, device)
        labels = predictions["labels"]
        cal_loss = float(F.cross_entropy(predictions["body_logits"], labels)
                         + F.cross_entropy(predictions["no_hand_logits"], labels))
        cal_top1 = float((predictions["logits"].argmax(1) == labels).float().mean())
        if cal_loss < best_loss:
            best_loss, best_epoch = cal_loss, epoch
            _save_checkpoint(save_dir / "heads_best.pt", model, args, epoch,
                             selection_metric="calibration_body_plus_no_hand_CE", calibration_loss=cal_loss)
        _save_checkpoint(save_dir / "last.pt", model, args, epoch,
                         best_epoch=best_epoch, calibration_loss=cal_loss)
        records.epoch({"epoch": epoch, "stage": "class_hand_auxiliary_heads", "lr": lr,
                       "train_loss": loss_sum / total, "train_top1": correct / total,
                       "calibration_loss": cal_loss, "calibration_top1": cal_top1,
                       "best_val_acc": None, "best_epoch": best_epoch,
                       "total_elapsed_seconds": time.perf_counter() - training_started})
        print(f"epoch={epoch} calibration_head_loss={cal_loss:.4f} best_epoch={best_epoch}", flush=True)
    best = torch.load(save_dir / "heads_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    predictions = collect_predictions(model, calibration_loader, device)
    table = build_hand_requirements(
        predictions["logits"], predictions["no_hand_logits"], predictions["labels"],
        margin=args.hand_requirement_margin, temperature=args.hand_requirement_temperature,
        min_samples=args.hand_requirement_min_samples, metadata=metadata)
    write_json(save_dir / "hand_requirement.json", table)
    with (save_dir / "hand_requirement.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table["classes"][0]))
        writer.writeheader()
        writer.writerows(table["classes"])
    model.set_hand_requirements(table)
    # Only now create/read the official validation partition.
    val_data = RTMWZipDataset(archive, split="val", augment=False, **dataset_options)
    val_loader = loader(val_data)
    metrics = evaluate_routing(model, val_loader, device)
    records.update_config(dataset={"train": {"samples": len(fit)},
                                   "calibration": {"samples": len(calibration)},
                                   "val": {"samples": len(val_data), "batches": len(val_loader)}})
    records.state.update(best_epoch=best_epoch, best_val_acc=metrics["top1"])
    write_json(save_dir / "evaluation.json", metrics)
    _save_checkpoint(save_dir / "best.pt", model, args, best_epoch,
                     best_epoch=best_epoch, val_accuracy=metrics["top1"], best_accuracy=metrics["top1"],
                     val_metrics=metrics, val_loss=metrics["loss"], hand_requirement_table=table,
                     selection_metric="calibration_body_plus_no_hand_CE")
    print(f"best epoch={best_epoch} Top1={metrics['top1']:.2%} Top5={metrics['top5']:.2%} "
          f"hand_call_rate={metrics['hand_call_rate']:.2%} checkpoints={save_dir}", flush=True)
