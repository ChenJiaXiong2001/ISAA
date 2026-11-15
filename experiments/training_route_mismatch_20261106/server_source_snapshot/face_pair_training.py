"""Scratch training of independent HF/BF heads and internal-holdout routing."""
from __future__ import annotations

import csv
import hashlib
from functools import partial
from pathlib import Path
import re
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from isaa.data.rtmw_zip_dataset import RTMWZipDataset
from isaa.models.face_pair_fusion import (
    FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT, FACE_PAIR_HAND_MOTION_VARIANT,
    FACE_PAIR_RAW_XY_VARIANT, FACE_PAIR_VARIANT,
    FacePairFusion, FacePairHandFaceGeometryFusion, FacePairHandMotionFusion, FacePairRawXYFusion,
)
from isaa.utils.batch_cosine import BatchWarmupCosine
from isaa.utils.experiment import write_json
from isaa.utils.hand_requirements import split_routing_calibration


COMPILE_SCOPE = {
    "enabled": True, "required": True, "backend": "inductor",
    "suppress_errors": False, "fail_on_recompile_limit_hit": True,
    "whole_model_eager_fallback": False, "triton_cudagraphs": False,
    "scope": "body/hand encoders, masked pooling, pair concatenation, both classifiers, probability fusion",
    "inherited_eager_boundary": "face LocalSTBlock.forward (unchanged baseline boundary)",
    "outside_forward": "CPU accuracy counting and routing buffer update between epochs",
}


def compile_scope(args):
    scope = dict(COMPILE_SCOPE)
    if getattr(args, "model_variant", None) == FACE_PAIR_RAW_XY_VARIANT:
        scope["scope"] += "; expanded 8-channel hand input and 5-channel face input block"
        scope["inherited_eager_boundary"] = "face LocalSTBlock blocks 1..3; expanded input block 0 compiled"
    elif getattr(args, "model_variant", None) == FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT:
        scope["scope"] += "; explicit hand-face geometry and hand temporal-difference channels"
        scope["inherited_eager_boundary"] = "face LocalSTBlock.forward (unchanged baseline boundary)"
    elif getattr(args, "model_variant", None) == FACE_PAIR_HAND_MOTION_VARIANT:
        scope["scope"] += "; per-hand first-order XY motion channels in HF branch"
        scope["inherited_eager_boundary"] = "face LocalSTBlock.forward (unchanged baseline boundary)"
    return scope


def compile_model(model, mode):
    if not hasattr(torch, "compile"):
        raise RuntimeError("Face-pair training requires torch.compile; no eager fallback")
    torch._dynamo.config.suppress_errors = False
    if not hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        raise RuntimeError("Required compilation needs fail_on_recompile_limit_hit support")
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
    options = dict(torch._inductor.list_mode_options(mode))
    options["triton.cudagraphs"] = False
    return torch.compile(model, backend="inductor", options=options)


def pair_loss_components(output, labels, *, include_consistency=False):
    """Return supervised losses plus an optional branch-agreement diagnostic.

    The KL term is intentionally exposed as a separate component so a caller
    can enable it for a new experiment without changing the default objective.
    It compares normalized HF/BF distributions in both directions and remains
    finite for extreme logits through ``log_softmax``.
    """
    components = {"fusion_loss": F.cross_entropy(output["logits"], labels),
            "hand_face_loss": F.cross_entropy(output["hand_face_logits"], labels),
            "body_face_loss": F.cross_entropy(output["body_face_logits"], labels)}
    if include_consistency:
        hand_face_log_prob = F.log_softmax(output["hand_face_logits"], dim=-1)
        body_face_log_prob = F.log_softmax(output["body_face_logits"], dim=-1)
        components["consistency_loss"] = 0.5 * (
            F.kl_div(hand_face_log_prob, body_face_log_prob.exp(), reduction="batchmean")
            + F.kl_div(body_face_log_prob, hand_face_log_prob.exp(), reduction="batchmean")
        )
    return components


def _weighted_pair_loss(components, head_weight, fusion_weight, consistency_weight=0.0):
    # Preserve the original three-way objective without weight normalization.
    consistency = components.get("consistency_loss")
    if consistency is None:
        if consistency_weight != 0.0:
            raise ValueError("consistency_loss is required when consistency_weight is nonzero")
        consistency = components["fusion_loss"].new_zeros(())
    return (fusion_weight * components["fusion_loss"]
            + head_weight * (components["hand_face_loss"] + components["body_face_loss"])
            + consistency_weight * consistency)


def pair_loss(output, labels, *, head_weight, fusion_weight, consistency_weight=0.0):
    """Direct supervision keeps both heads learning under hard class selection."""
    components = pair_loss_components(output, labels, include_consistency=consistency_weight > 0)
    return _weighted_pair_loss(components, head_weight, fusion_weight, consistency_weight)


def numerical_stability_config(args):
    return {"version": 2, "base_lr": args.lr, "max_grad_norm": args.max_grad_norm,
            "gradient_norm_timing": "before_clipping",
            "nonfinite_policy": "save_failure_snapshot_and_raise_no_skipped_batch",
            "loss_weight_normalization": False,
            "loss_weights": {"fusion": args.pair_fusion_loss_weight,
                             "hand_face": args.pair_head_loss_weight,
                             "body_face": args.pair_head_loss_weight,
                             "consistency": getattr(args, "pair_consistency_loss_weight", 0.0)},
            "fusion_domain": "log_softmax_logaddexp_shifted_logsumexp"}


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_snapshot(item) for item in value)
    return value


def save_failure_snapshot(records, compiled, optimizer, scheduler, args, epoch, phase, step,
                          features, labels, frame_mask, output, components, loss, reason):
    """Save the failing batch before an optimizer update, then let training abort."""
    model = getattr(compiled, "_orig_mod", compiled)
    path = records.directory / "failure.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(_cpu_snapshot({
        "schema_version": 1, "reason": reason, "epoch": epoch, "phase": phase, "step": step,
        "completed_updates": scheduler.completed_updates if scheduler is not None else None,
        "args": vars(args), "x": features, "labels": labels, "frame_mask": frame_mask,
        "output": output, "logits": output["logits"], "loss": loss,
        "loss_components": components, "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_schedule": scheduler.state_dict() if scheduler is not None else None,
        "gradients": {name: parameter.grad for name, parameter in model.named_parameters()
                      if parameter.grad is not None} if optimizer is not None else {},
        "max_grad_norm": args.max_grad_norm,
        "architecture": getattr(model, "ARCHITECTURE", None),
        "model_config": getattr(model, "model_config", None),
        "compile": {**compile_scope(args), "mode": args.compile_mode},
        "numerical_stability": numerical_stability_config(args),
    }), temporary)
    temporary.replace(path)
    records.state["failure_batch"] = {"epoch": epoch, "phase": phase, "step": step,
                                      "snapshot": path.name, "reason": reason}
    print(f"TRAINING_ABORTED {reason}; diagnostic_snapshot={path}", flush=True)
    return path


def routing_report(model, epoch):
    total = model.routing_class_counts.detach().cpu().tolist()
    hf = model.routing_hand_face_correct.detach().cpu().tolist()
    bf = model.routing_body_face_correct.detach().cpu().tolist()
    weights = model.hand_face_weight.detach().cpu().tolist()
    rows = []
    for index, (count, hand, body, weight) in enumerate(zip(total, hf, bf, weights)):
        rows.append({"class_index": index, "action_id": f"A{index + 1:03}",
                     "samples": count, "hand_face_correct": hand, "body_face_correct": body,
                     "hand_face_accuracy": hand / count if count else None,
                     "body_face_accuracy": body / count if count else None,
                     "selected": "hand_face" if weight == 1 else "body_face" if weight == 0 else "average",
                     "hand_face_weight": weight})
    return {"schema_version": 1, "epoch": epoch, "source": "training_subject_holdout_only",
            "selection": "higher_per_class_top1_accuracy", "tie_or_unseen": "equal_probability_average",
            "classes": rows}


def save_routing(path, report):
    write_json(path, report)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["classes"][0]))
        writer.writeheader()
        writer.writerows(report["classes"])


def _dataset(args, split, augment):
    return RTMWZipDataset(
        args.archive, split=split, split_protocol=args.split, window_size=args.window_size,
        num_joints=133, num_classes=args.num_classes, layout="rtmw_133",
        max_persons=args.max_persons, feature_mode="raw", max_samples=args.max_samples,
        augment=augment, temporal_compression="confidence_weighted",
        augmentation_config={"random_temporal_crop": False,
                             "joint_dropout_probability": args.joint_dropout_probability,
                             "grouped_joint_dropout_probability": args.grouped_joint_dropout_probability,
                             "precompression_anchor_dropout_probability": (
                                 args.precompression_anchor_dropout_probability
                                 if split == "train" and augment else 0.0),
                             "precompression_anchor_clean_sample_probability": (
                                 args.precompression_anchor_clean_sample_probability
                                 if split == "train" and augment else 0.0)})


def _loader(args, dataset, device, training):
    # Top-level collator is picklable; spawn avoids forking initialized CUDA.
    from isaa.train import collate_rtmw
    options = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda",
               "persistent_workers": args.num_workers > 0,
               "collate_fn": partial(collate_rtmw, feature_mode="raw")}
    if args.num_workers:
        options.update(prefetch_factor=args.prefetch_factor, multiprocessing_context="spawn")
    return DataLoader(dataset, batch_size=args.batch_size if training else args.test_batch_size,
                      shuffle=training, drop_last=bool(training and args.drop_last), **options)


def _split_manifest(args, train, val, fit, calibration, subjects):
    fit_members = [train.members[index] for index in fit]
    calibration_members = [train.members[index] for index in calibration]
    validation_members = list(val.members)
    if set(fit_members) & set(calibration_members) or set(train.members) & set(validation_members):
        raise RuntimeError("Training/calibration/validation members overlap")
    fit_subjects = sorted({int(re.search(r"P(\d{3})", member).group(1)) for member in fit_members})
    if set(fit_subjects) & set(subjects):
        raise RuntimeError("Gradient fitting and calibration subjects overlap")
    def describe(members):
        return {"count": len(members), "sha256": hashlib.sha256("\n".join(members).encode()).hexdigest(),
                "members": members}
    return {"protocol": args.split, "seed": args.seed, "fraction_of_training_subjects": args.pair_calibration_fraction,
            "fit_subjects": fit_subjects, "calibration_subjects": subjects,
            "official_validation_used_for_routing": False,
            "fit": describe(fit_members), "calibration": describe(calibration_members),
            "validation": describe(validation_members)}


def run_epoch(compiled, loader, device, args, records, epoch, phase,
              optimizer=None, scheduler=None):
    training = optimizer is not None
    compiled.train(training)
    started = last_print = time.perf_counter()
    # loss, fused/HF/BF correct, sample count, then all loss components
    consistency_weight = float(getattr(args, "pair_consistency_loss_weight", 0.0))
    include_consistency = consistency_weight > 0.0
    totals = torch.zeros(9 if include_consistency else 8, dtype=torch.float64, device=device)
    logit_max = torch.zeros(2, device=device)
    counts = torch.zeros(args.num_classes, dtype=torch.long, device=device)
    hf_correct, bf_correct = torch.zeros_like(counts), torch.zeros_like(counts)
    last_grad = None
    for step, (features, labels, frame_mask) in enumerate(loader, 1):
        batch_started = time.perf_counter()
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        frame_mask = frame_mask.to(device, non_blocking=True)
        if training:
            lr = scheduler.lr_at(scheduler.completed_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = compiled(features, frame_mask)
            components = pair_loss_components(output, labels, include_consistency=include_consistency)
            loss = _weighted_pair_loss(components, args.pair_head_loss_weight,
                                       args.pair_fusion_loss_weight,
                                       consistency_weight)
            head_max = torch.stack((output["hand_face_logits"].detach().abs().max(),
                                    output["body_face_logits"].detach().abs().max()))

            def abort(reason):
                snapshot = save_failure_snapshot(records, compiled, optimizer, scheduler, args,
                    epoch, phase, step, features, labels, frame_mask, output, components, loss, reason)
                raise RuntimeError(f"{reason}; diagnostic snapshot saved to {snapshot}")

            values_to_check = torch.stack((loss.detach(), *(value.detach() for value in components.values()),
                                           *head_max.unbind()))
            if not bool(torch.isfinite(values_to_check).all()):
                abort(f"Nonfinite {phase} loss/logits at epoch={epoch} step={step}")
            if training:
                loss.backward()
                try:
                    # The returned norm is measured before clipping.
                    norm = torch.nn.utils.clip_grad_norm_(compiled.parameters(), args.max_grad_norm,
                                                         error_if_nonfinite=True)
                except RuntimeError as exc:
                    abort(f"Gradient validation/clipping failed at epoch={epoch} step={step}: {exc}")
                if not bool(torch.isfinite(norm)):
                    abort(f"Nonfinite {phase} gradient norm at epoch={epoch} step={step}")
                last_grad = float(norm)
                optimizer.step()
                scheduler.completed_updates += 1
        with torch.no_grad():
            hf_right = output["hand_face_logits"].argmax(-1) == labels
            bf_right = output["body_face_logits"].argmax(-1) == labels
            totals += torch.stack((loss.detach().double() * labels.numel(),
                                   (output["logits"].argmax(-1) == labels).sum(),
                                   hf_right.sum(), bf_right.sum(), labels.new_tensor(labels.numel()),
                                   *(value.detach().double() * labels.numel() for value in components.values())))
            logit_max = torch.maximum(logit_max, head_max)
            counts += torch.bincount(labels, minlength=args.num_classes)
            hf_correct += torch.bincount(labels[hf_right], minlength=args.num_classes)
            bf_correct += torch.bincount(labels[bf_right], minlength=args.num_classes)
        row = {"epoch": epoch, "phase": phase, "step": step, "steps": len(loader),
               "loss": float(loss.detach()), "samples": labels.numel(),
               "seconds": time.perf_counter() - batch_started,
               **{key: float(value.detach()) for key, value in components.items()},
               "hand_face_logits_absmax": float(head_max[0]),
               "body_face_logits_absmax": float(head_max[1])}
        if training:
            row.update(lr=lr, gradient_norm=last_grad, max_grad_norm=args.max_grad_norm,
                       completed_updates=scheduler.completed_updates)
        records.batch(row)
        now = time.perf_counter()
        if step == 1 or step == len(loader) or now - last_print >= args.log_interval:
            values = totals.cpu().tolist()
            message = (f"epoch={epoch}/{args.epochs} {phase} step={step}/{len(loader)} "
                       f"loss={values[0]/values[4]:.5f} acc={values[1]/values[4]:.4f} "
                       f"HF={values[2]/values[4]:.4f} BF={values[3]/values[4]:.4f} "
                       f"CE_fused={values[5]/values[4]:.5f} CE_HF={values[6]/values[4]:.5f} "
                       f"CE_BF={values[7]/values[4]:.5f} ")
            if include_consistency:
                message += f"KL={values[8]/values[4]:.5f} "
            message += (f"HF_logit_max={row['hand_face_logits_absmax']:.3f} "
                        f"BF_logit_max={row['body_face_logits_absmax']:.3f} "
                        + (f"lr={lr:.7f} grad_preclip={last_grad:.4f} clip={args.max_grad_norm:g} "
                           if training else "")
                        + f"elapsed={now-started:.1f}s")
            print(message, flush=True)
            records.finish("running")
            last_print = now
    values = totals.cpu().tolist()
    if not values[4]:
        raise RuntimeError(f"Empty {phase} loader; reduce batch size or disable drop-last")
    result = {"loss": values[0] / values[4], "acc": values[1] / values[4],
            "hand_face_acc": values[2] / values[4], "body_face_acc": values[3] / values[4],
            **{key: values[index] / values[4] for index, key in enumerate(
                ("fusion_loss", "hand_face_loss", "body_face_loss"), 5)},
            "hand_face_logits_absmax": float(logit_max[0]), "body_face_logits_absmax": float(logit_max[1]),
            "samples": int(values[4]), "seconds": time.perf_counter() - started,
            "gradient_norm": last_grad, "max_grad_norm": args.max_grad_norm if training else None,
            "class_counts": counts.cpu(),
            "hand_face_correct": hf_correct.cpu(), "body_face_correct": bf_correct.cpu()}
    if include_consistency:
        result["consistency_loss"] = values[8] / values[4]
    return result


def _save_checkpoint(path, model, optimizer, scheduler, args, epoch, metrics, routing):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"architecture": model.ARCHITECTURE, "model_config": model.model_config,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "args": vars(args), "epoch": epoch, "lr_schedule": scheduler.state_dict(),
                "metrics": metrics, "routing": routing,
                "numerical_stability": numerical_stability_config(args),
                "compile": {**compile_scope(args), "mode": args.compile_mode}}, temporary)
    temporary.replace(path)


def run_face_pair_training(args, save_dir, records, device):
    if not args.compile or args.init_checkpoint or args.resume_checkpoint:
        raise ValueError("Face-pair training requires compilation and random initialization")
    if not 0 < args.max_grad_norm < float("inf"):
        raise ValueError("Face-pair training requires a positive finite max-grad-norm")
    if args.model_variant == FACE_PAIR_RAW_XY_VARIANT:
        model_class = FacePairRawXYFusion
    elif args.model_variant == FACE_PAIR_HAND_FACE_GEOMETRY_VARIANT:
        model_class = FacePairHandFaceGeometryFusion
    elif args.model_variant == FACE_PAIR_HAND_MOTION_VARIANT:
        model_class = FacePairHandMotionFusion
    elif args.model_variant == FACE_PAIR_VARIANT:
        model_class = FacePairFusion
    else:
        raise ValueError(f"Unsupported face-pair training variant: {args.model_variant}")
    model = model_class(args.num_classes).to(device)
    compiled = compile_model(model, args.compile_mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                nesterov=args.nesterov, weight_decay=args.weight_decay)
    records.update_config(architecture=model.ARCHITECTURE, model_config=model.model_config,
                          compile={**compile_scope(args), "mode": args.compile_mode},
                          initialization="random_scratch_no_checkpoint",
                          numerical_stability=numerical_stability_config(args),
                          environment={"torch": torch.__version__, "cuda": torch.version.cuda,
                                       "device": str(device), "gpu": torch.cuda.get_device_name(device)
                                       if device.type == "cuda" else None},
                          reference={"baseline": "BodyLocalHandCTRWideRelativeFusion",
                                     "inputs": ("raw RTMW133; weighted 100 frames; torso-relative plus raw hand/face xy, score, cross-hand distance/direction"
                                                if model.HAND_FACE_COORDINATE_MODE == "torso_relative_plus_raw_xy" else
                                                "raw RTMW133; weighted 100 frames; torso-relative hand xy/score/distance/direction"),
                                     "new_readout": "Hand+Face and Body+Face independently classified"})
    print(f"Face-pair scratch training on {device}; torch.compile mandatory ({args.compile_mode})", flush=True)
    print(f"Compile scope: {compile_scope(args)['scope']}; {compile_scope(args)['inherited_eager_boundary']}", flush=True)
    if args.dry_run:
        x = torch.rand(2, 3, args.window_size, 133, args.max_persons, device=device)
        labels = torch.arange(2, device=device) % args.num_classes
        dry_scheduler = BatchWarmupCosine(epochs=2, batches_per_epoch=1, warmup_epochs=0,
            base_lr=args.lr, warmup_start_lr=args.warmup_start_lr, min_lr=args.min_lr)
        result = run_epoch(compiled,
            [(x, labels, torch.ones(2, args.window_size, dtype=torch.bool, device=device))],
            device, args, records, 0, "dry_run", optimizer, dry_scheduler)
        write_json(save_dir / "dry_run.json", {"compiled_forward_backward": True,
                    **{key: value for key, value in result.items() if not isinstance(value, torch.Tensor)}})
        return

    train, clean_train, validation = _dataset(args, "train", True), _dataset(args, "train", False), _dataset(args, "val", False)
    if train.members != clean_train.members:
        raise RuntimeError("Training and clean calibration member ordering differ")
    fit, calibration, subjects = split_routing_calibration(train.members, args.pair_calibration_fraction, args.seed)
    manifest = _split_manifest(args, train, validation, fit, calibration, subjects)
    write_json(save_dir / "split_manifest.json", manifest)
    loaders = {"train": _loader(args, Subset(train, fit), device, True),
               "calibration": _loader(args, Subset(clean_train, calibration), device, False),
               "val": _loader(args, validation, device, False)}
    scheduler = BatchWarmupCosine(epochs=args.epochs, batches_per_epoch=len(loaders["train"]),
        warmup_epochs=args.warmup_epochs, base_lr=args.lr, warmup_start_lr=args.warmup_start_lr, min_lr=args.min_lr)
    records.update_config(lr_schedule=scheduler.state_dict(),
        data_split={key: {"count": manifest[key]["count"], "sha256": manifest[key]["sha256"]}
                    for key in ("fit", "calibration", "validation")},
        calibration={"subjects": subjects, "selection": "per_class_top1_accuracy",
                     "start_epoch": args.pair_calibration_start_epoch,
                     "fusion_accuracy_timing": "before_current_epoch_routing_update",
                     "temperatures": "fixed_at_1_softmax_probabilities",
                     "official_validation_used_for_routing": False})
    print(f"fit={len(fit)} calibration={len(calibration)} val={len(validation)} "
          f"calibration_subjects={subjects}; output={save_dir}", flush=True)
    best_acc, best_epoch = -1., 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(compiled, loaders["train"], device, args, records, epoch, "train", optimizer, scheduler)
        calibration_metrics = run_epoch(compiled, loaders["calibration"], device, args, records, epoch, "calibration")
        if epoch >= args.pair_calibration_start_epoch:
            model.set_routing_counts(calibration_metrics["hand_face_correct"],
                                     calibration_metrics["body_face_correct"], calibration_metrics["class_counts"])
        routing = routing_report(model, epoch)
        routing["active"] = epoch >= args.pair_calibration_start_epoch
        save_routing(save_dir / "routing_latest.json", routing)
        save_routing(save_dir / f"routing_epoch_{epoch:03}.json", routing)
        val_metrics = run_epoch(compiled, loaders["val"], device, args, records, epoch, "val")
        improved = val_metrics["acc"] > best_acc
        if improved:
            best_acc, best_epoch = val_metrics["acc"], epoch
        metrics = {"best_epoch": best_epoch, "best_val_acc": best_acc}
        for phase, result in (("train", train_metrics), ("calibration", calibration_metrics), ("val", val_metrics)):
            metrics[phase] = {key: value for key, value in result.items() if not isinstance(value, torch.Tensor)}
        _save_checkpoint(save_dir / "last.pt", model, optimizer, scheduler, args, epoch, metrics, routing)
        if improved:
            _save_checkpoint(save_dir / "best.pt", model, optimizer, scheduler, args, epoch, metrics, routing)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "completed_updates": scheduler.completed_updates, "routing_active": routing["active"],
               "hand_face_classes": sum(value["selected"] == "hand_face" for value in routing["classes"]),
               "body_face_classes": sum(value["selected"] == "body_face" for value in routing["classes"]),
               "average_classes": sum(value["selected"] == "average" for value in routing["classes"]),
               "best_epoch": best_epoch, "best_val_acc": best_acc}
        for phase, result in (("train", train_metrics), ("calibration", calibration_metrics), ("val", val_metrics)):
            row.update({f"{phase}_{key}": result[key] for key in
                        ("loss", "fusion_loss", "hand_face_loss", "body_face_loss")
                        + (("consistency_loss",) if "consistency_loss" in result else ())
                        + ("acc",
                         "hand_face_acc", "body_face_acc", "hand_face_logits_absmax",
                         "body_face_logits_absmax", "samples", "seconds")})
        records.epoch(row)
        records.update_config(lr_schedule=scheduler.state_dict())
        print(f"EPOCH_SAVED epoch={epoch} val={val_metrics['acc']:.4f} best={best_acc:.4f} "
              f"HF_classes={row['hand_face_classes']} BF_classes={row['body_face_classes']} "
              f"average_classes={row['average_classes']} last.pt best.pt", flush=True)
