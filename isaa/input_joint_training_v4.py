"""Per-input conditional joint routing; dense hard masking, no fixed subset export."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from isaa.models.input_joint_selection import OPTIONAL_BODY
from isaa.models.input_joint_selection_v4 import INPUT_SELECTION_V4_VARIANT as INPUT_SELECTION_VARIANT, InputJointSelectionV4Fusion
from isaa.input_joint_training_v2 import make_loader as _make_loader, gate_weight
from isaa.utils.batch_cosine import BatchWarmupCosine
from isaa.utils.console_logging import mirror_console_to_file
from isaa.utils.experiment import RunRecords, create_run_directory, write_json
from isaa.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
COMPILE_SCOPE = {
    "enabled": True, "required": True, "suppress_errors": False,
    "fail_on_recompile_limit_hit": True, "whole_model_eager_fallback": False,
    "scope": "input-conditioned descriptors for body, paired hand and six face tokens; neural router, stochastic/straight-through hard gates, node cost, raw geometry, body/hand encoders, face-token gates, pooling, fusion and classifier",
    "eager_control": "preserved original face LocalSTBlock boundary",
}

def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--model-variant", choices=(INPUT_SELECTION_VARIANT,), default=INPUT_SELECTION_VARIANT)
    p.add_argument("--stage", choices=("select",), default="select")
    p.add_argument("--selection-config", type=Path, default=ROOT / "configs/input_joint_selection_v4.json")
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
    p.add_argument("--gate-temperature", type=float)
    p.add_argument("--gate-lr", type=float)
    p.add_argument("--gate-noise-ramp-epochs", type=int)
    p.add_argument("--hard-eval-interval", type=int)
    p.add_argument("--gate-health-patience", type=int)
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
    if cfg.pop("version", None) != 4:
        p.error("Unsupported selection config version")
    expected = {"gate_temperature", "sparsity_weight", "gate_warmup_epochs", "gate_ramp_epochs", "export_threshold", "gate_lr", "gate_noise_ramp_epochs", "hard_eval_interval", "gate_health_patience"}
    if set(cfg) != expected:
        p.error("Selection config has unknown or missing settings")
    for key in expected:
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
    if not math.isfinite(args.gate_lr) or args.gate_lr <= 0 or not math.isfinite(args.gate_temperature) or args.gate_temperature <= 0:
        p.error("Gate learning rate and temperature must be finite and positive")
    if any(type(v) is not int or v < 1 for v in (args.gate_noise_ramp_epochs, args.hard_eval_interval, args.gate_health_patience)):
        p.error("Gate transition, hard evaluation interval and health patience must be positive integers")
    if args.num_classes is None:
        args.num_classes = 60 if args.split.endswith("60") else 120
    if min(args.window_size, args.max_persons, args.epochs, args.batch_size, args.test_batch_size, args.num_classes, args.prefetch_factor) < 1:
        p.error("Frame, person, epoch, class and batch counts must be positive")
    if args.num_workers < 0 or args.max_samples < 0:
        p.error("Workers/max-samples cannot be negative")
    if not isinstance(args.gate_warmup_epochs, int) or args.gate_warmup_epochs < 0 or not isinstance(args.gate_ramp_epochs, int) or args.gate_ramp_epochs < 1:
        p.error("Gate warmup must be nonnegative and ramp must be positive integers")
    numeric = (args.gate_temperature, args.sparsity_weight, args.export_threshold,
               args.lr, args.warmup_start_lr, args.min_lr, args.weight_decay, args.momentum,
               args.max_grad_norm, args.log_interval)
    if not all(math.isfinite(v) for v in numeric) or args.sparsity_weight < 0 or not 0 <= args.export_threshold < 1:
        p.error("Invalid gate/loss/threshold settings; all numeric settings must be finite")
    if not 0 <= args.warmup_epochs < args.epochs or not 0 <= args.min_lr <= args.lr or not 0 < args.warmup_start_lr <= args.lr:
        p.error("Invalid cosine learning-rate schedule")
    if args.weight_decay < 0 or args.max_grad_norm < 0 or args.momentum < 0 or args.log_interval <= 0 or (args.nesterov and args.momentum <= 0):
        p.error("Invalid optimizer or logging settings")
    if args.epochs <= args.gate_warmup_epochs and not args.evaluate and not args.dry_run:
        p.error("Training must extend beyond frozen gate warmup")
    if args.save_dir is None:
        args.save_dir = ROOT / f"outputs/input_joint_v4_dynamic_{args.stage}_100_ntu60_xsub"
    args.compile = True
    return args


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


def build_model(args):
    return InputJointSelectionV4Fusion(args.num_classes, gate_temperature=args.gate_temperature)


def make_loader(args, model, phase):
    loader = _make_loader(args, model, phase)
    # Validation workers start after training has initialized CUDA/Inductor.
    # Spawn avoids inheriting CUDA modules and Triton destructor state via fork.
    if loader.num_workers > 0:
        loader.multiprocessing_context = "spawn"
    return loader


def make_optimizers(model, args):
    gate_parameters = list(model.gates.parameters())
    gate_ids = {id(p) for p in gate_parameters}
    backbone = torch.optim.SGD([p for p in model.parameters() if id(p) not in gate_ids],
        lr=args.lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)
    router = torch.optim.Adam(gate_parameters, lr=args.gate_lr, weight_decay=0.)
    return backbone, router


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != InputJointSelectionV4Fusion.ARCHITECTURE:
        raise ValueError("A matching dynamic v4 checkpoint is required")
    model = InputJointSelectionV4Fusion(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def run_epoch(compiled, loader, device, *, epoch, args, optimizer=None, router_optimizer=None,
              scheduler=None, weight=0., gate_active=False, records=None, phase="val", hard=False):
    model = compiled._orig_mod
    model.gates.hard_eval.fill_(hard)
    compiled.train(optimizer is not None)
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    gate_sum = torch.zeros_like(model.gates.cost, dtype=torch.float64)
    sampled_gate_sum = torch.zeros_like(gate_sum)
    class_sum = torch.zeros(args.num_classes, gate_sum.numel(), device=device, dtype=torch.float64)
    class_count = torch.zeros(args.num_classes, device=device, dtype=torch.float64)
    class_correct = torch.zeros_like(class_count)
    retain_min, retain_max = torch.tensor(60., device=device), torch.tensor(0., device=device)
    retain_sum = torch.zeros((), device=device, dtype=torch.float64)
    sampled_retain_sum = torch.zeros_like(retain_sum)
    all_open = all_closed = grad_norm_sum = 0.
    samples = steps = 0
    start = last_log = time.monotonic()
    try:
        with torch.set_grad_enabled(optimizer is not None):
            for step, (x, labels, frame_mask) in enumerate(loader, 1):
                x, labels, frame_mask = (v.to(device, non_blocking=True) for v in (x, labels, frame_mask))
                if optimizer is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = scheduler.lr_at(scheduler.completed_updates)
                    optimizer.zero_grad(set_to_none=True)
                    router_optimizer.zero_grad(set_to_none=True)
                logits, penalty, aux = compiled(x, frame_mask, return_gate_values=True)
                ce = nn.functional.cross_entropy(logits, labels)
                loss = ce + weight * penalty if optimizer is not None else ce
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at epoch={epoch}, step={step}")
                if optimizer is not None:
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.gates.parameters(), float("inf"), error_if_nonfinite=True)
                    grad_norm_sum += float(norm)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm or float("inf"), error_if_nonfinite=True)
                    optimizer.step()
                    if gate_active:
                        router_optimizer.step()
                    scheduler.completed_updates += 1
                with torch.no_grad():
                    count = labels.size(0)
                    top = logits.topk(min(5, logits.size(1)), dim=1).indices
                    correct = (top[:, 0] == labels).double()
                    totals += torch.stack((loss.detach(), ce.detach(), penalty.detach(), correct.mean(),
                        (top == labels[:, None]).any(1).double().mean())) * count
                    gates = aux["gate_values"].detach().double()
                    binary = (gates > args.export_threshold).double()
                    retained = (binary * model.gates.cost).sum(1)
                    sampled_retained = ((aux["sampled_gate_values"].detach() > args.export_threshold) * model.gates.cost).sum(1)
                    sampled_retain_sum += sampled_retained.sum()
                    retain_min = torch.minimum(retain_min, retained.min())
                    retain_max = torch.maximum(retain_max, retained.max())
                    retain_sum += retained.sum()
                    gate_sum += binary.sum(0)
                    sampled_gate_sum += (aux["sampled_gate_values"].detach() > args.export_threshold).double().sum(0)
                    class_sum.index_add_(0, labels, binary)
                    class_count.index_add_(0, labels, torch.ones(count, device=device, dtype=torch.float64))
                    class_correct.index_add_(0, labels, correct)
                    all_open += int((binary.sum(1) == 40).sum())
                    all_closed += int((binary.sum(1) == 0).sum())
                samples += count
                steps += 1
                now = time.monotonic()
                if now - last_log >= args.log_interval or step == len(loader):
                    metrics = (totals / samples).tolist()
                    batch_retained = dict(mean=float(retained.mean()), min=float(retained.min()), max=float(retained.max()))
                    row = dict(epoch=epoch, phase=phase, step=step, loss=metrics[0], classification_loss=metrics[1],
                        sparsity_penalty=metrics[2], top1=metrics[3], top5=metrics[4], elapsed_seconds=now-start,
                        retained_mean=float(retain_sum / samples), retained_min=float(retain_min), retained_max=float(retain_max),
                        sampled_retained_mean=float(sampled_retain_sum / samples),
                        batch_retained=batch_retained, open_mix=float(model.gates.open_mix), noise_scale=float(model.gates.noise_scale),
                        hard_training=bool(model.gates.hard_train), hard_evaluation=hard, gate_total_grad_norm=grad_norm_sum / steps,
                        gate_std_across_batch=float(gates.std(0, unbiased=False).mean()), gate_mean=float(gates.mean()),
                        all_open_fraction=all_open/samples, all_optional_closed_fraction=all_closed/samples,
                        face_tokens_mean=float(gate_sum[34:].sum()/samples),
                        sampled_face_tokens_mean=float(sampled_gate_sum[34:].sum()/samples),
                        lr=optimizer.param_groups[0]["lr"] if optimizer else 0., gate_lr=args.gate_lr if optimizer else 0.)
                    if records:
                        records.batch(row)
                        records.finish("running")
                    print(f"{phase} epoch={epoch} step={step}/{len(loader)} loss={metrics[0]:.4f} top1={metrics[3]:.4f} "
                          f"active_optional_cost={row['retained_mean']:.2f}/60 batch_range=[{batch_retained['min']:.0f},{batch_retained['max']:.0f}] "
                          f"face_tokens={row['face_tokens_mean']:.2f}/6 sampled_face={row['sampled_face_tokens_mean']:.2f}/6 "
                          f"gate_std={row['gate_std_across_batch']:.4f} gate_grad={row['gate_total_grad_norm']:.5f}", flush=True)
                    last_log = now
        if not samples:
            raise ValueError("Empty loader")
        loss, ce, penalty, top1, top5 = (totals / samples).tolist()
        return dict(loss=loss, classification_loss=ce, sparsity_penalty=penalty, top1=top1, top5=top5,
            samples=samples, samples_per_second=samples/(time.monotonic()-start),
            retained_mean=float(retain_sum/samples), retained_min=float(retain_min), retained_max=float(retain_max),
            sampled_retained_mean=float(sampled_retain_sum/samples),
            all_open_fraction=all_open/samples, all_optional_closed_fraction=all_closed/samples,
            gate_total_grad_norm=grad_norm_sum/max(1,steps),
            body_gates_mean=float(gate_sum[:14].sum()/samples),
            hand_pair_gates_mean=float(gate_sum[14:34].sum()/samples),
            face_tokens_mean=float(gate_sum[34:].sum()/samples),
            sampled_face_tokens_mean=float(sampled_gate_sum[34:].sum()/samples),
            sampled_gate_open_rates=(sampled_gate_sum/samples).tolist(),
            gate_open_rates=(gate_sum/samples).tolist(), class_samples=class_count.tolist(),
            class_top1=(class_correct/class_count.clamp_min(1.)).tolist(),
            class_gate_open_rates=(class_sum/class_count.clamp_min(1.)[:,None]).tolist())
    finally:
        model.gates.hard_eval.zero_()


def _scalars(metrics):
    return {k:v for k,v in metrics.items() if not isinstance(v,list)}


def run(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no fallback")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.evaluate:
        model, checkpoint = load_checkpoint(args.evaluate)
        for key in ("split","window_size","temporal_compression","max_persons","num_classes","export_threshold"):
            setattr(args,key,checkpoint["args"][key])
        compiled = compile_model(model.to(device),args.compile_mode)
        loader = make_loader(args,model,"val")
        soft = run_epoch(compiled,loader,device,epoch=checkpoint["epoch"],args=args)
        hard = run_epoch(compiled,loader,device,epoch=checkpoint["epoch"],args=args,phase="val_hard",hard=True)
        print(json.dumps(dict(soft=soft,hard=hard),indent=2),flush=True)
        return
    model = build_model(args).to(device)
    model_config = dict(num_classes=args.num_classes,gate_temperature=args.gate_temperature)
    directory = create_run_directory(args.save_dir)
    serialized = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    records = RunRecords(directory,serialized,ROOT)
    records.update_config(architecture=model.ARCHITECTURE,model_config=model_config,
        initialization="random_from_scratch; no inherited checkpoints",compile={**COMPILE_SCOPE,"mode":args.compile_mode},
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=str(device)),
        loader_process_start_method="spawn" if args.num_workers else None,
        dynamic_selection=dict(granularity="one input-dependent mask per clip shared by persons and frames",
            labels_used_for_routing=False,fixed_node_budget=None,physical_compaction=False,
            candidate_inputs=129,fixed_dependencies=75,fixed_face_points=68,optional_raw_nodes=54,
            gate_count=40,gate_groups=dict(body_points=14,paired_hand_groups=20,face_tokens=6),
            face_token_groups=[[23,33],[34,44],[45,55],[56,66],[67,78],[79,90]],
            face_token_switches_control_encoded_readout=True,face_encoding_dense=True,
            raw_face_observations_retained_for_pooling=True,
            input_required="all candidate observations for light router; backbone masked per clip",
            training=dict(frozen_full_open_epochs=args.gate_warmup_epochs,
                open_mix_ramp_epochs=args.gate_noise_ramp_epochs,
                stochastic_straight_through_after=args.gate_warmup_epochs+args.gate_noise_ramp_epochs),
            penalty="expected binary-open cost: body=1, paired hand=2, face token=1; total=60; not a retained raw-input count",
            threshold=args.export_threshold,hard_evaluation="dense graph binary masking; no speedup claim"))
    print(f"run={directory} dynamic input-conditioned selection; random initialization",flush=True)
    try:
        with mirror_console_to_file(directory/"console.log"):
            compiled = compile_model(model,args.compile_mode)
            if args.dry_run:
                model.gates.begin_epoch(10,5,5)
                x = torch.randn(2,3,args.window_size,129,args.max_persons,device=device)*.05
                x[:,2]=1.
                logits,penalty,aux = compiled(x,return_gate_values=True)
                (logits.square().mean()+args.sparsity_weight*penalty).backward()
                nn.utils.clip_grad_norm_(model.parameters(),float("inf"),error_if_nonfinite=True)
                records.update_config(dry_run_result=dict(compiled_forward_backward=True))
                records.finish("dry_run_passed")
                return
            train,val = make_loader(args,model,"train"),make_loader(args,model,"val")
            optimizer,router = make_optimizers(model,args)
            scheduler = BatchWarmupCosine(epochs=args.epochs,batches_per_epoch=len(train),warmup_epochs=args.warmup_epochs,
                base_lr=args.lr,warmup_start_lr=args.warmup_start_lr,min_lr=args.min_lr)
            model.gates.threshold.fill_(args.export_threshold)
            records.update_config(dataset=dict(train=len(train.dataset),val=len(val.dataset)),
                optimizer=dict(backbone="SGD",router="Adam",gate_lr=args.gate_lr),lr_schedule=scheduler.state_dict())
            best,best_hard,best_sparse,best_epoch = -1.,-1.,-1.,0
            no_reduction = 0
            for epoch in range(1,args.epochs+1):
                model.gates.begin_epoch(epoch,args.gate_warmup_epochs,args.gate_noise_ramp_epochs)
                active = epoch > args.gate_warmup_epochs
                weight = gate_weight(epoch,args)
                tr = run_epoch(compiled,train,device,epoch=epoch,args=args,optimizer=optimizer,router_optimizer=router,
                    scheduler=scheduler,weight=weight,gate_active=active,records=records,phase="train")
                va = run_epoch(compiled,val,device,epoch=epoch,args=args,records=records)
                vh = None
                if epoch == args.epochs or (active and epoch % args.hard_eval_interval == 0):
                    vh = run_epoch(compiled,val,device,epoch=epoch,args=args,records=records,phase="val_hard",hard=True)
                ready = epoch >= args.gate_warmup_epochs + max(args.gate_ramp_epochs,args.gate_noise_ramp_epochs)
                no_reduction = no_reduction + 1 if ready and va["retained_mean"] >= 60-1e-6 else 0
                if no_reduction >= args.gate_health_patience or (ready and va["all_optional_closed_fraction"] > .99):
                    print("GATE HEALTH WARNING: inspect full-open/full-closed fractions and per-class usage",flush=True)
                diagnostic = dict(epoch=epoch,soft=va,hard=vh,train=tr,weight=weight,
                    optional_body_indices=list(OPTIONAL_BODY),
                    hand_local_indices=list(range(1,21)),threshold=args.export_threshold,
                    face_token_original_indices=[[model.selection["input_indices"][p] for p, valid in zip(row, members) if valid]
                        for row, members in zip(model.face_positions.tolist(), model.face_members.tolist())],
                    gate_groups=dict(body=[0,14],paired_hands=[14,34],face_tokens=[34,40]),
                    labels_used_for_diagnostics_only=True,physical_compaction=False)
                with (directory/"selection_history.jsonl").open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostic,allow_nan=False)+"\n")
                write_json(directory/"latest_gate_usage.json",diagnostic)
                names=["last"]
                if va["top1"] > best:
                    best,best_epoch = va["top1"],epoch
                    names.append("best")
                if vh is not None and epoch >= args.gate_warmup_epochs+args.gate_noise_ramp_epochs:
                    if vh["top1"] > best_hard:
                        best_hard=vh["top1"]
                        names.append("best_hard")
                    if ready and vh["retained_mean"] < 60-1e-6 and vh["top1"] > best_sparse:
                        best_sparse=vh["top1"]
                        names.append("best_sparse")
                checkpoint=dict(epoch=epoch,architecture=model.ARCHITECTURE,model_config=model_config,model=model.state_dict(),
                    args=serialized,optimizer=optimizer.state_dict(),gate_optimizer=router.state_dict(),lr_schedule=scheduler.state_dict(),
                    val_accuracy=va["top1"],best_accuracy=best,best_epoch=best_epoch,train_metrics=tr,val_metrics=va,hard_val_metrics=vh)
                for name in names:
                    checkpoint["checkpoint_selection_metric"]="hard_val_top1" if name in ("best_hard","best_sparse") else "soft_val_top1"
                    tmp=directory/f"{name}.pt.tmp"
                    torch.save(checkpoint,tmp)
                    tmp.replace(directory/f"{name}.pt")
                    write_json(directory/f"{name}_gate_usage.json",diagnostic)
                records.epoch(dict(epoch=epoch,stage="dynamic",sparsity_weight=weight,
                    **{f"train_{k}":v for k,v in _scalars(tr).items()},**{f"val_{k}":v for k,v in _scalars(va).items()},
                    hard_val_top1=vh["top1"] if vh else None,hard_retained_mean=vh["retained_mean"] if vh else None,
                    best_epoch=best_epoch,best_val_acc=best))
                print(f"epoch={epoch} soft_val={va['top1']:.4f} retained_mean={va['retained_mean']:.2f} "
                      f"hard_val={vh['top1'] if vh else None}",flush=True)
            write_json(directory/"selection_result.json",dict(dynamic_selection=True,physical_compaction=False,
                final_mean_retained=va["retained_mean"],best_soft_top1=best,best_hard_top1=best_hard if best_hard>=0 else None,
                best_sparse_top1=best_sparse if best_sparse>=0 else None))
            records.finish("finished")
    except BaseException as exc:
        records.finish("failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        records.close()
