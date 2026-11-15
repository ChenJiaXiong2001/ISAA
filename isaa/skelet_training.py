"""Independent SkeleT-GCN train/test entry; no changes to existing experiments."""
import argparse
import hashlib
import json
import math
import os
from datetime import timedelta
from functools import partial
from pathlib import Path
import random
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from isaa.data.skelet_dataset import SkeleTRTMWDataset, SkeleTPoseDataset, SPLITS, STREAMS
from isaa.models.skelet import SkeleTGCN
from isaa.models.skelet.model import VARIANTS, REFERENCE_COMMIT
from isaa.utils.experiment import RunRecords, create_run_directory, write_json

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/skelet_gcn.json"


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path, default=PRESET)
    selected, _ = selector.parse_known_args(argv)
    defaults = json.loads(selected.config.read_text(encoding="utf-8"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(selected.config))
    source = p.add_mutually_exclusive_group()
    source.add_argument("--archive", default="data/ntu60_skeletons_rtmw.zip")
    source.add_argument("--ann-file", help="Official PYSKL annotation pickle")
    p.add_argument("--backbone", choices=VARIANTS)
    p.add_argument("--split", choices=SPLITS)
    p.add_argument("--stream", choices=STREAMS)
    for name in ("window_size", "max_persons", "batch_size", "test_batch_size", "epochs", "repeat",
                 "seed", "image_height", "image_width"):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("lr", "momentum", "weight_decay"):
        p.add_argument("--" + name.replace("_", "-"), type=float)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--expected-world-size", type=int, default=1,
                   help="Required number of torchrun ranks; batch-size is per rank")
    p.add_argument("--distributed-backend", choices=("gloo", "nccl"), default="gloo")
    p.add_argument("--checkpoint-keep", type=int, default=0,
                   help="Keep this many numbered epoch checkpoints, plus best/last")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--instance-pooling", action="store_true", help="Optional early person fusion; off for NTU official preset")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-dir", default="outputs/skelet_gcn")
    p.add_argument("--dry-run", action="store_true", help="Two compiled synthetic forward/backward SGD updates")
    p.add_argument("--evaluate", help="Evaluate a matching checkpoint; never initializes training")
    p.add_argument("--num-clips", type=int, default=10, help="Test-time clips for --evaluate; validation uses one")
    p.add_argument("--predictions", help="Optional evaluation NPZ with sample identities and probabilities")
    p.add_argument("--test-best", action=argparse.BooleanOptionalAction, default=False,
                   help="Run official ten-clip best-checkpoint evaluation after training")
    p.set_defaults(**defaults)
    args = p.parse_args(argv)
    if not args.compile or "--no-compile" in argv:
        p.error("SkeleT-GCN requires torch.compile; --no-compile is not allowed")
    for name in ("window_size", "max_persons", "batch_size", "test_batch_size", "epochs", "repeat",
                 "image_height", "image_width", "num_clips"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.num_workers < 0 or args.max_samples < 0 or args.lr <= 0 or args.weight_decay < 0:
        p.error("Invalid worker count, sample limit, learning rate or weight decay")
    if not 0 < args.momentum < 1:
        p.error("Nesterov momentum must be between zero and one")
    if args.dry_run and args.evaluate:
        p.error("--dry-run and --evaluate are mutually exclusive")
    if args.expected_world_size < 1:
        p.error("expected_world_size must be positive")
    if args.seed < 0 or args.checkpoint_keep < 0:
        p.error("seed and checkpoint_keep must be nonnegative")
    if int(os.environ.get("WORLD_SIZE", "1")) != args.expected_world_size:
        p.error(f"This preset requires {args.expected_world_size} torchrun ranks")
    args.num_classes = 60 if args.split.endswith("60") else 120
    return args


def compile_model(model):
    if not hasattr(torch, "compile"):
        raise RuntimeError("SkeleT-GCN requires a torch.compile-capable PyTorch")
    torch._dynamo.config.suppress_errors = False
    # Fullgraph raises on graph breaks and on unsupported operations. Disabling
    # CUDA Graph capture still compiles all neural computations with Inductor.
    try:
        return torch.compile(model, backend="inductor", fullgraph=True,
                             options={"triton.cudagraphs": False})
    except Exception as exc:
        raise RuntimeError("Required SkeleT torch.compile initialization failed; no eager fallback") from exc


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def official_worker_seed(worker_id, *, num_workers, rank, seed):
    # Same NumPy/Python worker seed formula as the public PYSKL loader.
    worker_rng_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_rng_seed)
    random.seed(worker_rng_seed)


def synchronize_bn_for_evaluation(model):
    # MMCV DistEvalHook broadcasts rank-zero BN means/variances before eval.
    if dist.is_initialized():
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
                dist.broadcast(module.running_mean, 0)
                dist.broadcast(module.running_var, 0)


class RepeatDataset(Dataset):
    def __init__(self, dataset, times):
        self.dataset, self.times = dataset, times

    def __len__(self):
        return len(self.dataset) * self.times

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]


def make_dataset(args, training, num_clips=1):
    kwargs = dict(protocol=args.split, training=training, stream=args.stream,
                  clip_len=args.window_size, num_person=args.max_persons, num_clips=num_clips,
                  image_shape=(args.image_height, args.image_width), max_samples=args.max_samples)
    path = Path(args.ann_file or args.archive)
    if not path.is_absolute():
        path = ROOT / path
    return (SkeleTPoseDataset(path, **kwargs) if args.ann_file else SkeleTRTMWDataset(path, **kwargs))


def make_loader(args, dataset, training):
    distributed = dist.is_initialized()
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(),
                                 shuffle=training, seed=args.seed) if distributed else None
    seed_fn = partial(official_worker_seed, num_workers=args.num_workers,
                      rank=dist.get_rank(), seed=args.seed) if distributed else worker_seed
    return DataLoader(dataset, batch_size=args.batch_size if training else args.test_batch_size,
                      shuffle=training and sampler is None, sampler=sampler,
                      num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
                      worker_init_fn=seed_fn,
                      generator=None if distributed else torch.Generator().manual_seed(args.seed),
                      persistent_workers=args.persistent_workers and args.num_workers > 0)


def compiled_updates(model, compiled, x, num_classes, optimizer):
    """Two true backward/update steps; also verifies that mappings receive gradients."""
    y = torch.arange(x.shape[0], device=x.device) % num_classes
    results = []
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = compiled(x)
        loss = nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite SkeleT compiled loss")
        loss.backward()
        mapping = model.backbone.gcn[4].gcn.M
        if mapping.grad is None or not torch.isfinite(mapping.grad).all() or mapping.grad.abs().sum() == 0:
            raise RuntimeError("65->27 mapping did not receive finite, nonzero compiled gradients")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Non-finite compiled parameter gradient")
        results.append(dict(step=step + 1, loss=loss.item(), logits_shape=list(logits.shape),
                            mapping_grad_norm=mapping.grad.norm().item()))
        optimizer.step()
    if x.is_cuda:
        torch.cuda.synchronize()
    return results


def evaluate(model, loader, device, predictions=None):
    model.eval()
    loss_sum = correct = correct5 = count = 0
    all_probs, all_labels = [], []
    offset = 0
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            if x.ndim == 6:
                # Score averaging in probability space, matching RecognizerGCN.
                clip_scores = [model(x[:, k].contiguous()).softmax(-1) for k in range(x.shape[1])]
                probs = torch.stack(clip_scores).mean(0)
                loss = nn.functional.nll_loss(probs.clamp_min(1e-12).log(), labels)
            else:
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, labels)
                probs = logits.softmax(-1)
            if dist.is_initialized():
                # DistributedSampler pads its tail. Exclude padding from metrics.
                positions = (torch.arange(offset, offset + len(labels), device=device)
                             * dist.get_world_size() + dist.get_rank())
                keep = positions < len(loader.dataset)
                probs, labels = probs[keep], labels[keep]
                per_sample = nn.functional.nll_loss(probs.clamp_min(1e-12).log(), labels, reduction="sum")
                loss_sum += per_sample.item()
                offset += len(keep)
            else:
                loss_sum += loss.item() * len(labels)
            n = len(labels)
            count += n
            correct += (probs.argmax(-1) == labels).sum().item()
            correct5 += (probs.topk(min(5, probs.shape[-1]), -1).indices == labels[:, None]).any(-1).sum().item()
            if predictions is not None:
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    if dist.is_initialized():
        totals = torch.tensor([loss_sum, correct, correct5, count], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        loss_sum, correct, correct5, count = totals.tolist()
    metrics = dict(loss=loss_sum / count, top1=correct / count, top5=correct5 / count, samples=int(count))
    if predictions is not None:
        ds = loader.dataset
        names = ds.members if hasattr(ds, "members") else [a["frame_dir"] for a in ds.annotations]
        output = Path(predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        probabilities, labels = np.concatenate(all_probs), np.concatenate(all_labels)
        if dist.is_initialized():
            shards = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
            dist.gather_object((probabilities, labels), shards, dst=0)
            if dist.get_rank() != 0:
                return metrics
            probabilities = np.empty((len(names), probabilities.shape[-1]), dtype=probabilities.dtype)
            labels = np.empty(len(names), dtype=labels.dtype)
            for rank, (shard_probs, shard_labels) in enumerate(shards):
                probabilities[rank::dist.get_world_size()] = shard_probs
                labels[rank::dist.get_world_size()] = shard_labels
        np.savez_compressed(output, probabilities=probabilities, labels=labels, sample_names=np.asarray(names))
    return metrics


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; SkeleT training cannot silently change device")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    if world_size > 1:
        # On this single-GPU server all ranks explicitly use the requested GPU.
        # Gloo supports CUDA all-reduce without NCCL's duplicate-device restriction.
        dist.init_process_group(args.distributed_backend, timeout=timedelta(minutes=30))
        torch._dynamo.config.optimize_ddp = False
    model = SkeleTGCN(args.num_classes, args.backbone, args.max_persons, args.instance_pooling).to(device)
    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=False)
        saved = checkpoint["args"]
        for key in ("backbone", "num_classes", "max_persons", "instance_pooling", "stream", "split",
                    "window_size", "image_height", "image_width"):
            if saved[key] != getattr(args, key):
                raise ValueError(f"Checkpoint {key}={saved[key]!r} differs from requested {getattr(args, key)!r}")
        model.load_state_dict(checkpoint["model"], strict=True)
    run_dir = create_run_directory(ROOT / args.save_dir if not Path(args.save_dir).is_absolute() else Path(args.save_dir)) if rank == 0 else None
    if world_size > 1:
        shared_path = [str(run_dir) if rank == 0 else None]
        dist.broadcast_object_list(shared_path, src=0)
        run_dir = Path(shared_path[0])
    records = RunRecords(run_dir, vars(args), ROOT) if rank == 0 else None
    if records is not None:
        records.update_config(
        reference=dict(repository="https://github.com/YijieYang23/SkeleT-GCN", commit=REFERENCE_COMMIT,
                       paper="https://arxiv.org/abs/2406.18011", config=f"configs/skelet_{'ctrgcn' if args.backbone.startswith('ctrgcn') else args.backbone}/ntu60_xsub/{args.stream}.py",
                       adaptation="Native PyTorch; RTMW ZIP or official pose annotations; see docs/skelet_gcn.md"),
        backbone=model.config,
        compile=dict(enabled=True, required=True, backend="inductor", fullgraph=True, cudagraphs=False,
                     suppress_errors=False, scope="complete SkeleT model including mappings, IP, neural encoders and classifier"),
        initialization="matching checkpoint for evaluation" if args.evaluate else "random; no pretrained weights",
        schedule=dict(type="cosine_per_iteration", logical_epochs=args.epochs,
                      dataset_repeat=args.repeat, effective_dataset_passes=args.epochs * args.repeat),
        distributed=dict(world_size=world_size, backend=args.distributed_backend if world_size > 1 else None,
                         per_rank_batch=args.batch_size, total_batch=args.batch_size * world_size,
                         sync_batchnorm=False, broadcast_buffers=False,
                         bn_eval_buffers="broadcast rank-zero means/variances" if world_size > 1 else "single rank",
                         worker_seed="PYSKL rank/worker formula" if world_size > 1 else "torch initial seed"),
        protocol_differences=["Required torch.compile fullgraph; public entry does not configure compile",
                              "RTMW input replaces author's pose annotations",
                              "Four logical ranks share one GPU with Gloo rather than four GPUs with NCCL"] if world_size > 1 else [],
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, device=str(device)))
    # Include the independent entry, preset and license, not just generic main.py.
    if rank == 0:
        with ZipFile(run_dir / "source.zip", "a", ZIP_DEFLATED) as archive:
            for path in (ROOT / "train_skelet_gcn.py", Path(args.config).resolve(), ROOT / "docs/skelet_gcn.md",
                         ROOT / "isaa/models/skelet/LICENSE", ROOT / "isaa/models/skelet/NOTICE.md",
                         ROOT / "tools/train_skelet_official_server.sh"):
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT).as_posix())
    if rank == 0:
        print(f"run={run_dir}\nSkeleT-{args.backbone}: 65->27->11, groups 1->2->4; torch.compile Inductor fullgraph required; ranks={world_size}, batch={args.batch_size}x{world_size}", flush=True)
    try:
        compiled = compile_model(model)
        train_model = DistributedDataParallel(compiled, device_ids=[device.index or 0] if device.type == "cuda" else None,
                                              broadcast_buffers=False, find_unused_parameters=True) if world_size > 1 else compiled
        if args.dry_run:
            generator = torch.Generator(device=device).manual_seed(args.seed + rank + 100)
            x = torch.randn(args.batch_size if world_size > 1 else 2, 3, args.window_size, 65, args.max_persons,
                            device=device, generator=generator)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
            proof = dict(device=str(device), torch=torch.__version__, backend="inductor", fullgraph=True,
                         backbone=args.backbone, instance_pooling=args.instance_pooling,
                         rank=rank, world_size=world_size,
                         updates=compiled_updates(model, train_model, x, args.num_classes, optimizer))
            if world_size > 1:
                proof["parameter_hash"] = hashlib.sha256(b"".join(
                    p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
                proof["train_bn_hash"] = hashlib.sha256(b"".join(
                    b.detach().cpu().numpy().tobytes() for name, b in model.named_buffers()
                    if "running_" in name)).hexdigest()
                synchronize_bn_for_evaluation(model)
                proof["eval_bn_hash"] = hashlib.sha256(b"".join(
                    b.detach().cpu().numpy().tobytes() for name, b in model.named_buffers()
                    if "running_" in name)).hexdigest()
                proofs = [None] * world_size if rank == 0 else None
                dist.gather_object(proof, proofs, dst=0)
                if rank == 0:
                    if len({p["parameter_hash"] for p in proofs}) != 1:
                        raise RuntimeError("DDP parameters diverged after SGD")
                    if len({p["train_bn_hash"] for p in proofs}) != world_size:
                        raise RuntimeError("Training BN must retain per-rank statistics")
                    if len({p["eval_bn_hash"] for p in proofs}) != 1:
                        raise RuntimeError("Rank-zero evaluation BN broadcast failed")
                proof = dict(ranks=proofs, total_batch=args.batch_size * world_size, all_neural_paths_compiled=True)
            if rank == 0:
                write_json(run_dir / "compile_check.json", proof)
                print(json.dumps(proof, indent=2), flush=True)
                records.finish("compile_check_passed")
            return
        if args.evaluate:
            val_ds = make_dataset(args, False, args.num_clips)
            synchronize_bn_for_evaluation(model)
            metrics = evaluate(compiled, make_loader(args, val_ds, False), device, args.predictions)
            if rank == 0:
                records.update_config(evaluation=metrics, test_clips=args.num_clips)
                records.finish("evaluated")
                print(json.dumps(metrics), flush=True)
            return
        train_ds, val_ds = make_dataset(args, True), make_dataset(args, False)
        loaders = (make_loader(args, RepeatDataset(train_ds, args.repeat), True), make_loader(args, val_ds, False))
        if rank == 0:
            records.update_config(dataset=dict(train_samples=len(train_ds), val_samples=len(val_ds),
                                           repeated_train_samples=len(train_ds) * args.repeat,
                                           training_clips=1, validation_clips=1, stream=args.stream))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
        total_steps = args.epochs * len(loaders[0])
        best, best_epoch, global_step = -1.0, 0, 0
        for epoch in range(args.epochs):
            train_model.train()
            if world_size > 1:
                loaders[0].sampler.set_epoch(epoch)
            started = time.perf_counter()
            loss_sum = correct = count = 0
            for step, (x, y) in enumerate(loaders[0]):
                lr = args.lr * (1 + math.cos(math.pi * global_step / total_steps)) / 2
                for group in optimizer.param_groups:
                    group["lr"] = lr
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = train_model(x)
                loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite SkeleT training loss")
                loss.backward()
                optimizer.step()
                n = len(y)
                loss_sum += loss.item() * n
                correct += (logits.argmax(-1) == y).sum().item()
                count += n
                global_step += 1
                if rank == 0:
                    records.batch(dict(epoch=epoch + 1, phase="train", step=step + 1, lr=lr, loss=loss.item()))
                if rank == 0 and (step == 0 or (step + 1) % 100 == 0):
                    print(f"epoch={epoch+1}/{args.epochs} batch={step+1}/{len(loaders[0])} loss={loss.item():.4f} lr={lr:.6f}", flush=True)
            if world_size > 1:
                totals = torch.tensor([loss_sum, correct, count], dtype=torch.float64, device=device)
                dist.all_reduce(totals)
                loss_sum, correct, count = totals.tolist()
            synchronize_bn_for_evaluation(model)
            val = evaluate(compiled, loaders[1], device)
            improved = val["top1"] > best
            if improved:
                best, best_epoch = val["top1"], epoch + 1
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), args=vars(args),
                         backbone=model.config, epoch=epoch + 1, global_step=global_step,
                         best_val_acc=best, best_epoch=best_epoch)
            if rank == 0:
                torch.save(state, run_dir / "last.pt")
                if args.checkpoint_keep:
                    torch.save(state, run_dir / f"epoch_{epoch + 1}.pt")
                    expired = epoch + 1 - args.checkpoint_keep
                    if expired > 0:
                        (run_dir / f"epoch_{expired}.pt").unlink(missing_ok=True)
                if improved:
                    torch.save(state, run_dir / "best.pt")
                records.epoch(dict(epoch=epoch + 1, train_loss=loss_sum / count, train_acc=correct / count,
                               val_loss=val["loss"], val_acc=val["top1"], val_top5=val["top5"],
                               best_val_acc=best, best_epoch=best_epoch, lr=lr,
                               seconds=time.perf_counter() - started))
                print(f"epoch={epoch+1} train_top1={correct/count:.4%} val_top1={val['top1']:.4%} best={best:.4%}", flush=True)
        if args.test_best:
            if world_size > 1:
                dist.barrier()
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            synchronize_bn_for_evaluation(model)
            test_ds = make_dataset(args, False, args.num_clips)
            metrics = evaluate(compiled, make_loader(args, test_ds, False), device,
                               run_dir / "best_test_predictions.npz")
            if rank == 0:
                write_json(run_dir / "best_test_metrics.json", dict(checkpoint="best.pt", clips=args.num_clips, **metrics))
                records.update_config(best_test=dict(checkpoint="best.pt", clips=args.num_clips, **metrics))
                print(f"best_checkpoint_test clips={args.num_clips} top1={metrics['top1']:.4%}", flush=True)
        if rank == 0:
            records.finish("completed")
    except BaseException as exc:
        if records is not None:
            records.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if records is not None:
            records.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
