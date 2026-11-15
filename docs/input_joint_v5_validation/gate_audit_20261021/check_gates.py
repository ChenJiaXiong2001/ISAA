"""Read-only V5 checkpoint audit; never updates weights or training artifacts."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, '/home/xiong/Desktop/ISAA')
from isaa.input_joint_training_v5 import load_checkpoint, compile_model, make_loader, COMPILE_SCOPE
from isaa.models.input_joint_selection import OPTIONAL_BODY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.run / 'best_sparse.pt')
    config = argparse.Namespace(**checkpoint['args'])
    config.archive = Path(config.archive)
    config.num_workers = 0
    config.max_samples = 0
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no fallback')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = config.tf32
    torch.backends.cuda.matmul.allow_tf32 = config.tf32
    model.to(device).eval()
    model.gates.hard_eval.fill_(True)
    model.gates.forced_mode.zero_()
    model.gates.threshold.fill_(config.export_threshold)
    compiled = compile_model(model, config.compile_mode)
    loader = make_loader(config, model, 'val')
    print(f'AUDIT_START epoch={checkpoint["epoch"]} samples={len(loader.dataset)} workers=0 threshold={config.export_threshold}', flush=True)
    probabilities, raw_logits, labels_all, predictions, penalties = [], [], [], [], []
    benchmark_batch = None
    start = last_log = time.monotonic()
    with torch.inference_mode():
        for step, (x, labels, mask) in enumerate(loader, 1):
            x, labels, mask = (v.to(device, non_blocking=True) for v in (x, labels, mask))
            logits, penalty, aux = compiled(x, mask, return_gate_values=True)
            probabilities.append(aux['gate_values'].float().cpu())
            raw_logits.append(aux['router_raw_logits'].float().cpu())
            labels_all.append(labels.cpu())
            predictions.append(logits.argmax(1).cpu())
            penalties.append((float(penalty), len(labels)))
            if benchmark_batch is None:
                benchmark_batch = (x, mask)
            now = time.monotonic()
            if now-last_log >= 20 or step == len(loader):
                print(f'AUDIT_PROGRESS step={step}/{len(loader)} elapsed={now-start:.1f}s', flush=True)
                last_log = now
    elapsed = time.monotonic()-start
    p, raw = torch.cat(probabilities).double(), torch.cat(raw_logits).double()
    labels, prediction = torch.cat(labels_all), torch.cat(predictions)
    hard = p > config.export_threshold
    patterns, counts = torch.unique(hard, dim=0, return_counts=True)
    class_means = torch.stack([p[labels==c].mean(0) for c in range(config.num_classes)])
    class_hard_patterns = [int(torch.unique(hard[labels==c],dim=0).shape[0]) for c in range(config.num_classes)]
    ids = list(OPTIONAL_BODY)
    per_gate = []
    for j in range(40):
        group = 'body' if j<14 else 'paired_hands' if j<34 else 'face_token'
        original_ids = [ids[j]] if j<14 else [91+j-14+1,112+j-14+1] if j<34 else None
        per_gate.append(dict(gate_index=j,group=group,original_rtmw_indices=original_ids,
            mean=float(p[:,j].mean()),std=float(p[:,j].std(unbiased=False)),
            min=float(p[:,j].min()),max=float(p[:,j].max()),hard_open_fraction=float(hard[:,j].double().mean()),
            class_mean_range=float(class_means[:,j].max()-class_means[:,j].min()),
            raw_logit_min=float(raw[:,j].min()),raw_logit_max=float(raw[:,j].max()),
            at_logit_bound_fraction=float((raw[:,j].abs()>=model.gates.logit_bound).double().mean())))
    benchmark = {}
    with torch.inference_mode():
        for name, forced in [('learned_hard',0),('all_optional_open',1),('all_optional_closed',2)]:
            model.gates.forced_mode.fill_(forced)
            for _ in range(20):
                compiled(*benchmark_batch, return_gate_values=True)
            torch.cuda.synchronize()
            timings = []
            for _ in range(5):
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
                for _ in range(20):
                    compiled(*benchmark_batch, return_gate_values=True)
                event_end.record()
                event_end.synchronize()
                timings.append(event_start.elapsed_time(event_end)/20)
            ms = float(torch.tensor(timings).median())
            benchmark[name] = dict(batch_size=benchmark_batch[0].shape[0],median_ms_per_batch=ms,
                samples_per_second=1000*benchmark_batch[0].shape[0]/ms,repeats_ms_per_batch=timings)
            print(f'AUDIT_BENCHMARK {name} ms_per_batch={ms:.3f}',flush=True)
    model.gates.forced_mode.zero_()
    model.gates.hard_eval.zero_()
    result = dict(checkpoint=str(args.run/'best_sparse.pt'),epoch=checkpoint['epoch'],
        samples=len(labels),hard_top1=float((labels==prediction).double().mean()),
        recorded_hard_top1=checkpoint['hard_val_metrics']['top1'],recorded_soft_top1=checkpoint['val_metrics']['top1'],
        threshold=config.export_threshold,unique_hard_patterns=len(patterns),
        hard_patterns=[dict(open_gate_indices=torch.where(row)[0].tolist(),samples=int(count)) for row,count in zip(patterns,counts)],
        per_class_unique_hard_patterns=class_hard_patterns,
        mean_gate_std_across_samples=float(p.std(0,unbiased=False).mean()),
        max_gate_std_across_samples=float(p.std(0,unbiased=False).max()),
        max_gate_probability_range=float((p.max(0).values-p.min(0).values).max()),
        nearest_probability_to_threshold=float((p-config.export_threshold).abs().min()),
        mean_expected_weighted_fraction=sum(v*n for v,n in penalties)/len(labels),
        probability_bounds=[model.gates.exploration+(1-2*model.gates.exploration)*float(torch.sigmoid(torch.tensor(v/model.gates.temperature))) for v in [-model.gates.logit_bound,model.gates.logit_bound]],
        per_gate=per_gate,benchmark=benchmark,gpu=torch.cuda.get_device_name(),torch=torch.__version__,compile=COMPILE_SCOPE,
        data_inclusive_elapsed_seconds=elapsed,data_inclusive_samples_per_second=len(labels)/elapsed,
        benchmark_note='Warmed compiled V5 batch-32 full forwards, CUDA events; excludes data loading and compilation. Compares masks within the SAME dense architecture; NOT a baseline/compact speedup.',
        physical_compaction=False,training_artifacts_modified=False)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('AUDIT_RESULT '+json.dumps({k:v for k,v in result.items() if k not in ('per_gate','compile','per_class_unique_hard_patterns')},allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
