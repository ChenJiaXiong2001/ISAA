"""Inventory finished training runs without moving or deleting experiment data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re


SKIP = {'.git', '.venv', 'node_modules', '__pycache__', 'code', 'source',
        'source_snapshot', '.codex-backups', '.sync-work', 'completed_training'}


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def number(value):
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return 0


def classify(config, status, rows):
    """Require the intended schedule and a complete per-epoch record."""
    args = config.get('args', config)
    if not isinstance(args, dict):
        args = {}
    planned = number(args.get('epochs', args.get('num_epoch', 0)))
    epochs = {number(row.get('epoch')) for row in rows} - {0}
    actual = max(epochs, default=0)
    state = str(status.get('status', '')).lower()
    if args.get('dry_run') or 'dry_run' in state:
        return False, 'dry_run', planned, actual
    if any(args.get(k) for k in ('smoke', 'cpu_smoke', 'debug')):
        return False, 'smoke_or_debug', planned, actual
    if state in ('failed', 'error', 'interrupted', 'running', 'stopped', 'cancelled', 'aborted'):
        return False, 'nonterminal_or_unsuccessful_status', planned, actual
    if planned <= 0:
        return False, 'unknown_planned_epochs', planned, actual
    if actual < planned or not set(range(1, planned + 1)).issubset(epochs):
        return False, 'incomplete_epoch_history', planned, actual
    if state and state not in ('completed', 'finished', 'success', 'succeeded'):
        return False, 'unrecognized_completion_status', planned, actual
    completed = number(status.get('last_completed_epoch', actual))
    if completed < planned:
        return False, 'status_epoch_disagrees', planned, actual
    return True, 'completed_status_and_full_history' if state else 'full_epoch_history_without_status', planned, actual


def scan(root, host):
    runs, rejected = [], []
    for top in ('outputs', 'experiments', 'docs'):
        start = root / top
        if not start.exists():
            continue
        for current, directories, files in os.walk(start, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in SKIP and not (Path(current)/d).is_symlink())
            directory = Path(current)
            if not {'config.json', 'status.json', 'epochs.csv'}.intersection(files):
                continue
            if 'epochs.csv' not in files and not ('config.json' in files and 'status.json' in files):
                continue
            config = read_json(directory / 'config.json')
            status = read_json(directory / 'status.json')
            try:
                raw = (directory / 'epochs.csv').read_bytes().replace(b'\r\n', b'\n')
                rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
            except (OSError, UnicodeError, csv.Error):
                raw, rows = b'', []
            ok, reason, planned, actual = classify(config, status, rows)
            relative = directory.relative_to(root).as_posix()
            # Short planned screens are still finished runs, but have their own
            # section rather than being presented as full-scale training.
            scope = 'short_screen' if planned < 24 else 'full_schedule'
            if re.search(r'(^|[/_-])(smoke\d*|dryrun|dry_run)([/_-]|$)', relative.lower()):
                ok, reason = False, 'smoke_or_dryrun_directory'
            if not ok:
                rejected.append(dict(host=host, path=relative, reason=reason,
                                     planned_epochs=planned, recorded_epochs=actual,
                                     status=status.get('status')))
                continue
            args = config.get('args', config)
            variant = str(args.get('model_variant', config.get('model_variant', 'unspecified')))
            if 'SkeleT-GCN' in str(config.get('reference', {}).get('repository', '')):
                variant = 'SkeleT-CTRGCN/' + str(args.get('backbone', 'unspecified'))
            signature = hashlib.sha256(variant.encode()+b'\0'+raw).hexdigest()
            metrics = []
            for row in rows:
                try:
                    value = float(row.get('val_acc', row.get('val_top1')))
                    if math.isfinite(value):
                        metrics.append((value, number(row['epoch'])))
                except (KeyError, TypeError, ValueError):
                    pass
            best = max(metrics, default=(None, None))
            artifacts = []
            for folder, children, names in os.walk(directory, followlinks=False):
                children[:] = sorted(d for d in children if d not in SKIP and not (Path(folder)/d).is_symlink())
                for name in sorted(names):
                    p = Path(folder)/name
                    if p.is_symlink():
                        continue
                    try:
                        stat = p.stat()
                    except OSError:
                        continue
                    suffix = p.suffix.lower()
                    kind = 'checkpoint' if suffix in ('.pt', '.pth', '.ckpt') else 'source_snapshot' if suffix in ('.zip', '.gz', '.tar') else 'record'
                    artifacts.append(dict(path=p.relative_to(root).as_posix(), size=stat.st_size,
                                          kind=kind, mtime_ns=stat.st_mtime_ns))
            runs.append(dict(host=host, path=relative, absolute_path=str(directory),
                             run_id=signature[:16], model_variant=variant,
                             planned_epochs=planned, recorded_epochs=actual,
                             status=status.get('status', 'history_complete'), completion_evidence=reason,
                             scope=scope, split=args.get('split'), seed=args.get('seed'),
                             window_size=args.get('window_size'),
                             max_samples=args.get('max_samples', 0), repeat=args.get('repeat', 1),
                             accuracy_units=config.get('accuracy_units', 'fraction'),
                             best_logged_val_acc=best[0], best_logged_epoch=best[1],
                             updated_at=status.get('updated_at'),
                             epochs_sha256=hashlib.sha256(raw).hexdigest(), artifacts=artifacts))
    return dict(host=host, root=str(root), completed=runs, excluded=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--host', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = scan(args.root.resolve(), args.host)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(host=args.host, completed=len(result['completed']),
                         excluded=len(result['excluded'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
