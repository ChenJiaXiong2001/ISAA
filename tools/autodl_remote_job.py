"""Remote lifecycle helper: no credentials, installed separately for each job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time

ERROR = re.compile(r'Traceback \(most recent call last\)|RuntimeError:|OutOfMemoryError:|CUDA out of memory|PassManager::run failed|(?:^|\n).*\bERROR\b|Non-finite (?:loss|gradient)')


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value), encoding='utf-8')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def alive(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text.split(') ', 1)[1].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def run(root):
    config = read(root / 'job.json')
    command = [sys.executable, '-u', str(root / 'code/train_hand_ctr_wrist_fusion.py'),
               '--archive', config['archive'], '--split', 'xsub60', '--device', 'cuda',
               '--batch-size', str(config['batch_size']), '--test-batch-size', str(config['batch_size']),
               '--num-workers', '8', '--prefetch-factor', '4', '--save-dir', str(root / 'outputs')]
    result = {'started_at': time.time(), 'command': command}
    save(root / 'runner.json', {'pid': os.getpid(), **result})
    try:
        with (root / 'training_launcher.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=root / 'code', stdout=log, stderr=subprocess.STDOUT)
            save(root / 'runner.json', {'pid': os.getpid(), 'child_pid': child.pid, **result})
            result['exit_code'] = child.wait()
    except BaseException as exc:
        result.update(exit_code=-1, error=repr(exc))
    finally:
        save(root / 'exit.json', {**result, 'finished_at': time.time()})


def status(root):
    runner = read(root / 'runner.json')
    pid = runner.get('pid')
    result = {'remote_time': time.time(), 'alive': bool(pid and alive(pid)),
              'pid': pid, 'child_pid': runner.get('child_pid'), 'exit': read(root / 'exit.json')}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    result['gpu_error'] = gpu.stderr[-500:] if gpu.returncode else ''
    result['gpus'] = []
    for line in gpu.stdout.splitlines():
        try:
            name, util, used, total = [s.strip() for s in line.split(',')]
            result['gpus'].append(dict(name=name, utilization=float(util), used_mib=float(used), total_mib=float(total)))
        except ValueError:
            result['gpu_error'] = 'unparseable GPU telemetry'
    log = root / 'training_launcher.log'
    scan = read(root / 'scan.json')
    result.update(log_size=0, log_mtime=0, tail='', error=scan.get('error'))
    if log.exists():
        st = log.stat(); result.update(log_size=st.st_size, log_mtime=st.st_mtime)
        offset = min(scan.get('offset', 0), st.st_size)
        with log.open('rb') as f:
            f.seek(max(0, offset - 512))
            recent = f.read(65536).decode(errors='replace')
            offset = f.tell()
            f.seek(max(0, st.st_size - 4096))
            tail = f.read().decode(errors='replace')
        match = ERROR.search(recent) or ERROR.search(tail)
        result['error'] = scan.get('error') or (match.group(0)[-500:] if match else None)
        result['tail'] = '\n'.join(tail.splitlines()[-6:])
        save(root / 'scan.json', {'offset': offset, 'error': result['error']})
    runs = sorted((root / 'outputs').glob('*/status.json'))
    result['training_status'] = read(runs[-1]) if runs else {}
    print(json.dumps(result))


def stop(root):
    info = read(root / 'runner.json'); pid = info.get('pid')
    if pid and alive(pid):
        # A separate session is created at start; only this job's process group.
        if os.getpgid(pid) != pid:
            raise RuntimeError('runner process group mismatch')
        os.killpg(pid, signal.SIGTERM)
        for _ in range(10):
            if not alive(pid): break
            time.sleep(1)
        # Kill remaining workers even when their runner has exited.
        try: os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError: pass
    print(json.dumps({'stopped': True}))


def archive(root):
    suffixes = {'.log', '.json', '.jsonl', '.csv', '.txt', '.yaml', '.yml', '.md'}
    paths = [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
             and 'code' not in p.relative_to(root).parts and p.suffix in suffixes]
    manifest = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk)
        manifest.append({'path': str(path.relative_to(root)), 'size': path.stat().st_size, 'sha256': digest.hexdigest()})
    target = root / 'logs.tar.gz'
    with tarfile.open(target, 'w:gz') as tar:
        for p in paths: tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
    digest = hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk)
    print(json.dumps({'path': str(target), 'size': target.stat().st_size,
                      'sha256': digest.hexdigest(), 'files': manifest}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action'); parser.add_argument('root', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'start':
        with (root / 'runner.log').open('ab') as log:
            p = subprocess.Popen([sys.executable, __file__, 'run', str(root)], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': p.pid}))
    elif args.action == 'run': run(root)
    elif args.action == 'status': status(root)
    elif args.action == 'stop': stop(root)
    elif args.action == 'archive': archive(root)
    else: raise ValueError(args.action)
