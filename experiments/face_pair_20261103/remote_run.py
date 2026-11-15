"""Launch/inspect only the isolated face-pair experiment on its training host."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

BASE = Path('/home/xiong/Desktop/ISAA/experiments/face_pair_20261103')
CODE = BASE / 'code'
VERIFY = BASE / 'verification'
PYTHON = '/home/xiong/venvs/isaa-cu126/bin/python'
DATA = '/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip'
OUTPUT = Path('/home/xiong/Desktop/ISAA/outputs/best_face_pair_100_20261103')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('probe', 'smoke', 'full', 'status', 'verify-source', 'tests'))
    args = parser.parse_args()
    if args.action == 'verify-source':
        manifest = json.loads((BASE / 'source_manifest.json').read_text())
        invalid = [name for name, digest in manifest.items()
                   if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != digest]
        if invalid:
            raise RuntimeError(f'Source checksum mismatch: {invalid}')
        print(json.dumps({'source_files_verified': len(manifest)}))
        return
    if args.action == 'status':
        for filename in sorted(VERIFY.glob('launch_*.json')):
            launch = json.loads(filename.read_text())
            try:
                os.kill(launch['pid'], 0)
                launch['alive'] = True
            except ProcessLookupError:
                launch['alive'] = False
            log = Path(launch['log'])
            launch['log_tail'] = log.read_text(errors='replace')[-5000:] if log.exists() else ''
            parent = Path(launch.get('output_parent') or '/nonexistent')
            runs = sorted(parent.glob('*/status.json'))
            if runs:
                launch['run_directory'] = str(runs[-1].parent)
                launch['training_status'] = json.loads(runs[-1].read_text())
            print(json.dumps(launch))
        return
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TORCHINDUCTOR_COMPILE_THREADS='4', PYTHONUNBUFFERED='1')
    if args.action == 'tests':
        with (VERIFY / 'unittests.log').open('w') as log:
            for pattern in ('test_face_pair_fusion.py', 'test_best_action_hand_body.py', 'test_best_weighted_100.py'):
                command = [PYTHON, '-s', '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v']
                result = subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{pattern} failed: {result.returncode}')
        print((VERIFY / 'unittests.log').read_text())
        return
    launch_file = VERIFY / f'launch_{args.action}.json'
    if launch_file.exists():
        raise RuntimeError(f'Launch already recorded: {launch_file}')
    if args.action == 'probe':
        command = [PYTHON, '-s', '-u', str(CODE / 'tools/check_face_pair_compile.py'),
                   '--archive', DATA, '--batch-size', '2', '--output', str(VERIFY / 'cuda_compile.json')]
        parent = None
    else:
        parent = VERIFY / 'smoke_outputs' if args.action == 'smoke' else OUTPUT
        command = [PYTHON, '-s', '-u', str(CODE / 'train_best_face_pair_100.py'),
                   '--archive', DATA, '--device', 'cuda', '--compile', '--compile-mode', 'default',
                   '--batch-size', '32', '--test-batch-size', '32', '--log-interval', '10',
                   '--save-dir', str(parent)]
        if args.action == 'smoke':
            command.extend(['--epochs', '2', '--warmup-epochs', '0', '--max-samples', '256',
                            '--num-workers', '2', '--pair-calibration-start-epoch', '1'])
        else:
            if not (VERIFY / 'cuda_compile.json').exists():
                raise RuntimeError('CUDA compile evidence missing')
            statuses = sorted((VERIFY / 'smoke_outputs').glob('*/status.json'))
            if not statuses or json.loads(statuses[-1].read_text())['status'] != 'completed':
                raise RuntimeError('Real-data smoke did not complete')
            if not (VERIFY / 'checkpoint_validation.json').exists():
                raise RuntimeError('Checkpoint validation missing')
            command.extend(['--epochs', '100', '--num-workers', '8'])
    log_path = VERIFY / f'{args.action}.log'
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {'pid': process.pid, 'command': command, 'code_directory': str(CODE),
              'log': str(log_path), 'output_parent': str(parent) if parent else None,
              'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    launch_file.write_text(json.dumps(launch, indent=2))
    print(json.dumps(launch, indent=2))


if __name__ == '__main__':
    main()
