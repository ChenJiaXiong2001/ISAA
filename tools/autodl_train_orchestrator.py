"""Run the AutoDL Pro training lifecycle independently of the Codex session.

The process creates one Beijing-B instance, starts the fixed hand/wrist training
entry point, performs lightweight five-minute checks, downloads the run logs,
then powers the instance off and verifies the final state.  The API token is
read from ``%USERPROFILE%/.autodl/config.json`` and is never written to logs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paramiko


PROJECT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT / ".autodl_run"
LOG_DIR = PROJECT / "autodl_training_logs"
API_BASE = "https://api.autodl.com"
GPU_SPEC = "5090-p"
DEFAULT_IMAGE = "image-583fc38ec7"
REMOTE_PROJECT = "/root/autodl-fs/ISAA"
REMOTE_SAVE_DIR = f"{REMOTE_PROJECT}/outputs/hand_ctr_wrist_t100_e100_cosine_ntu60_xsub"
REMOTE_ARCHIVE = "/root/autodl-fs/ntu60_skeletons_rtmw.zip"
REMOTE_LOG = f"{REMOTE_PROJECT}/training_launcher.log"
STALE_LOG_SECONDS = 900
STARTUP_GRACE_SECONDS = 900


@dataclass
class Instance:
    uuid: str
    host: str
    port: int
    password: str


class Api:
    def __init__(self, token: str):
        self.token = token

    def call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            API_BASE + path,
            data=data,
            method=method,
            headers={"Authorization": self.token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode())
        if result.get("code") != "Success":
            raise RuntimeError(f"AutoDL API {path} failed: {result.get('code')} {result.get('msg', '')}")
        return result

    def create(self, image_uuid: str) -> str:
        result = self.call("POST", "/api/v1/dev/instance/pro/create", {
            "data_center_list": ["beijingDC2"],
            "req_gpu_amount": 1,
            "expand_system_disk_by_gb": 0,
            "gpu_spec_uuid": GPU_SPEC,
            "image_uuid": image_uuid,
            "cuda_v_from": 128,
            "instance_name": "ISAA-hand-wrist-T100-E100-orchestrated",
            "start_command": "sleep 1",
        })
        return str(result["data"])

    def status(self, uuid: str) -> str:
        return str(self.call("GET", f"/api/v1/dev/instance/pro/status?instance_uuid={uuid}")["data"])

    def snapshot(self, uuid: str) -> dict[str, Any]:
        return dict(self.call("GET", f"/api/v1/dev/instance/pro/snapshot?instance_uuid={uuid}")["data"])

    def power_off(self, uuid: str) -> None:
        self.call("POST", "/api/v1/dev/instance/pro/power_off", {"instance_uuid": uuid})


def read_config() -> tuple[dict[str, Any], str]:
    path = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".autodl" / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    token = str(config["token"])
    if not token:
        raise RuntimeError(f"missing token in {path}")
    return config, token


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


@contextlib.contextmanager
def singleton():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / 'orchestrator.lock').open('a+b') as handle:
        handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            handle.seek(0)
            if os.name == 'nt': msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle, fcntl.LOCK_UN)


def setup_logging(directory: Path | None = None) -> logging.Logger:
    directory = directory or STATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('autodl-orchestrator'); logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]: handler.close(); logger.removeHandler(handler)
    for handler in (logging.FileHandler(directory / 'orchestrator.log', encoding='utf-8'), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s')); logger.addHandler(handler)
    return logger


def retry(label, function, logger, attempts=3, delay=10):
    for attempt in range(1, attempts + 1):
        try: return function()
        except Exception as exc:
            logger.warning('%s attempt=%s/%s failed: %s', label, attempt, attempts, exc)
            if attempt == attempts: raise
            time.sleep(delay)


def wait_for(api, uuid, expected, logger, attempts=36):
    last = 'unknown'
    for _ in range(attempts):
        try:
            last = api.status(uuid); logger.info('instance=%s status=%s', uuid, last)
            if last == expected: return last
        except Exception as exc: logger.warning('status query failed: %s', exc)
        time.sleep(10)
    raise RuntimeError(f'Instance did not reach {expected}; last={last}')


def ssh_connect(instance):
    client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(instance.host, port=instance.port, username='root', password=instance.password,
                       timeout=20, auth_timeout=20, banner_timeout=20)
        client.get_transport().set_keepalive(30)
        return client
    except BaseException:
        client.close(); raise


def remote_exec(client, command, timeout=60):
    # Drain both streams while waiting; recv_exit_status first can deadlock.
    channel = client.get_transport().open_session(timeout=20)
    out, err = bytearray(), bytearray(); deadline = time.monotonic() + timeout
    try:
        channel.exec_command(command)
        while True:
            if channel.recv_ready(): out.extend(channel.recv(65536))
            if channel.recv_stderr_ready(): err.extend(channel.recv_stderr(65536))
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready(): break
            if time.monotonic() > deadline: raise TimeoutError('remote command exceeded deadline')
            time.sleep(0.03)
        return channel.recv_exit_status(), out.decode(errors='replace'), err.decode(errors='replace')
    finally: channel.close()


def checked(client, command, timeout=60):
    code, out, err = remote_exec(client, command, timeout)
    if code: raise RuntimeError(f'Remote exit={code}: {err[-2000:]} {out[-1000:]}')
    return out


def helper(client, root, action, timeout=60):
    return json.loads(checked(client, f'/root/miniconda3/bin/python {shlex.quote(root + "/job_helper.py")} {action} {shlex.quote(root)}', timeout))


def deploy(client, root, directory, batch_size, logger):
    bundle = directory / 'current_code.bundle'
    subprocess.run(['git', 'bundle', 'create', str(bundle), 'HEAD'], cwd=PROJECT, check=True, capture_output=True)
    checked(client, f'mkdir -p {shlex.quote(root)}')
    sftp = client.open_sftp()
    try:
        sftp.get_channel().settimeout(60)
        sftp.put(str(bundle), root + '/code.bundle')
        sftp.put(str(PROJECT / 'tools/autodl_remote_job.py'), root + '/job_helper.py')
        with sftp.file(root + '/job.json', 'w') as f:
            f.write(json.dumps({'archive': REMOTE_ARCHIVE, 'batch_size': batch_size}))
    finally:
        sftp.close()
    # Isolate each run: never checkout -f or reset someone else's experiments.
    code = shlex.quote(root + '/code'); src = shlex.quote(root + '/code.bundle')
    checked(client, f'git init {code} && git -C {code} fetch {src} HEAD && git -C {code} checkout --detach FETCH_HEAD', 180)
    commit = checked(client, f'git -C {code} rev-parse HEAD').strip()
    logger.info('deployed commit=%s remote_root=%s', commit, root)
    check = "import torch,triton,json,os; print(json.dumps(dict(torch=torch.__version__,triton=triton.__version__,cuda=torch.cuda.is_available(),count=torch.cuda.device_count(),gpu=torch.cuda.get_device_name(0),archive_exists=os.path.isfile(" + repr(REMOTE_ARCHIVE) + "))))"
    environment = json.loads(checked(client, '/root/miniconda3/bin/python -c ' + shlex.quote(check)))
    if not environment['cuda'] or environment['count'] != 1 or '5090' not in environment['gpu'] or not environment['archive_exists']:
        raise RuntimeError(f'Environment validation failed: {environment}')
    save_json(directory / 'environment.json', environment)
    return commit, environment


def classify(check, *, now, started, idle_checks=0, startup_grace=1200, stale_seconds=900):
    """Do not classify one idle GPU snapshot as failure during compilation/I/O."""
    exit_info = check.get('exit', {})
    status = check.get('training_status', {})
    if check.get('error'): return 'training_error: ' + check['error'], idle_checks
    if exit_info:
        if exit_info.get('exit_code') == 0 and status.get('status') == 'completed' and status.get('last_completed_epoch') == 100:
            return 'completed', idle_checks
        return f'training_exit={exit_info.get("exit_code")}; status={status.get("status")}; epoch={status.get("last_completed_epoch")}', idle_checks
    if not check.get('alive'): return 'training_process_exited_without_completion', idle_checks
    if status.get('status') in ('failed', 'error'): return 'training_status_failed', idle_checks
    gpus = check.get('gpus', [])
    if check.get('gpu_error') or len(gpus) != 1 or '5090' not in gpus[0]['name']:
        return 'GPU unavailable or configuration mismatch', idle_checks
    gpu = gpus[0]
    if not (0 <= gpu['utilization'] <= 100 and 0 <= gpu['used_mib'] <= gpu['total_mib'] and gpu['total_mib'] > 0):
        return 'invalid GPU telemetry', idle_checks
    if now - started < startup_grace: return None, 0
    # Use the remote clock to avoid host timezone / clock differences.
    age = check['remote_time'] - check.get('log_mtime', 0)
    if check.get('log_size', 0) == 0 or age > stale_seconds:
        return f'training_log_stale age_seconds={age:.0f}', idle_checks
    idle_checks = idle_checks + 1 if gpu['utilization'] == 0 else 0
    if gpu['used_mib'] < 100 or idle_checks >= 3:
        return f'GPU inactive used_mib={gpu["used_mib"]} idle_checks={idle_checks}', idle_checks
    return None, idle_checks


def download_logs(client, root, destination, logger):
    destination.mkdir(parents=True, exist_ok=True)
    meta = helper(client, root, 'archive', timeout=180)
    target = destination / 'logs.tar.gz'; part = target.with_suffix('.partial')
    sftp = client.open_sftp()
    try:
        sftp.get_channel().settimeout(60)
        sftp.get(meta['path'], str(part))
    finally:
        sftp.close()
    digest = hashlib.sha256()
    with part.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk)
    if part.stat().st_size != meta['size'] or digest.hexdigest() != meta['sha256']:
        raise RuntimeError('Downloaded archive size/SHA256 mismatch')
    with tarfile.open(part, 'r:gz') as tar:
        for entry in meta['files']:
            member = tar.extractfile(entry['path'])
            if member is None: raise RuntimeError('missing archive member')
            sha = hashlib.sha256(); size = 0
            with member:
                for chunk in iter(lambda: member.read(1024 * 1024), b''):
                    sha.update(chunk); size += len(chunk)
            if size != entry['size'] or sha.hexdigest() != entry['sha256']:
                raise RuntimeError(f'Archive member verification failed: {entry["path"]}')
    if not meta['files']: raise RuntimeError('No logs in archive')
    part.replace(target); save_json(destination / 'manifest.json', meta)
    logger.info('logs verified SHA256=%s files=%d path=%s', meta['sha256'], len(meta['files']), target)
    return str(target)


def shutdown_with_retry(api, uuid, logger):
    result = {'requested': False, 'confirmed': False, 'final_status': 'unknown'}
    # A timed-out request may already have succeeded: query before retrying.
    for attempt in range(3):
        try:
            result['final_status'] = api.status(uuid)
            if result['final_status'] == 'shutdown':
                result['confirmed'] = True; return result
        except Exception as exc: logger.warning('pre-shutdown query failed: %s', exc)
        try:
            api.power_off(uuid); result['requested'] = True
            logger.info('power_off accepted attempt=%d', attempt + 1)
        except Exception as exc:
            result['request_error'] = str(exc); logger.error('power_off failed: %s', exc)
        for _ in range(6):
            try:
                result['final_status'] = api.status(uuid)
                logger.info('final instance status=%s', result['final_status'])
                if result['final_status'] == 'shutdown':
                    result['confirmed'] = True; return result
            except Exception as exc: logger.warning('shutdown verification failed: %s', exc)
            time.sleep(10)
    logger.critical('Shutdown NOT confirmed: %s', result)
    return result


def lifecycle(args):
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + str(os.getpid())
    directory = STATE_DIR / run_id; logger = setup_logging(directory)
    state = dict(run_id=run_id, local_pid=os.getpid(), phase='starting', created_at=time.time(),
                 region='beijingDC2', gpu_count=1, gpu_spec_uuid=GPU_SPEC, expand_system_disk_by_gb=0,
                 check_interval=args.check_interval, batch_size=args.batch_size, log_download_verified=False)
    def record(**fields):
        state.update(fields)
        save_json(directory / 'state.json', state); save_json(STATE_DIR / 'instance.json', state)
    record()
    uuid = None; instance = None; client = None; api = None; root = None; job_started = False; reason = 'setup_failed'
    def connect():
        nonlocal client
        if client: client.close()
        client = ssh_connect(instance)
        return client
    try:
        config, token = read_config(); api = Api(token)
        # Never blindly retry an ambiguous create timeout (could allocate twice).
        uuid = api.create(str(config.get('image_uuid') or DEFAULT_IMAGE))
        root = '/root/autodl-fs/isaa_runs/' + uuid
        record(instance_uuid=uuid, remote_root=root, phase='creating')
        logger.info('created instance=%s Beijing B RTX5090 x1 expansion=0', uuid)
        wait_for(api, uuid, 'running', logger)
        snap = retry('snapshot', lambda: api.snapshot(uuid), logger)
        if snap.get('region_sign') not in ('bj-B2',) or '5090' not in snap.get('snapshot_gpu_alias_name', ''):
            raise RuntimeError('API snapshot does not match Beijing B RTX5090')
        if snap.get('expand_system_disk_size') != snap.get('system_init_disk_size'):
            raise RuntimeError('API snapshot indicates expanded system disk')
        instance = Instance(uuid, str(snap['proxy_host']), int(snap['ssh_port']), str(snap['root_password']))
        record(ssh_host=instance.host, ssh_port=instance.port, phase='connecting',
               tail_command=f'ssh -p {instance.port} root@{instance.host} "tail -F {root}/training_launcher.log"')
        retry('SSH startup', connect, logger, attempts=6)
        commit, environment = deploy(client, root, directory, args.batch_size, logger)
        record(commit=commit, environment=environment, phase='deployed')
        pid = helper(client, root, 'start')['pid']; job_started = True
        record(remote_pid=pid, phase='training', training_started_at=time.time())
        started = time.monotonic(); idle = 0
        logger.info('training started pid=%s; monitor=%ss startup_grace=%ss', pid, args.check_interval, args.startup_grace)
        # Let runner.json be committed before the first status read.
        time.sleep(3)
        while True:
            def sample():
                if not client or not client.get_transport() or not client.get_transport().is_active(): connect()
                return helper(client, root, 'status')
            check = retry('lightweight check', sample, logger, attempts=2)
            reason, idle = classify(check, now=time.monotonic(), started=started, idle_checks=idle,
                                    startup_grace=args.startup_grace, stale_seconds=args.stale_seconds)
            record(last_check=check, checked_at=time.time())
            logger.info('check alive=%s gpu=%s size=%s mtime=%s exit=%s tail=%s',
                        check['alive'], check['gpus'], check['log_size'], check['log_mtime'],
                        check.get('exit'), check['tail'][-600:])
            if reason:
                logger.log(logging.INFO if reason == 'completed' else logging.ERROR, 'monitor stopped: %s', reason)
                break
            time.sleep(args.check_interval)
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'; logger.exception('Lifecycle exception; entering cleanup')
    finally:
        # All exits, including partial deployment, pass through this single cleanup.
        try:
            record(phase='cleanup', stop_reason=reason)
            if uuid and instance and root:
                def collect():
                    c = connect()
                    tail = checked(c, f'tail -c 32768 {shlex.quote(root + "/training_launcher.log")} 2>/dev/null || true')
                    (directory / 'final_tail.log').write_text(tail, encoding='utf-8')
                    if job_started and reason != 'completed': helper(c, root, 'stop', timeout=30)
                    return download_logs(c, root, LOG_DIR / uuid, logger)
                try:
                    archive = retry('save/download logs', collect, logger)
                    record(log_download_verified=True, log_archive=archive)
                except Exception as exc:
                    logger.error('Log download exhausted retries; proceeding to shutdown: %s', exc)
                    record(log_download_error=str(exc))
        finally:
            if client: client.close()
            if uuid and api:
                shutdown = shutdown_with_retry(api, uuid, logger)
                record(shutdown=shutdown, phase='shutdown' if shutdown['confirmed'] else 'shutdown_unconfirmed', finished_at=time.time())
            else: record(phase='create_failed', finished_at=time.time())
    return 0 if reason == 'completed' and state['log_download_verified'] and state.get('shutdown', {}).get('confirmed') else 1


def tail_latest():
    state_file = STATE_DIR / 'instance.json'
    state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {}
    if 'instance_uuid' not in state:
        print(f'当前没有可查看的运行实例；请查看 {state_file} 和 {STATE_DIR / "launcher_stderr_latest.log"}')
        return
    _, token = read_config(); api = Api(token)
    if api.status(state['instance_uuid']) != 'running':
        print(f'实例 {state["instance_uuid"]} 当前不在 running 状态；本地日志目录：{LOG_DIR / state["instance_uuid"]}')
        return
    snap = api.snapshot(state['instance_uuid'])
    c = ssh_connect(Instance(state['instance_uuid'], snap['proxy_host'], int(snap['ssh_port']), snap['root_password']))
    try:
        _, stdout, _ = c.exec_command('tail -n 30 -F ' + shlex.quote(state['remote_root'] + '/training_launcher.log'))
        for line in stdout: print(line, end='', flush=True)
    finally: c.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--check-interval', type=int, default=300)
    parser.add_argument('--startup-grace', type=int, default=1200)
    parser.add_argument('--stale-seconds', type=int, default=900)
    parser.add_argument('--tail', action='store_true', help='Stream latest remote training log without allocating an instance')
    args = parser.parse_args()
    if args.tail: tail_latest(); return 0
    if min(args.batch_size, args.check_interval, args.startup_grace, args.stale_seconds) <= 0: parser.error('positive values required')
    with singleton(): return lifecycle(args)


if __name__ == '__main__':
    raise SystemExit(main())


