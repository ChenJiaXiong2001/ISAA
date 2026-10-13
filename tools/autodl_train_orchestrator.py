"""Run the AutoDL Pro training lifecycle independently of the Codex session.

The process creates one Beijing-B instance, starts the fixed hand/wrist training
entry point, performs lightweight five-minute checks, downloads the run logs,
then powers the instance off and verifies the final state.  The API token is
read from ``%USERPROFILE%/.autodl/config.json`` and is never written to logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
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


def setup_logging() -> logging.Logger:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("autodl-orchestrator")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(STATE_DIR / "orchestrator.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(console)
    return logger


def wait_for(api: Api, uuid: str, expected: str, logger: logging.Logger, attempts: int = 36) -> str:
    value = "unknown"
    for _ in range(attempts):
        try:
            value = api.status(uuid)
            logger.info("instance=%s status=%s", uuid, value)
            if value == expected:
                return value
        except Exception as exc:  # retry transient API failures
            logger.warning("status check failed: %s", exc)
        time.sleep(10)
    raise RuntimeError(f"instance {uuid} did not reach {expected}; last={value}")


def ssh_connect(instance: Instance) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(instance.host, port=instance.port, username="root", password=instance.password,
                   timeout=30, auth_timeout=30, banner_timeout=30)
    return client


def remote_exec(client: paramiko.SSHClient, command: str, timeout: int = 60) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    return stdout.channel.recv_exit_status(), stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")


def deploy_bundle(client: paramiko.SSHClient, logger: logging.Logger) -> None:
    bundle = STATE_DIR / "current_code.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=PROJECT, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sftp = client.open_sftp()
    remote_bundle = "/root/autodl-fs/isaa_current.bundle"
    sftp.put(str(bundle), remote_bundle)
    sftp.close()
    command = (
        f"set -e; mkdir -p {REMOTE_PROJECT}; "
        f"if [ ! -d {REMOTE_PROJECT}/.git ]; then git init {REMOTE_PROJECT}; fi; "
        f"git -C {REMOTE_PROJECT} fetch {remote_bundle} HEAD:refs/remotes/bundle/current; "
        f"git -C {REMOTE_PROJECT} checkout -f refs/remotes/bundle/current; "
        f"git -C {REMOTE_PROJECT} log -1 --oneline"
    )
    code, out, err = remote_exec(client, command)
    if code:
        raise RuntimeError(f"remote code deployment failed: {err[-1000:]}")
    logger.info("deployed code: %s", out.strip().splitlines()[-1])


def start_training(client: paramiko.SSHClient, logger: logging.Logger, batch_size: int) -> int:
    command = (
        f"rm -f {REMOTE_LOG}; mkdir -p {REMOTE_SAVE_DIR}; "
        f"nohup /root/miniconda3/bin/python {REMOTE_PROJECT}/train_hand_ctr_wrist_fusion.py "
        f"--archive {REMOTE_ARCHIVE} --split xsub60 --device cuda --batch-size {batch_size} "
        f"--test-batch-size {batch_size} --num-workers 8 --prefetch-factor 4 "
        f"--save-dir {REMOTE_SAVE_DIR} > {REMOTE_LOG} 2>&1 < /dev/null & echo $!"
    )
    code, out, err = remote_exec(client, command)
    if code:
        raise RuntimeError(f"training start failed: {err[-1000:]}")
    match = re.search(r"(\d+)", out)
    if not match:
        raise RuntimeError(f"training PID missing: {out[-500:]}")
    pid = int(match.group(1))
    logger.info("training started pid=%s", pid)
    return pid


def lightweight_check(client: paramiko.SSHClient, pid: int) -> dict[str, Any]:
    command = (
        f"alive=0; ps -p {pid} -o stat=,cmd= >/tmp/isaa_ps 2>/dev/null && alive=1; "
        "gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1); "
        f"log_size=$(stat -c %s {REMOTE_LOG} 2>/dev/null || echo 0); "
        f"log_mtime=$(stat -c %Y {REMOTE_LOG} 2>/dev/null || echo 0); "
        "printf 'alive=%s\\ngpu=%s\\nlog_size=%s\\nlog_mtime=%s\\n' \"$alive\" \"$gpu\" \"$log_size\" \"$log_mtime\"; "
        "cat /tmp/isaa_ps 2>/dev/null || true"
    )
    code, out, _ = remote_exec(client, command)
    if code:
        raise RuntimeError("remote lightweight check failed")
    values: dict[str, Any] = {"raw": out.strip()}
    for line in out.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip()
    values["alive"] = values.get("alive") == "1"
    return values


def download_logs(client: paramiko.SSHClient, destination: Path, logger: logging.Logger) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        def get(path: str, target: Path) -> None:
            try:
                sftp.get(path, str(target))
            except FileNotFoundError:
                pass

        get(REMOTE_LOG, destination / "training_launcher.log")
        command = f"find {REMOTE_SAVE_DIR} -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-"
        _, out, _ = remote_exec(client, command)
        run_dir = out.strip()
        if run_dir:
            (destination / "run_dir.txt").write_text(run_dir, encoding="utf-8")
            for name in ("console.log", "config.json", "epochs.csv", "batches.jsonl", "status.json"):
                get(f"{run_dir}/{name}", destination / name)
    finally:
        sftp.close()
    archive = destination.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(destination, arcname=destination.name)
    if archive.stat().st_size <= 0:
        raise RuntimeError("downloaded log archive is empty")
    logger.info("logs downloaded: %s (%d bytes)", archive, archive.stat().st_size)
    return archive


def shutdown_with_retry(api: Api, uuid: str, logger: logging.Logger) -> None:
    for attempt in range(1, 4):
        try:
            api.power_off(uuid)
            logger.info("power_off requested attempt=%d", attempt)
            break
        except Exception as exc:
            logger.warning("power_off attempt=%d failed: %s", attempt, exc)
            time.sleep(10)
    else:
        logger.error("power_off request failed after retries")
        return
    for _ in range(18):
        try:
            status = api.status(uuid)
            logger.info("final instance status=%s", status)
            if status == "shutdown":
                return
        except Exception as exc:
            logger.warning("final status check failed: %s", exc)
        time.sleep(10)
    logger.error("instance did not reach shutdown within the verification window")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-interval", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-checks", type=int, default=0, help="0 means until training exits")
    args = parser.parse_args()
    logger = setup_logging()
    config, token = read_config()
    api = Api(token)
    uuid = None
    client = None
    instance = None
    run_state = STATE_DIR / "instance.json"
    try:
        image_uuid = str(config.get("image_uuid") or DEFAULT_IMAGE)
        uuid = api.create(image_uuid)
        run_state.write_text(json.dumps({"instance_uuid": uuid, "created_at": time.time(),
                                         "region": "beijingDC2", "gpu_spec_uuid": GPU_SPEC,
                                         "gpu_count": 1, "expand_system_disk_by_gb": 0}, indent=2), encoding="utf-8")
        logger.info("created instance=%s region=beijingDC2 gpu=RTX5090 count=1 expansion=0", uuid)
        wait_for(api, uuid, "running", logger)
        snap = api.snapshot(uuid)
        instance = Instance(uuid, str(snap["proxy_host"]), int(snap["ssh_port"]), str(snap["root_password"]))
        client = ssh_connect(instance)
        deploy_bundle(client, logger)
        pid = start_training(client, logger, args.batch_size)
        logger.info("monitor interval=%ss", args.check_interval)
        checks = 0
        while True:
            try:
                check = lightweight_check(client, pid)
                logger.info("check alive=%s gpu=%s log_size=%s log_mtime=%s", check.get("alive"),
                            check.get("gpu", "unknown"), check.get("log_size", "0"), check.get("log_mtime", "0"))
                if not check.get("alive"):
                    break
            except Exception as exc:
                logger.warning("lightweight check failed: %s", exc)
                try:
                    client.close()
                except Exception:
                    pass
                client = ssh_connect(instance)
            checks += 1
            if args.max_checks and checks >= args.max_checks:
                logger.info("max-checks reached; orchestrator remains responsible for cleanup")
                return 0
            time.sleep(args.check_interval)
        abnormal_log = None
        if client is not None:
            try:
                _, tail, _ = remote_exec(client, f"tail -n 120 {REMOTE_LOG}")
                abnormal_log = tail
                if "Traceback" in tail or "Error" in tail or "Exception" in tail:
                    logger.error("training log contains a likely error; see downloaded logs")
            except Exception as exc:
                logger.warning("could not read final training tail: %s", exc)
        destination = LOG_DIR / uuid
        try:
            if client is None:
                client = ssh_connect(instance)
            download_logs(client, destination, logger)
        except Exception as exc:
            logger.error("log download failed: %s", exc)
        finally:
            if client is not None:
                client.close()
            shutdown_with_retry(api, uuid, logger)
        return 0
    except Exception:
        logger.exception("orchestrator failed")
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if uuid:
            shutdown_with_retry(api, uuid, logger)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
