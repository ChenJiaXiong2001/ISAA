"""Download the original pooled 100-frame runtime artifacts over SSH.

The server address and credentials are intentionally command-line inputs.  The
script does not embed a private key, password, or a machine-specific hostname.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_CHECKPOINT = (
    "/home/xiong/Desktop/ISAA/outputs/"
    "best_weighted_100_cosine_ntu60_xsub/20261014-074736-ab97a386/best.pt"
)
DEFAULT_CONFIG = "/home/xiong/Desktop/ISAA/outputs/best_weighted_100_cosine_ntu60_xsub/20261014-074736-ab97a386/config.json"
DEFAULT_ARCHIVE = "/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip"


def scp(remote: str, destination: Path, *, host: str, user: str, port: int, key: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["scp", "-P", str(port)]
    if key:
        command += ["-i", key]
    command += [f"{user}@{host}:{remote}", str(destination)]
    print("Downloading", remote, "->", destination, flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True, help="SSH host or an alias from ~/.ssh/config")
    parser.add_argument("--ssh-user", default="xiong")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--ssh-key", help="Private key passed to scp")
    parser.add_argument("--remote-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--remote-config", default=DEFAULT_CONFIG)
    parser.add_argument("--remote-archive", default=None,
                        help="Optional large RTMW ZIP; not downloaded unless provided")
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/pool_100"))
    args = parser.parse_args()

    scp(args.remote_checkpoint, args.output_dir / "best.pt", host=args.ssh_host,
        user=args.ssh_user, port=args.ssh_port, key=args.ssh_key)
    scp(args.remote_config, args.output_dir / "config.json", host=args.ssh_host,
        user=args.ssh_user, port=args.ssh_port, key=args.ssh_key)
    if args.remote_archive:
        scp(args.remote_archive, args.output_dir / "ntu60_skeletons_rtmw.zip",
            host=args.ssh_host, user=args.ssh_user, port=args.ssh_port, key=args.ssh_key)
    print("Runtime artifacts are ready in", args.output_dir.resolve())


if __name__ == "__main__":
    main()
