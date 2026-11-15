#!/usr/bin/env bash
# Use the server environment verified with real CUDA/Inductor forward/backward.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${ISAA_SKELET_PYTHON:-${HOME}/venvs/isaa-cu126/bin/python}"

if [[ ! -x "$python_bin" ]]; then
    printf 'SkeleT server Python is unavailable: %s\n' "$python_bin" >&2
    printf 'Set ISAA_SKELET_PYTHON to a verified CUDA PyTorch interpreter.\n' >&2
    exit 1
fi

exec "$python_bin" -s "$project_dir/train_skelet_gcn.py" "$@"
