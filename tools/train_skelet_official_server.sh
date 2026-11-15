#!/usr/bin/env bash
# Preserve four per-rank BN batches and averaged DDP gradients on one GPU.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${ISAA_SKELET_PYTHON:-${HOME}/venvs/isaa-cu126/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    printf 'SkeleT server Python is unavailable: %s\n' "$python_bin" >&2
    exit 1
fi
exec "$python_bin" -s -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$project_dir/train_skelet_gcn.py" \
    --config "$project_dir/configs/skelet_ctrgcn_official.json" "$@"
