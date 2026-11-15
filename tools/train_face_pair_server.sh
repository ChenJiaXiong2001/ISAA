#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${ISAA_FACE_PAIR_PYTHON:-${HOME}/venvs/isaa-cu126/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    printf 'Verified CUDA Python unavailable: %s\n' "$python_bin" >&2
    exit 1
fi
exec "$python_bin" -s -u "$project_dir/train_best_face_pair_100.py" "$@"
