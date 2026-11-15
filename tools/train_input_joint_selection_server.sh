#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${ISAA_INPUT_SELECTION_PYTHON:-${HOME}/venvs/isaa-cu126/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    printf 'Verified CUDA Python unavailable: %s\n' "$python_bin" >&2
    exit 1
fi
exec "$python_bin" -s "$project_dir/train_best_input_selection_100.py" "$@"
