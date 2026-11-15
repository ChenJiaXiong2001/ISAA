#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${ISAA_FACE_PAIR_PYTHON:-${HOME}/venvs/isaa-cu126/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    printf 'Verified CUDA Python unavailable: %s\n' "$python_bin" >&2
    exit 1
fi
exec "$python_bin" -s -u "$project_dir/train_best_face_pair_handface_geometry_100.py" \
    --archive "$project_dir/data/ntu60_skeletons_rtmw.zip" \
    --feature-mode raw --split xsub60 --window-size 100 --max-persons 2 \
    --temporal-compression confidence_weighted --device cuda \
    --compile --compile-mode default --batch-size 32 --test-batch-size 32 \
    --log-interval 999999 --epochs 8 --warmup-epochs 1 --num-workers 8 \
    --seed 17 --max-samples 4096 --pair-calibration-start-epoch 1 \
    --pair-head-loss-weight 0.5 --pair-fusion-loss-weight 1.0 \
    --save-dir "$project_dir/outputs/face_pair_handface_geometry_8ep4096_20261105" "$@"
