#!/usr/bin/env bash
set -euo pipefail
CODE=/home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/code
ARCHIVE=/home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip
ROOT=/home/xiong/Desktop/ISAA/experiments/face_pair_loss_screen_20261105
OUTROOT=/home/xiong/Desktop/ISAA/outputs/face_pair_loss_screen_20261105
mkdir -p "$ROOT" "$OUTROOT"
run_one() {
 name="$1"; headw="$2"; save="$OUTROOT/$name"; log="$ROOT/$name.log"
 echo "START $name $(date -Is)" | tee -a "$ROOT/screen.log"
 /home/xiong/venvs/isaa-cu126/bin/python -s "$CODE/train_best_face_pair_100.py" \
 --archive "$ARCHIVE" --device cuda --compile --compile-mode default --batch-size 32 --test-batch-size 32 \
 --log-interval 999999 --save-dir "$save" --epochs 4 --warmup-epochs 1 --num-workers 8 --seed 17 \
 --max-samples 1024 --pair-calibration-start-epoch 1 --pair-head-loss-weight "$headw" --pair-fusion-loss-weight 1.0 \
 >"$log" 2>&1
 echo "DONE $name $(date -Is)" | tee -a "$ROOT/screen.log"
}
run_one loss_h075_f100 0.75
run_one loss_h100_f100 1.0