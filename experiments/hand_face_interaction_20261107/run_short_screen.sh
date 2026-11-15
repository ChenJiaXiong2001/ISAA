#!/usr/bin/env bash
set -euo pipefail

# Start only after the currently running matched control has released the A6000.
cd "$(dirname "$0")/../.."
python experiments/hand_face_interaction_20261107/check_cuda_compile.py
python experiments/hand_face_interaction_20261107/train.py \
  --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
  --feature-mode raw --split xsub60 --window-size 100 --max-persons 2 \
  --temporal-compression confidence_weighted --device cuda --compile \
  --compile-mode default --batch-size 32 --test-batch-size 32 --epochs 8 \
  --warmup-epochs 1 --num-workers 8 --seed 17 --max-samples 4096 --lr 0.025 \
  --pair-calibration-start-epoch 1 --pair-head-loss-weight 0.5 \
  --pair-fusion-loss-weight 1.0 --log-interval 20 \
  --save-dir outputs/hand_face_temporal_interaction_8ep4096_seed17_20261107

