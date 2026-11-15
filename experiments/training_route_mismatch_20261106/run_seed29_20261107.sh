#!/usr/bin/env bash
set -euo pipefail
cd /home/xiong/Desktop/ISAA
python_bin=/home/xiong/venvs/isaa-cu126/bin/python
experiment=experiments/training_route_mismatch_20261106
control=outputs/soft_route_matched_control_20ep4096_seed29_20261107
candidate=outputs/soft_route_candidate_20ep4096_seed29_20261107
for path in "$control" "$candidate" "$experiment/control_check_seed29_20261107.json" "$experiment/candidate_check_seed29_20261107.json"; do
    if [[ -e "$path" ]]; then echo "Refusing existing path: $path"; exit 1; fi
done
common=(--archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip --feature-mode raw
        --split xsub60 --window-size 100 --max-persons 2 --temporal-compression confidence_weighted
        --device cuda --compile --compile-mode default --batch-size 32 --test-batch-size 32
        --epochs 20 --warmup-epochs 1 --num-workers 8 --seed 29 --max-samples 4096
        --lr 0.025 --pair-calibration-start-epoch 1 --pair-head-loss-weight 0.5
        --pair-fusion-loss-weight 1.0 --log-interval 20)
echo "START_CONTROL $(date -Is)"
"$python_bin" -s -u train_best_face_pair_100.py "${common[@]}" --save-dir "$control" > "$experiment/control_train_seed29_20261107.log" 2>&1
shopt -s nullglob
control_dirs=("$control"/*/config.json)
[[ ${#control_dirs[@]} -eq 1 ]] || { echo 'Expected exactly one control run'; exit 1; }
control_run=${control_dirs[0]%/config.json}
"$python_bin" -s -u tools/check_face_pair_checkpoint.py --run-dir "$control_run" --output "$experiment/control_check_seed29_20261107.json"
echo "CONTROL_CHECK_OK $(date -Is) $control_run"
echo "START_CANDIDATE $(date -Is)"
"$python_bin" -s -u "$experiment/train_soft_route.py" "${common[@]}" --save-dir "$candidate" > "$experiment/candidate_train_seed29_20261107.log" 2>&1
candidate_dirs=("$candidate"/*/config.json)
[[ ${#candidate_dirs[@]} -eq 1 ]] || { echo 'Expected exactly one candidate run'; exit 1; }
candidate_run=${candidate_dirs[0]%/config.json}
"$python_bin" -s -u "$experiment/check_soft_run.py" --run-dir "$candidate_run" --output "$experiment/candidate_check_seed29_20261107.json"
echo "CANDIDATE_CHECK_OK $(date -Is) $candidate_run"
"$python_bin" -s -u "$experiment/paired_evaluate.py" --control "$control_run/last.pt" --candidate "$candidate_run/last.pt" --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip --partition calibration --smoke-samples 2 --num-workers 0 --output-dir "$experiment/paired_smoke_seed29_20261107" > "$experiment/paired_smoke_seed29_20261107.log" 2>&1
echo "PAIRED_SMOKE_OK $(date -Is)"
"$python_bin" -s -u "$experiment/paired_evaluate.py" --control "$control_run/last.pt" --candidate "$candidate_run/last.pt" --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip --partition calibration --output-dir "$experiment/paired_calibration_seed29_20261107" > "$experiment/paired_calibration_seed29_20261107.log" 2>&1
echo "CALIBRATION_EVALUATION_OK $(date -Is)"
"$python_bin" -s -u "$experiment/paired_evaluate.py" --control "$control_run/last.pt" --candidate "$candidate_run/last.pt" --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip --partition validation --output-dir "$experiment/paired_validation_seed29_20261107" > "$experiment/paired_validation_seed29_20261107.log" 2>&1
echo "VALIDATION_EVALUATION_OK $(date -Is)"