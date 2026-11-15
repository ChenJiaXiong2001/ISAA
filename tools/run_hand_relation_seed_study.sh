#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${ISAA_PYTHON:-$HOME/venvs/isaa-cu126/bin/python}"
study_dir="outputs/hand_relation_seed_study_20261016"
mkdir -p "$study_dir"

current_run="setup"
trap 'code=$?; printf "%s failed %s exit=%s\n" "$(date -Is)" "$current_run" "$code" | tee -a "$study_dir/progress.log"' ERR

run_one() {
    local variant="$1"
    local seed="$2"
    local entry checkpoint output_dir
    current_run="${variant}_seed${seed}"
    output_dir="$study_dir/$current_run"
    if [[ -e "$output_dir" ]]; then
        printf 'Refusing to reuse existing output directory: %s\n' "$output_dir" >&2
        return 1
    fi

    if [[ "$variant" == "baseline" ]]; then
        entry="train_best_weighted_100_cosine.py"
    else
        entry="train_best_hand_relation_100_cosine.py"
    fi

    printf '%s starting %s\n' "$(date -Is)" "$current_run" | tee -a "$study_dir/progress.log"
    "$python_bin" "$entry" --device cuda --seed "$seed" --compile \
        --save-dir "$output_dir" > "$study_dir/$current_run.log" 2>&1

    checkpoint="$(find "$output_dir" -mindepth 2 -maxdepth 2 -type f -name best.pt -print -quit)"
    if [[ -z "$checkpoint" ]]; then
        printf 'Missing best checkpoint for %s\n' "$current_run" >&2
        return 1
    fi

    "$python_bin" tools/evaluate_best.py --checkpoint "$checkpoint" --device cuda \
        --num-workers 8 --output-dir "$output_dir/evaluation_adaptive" \
        > "$study_dir/${current_run}_eval.log" 2>&1
    if [[ "$variant" == "relation" ]]; then
        "$python_bin" tools/evaluate_best.py --checkpoint "$checkpoint" --device cuda \
            --num-workers 8 --interaction-mode none \
            --output-dir "$output_dir/evaluation_none" \
            > "$study_dir/${current_run}_eval_none.log" 2>&1
    fi
    printf '%s completed %s checkpoint=%s\n' "$(date -Is)" "$current_run" "$checkpoint" \
        | tee -a "$study_dir/progress.log"
}

run_one baseline 2
run_one relation 2
run_one relation 3
run_one baseline 3
printf '%s all studies completed\n' "$(date -Is)" | tee -a "$study_dir/progress.log"
