#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

if [[ -n "${WAIT_PID:-}" ]]; then
    echo "等待训练进程 PID=${WAIT_PID} 完成..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 30
    done
else
    echo "查找当前 train_best_no_pool.py 训练进程..."
    while true; do
        pids="$(pgrep -f '[p]ython([^ ]*)?.*train_best_no_pool\.py([[:space:]]|$)' || true)"
        if [[ -z "$pids" ]]; then
            break
        fi
        echo "仍在训练，PID=${pids//$'\n'/, }；30秒后重试..."
        sleep 30
    done
fi

echo "当前训练已结束，启动100帧置信度加权压缩训练。"
exec python train_best_no_pool_weighted_100.py "$@"
