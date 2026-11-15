#!/usr/bin/env bash
set -u
PID=1803375
RUN=/home/xiong/Desktop/ISAA/outputs/best_face_pair_100_seed2_20261105/20261105-051202-2fcd5590
CODE=/home/xiong/Desktop/ISAA/experiments/face_pair_stable_20261103/code
VERIFY=/home/xiong/Desktop/ISAA/experiments/face_pair_seed2_20261105/verification
LOG=/home/xiong/Desktop/ISAA/experiments/face_pair_seed2_20261105/monitor.log
exec >>"$LOG" 2>&1
echo "MONITOR_START $(date -Is) pid=$PID"
while kill -0 "$PID" 2>/dev/null; do
  echo "MONITOR_TICK $(date -Is)"
  sleep 60
done
sleep 3
STATUS="$RUN/status.json"
echo "MONITOR_EXIT $(date -Is)"
cat "$STATUS" 2>/dev/null || true
if [ -f "$RUN/best.pt" ] && [ -f "$STATUS" ] && grep -q 'completed' "$STATUS"; then
  mkdir -p "$VERIFY"
  OUT="$VERIFY/accuracy_weighted"
  if [ ! -e "$OUT/summary.json" ]; then
    /home/xiong/venvs/isaa-cu126/bin/python -s "$CODE/tools/evaluate_face_pair_accuracy_weighted.py" \
      --checkpoint "$RUN/best.pt" \
      --output-dir "$OUT" \
      --archive /home/xiong/Desktop/ISAA/data/ntu60_skeletons_rtmw.zip \
      --batch-size 32 --num-workers 8 --device cuda
  else
    echo "EVAL_ALREADY_PRESENT $OUT"
  fi
else
  echo "MONITOR_NO_EVAL best_or_completed_missing"
fi