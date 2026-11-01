#!/usr/bin/env bash
# Usage: bash wave1.sh <queue-name> <arm> [<arm> ...]   (each arm: train2 then run_eval2, 2 threads each)
cd /tmp/pvcexp
export TF_CPP_MIN_LOG_LEVEL=2
F="AutoGraph\|autograph\|To silence\|Cause:\|oneDNN\|absl"
Q=$1; shift
for arm in "$@"; do
  echo "=== $(date -u +%H:%M:%S) [$Q] train $arm s42"
  python3 train2.py "$arm" --seed 42 --epochs 8 --threads 2 2>&1 | grep --line-buffered -v "$F" > "logs/train2_${arm}_s42.log"
  echo "=== $(date -u +%H:%M:%S) [$Q] eval $arm s42"
  python3 run_eval2.py "runs/${arm}_s42" --threads 2 2>&1 | grep --line-buffered -v "$F" > "logs/eval2_${arm}_s42.log"
  grep "^TEST" "logs/eval2_${arm}_s42.log" | cut -c1-600
done
echo "=== $(date -u +%H:%M:%S) [$Q] QUEUE DONE"
