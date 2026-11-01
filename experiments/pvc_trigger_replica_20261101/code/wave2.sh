#!/usr/bin/env bash
# Usage: bash wave2.sh <queue-name> <arm:seed> [<arm:seed> ...]   (train2 then run_eval2, 2 threads each)
cd /tmp/pvcexp
export TF_CPP_MIN_LOG_LEVEL=2
F="AutoGraph\|autograph\|To silence\|Cause:\|oneDNN\|absl"
Q=$1; shift
for job in "$@"; do
  arm=${job%%:*}; seed=${job##*:}
  echo "=== $(date -u +%H:%M:%S) [$Q] train $arm s$seed"
  python3 train2.py "$arm" --seed "$seed" --epochs 8 --threads 2 2>&1 | grep --line-buffered -v "$F" > "logs/train2_${arm}_s${seed}.log"
  echo "=== $(date -u +%H:%M:%S) [$Q] eval $arm s$seed"
  python3 run_eval2.py "runs/${arm}_s${seed}" --threads 2 2>&1 | grep --line-buffered -v "$F" > "logs/eval2_${arm}_s${seed}.log"
  grep "^TEST" "logs/eval2_${arm}_s${seed}.log" | cut -c1-700
done
echo "=== $(date -u +%H:%M:%S) [$Q] QUEUE DONE"
