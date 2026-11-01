#!/usr/bin/env bash
# Sequential queue: train then evaluate each (arm, seed).
cd /tmp/pvcexp
export TF_CPP_MIN_LOG_LEVEL=2
for job in "ref 42" "trig 42" "trig_noQ 42" "ref 43" "trig 43"; do
  set -- $job
  echo "=== $(date -u +%H:%M:%S) train $1 s$2"
  python3 train.py "$1" --seed "$2" --epochs 8 2>&1 | grep -v "AutoGraph\|autograph\|To silence\|Cause:" > "logs/train_$1_s$2.log"
  echo "=== $(date -u +%H:%M:%S) eval $1 s$2"
  python3 run_eval.py "runs/$1_s$2" 2>&1 | grep -v "AutoGraph\|autograph\|To silence\|Cause:" > "logs/eval_$1_s$2.log"
  grep "^TEST" "logs/eval_$1_s$2.log" | cut -c1-400
done
echo "=== $(date -u +%H:%M:%S) ALL DONE"
