#!/usr/bin/env bash
# Takes over the queue from pipeline.sh after its first training job (PID passed as $1) ends.
cd /tmp/pvcexp
export TF_CPP_MIN_LOG_LEVEL=2
F="AutoGraph\|autograph\|To silence\|Cause:"
while kill -0 "$1" 2>/dev/null; do sleep 15; done
evaluate() {
  python3 run_eval.py "runs/$1_s$2" 2>&1 | grep --line-buffered -v "$F" > "logs/eval_$1_s$2.log"
  grep "^TEST" "logs/eval_$1_s$2.log" | cut -c1-400
}
echo "=== $(date -u +%H:%M:%S) eval ref s42"
evaluate ref 42
for job in "trig 42" "trigNV 42" "trig_noQ 42" "ref 43" "trig 43" "trigNV 43"; do
  set -- $job
  echo "=== $(date -u +%H:%M:%S) train $1 s$2"
  python3 train.py "$1" --seed "$2" --epochs 8 2>&1 | grep --line-buffered -v "$F" > "logs/train_$1_s$2.log"
  echo "=== $(date -u +%H:%M:%S) eval $1 s$2"
  evaluate "$1" "$2"
done
echo "=== $(date -u +%H:%M:%S) ALL DONE"
