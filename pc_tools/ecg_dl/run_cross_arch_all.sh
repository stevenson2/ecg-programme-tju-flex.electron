#!/bin/bash
# run_cross_arch_all.sh — 跨架构部署链失配对照：训练 3 架构 × 2 链 + 评估
# 顺序执行避免 GPU OOM；日志保存在 pc_tools/ecg_dl/models/cross_arch/logs/
set -e
cd "$(dirname "$0")"

LOG_DIR="models/cross_arch/logs"
mkdir -p "$LOG_DIR"

ARCHS=(lstm_cnn cnn_standard resnet1d)
CHAINS=(baseline deploy)

for arch in "${ARCHS[@]}"; do
  for chain in "${CHAINS[@]}"; do
    echo "=== $(date) TRAIN ${arch}/${chain} ===" | tee -a "$LOG_DIR/run_all.log"
    python3 -u train_cross_arch.py --arch "$arch" --chain "$chain" \
        --epochs 30 --patience 10 --batch-size 256 \
        --steps-per-epoch 200 --val-steps 0 \
        > "$LOG_DIR/${arch}_${chain}.log" 2>&1
    echo "=== $(date) DONE ${arch}/${chain} ===" | tee -a "$LOG_DIR/run_all.log"
  done
done

echo "=== $(date) EVAL ===" | tee -a "$LOG_DIR/run_all.log"
python3 -u eval_cross_arch.py > "$LOG_DIR/eval.log" 2>&1
echo "=== $(date) ALL DONE ===" | tee -a "$LOG_DIR/run_all.log"
