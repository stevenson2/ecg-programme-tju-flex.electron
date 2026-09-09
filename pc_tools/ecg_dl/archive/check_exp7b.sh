#!/bin/bash
# check_exp7b.sh - 检查 exp7b 训练状态 (避免 PowerShell 引号地狱)
echo "=== run log (exp7b_run.log) ==="
tail -40 /home/devcontainers/exp7b_run.log 2>/dev/null
echo ""
echo "=== train log tail (exp7b_train.log) ==="
tail -50 /home/devcontainers/exp7b_train.log 2>/dev/null
echo ""
echo "=== train.py process ==="
pgrep -af train.py || echo "(no train.py running)"
echo ""
echo "=== models dir exp7b / train_history ==="
cd "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl/models"
ls -la | grep -iE 'exp7b|train_history\.csv|scratch'
