#!/bin/bash
# eval_binary_all.sh — 统一二分类评估 (CPU, 全模型含 bal_mixed)
# 输出: models/binary_class_eval_all.json
set -e
export HOME=/home/devcontainers
export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd /mnt/c/Users/cai/OneDrive/Desktop/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

if pgrep -f "eval_binary_all.py" > /dev/null; then
    echo "SKIP: another eval_binary_all.py is running"
    exit 0
fi

CUDA_VISIBLE_DEVICES="" python3 eval_binary_all.py
echo "EVAL_BINARY_ALL_DONE rc=$?"
