#!/bin/bash
# run_eval_aami_3beat.sh — 3-beat AAMI 分报评估启动脚本
# 用法: bash run_eval_aami_3beat.sh [model] [tag] [beat|patient]
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

MODEL=${1:-final_cnn_m_large.h5}
TAG=${2:-exp6_3beat}
MODE=${3:-patient}
EXTRA=""
if [ "$MODE" = "beat" ]; then
    EXTRA="--beat-level"
fi

python3 eval_aami_3beat.py --model "$MODEL" --tag "$TAG" $EXTRA
