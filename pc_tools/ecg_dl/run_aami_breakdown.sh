#!/bin/bash
# run_aami_breakdown.sh — AAMI 类别分报评估启动脚本
# 用法: bash run_aami_breakdown.sh [model_name] [tag] [deploy_suffix]
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

MODEL=${1:-best_resnet_large_exp6_deploy.h5}
TAG=${2:-exp6_deploy}
SUFFIX=${3:-_deploy}
MODE=${4:-}   # 传 "beat" 则跑 beat 级 (全拍), 否则患者级

EXTRA=""
if [ "$MODE" = "beat" ]; then
    EXTRA="--beat-level"
fi

python3 eval_aami_breakdown.py --model "$MODEL" --tag "$TAG" --deploy-suffix "$SUFFIX" $EXTRA
