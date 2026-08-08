#!/bin/bash
# run_exp6_phase.sh — T2-5: 全类相位扰动增强训练 (参数化: $1 = max_shift)
# 部署链口径 + SGD + batch 64 (与 exp6_sgd 完全同配置) + ±$1 样本 batch 统一相位扰动
set -e
SHIFT="${1:-5}"
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=$HOME/ecg_data
cd "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl"

if pgrep -f "train.py.*--deploy-chain" > /dev/null; then
    echo "SKIP: another --deploy-chain training is running"
    exit 0
fi

nohup python3 train.py --resnet-large --incart --ptb-beat --ptb-abn-max 10000 --domain-balanced --patient-split --epochs 200 --deploy-chain --patience 40 --optimizer sgd --lr 0.01 --phase-shift $SHIFT > ~/exp6_phase${SHIFT}_train.log 2>&1 &
echo "TRAIN_PID=$! (shift=$SHIFT)"
