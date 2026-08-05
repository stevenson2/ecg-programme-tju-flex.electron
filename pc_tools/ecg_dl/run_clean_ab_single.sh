#!/bin/bash
# run_clean_ab_single.sh — 单拍 (250pt) 干净对照臂 A
# 数据: mit_incart_merged deploy (MIT+INCART, 无 PTB) — 与臂 B 相同数据源
# 模型: ResNet-L, SGD lr=0.01 patience 40, 患者级 seed42
# 与臂 B (3-beat 750pt CNN-M-Large) 唯一变量 = 窗口。
# 注意: 单拍最佳对照模型应含 PTB 域平衡 (exp6 口径)? 否 — 干净对照排除 PTB,
#   聚焦"窗口"单变量。模型权重存 best_resnet_large.h5 (checkpoint 已确保)。
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=$HOME/ecg_data
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

if pgrep -f "train.py.*--resnet-large.*--incart" > /dev/null; then
    echo "SKIP: single-beat clean-A training already running"
    exit 0
fi

nohup python3 train.py --resnet-large --incart --deploy-chain --patient-split \
    --epochs 200 --patience 40 --optimizer sgd --lr 0.01 \
    > ~/exp6_cleanab_single_train.log 2>&1 &
echo "TRAIN_PID=$!"
