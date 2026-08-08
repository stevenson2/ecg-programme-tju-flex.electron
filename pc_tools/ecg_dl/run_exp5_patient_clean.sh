#!/bin/bash
# run_exp5_patient_clean.sh — exp5 (患者级清洁) 复现脚本 (T3-7/H3)
# exp5: PTB 训练拍 cap 10,000 (受控配比), 无域平衡; AdamW lr 5e-4; seed42 患者级
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=$HOME/ecg_data
cd "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl"

if pgrep -f "train.py.*--deploy-chain" > /dev/null; then
    echo "SKIP: another training is running"
    exit 0
fi

# 训练链口径 (filtfilt) + 患者级划分 + PTB cap 10000 (exp5 配置)
nohup python3 train.py --resnet-large --incart --ptb-beat --ptb-abn-max 10000 --patient-split --epochs 200 --patience 20 > ~/exp5_patient_clean_train.log 2>&1 &
echo "TRAIN_PID=$!"
