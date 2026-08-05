#!/bin/bash
# run_bal_mixed.sh — 平衡混合单模型 (MIT+INCART + PTB abnormal, π=0.3)
# One ResNet-Large model, class-balanced to abnormal fraction 0.30.
# SGD+Nesterov (lr 0.01, momentum 0.9, wd 1e-4) per deploy standard.
# GPU 独占: pgrep guard against concurrent train_mixed_balanced.py runs.
set -e
# HOME 显式固定: Windows 环境变量继承会把它弄成 C:Userscai (WSL 路径错乱)
export HOME=/home/devcontainers
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=$HOME/ecg_data
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

if pgrep -f "train_mixed_balanced.py" > /dev/null; then
    echo "SKIP: another train_mixed_balanced.py training is running"
    exit 0
fi

nohup python3 train_mixed_balanced.py --abn-frac 0.30 --epochs 200 --patience 40 > ~/bal_mixed_train.log 2>&1 &
echo "TRAIN_PID=$!"
