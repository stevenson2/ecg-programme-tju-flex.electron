#!/bin/bash
# run_exp6_sgd.sh — exp6 部署链 SGD 对照臂 (A/B: adamw vs sgd, TUNING_HISTORY 十三章)
# SGD+Nesterov (momentum 0.9, wd 1e-4), lr 0.01 (adamw 5e-4 的 ~20x, Wilson et al. 2017 泛化证据)
# 仅在当前 adamw run 结束后运行 (GPU 独占)
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=$HOME/ecg_data
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

if pgrep -f "train.py.*--deploy-chain" > /dev/null; then
    echo "SKIP: another --deploy-chain training is running"
    exit 0
fi

nohup python3 train.py --resnet-large --incart --ptb-beat --ptb-abn-max 10000 --domain-balanced --patient-split --epochs 200 --deploy-chain --patience 40 --optimizer sgd --lr 0.01 > ~/exp6_sgd_train.log 2>&1 &
echo "TRAIN_PID=$!"
