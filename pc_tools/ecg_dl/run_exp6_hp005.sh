#!/bin/bash
# run_exp6_hp005.sh — exp6 降HP 0.05Hz 部署链重训 (TUNING_HISTORY §8.3.1/§8.6)
# 固件+PC链已同步 0.05Hz, 数据已重建 (build_deploy_npz.py 全验证 PASS)。
# 配置 = exp6 SGD 臂 (PTB域平衡, --optimizer sgd --lr 0.01 --patience 40)。
# 预期: PTB D3 0.7697 (0.5Hz SGD) → 恢复 0.05Hz 形态后逼近 0.78-0.82。
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=$HOME/ecg_data
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

if pgrep -f "train.py.*--deploy-chain" > /dev/null; then
    echo "SKIP: another --deploy-chain training is running"
    exit 0
fi

nohup python3 train.py --resnet-large --incart --ptb-beat --ptb-abn-max 10000 \
    --domain-balanced --patient-split --epochs 200 --deploy-chain --patience 40 \
    --optimizer sgd --lr 0.01 > ~/exp6_hp005_train.log 2>&1 &
echo "TRAIN_PID=$!"
