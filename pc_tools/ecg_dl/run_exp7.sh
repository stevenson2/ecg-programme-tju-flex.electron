#!/bin/bash
# run_exp7.sh — P0-2 exp7 修正后因果链重训 (SGD, 与 exp6-SGD 完全同配置)
# 数据源: *_deploy_causal.npz (D3 部署链 + 因果 HP 0.5Hz@250Hz 修正系数)
# 配置: --resnet-large --incart --ptb-beat --ptb-abn-max 10000 --domain-balanced
#       --patient-split --epochs 200 --patience 40 --optimizer sgd --lr 0.01
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data
cd "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl"

if pgrep -f "train.py.*--deploy" > /dev/null; then
    echo "SKIP: another --deploy training is running"
    exit 0
fi

# 备份现有通用名 scratch 文件 (避免静默丢失)
TS=$(date +%Y%m%d_%H%M%S)
for f in best_resnet_large.h5 final_resnet_l.h5 train_history.csv; do
    if [ -f "models/$f" ]; then
        mv -f "models/$f" "models/${f%.*}_scratch_${TS}.${f##*.}"
        echo "backup: models/$f -> models/${f%.*}_scratch_${TS}.${f##*.}"
    fi
done

# 训练 (前台阻塞; set -e 下失败即退出不归档)
python3 train.py --resnet-large --incart --ptb-beat --ptb-abn-max 10000 \
    --domain-balanced --patient-split --epochs 200 --deploy-causal --patience 40 \
    --optimizer sgd --lr 0.01 > /home/devcontainers/exp7_train.log 2>&1

# 归档为 exp7 专属名
mv -f models/best_resnet_large.h5 models/best_resnet_large_exp7.h5
mv -f models/final_resnet_l.h5 models/final_resnet_l_exp7.h5
mv -f models/train_history.csv models/train_history_exp7.csv
echo "EXP7_ARCHIVED"
