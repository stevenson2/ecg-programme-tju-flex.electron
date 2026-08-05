#!/bin/bash
# run_3beat_deploy.sh — exp6 3-beat (CNN-M-Large) 部署链试点训练
# 验证假设: 多拍窗口 (3-beat, 750pt) 能否提升 SVEB/F recall vs 单拍 (250pt)。
# 配置对齐 exp6-deploy SGD 臂: SGD+Nesterov lr=0.01, patience=40, 患者级 seed42。
# 数据: mit_incart_3beat_deploy.npz (MIT+INCART, 不含 PTB — PTB 是 MI 域,
#   对 SVEB 无贡献; 试点聚焦 AAMI 分报对比, 后续如需 PTB 再扩展)。
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=$HOME/ecg_data
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

if pgrep -f "train.py.*--cnn-m-large" > /dev/null; then
    echo "SKIP: --cnn-m-large training already running"
    exit 0
fi

nohup python3 train.py --cnn-m-large --3beat --deploy-chain --patient-split \
    --epochs 200 --patience 40 --optimizer sgd --lr 0.01 \
    > ~/exp6_3beat_deploy_train.log 2>&1 &
echo "TRAIN_PID=$!"
