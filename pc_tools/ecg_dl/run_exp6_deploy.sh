#!/bin/bash
# run_exp6_deploy.sh — exp6 部署链试点重训启动脚本 (阶段 1.5, TUNING_HISTORY 十三章)
# 1) 显式 LD_LIBRARY_PATH (README 环境约定; bash -lc 不保证加载 ~/.bashrc)
# 2) 持久日志 $HOME/exp6_deploy_train.log (勿用 /tmp — WSL 重启即清空)
# 3) 幂等: 若已有 train.py --deploy-chain 进程则跳过 (防重复启动)
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export ECG_PROCESSED_DIR=$HOME/ecg_data   # WSL 本地数据 (避开 OneDrive/9p, 降主机内存压力)
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

if pgrep -f "train.py.*--deploy-chain" > /dev/null; then
    echo "SKIP: train.py already running (PID $(pgrep -f 'train.py.*--deploy-chain'))"
    exit 0
fi

nohup python3 train.py --resnet-large --incart --ptb-beat --ptb-abn-max 10000 --domain-balanced --patient-split --epochs 200 --deploy-chain --patience 40 > ~/exp6_deploy_train.log 2>&1 &
echo "TRAIN_PID=$!"
