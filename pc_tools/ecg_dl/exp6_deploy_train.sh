#!/bin/bash
# exp6_deploy_train.sh — Launch exp6-config retrain on deployment-chain data with GPU
# Logs to persistent location (models/) not /tmp to survive WSL restart
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

WORKDIR="/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl"
LOGFILE="$WORKDIR/models/exp6_deploy_train.log"

cd "$WORKDIR"

echo "=== exp6-deploy training started at $(date) ===" | tee "$LOGFILE"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a "$LOGFILE"
python3 -c 'import tensorflow as tf; print("GPU:", tf.config.list_physical_devices("GPU"))' 2>&1 | tee -a "$LOGFILE"

python3 train.py \
    --resnet-large \
    --incart \
    --ptb-beat \
    --ptb-abn-max 10000 \
    --domain-balanced \
    --patient-split \
    --epochs 200 \
    --deploy-chain 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=$?
echo "=== exp6-deploy training finished at $(date) with exit code $EXIT_CODE ===" | tee -a "$LOGFILE"
exit $EXIT_CODE
