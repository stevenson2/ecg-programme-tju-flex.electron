#!/bin/bash
# launch_exp7b.sh - 可靠后台启动 exp7b 训练 (setsid 完全脱离会话, 防 WSL VM 关闭误杀)
cd "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl"
# 清理任何残留训练进程 (防重复)
pkill -f "train.py.*--deploy" 2>/dev/null && sleep 2 || true
# setsid + nohup + stdin/out/err 全脱离: 进程进入新会话, 不随启动 shell 退出被杀
setsid nohup bash run_exp7b.sh > /home/devcontainers/exp7b_run.log 2>&1 < /dev/null &
echo "LAUNCHED run_exp7b.sh (setsid detach)"
sleep 5
echo "--- process check ---"
pgrep -af "train.py|run_exp7b" || echo "(训练进程尚未出现, 可能在数据加载)"
