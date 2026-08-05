#!/bin/bash
# run_kd_screen.sh — KD 网格粗筛: α{0.3,0.5,0.7} × T{1,3,5} = 9 runs, patience 15
# 并发控制: 保持最多 CONCURRENCY 个训练进程 (GPU 5.2GB, 2 并发已验证冒烟 OK)
# 每 run 完成立即跑 deploy eval. 全程 nohup 日志到 ~/kd_<name>.log
# 用法: bash run_kd_screen.sh [--concurrency N]
set -uo pipefail

CONCURRENCY=2
if [ "${1:-}" = "--concurrency" ]; then
    CONCURRENCY="$2"
fi

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data
cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl

# 前置检查: teacher logits 必须已生成
if ! ls models/teacher_logits_ssl_a_train.npy >/dev/null 2>&1; then
    echo "ERROR: teacher logits missing. Run precompute_teacher_logits.py first."
    exit 1
fi

ALPHAS=(0.3 0.5 0.7)
TEMPS=(1 3 5)

# 命名: alpha x100 整数 (a030), t 整数 (t1)
make_name() {
    local a="$1" t="$2"
    printf "kd_a%03d_t%d" "$(python3 -c "print(int(round($a*100)))")" "$t"
}

# 生成全部组合
COMBOS=()
for a in "${ALPHAS[@]}"; do
    for t in "${TEMPS[@]}"; do
        COMBOS+=("$a $t")
    done
done

echo "[$(date +%H:%M:%S)] SCREEN START: ${#COMBOS[@]} combos, concurrency=$CONCURRENCY"

# 运行中数组: pid -> "alpha temp"
declare -A RUNNING=()

run_one() {
    local a="$1" t="$2" name
    name=$(make_name "$a" "$t")
    echo "[$(date +%H:%M:%S)] LAUNCH $name (alpha=$a T=$t)"
    python3 train_kd.py --alpha "$a" --temperature "$t" \
        --epochs 200 --patience 15 \
        --model-name "${name}.h5" --log-suffix "$name" \
        > "$HOME/${name}.log" 2>&1 &
    RUNNING[$!]="$a $t"
}

eval_one() {
    local a="$1" t="$2" name
    name=$(make_name "$a" "$t")
    if [ -f "models/${name}.h5" ]; then
        python3 eval_exp6_deploy.py --model "${name}.h5" --out "${name}_eval.json" \
            > "$HOME/${name}_eval.log" 2>&1 \
            && echo "[$(date +%H:%M:%S)] EVAL OK $name" \
            || echo "[$(date +%H:%M:%S)] EVAL FAIL $name (see ~/${name}_eval.log)"
    else
        echo "[$(date +%H:%M:%S)] SKIP eval $name (checkpoint missing)"
    fi
}

# 主循环
for combo in "${COMBOS[@]}"; do
    read -r a t <<< "$combo"
    # 等待槽位: 若满, 收割完成的
    while [ ${#RUNNING[@]} -ge "$CONCURRENCY" ]; do
        for pid in "${!RUNNING[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null
                rc=$?
                read -r da dt <<< "${RUNNING[$pid]}"
                name=$(make_name "$da" "$dt")
                echo "[$(date +%H:%M:%S)] DONE $name (rc=$rc)"
                unset "RUNNING[$pid]"
                eval_one "$da" "$dt"
            fi
        done
        [ ${#RUNNING[@]} -lt "$CONCURRENCY" ] && break
        sleep 15
    done
    run_one "$a" "$t"
done

# 等待剩余
while [ ${#RUNNING[@]} -gt 0 ]; do
    for pid in "${!RUNNING[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            rc=$?
            read -r da dt <<< "${RUNNING[$pid]}"
            name=$(make_name "$da" "$dt")
            echo "[$(date +%H:%M:%S)] DONE $name (rc=$rc)"
            unset "RUNNING[$pid]"
            eval_one "$da" "$dt"
        fi
    done
    [ ${#RUNNING[@]} -eq 0 ] && break
    sleep 15
done

echo "[$(date +%H:%M:%S)] SCREEN COMPLETE"
