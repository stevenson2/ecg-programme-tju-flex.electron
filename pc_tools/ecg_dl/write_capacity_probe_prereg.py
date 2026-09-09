#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""write_capacity_probe_prereg.py — 容量/架构探针预注册 (TH §109)
================================================================================
探针问题 (承接 §105/P2): 在 v3 A 数据包与 SplitGuard 纪律下, 从零训练
更大/更深/更宽架构, 检验"62,834 参数模型从零容量不足"是否为 fc1 嵌入
不可分离 (嵌入 AUC≈0.5433) 与 val 侧正常拍高置信误报的结构性原因。

铁律:
  - 本脚本只在任何训练/评估运行之前执行一次, 生成 models/deploy_match/
    capacity_probe_prereg.json; 门常量与冻结操作点写入后不得更改。
  - 数据来源锁定 distill_pilot_teacher_targets.npz 的 x_train_perm / y_train /
    x_val / y_val / x_real_holdout (与 v3 A 训练集逐拍同源); 不读取教师
    p_train/p_val; 测试集零接触 (禁止加载 mit/ptb_deploy_causal_match.npz)。
  - 判定只允许 PASS / PARTIAL / UNPROVEN; 不使用"证伪"表述。

运行: python3 write_capacity_probe_prereg.py   (WSL 或任意有 python3 的环境)
输出: models/deploy_match/capacity_probe_prereg.json
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
OUT = CACHE / "capacity_probe_prereg.json"

FROZEN = {"theta": 0.95, "k": 1, "n": 5, "cooldown": 20}

GATES = {
    "primary_relative_reduction_min": 0.30,
    "guardrail_auc_max_drop": 0.02,
    "guardrail_event_recall_max_drop": 0.02,
    "guardrail_alert_blocks_min": 3,
    "real_holdout_mean_prob_max": 0.10,
    "real_holdout_frac_gt_0.5_max": 0.0,
}

VARIANTS = {
    "cap_w2": {
        "description": "宽度×2, 深度不变 (large 预设同深度)",
        "filters": [32, 64, 128, 256],
        "blocks_per_stage": [2, 3, 3, 1],
        "kernel_sizes": [7, 5, 3, 3],
        "strides": [1, 2, 2, 1],
        "dropout": 0.4,
        "params_frozen": 222018,
        "params_ratio_vs_v3A": 3.53,
    },
    "cap_d2": {
        "description": "宽度不变, 加深 2 个残差块 (stage C +1, stage D +1)",
        "filters": [16, 32, 64, 128],
        "blocks_per_stage": [2, 3, 4, 2],
        "kernel_sizes": [7, 5, 3, 3],
        "strides": [1, 2, 2, 1],
        "dropout": 0.4,
        "params_frozen": 95906,
        "params_ratio_vs_v3A": 1.53,
    },
    "cap_hybrid": {
        "description": "宽度×1.5 + 深度+1 (stage A +1); 优先变体",
        "filters": [24, 48, 96, 192],
        "blocks_per_stage": [3, 3, 3, 1],
        "kernel_sizes": [7, 5, 3, 3],
        "strides": [1, 2, 2, 1],
        "dropout": 0.4,
        "params_frozen": 131808,
        "params_ratio_vs_v3A": 2.10,
    },
}

TRAINING_CONFIG = {
    "optimizer": "adam",
    "lr_schedule": "cosine decay from 3e-4 over 80 epochs",
    "epochs": 80,
    "batch_size": 32,
    "early_stopping_patience": 20,
    "monitor": "val_auc (custom epoch callback, best h5 saved)",
    "restore_best_weights": True,
    "class_weight": None,
    "loss": "sparse_categorical_crossentropy",
    "init": "from scratch, no pretrained weights, no KD / teacher targets",
    "head": "GAP -> Dense(64,relu) fc1 -> Dropout(0.4) -> Dense(2,softmax)",
    "input_output": "250x1 -> 2 classes (数据口径不变)",
}


def payload():
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §109 容量/架构探针预注册: 容量提高能否修复 fc1 嵌入"
                   "不可分离与 val 侧正常拍高置信误报 (v3 A 62,834 参数对照)",
        "baseline_ref": {
            "model": "models/best_resnet_large_clean_baseline_v3.h5",
            "params": 62834,
            "role": "只作对照, 不重训",
            "baseline_references_from_disk": {
                "p1_mit_frac_gt_0.95": 0.24480296159996495,
                "p1_incart_frac_gt_0.95": 0.2735855129057867,
                "p1_mit_auc": 0.7679994798154276,
                "p1_ptb_auc": 0.8262335343912346,
                "sweep_mit_event_recall": 0.9807,
                "sweep_mit_alert_blocks": 401,
                "p2_embed_auc": 0.543305347518394,
                "p2_fp_frac_nearest_abn": 0.996952608794714,
            },
        },
        "data": {
            "source": "models/deploy_match/distill_pilot_teacher_targets.npz",
            "keys_used": ["x_train_perm", "y_train", "x_val", "y_val",
                          "x_real_holdout"],
            "keys_forbidden": ["p_train", "p_val"],
            "identity_anchor": "x_train_perm 5942 拍 (abn 2500 / norm 3442), "
                               "x_val 4040 拍, holdout 40 拍 — 与 v3 A "
                               "train_clean_baseline_v3.json 逐项一致, "
                               "训练脚本硬断言",
            "zero_test_contact": "禁止加载 mit_deploy_causal_match.npz / "
                                 "ptb_deploy_causal_match.npz / 任何 test_mask",
        },
        "variants": VARIANTS,
        "run_order": ["cap_hybrid", "cap_w2", "cap_d2"],
        "seeds": {
            "primary": [42],
            "contingent": [43, 44],
            "rule": "每变体先 seed 42; 仅当该变体 seed 42 主门相对降幅>=0.30 "
                    "时, 补 seed 43/44 做稳健性",
        },
        "training_config": TRAINING_CONFIG,
        "frozen_operating_point": FROZEN,
        "gates": GATES,
        "gate_definitions": {
            "primary": "val (MIT+INCART val 患者全序列) 正常拍 pooled "
                       "frac>0.95 相对 v3 A 同进程重算值降幅 >= 0.30",
            "guardrail_A": "val MIT+INCART AUC 相对 v3 A drop <= 0.02",
            "guardrail_B": "冻结点 (0.95, 1-of-5, cooldown=20) 事件召回 "
                           "drop <= 0.02 且报警块 >= 3",
            "guardrail_C": "真实 AFE 留出 40 拍 mean prob < 0.10 且 "
                           "frac>0.5 = 0",
        },
        "verdict_semantics": {
            "PASS": "主门 + 护栏 A/B/C 全过",
            "PARTIAL": "主门过但至少一条护栏破 (或嵌入分离度改善但未过门)",
            "UNPROVEN": "主门未过 (相对降幅在 0–30% 区间或为负) 记未证明; "
                        "结果在种子噪声区间内不称证伪",
        },
        "embed_diagnostics": {
            "method": "与 p2_embed_separation.py 同口径: val MIT+INCART 全量拍"
                      "(SplitGuard val_mask), fc1 (Dense64 relu) 嵌入, "
                      "leave-one-record-out 原型, dist_ratio=dn/(dn+da)",
            "metrics": ["embedding_auc_label_vs_ratio", "FP frac_nearest_abn",
                        "FP mean_dist_ratio", "FP/TP 距离比"],
            "success_line_advisory": "嵌入 AUC 相对 v3 A 0.5433 有实质提升 "
                                     "(建议 >0.65), 或 FP 最近邻异常比例显著下降",
        },
        "outputs": {
            "prereg": "models/deploy_match/capacity_probe_prereg.json",
            "models": "models/deploy_match/best_resnet_capacity_<variant>_seed<seed>.h5",
            "train_json": "models/deploy_match/train_capacity_<variant>_seed<seed>.json",
            "train_csv": "models/deploy_match/train_history_capacity_<variant>_seed<seed>.csv",
            "train_log": "models/deploy_match/capacity_probe_train.log",
            "eval_json": "models/deploy_match/capacity_probe_eval.json",
            "embed_json": "models/deploy_match/capacity_probe_embed.json",
            "eval_log": "models/deploy_match/capacity_probe_eval.log",
        },
        "discipline": {
            "frozen_scripts": "不修改任何已冻结脚本; 全部新功能在新脚本中",
            "log_discipline": "stdout 重定向到仓库 .log, 用 [CAP]/[CP]/[DPE] "
                              "标记核对; 不相信包装退出码",
            "numbers": "数字只认盘上 artifact (JSON/日志)",
            "history": "结论只追加 02_文档与规划/_restored/TUNING_HISTORY.restored.md "
                       "§109; 公开仓库 docs 不回填",
        },
    }


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload(), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[CAP] prereg saved: {OUT}", flush=True)
    print(f"[CAP] variants: {list(VARIANTS)} seeds={payload()['seeds']['primary']}",
          flush=True)
    print(f"[CAP] gates: {GATES}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
