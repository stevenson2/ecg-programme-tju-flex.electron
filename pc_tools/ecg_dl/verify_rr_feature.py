#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可行性验证 (2026-08-03): 单拍 + RR 间期特征 对 SVEB/整体 Recall 的提升

方法:
1. 从 MIT-BIH .atr 注解恢复每个心拍的 R 峰采样位置 (ann_indices) 与拍类型 (symbols)
2. 对患者级测试集每个拍计算 pre-RR / post-RR / RR-ratio (与前后拍的间隔)
3. 基线: 只用现有单拍模型概率 (P2A) 做二分类
4. 实验: 单拍概率 + RR 特征 拼接, 训练轻量分类器 (逻辑回归), 对比 SVEB Recall/Precision
5. 评估: 患者级测试, θ=0.5 与最优阈值

核心问题: RR 间期 (SVEB 联律间期提前) 能否显著提升 SVEB 识别?
"""
import sys
from pathlib import Path
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged, add_channel_dim
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_aami_breakdown import recover_mit_symbols_per_record, align_symbols_to_npz

MODELS = Path(__file__).resolve().parent / "models"
DATA_RAW = Path(__file__).resolve().parent / "data" / "raw" / "mit-bih-arrhythmia-database"
import wfdb


def build_rr_features():
    """从 MIT .atr 恢复每个拍的 pre-RR/post-RR (单位: 采样点@360Hz)。返回 dict record->{pre,post}."""
    rec2rr = {}
    rec_dir = DATA_RAW
    if not rec_dir.exists():
        print(f"!! {rec_dir} 不存在")
        return rec2rr
    for atr in sorted(rec_dir.glob("*.atr")):
        rec = atr.stem  # '100', '101', ...
        try:
            ann = wfdb.rdann(str(atr.with_suffix("")), 'atr')
            samples = ann.sample.astype(np.int64)
            rr = np.diff(samples)
            pre = np.concatenate([[np.nan], rr.astype(float)])
            post = np.concatenate([rr.astype(float), [np.nan]])
            rec2rr[rec] = {"pre": pre, "post": post}
        except Exception as e:
            print(f"  !! {rec}: {e}")
    print(f"  已加载 {len(rec2rr)} 条 MIT 记录的 RR 序列")
    return rec2rr


def main():
    set_npz_suffix("_deploy")
    print("=" * 78)
    print("可行性: 单拍 + RR 间期 对 SVEB/Recall 提升 (患者级测试)")
    print("=" * 78)

    mit_inc = load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr, va, te, stats = patient_level_split(mit_inc["record_ids"], pmap)
    x_te, y_te = mit_inc["beats"][te], mit_inc["labels"][te]
    rids_te = mit_inc["record_ids"][te]
    print(f"patient test: {te.sum()} beats")

    # 恢复 AAMI 符号 (识别 SVEB='S')
    per_rec_syms = recover_mit_symbols_per_record()
    sym_full, n_incart_unknown = align_symbols_to_npz(per_rec_syms, mit_inc["record_ids"], 6)
    sym_te = sym_full[te]
    n_s = (sym_te == 'S').sum()
    print(f"  SVEB (S): {n_s} 拍")

    # 单拍模型 (用于全量预测)
    m_p2a = tf.keras.models.load_model(str(MODELS / "archived" / "final_resnet_l_p2a_backup.h5"), compile=False)

    # ---- 构造 RR 特征 (全量, 用于 LR 训练) ----
    # npz 布局: [原始拍 | 噪声拍 | 缩放拍×2 | 漂移拍×2] (按增强类型拼接)
    print("\n构造 RR 特征 (全量 834741)...")
    n_all = len(mit_inc["beats"])
    rr_pre_all = np.full(n_all, np.nan)
    rr_post_all = np.full(n_all, np.nan)
    rec2rr = build_rr_features()

    all_rids = mit_inc["record_ids"]
    from collections import defaultdict
    groups_all = defaultdict(list)
    for i, rid in enumerate(all_rids):
        groups_all[int(rid)].append(i)

    for rid, idxs in groups_all.items():
        rec_name = str(int(rid))
        if rec_name in rec2rr:
            info = rec2rr[rec_name]
            n_ann = len(info["pre"])
            pre_arr = info["pre"]
            post_arr = info["post"]
            start = idxs[0]  # 该记录在 npz 中的起点
            for i in idxs:
                local = i - start
                orig_k = local % n_ann
                if orig_k < len(pre_arr):
                    rr_pre_all[i] = pre_arr[orig_k]
                    rr_post_all[i] = post_arr[orig_k]

    valid_all = ~np.isnan(rr_pre_all) & ~np.isnan(rr_post_all)
    print(f"  全量有 RR 特征: {valid_all.sum()}/{n_all}")

    # 全量 P2A 概率 (需对全量预测, 慢但必要)
    print("  全量 P2A 预测 (CPU)...")
    p_all = m_p2a.predict(add_channel_dim(mit_inc["beats"]), verbose=0, batch_size=1024)[:, 1]

    # 特征拼接 (全量)
    rr_pre_n = (rr_pre_all - np.nanmean(rr_pre_all)) / np.nanstd(rr_pre_all)
    rr_post_n = (rr_post_all - np.nanmean(rr_post_all)) / np.nanstd(rr_post_all)
    ratio_all = rr_pre_all / np.maximum(rr_post_all, 1e-9)
    ratio_n = (ratio_all - np.nanmean(ratio_all)) / np.nanstd(ratio_all)
    X_all = np.stack([p_all, rr_pre_n, rr_post_n, ratio_n], axis=-1)
    y_all = mit_inc["labels"]

    # 训练/评估划分: 非测试拍训练, 测试拍评估 (患者级无泄漏)
    tr_mask = tr & valid_all   # 训练患者
    va_mask = va & valid_all   # 验证患者
    te_mask = te & valid_all   # 测试患者
    x_tr_rr = X_all[tr_mask]; y_tr = y_all[tr_mask]
    x_va_rr = X_all[va_mask]; y_va = y_all[va_mask]
    x_te_rr = X_all[te_mask]; y_te_v = y_all[te_mask]
    x_tr_s = p_all[tr_mask].reshape(-1, 1)
    x_te_s = p_all[te_mask].reshape(-1, 1)

    # SVEB mask
    s_tr = (sym_full[tr_mask] == 'S')
    s_te = (sym_full[te_mask] == 'S')

    print(f"\n  train={x_tr_rr.shape[0]} val={x_va_rr.shape[0]} test={x_te_rr.shape[0]}")
    print(f"  SVEB in test: {s_te.sum()}")

    # ---- 基线: 单拍概率直接阈值 ----
    def eval_binary(y, score, th=0.5):
        pred = (score >= th).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        return p, r, f

    print("\n--- 基线 (单拍 P2A 概率, θ=0.5) ---")
    p_pa = x_te_s[:, 0]  # 单拍概率 (测试集, valid)
    p0, r0, f0 = eval_binary(y_te_v, p_pa, 0.5)
    print(f"  ALL: P={p0:.3f} R={r0:.3f} F1={f0:.3f}")
    if s_te.sum() > 0:
        ps, rs, fs = eval_binary(s_te.astype(int), p_pa, 0.5)
        print(f"  SVEB: P={ps:.3f} R={rs:.3f} F1={fs:.3f}")

    # ---- 实验: 单拍 + RR (逻辑回归) ----
    print("\n--- 实验 (单拍 + RR 特征, 逻辑回归) ---")
    # 用 val 选阈值, test 评估
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_tr_rr, y_tr)
    score_va = lr.predict_proba(x_va_rr)[:, 1]
    score_te = lr.predict_proba(x_te_rr)[:, 1]

    # 阈值网格在 val 上选最优 F1
    best_th, best_f1 = 0.5, -1
    for th in np.arange(0.2, 0.8, 0.05):
        _, _, fv = eval_binary(y_va, score_va, th)
        if fv > best_f1:
            best_f1, best_th = fv, th
    print(f"  最优阈值 (val): θ={best_th:.2f}")

    p1, r1, f1 = eval_binary(y_te_v, score_te, best_th)
    print(f"  ALL: P={p1:.3f} R={r1:.3f} F1={f1:.3f}")
    if s_te.sum() > 0:
        ps, rs, fs = eval_binary(s_te.astype(int), score_te, best_th)
        print(f"  SVEB: P={ps:.3f} R={rs:.3f} F1={fs:.3f}")

    # 对比提升
    print(f"\n  提升: ALL-R {r0:.3f}→{r1:.3f} ({r1-r0:+.3f}) | "
          f"SVEB-R {rs if s_te.sum()>0 else 'N/A'}")

    # AUC
    print(f"  AUC: 单拍={roc_auc_score(y_te_v, p_pa):.4f} | 单拍+RR={roc_auc_score(y_te_v, score_te):.4f}")


if __name__ == "__main__":
    main()
