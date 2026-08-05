#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""低算力模拟: RR 间期节律模式对 SVEB 的检测能力 (2026-08-03, 方案A延伸)

不训练深度模型, 只用 numpy/sklearn 验证"节律模式" (RR 序列局部特征) 能否区分 SVEB。
模拟:
  S1: 3-RR 滑窗规则 (CinC 2002 风格) — 当前拍 pre/post RR 相对局部均值偏离
  S2: RR 比值特征 (pre/局部mean, post/局部mean) + 轻量规则
  S3: 单拍概率 + RR 特征 -> 随机森林 (非线性, 类别特定)
评估: 患者级测试, SVEB 的 Recall/Precision + ALL 对比单拍基线。
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged, add_channel_dim
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_aami_breakdown import recover_mit_symbols_per_record, align_symbols_to_npz
import wfdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

MODELS = Path(__file__).resolve().parent / "models"
rec_dir = Path(__file__).resolve().parent / "data" / "raw" / "mit-bih-arrhythmia-database"


def main():
    set_npz_suffix("_deploy")
    print("=" * 78)
    print("低算力模拟: RR 节律模式 vs SVEB")
    print("=" * 78)
    mi = load_mit_incart_merged()
    per = recover_mit_symbols_per_record()
    sym, nuk = align_symbols_to_npz(per, mi["record_ids"], 6)

    # 恢复全量 RR 序列 (含前 3 个 RR 的滑动窗口)
    n_all = len(mi["beats"])
    rr_pre = np.full(n_all, np.nan)
    rr_post = np.full(n_all, np.nan)
    rr_seq = np.full(n_all, np.nan)   # 当前拍的 RR (与下一拍)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, rid in enumerate(mi["record_ids"]):
        groups[int(rid)].append(i)

    for rid, idxs in groups.items():
        rec_name = str(int(rid))
        atr_path = rec_dir / f"{rec_name}.atr"
        if atr_path.exists():
            ann = wfdb.rdann(str(atr_path.with_suffix("")), 'atr')
            samples = ann.sample.astype(np.int64)
            rr = np.diff(samples).astype(float)
            pre = np.concatenate([[np.nan], rr])
            post = np.concatenate([rr, [np.nan]])
            # 前3个RR的均值 (滑动窗口, 用原始索引)
            n_ann = len(pre)
            start = idxs[0]
            for i in idxs:
                local = i - start
                orig_k = local % n_ann
                if orig_k < len(pre):
                    rr_pre[i] = pre[orig_k]
                    rr_post[i] = post[orig_k]
                    # 3-RR 局部窗口: 当前拍前后各1个RR + 当前
                    k = orig_k
                    if k >= 1 and k < len(rr) - 1:
                        w = rr[max(0,k-1):k+2]
                        rr_seq[i] = np.mean(w)

    # 患者级测试
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({r + 100000: "inc_" + p for r, p in build_incart_patient_map().items()})
    tr, va, te, _ = patient_level_split(mi["record_ids"], pmap)
    sym_te = sym[te]
    y_te = mi["labels"][te]
    valid = te & ~np.isnan(rr_pre) & ~np.isnan(rr_post) & ~np.isnan(rr_seq)

    # 单拍概率 (仅 valid 拍)
    m_p2a = tf.keras.models.load_model(str(MODELS / "archived" / "final_resnet_l_p2a_backup.h5"), compile=False)
    p_te = m_p2a.predict(add_channel_dim(mi["beats"][valid]), verbose=0, batch_size=1024)[:, 1]
    y_v = mi["labels"][valid]
    sym_v = sym[valid]
    rr_pre_v = rr_pre[valid]
    rr_post_v = rr_post[valid]
    rr_seq_v = rr_seq[valid]

    n_te = len(y_v)
    n_s = int((sym_v == "S").sum())
    print(f"test(valid): {n_te}, SVEB: {n_s} ({n_s/n_te*100:.2f}%)")

    # 特征
    # 相对局部均值: pre/seq 和 post/seq (早搏 = 显著小于1, 代偿 = 显著大于1)
    ratio_pre = rr_pre_v / np.maximum(rr_seq_v, 1e-9)
    ratio_post = rr_post_v / np.maximum(rr_seq_v, 1e-9)

    def eval_binary(y, score, th=0.5):
        pred = (score >= th).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        return p, r, f

    # 基线: 单拍概率
    print("\n--- 基线 (单拍 P2A) ---")
    p0, r0, f0 = eval_binary(y_v, p_te, 0.5)
    s_mask = (sym_v == "S").astype(int)
    p_s0, r_s0, f_s0 = eval_binary(s_mask, p_te, 0.5)
    print(f"  ALL: P={p0:.3f} R={r0:.3f} F1={f0:.3f} | SVEB: P={p_s0:.3f} R={r_s0:.3f} F1={f_s0:.3f}")

    # ---- S1: 3-RR 比值规则: 早搏(显著缩短) 或 代偿(显著延长) -> 异常 ----
    print("\n--- S1: 3-RR 比值规则 (pre/seq<0.7 或 >1.25 -> 异常) ---")
    s1 = ((ratio_pre < 0.7) | (ratio_pre > 1.25)).astype(int)
    # 规则触发率 (对正常拍)
    n_mask = sym_v == "N"
    print(f"  规则触发: 正常拍 {(s1[n_mask]==1).mean()*100:.1f}%, "
          f"SVEB拍 {(s1[s_mask]==1).mean()*100:.1f}%")
    # 合并单拍: 单拍报警 OR 规则报警
    combo1 = ((p_te >= 0.5) | (s1 == 1)).astype(int)
    p1, r1, f1 = eval_binary(y_v, combo1, 0.5)
    ps1, rs1, fs1 = eval_binary(s_mask, combo1, 0.5)
    print(f"  ALL: P={p1:.3f} R={r1:.3f} F1={f1:.3f} | SVEB: P={ps1:.3f} R={rs1:.3f} F1={fs1:.3f}")
    # 纯规则 (不用单拍)
    p1r, r1r, f1r = eval_binary(y_v, s1, 0.5)
    ps1r, rs1r, fs1r = eval_binary(s_mask, s1, 0.5)
    print(f"  纯规则: ALL: P={p1r:.3f} R={r1r:.3f} | SVEB: P={ps1r:.3f} R={rs1r:.3f}")

    # ---- S2: RR 比值特征 + 单拍, 规则 OR ----
    print("\n--- S2: RR 比值 (pre/seq, post/seq) 单独看 SVEB 区分度 ---")
    for name, feat in [("pre/seq", ratio_pre), ("post/seq", ratio_post)]:
        med_n = np.median(feat[sym_v == "N"])
        med_s = np.median(feat[sym_v == "S"])
        med_v = np.median(feat[sym_v == "V"])
        u, pval = stats.mannwhitneyu(feat[sym_v == "N"], feat[sym_v == "S"])
        print(f"  {name:<10} N中位={med_n:.3f} S中位={med_s:.3f} V中位={med_v:.3f} | NvsS p={pval:.2e}")

    # ---- S3: 随机森林 (单拍 + pre/post + 2 比值) ----
    print("\n--- S3: 随机森林 (单拍+RR特征) ---")
    # 全量 P2A 概率 (train+test 患者拍都要)
    # 特征: 全量
    rr_pre_all = rr_pre / 288.0
    rr_post_all = rr_post / 288.0
    ratio_pre_all = rr_pre / np.maximum(rr_seq, 1e-9)
    ratio_post_all = rr_post / np.maximum(rr_seq, 1e-9)

    # 需要 RR 特征的拍 (全量)
    have_rr = ~np.isnan(rr_pre) & ~np.isnan(rr_post) & ~np.isnan(rr_seq)
    idx_use = np.where(have_rr)[0]
    print(f"  全量有 RR 特征: {len(idx_use)} 拍")

    p_use = m_p2a.predict(add_channel_dim(mi["beats"][idx_use]), verbose=0, batch_size=1024)[:, 1]
    X_use = np.stack([p_use, rr_pre_all[idx_use], rr_post_all[idx_use],
                      ratio_pre_all[idx_use], ratio_post_all[idx_use]], axis=-1)
    y_use = mi["labels"][idx_use]
    sym_use = sym[idx_use]

    # 患者级划分: train=tr|va 患者, test=te 患者
    pos_in_use = np.zeros(n_all, dtype=bool)
    pos_in_use[idx_use] = True
    tr_pos = np.where(have_rr & (tr | va))[0]
    te_pos = np.where(have_rr & te)[0]
    # 映射到 idx_use 内位置
    idx_map = {orig: k for k, orig in enumerate(idx_use)}
    tr_k = np.array([idx_map[o] for o in tr_pos])
    te_k = np.array([idx_map[o] for o in te_pos])
    X_tr, y_tr = X_use[tr_k], y_use[tr_k]
    X_te, y_te2 = X_use[te_k], y_use[te_k]
    sym_te2 = sym_use[te_k]
    print(f"  RF train={len(tr_k)} test={len(te_k)} SVEB_test={(sym_te2=='S').sum()}")

    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=42, n_jobs=4)
    rf.fit(X_tr, y_tr)
    score_te = rf.predict_proba(X_te)[:, 1]
    p3, r3, f3 = eval_binary(y_te2, score_te, 0.5)
    s3_mask = (sym_te2 == "S").astype(int)
    ps3, rs3, fs3 = eval_binary(s3_mask, score_te, 0.5)
    print(f"  ALL: P={p3:.3f} R={r3:.3f} F1={f3:.3f} | SVEB: P={ps3:.3f} R={rs3:.3f} F1={fs3:.3f}")
    print(f"  AUC: {roc_auc_score(y_te2, score_te):.4f} (基线单拍 AUC={roc_auc_score(y_v, p_te):.4f})")

    print("\n" + "=" * 78)
    print("对比汇总:")
    print(f"  基线单拍    : ALL-R {r0:.3f} SVEB-R {r_s0:.3f} SVEB-P {p_s0:.3f}")
    print(f"  S1 规则OR   : ALL-R {r1:.3f} SVEB-R {rs1:.3f} SVEB-P {ps1:.3f}")
    print(f"  S3 随机森林 : ALL-R {r3:.3f} SVEB-R {rs3:.3f} SVEB-P {ps3:.3f}")


if __name__ == "__main__":
    main()
