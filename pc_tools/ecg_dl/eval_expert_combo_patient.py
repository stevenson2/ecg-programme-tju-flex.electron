#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双专家 OR 严谨口径实测 (4.2-0)
患者级划分 + 部署链数据 + 清洁模型:
  P2A (心律失常专家, θ1) OR exp5_clean (心梗专家, θ2) -> 组合报警
  CPU-only (不抢 KD 的 GPU)。含正常拍误报率 (报警疲劳指标)。
"""
import sys
from pathlib import Path
import numpy as np
import json

# ---- 强制 CPU (不干扰 KD 训练) ----
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged, add_channel_dim
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map, build_ptb_patient_map,
    patient_level_split,
)

MODELS = Path(__file__).resolve().parent / "models"


def combo_stats(y, p_a, p_b, th_a, th_b):
    """OR 融合: 任一专家超阈值即报警。返回 P/R/F1 + 正常拍误报率 + 总报警率。"""
    pred = ((p_a >= th_a) | (p_b >= th_b)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    alarm = float((pred == 1).mean())          # 总报警率
    fp_normal = fp / max(1, int((y == 0).sum()))  # 正常拍误报率
    return dict(prec=prec, rec=rec, f1=f1, alarm=alarm,
                fp_normal=fp_normal, tp=tp, fp=fp, fn=fn)


def single_stats(y, p, th):
    pred = (p >= th).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(prec=prec, rec=rec, f1=f1,
                fp_normal=fp / max(1, int((y == 0).sum())))


def main():
    set_npz_suffix("_deploy")
    print("=" * 78)
    print("双专家 OR 严谨实测 (患者级 / 部署链 / 清洁模型) — CPU mode")
    print("=" * 78)

    # ---- MIT+INCART 患者级测试 ----
    mit_inc = load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    tr, va, te, stats = patient_level_split(mit_inc["record_ids"], pmap)
    x_mit, y_mit = mit_inc["beats"][te], mit_inc["labels"][te]
    print(f"MIT test: {te.sum()} beats (patient-level, {stats['n_test']} patients)")

    # ---- PTB 患者级测试 (部署链 npz) ----
    from data.dataset import load_ptb_data
    ptb = load_ptb_data()
    pmap_ptb = build_ptb_patient_map()
    tr2, va2, te2, stats2 = patient_level_split(ptb["record_ids"], pmap_ptb)
    x_ptb, y_ptb = ptb["beats"][te2], ptb["labels"][te2]
    print(f"PTB test: {te2.sum()} beats (patient-level, {stats2['n_test']} patients)")

    # ---- 加载模型 (CPU) ----
    print("\nLoading P2A + exp5_clean (CPU) ...")
    m_p2a = tf.keras.models.load_model(
        str(MODELS / "archived" / "final_resnet_l_p2a_backup.h5"), compile=False)
    m_exp5 = tf.keras.models.load_model(
        str(MODELS / "best_resnet_large_exp5_patient_clean.h5"), compile=False)

    p2a_mit = m_p2a.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    exp5_mit = m_exp5.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    p2a_ptb = m_p2a.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]
    exp5_ptb = m_exp5.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]

    out = {"meta": {"models": ["P2A(archived)", "exp5_patient_clean"],
                    "split": "patient-level 60/20/20 seed42",
                    "data": "deploy-chain (_deploy npz)", "mode": "CPU"}}

    # ---- 单模型基线 ----
    print("\n--- 单模型基线 (θ=0.5) ---")
    for name, p_mit, p_ptb in [("P2A", p2a_mit, p2a_ptb),
                               ("exp5_clean", exp5_mit, exp5_ptb)]:
        s_mit = single_stats(y_mit, p_mit, 0.5)
        s_ptb = single_stats(y_ptb, p_ptb, 0.5)
        print(f"{name:<12} MIT P={s_mit['prec']:.3f} R={s_mit['rec']:.3f} "
              f"误报={s_mit['fp_normal']*100:.1f}% | PTB P={s_ptb['prec']:.3f} "
              f"R={s_ptb['rec']:.3f} 误报={s_ptb['fp_normal']*100:.1f}%")
        out.setdefault("single", {})[name] = {"mit": s_mit, "ptb": s_ptb}

    # ---- 双专家 OR 网格 ----
    print("\n--- 双专家 OR (P2A θ1 OR exp5 θ2) ---")
    print(f"{'θ1':<5}{'θ2':<6} | {'MIT-P':<7}{'MIT-R':<7}{'MIT误报':<8}"
          f"{'MIT报警':<8}| {'PTB-P':<7}{'PTB-R':<7}{'PTB误报':<8}{'PTB报警':<8}")
    out["or_grid"] = {}
    for th_a in [0.35, 0.50]:
        for th_b in [0.50, 0.65, 0.80]:
            sm = combo_stats(y_mit, p2a_mit, exp5_mit, th_a, th_b)
            sp = combo_stats(y_ptb, p2a_ptb, exp5_ptb, th_a, th_b)
            print(f"{th_a:<5}{th_b:<6} | {sm['prec']:<7.3f}{sm['rec']:<7.3f}"
                  f"{sm['fp_normal']*100:<8.1f}{sm['alarm']*100:<8.1f}| "
                  f"{sp['prec']:<7.3f}{sp['rec']:<7.3f}{sp['fp_normal']*100:<8.1f}"
                  f"{sp['alarm']*100:<8.1f}")
            out["or_grid"][f"t{th_a}_t{th_b}"] = {"mit": sm, "ptb": sp}

    # ---- AUC ----
    out["auc"] = {
        "p2a": {"mit": float(roc_auc_score(y_mit, p2a_mit)),
                "ptb": float(roc_auc_score(y_ptb, p2a_ptb))},
        "exp5": {"mit": float(roc_auc_score(y_mit, exp5_mit)),
                 "ptb": float(roc_auc_score(y_ptb, exp5_ptb))},
    }
    print(f"\nAUC: P2A MIT={out['auc']['p2a']['mit']:.4f} PTB={out['auc']['p2a']['ptb']:.4f} | "
          f"exp5 MIT={out['auc']['exp5']['mit']:.4f} PTB={out['auc']['exp5']['ptb']:.4f}")

    out_path = MODELS / "expert_combo_patient_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
