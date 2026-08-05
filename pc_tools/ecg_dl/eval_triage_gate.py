#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分诊式设计验证 (8.9): 正常拍关卡 + 双专家
用已有模型概率模拟"关卡"效果:
  关卡 = 先判 正常/异常, 正常拍不放行(不报警), 疑似异常进双专家 OR
  验证: 关卡能否通过过滤正常拍抬高 Precision (不预设结论, 严谨评估)

方案: 用 P2A 的概率分数本身模拟关卡 (P2A 是 MIT 域判别器)
  关卡判"异常" = p_p2a >= θ_gate (高置信疑似异常)
  进入关卡的拍再进双专家 OR (P2A θ1 OR exp5 θ2)
  即: 最终报警 = (p_p2a >= θ_gate) AND (p_p2a>=θ1 OR p_exp5>=θ2)
  其中 θ_gate > θ1 (关卡更保守, 挡正常拍)
"""
import sys
from pathlib import Path
import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged, add_channel_dim, load_ptb_data
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map, build_ptb_patient_map,
    patient_level_split,
)

MODELS = Path(__file__).resolve().parent / "models"


def stats(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(prec=prec, rec=rec, f1=f1, fp_normal=fp / max(1, int((y == 0).sum())),
                alarm=float((pred == 1).mean()), n_abn=int(y.sum()), n=len(y))


def main():
    set_npz_suffix("_deploy")
    print("=" * 78)
    print("分诊式: 关卡 + 双专家 OR (8.9)")
    print("=" * 78)

    mit_inc = load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr, va, te, stats_m = patient_level_split(mit_inc["record_ids"], pmap)
    x_mit, y_mit = mit_inc["beats"][te], mit_inc["labels"][te]

    ptb = load_ptb_data()
    pmap_ptb = build_ptb_patient_map()
    tr2, va2, te2, stats_p = patient_level_split(ptb["record_ids"], pmap_ptb)
    x_ptb, y_ptb = ptb["beats"][te2], ptb["labels"][te2]

    print("Loading P2A + exp5_clean (CPU) ...")
    m_p2a = tf.keras.models.load_model(str(MODELS / "archived" / "final_resnet_l_p2a_backup.h5"), compile=False)
    m_exp5 = tf.keras.models.load_model(str(MODELS / "best_resnet_large_exp5_patient_clean.h5"), compile=False)

    p2a_mit = m_p2a.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    exp5_mit = m_exp5.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    p2a_ptb = m_p2a.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]
    exp5_ptb = m_exp5.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]

    # 基线: 无双关卡, 纯双专家 OR (θ1=0.5, θ2=0.8)
    base_mit = stats(y_mit, ((p2a_mit >= 0.5) | (exp5_mit >= 0.8)).astype(int))
    base_ptb = stats(y_ptb, ((p2a_ptb >= 0.5) | (exp5_ptb >= 0.8)).astype(int))
    print(f"\n[基线] 纯双专家OR(0.5,0.8): MIT P={base_mit['prec']:.3f} R={base_mit['rec']:.3f} "
          f"误报={base_mit['fp_normal']*100:.1f}% | PTB P={base_ptb['prec']:.3f} "
          f"R={base_ptb['rec']:.3f} 误报={base_ptb['fp_normal']*100:.1f}%")

    # 关卡: gate 用 P2A 分数 (θ_gate 从 0.55~0.85), 双专家 OR 在放行拍上 (θ1=0.5, θ2=0.8)
    print(f"\n[关卡] 放行=p_p2a>=θg, 放行拍再 OR(0.5,0.8):")
    print(f"{'θg':<6}| {'MIT-P':<7}{'MIT-R':<7}{'MIT误报':<8}{'放行率':<8}"
          f"| {'PTB-P':<7}{'PTB-R':<7}{'PTB误报':<8}{'放行率':<8}")
    out = {"baseline_or": {"mit": base_mit, "ptb": base_ptb}, "gate": {}}
    for thg in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        gm = stats(y_mit, ((p2a_mit >= thg) & ((p2a_mit >= 0.5) | (exp5_mit >= 0.8))).astype(int))
        gp = stats(y_ptb, ((p2a_ptb >= thg) & ((p2a_ptb >= 0.5) | (exp5_ptb >= 0.8))).astype(int))
        print(f"{thg:<6}| {gm['prec']:<7.3f}{gm['rec']:<7.3f}{gm['fp_normal']*100:<8.1f}"
              f"{gm['alarm']*100:<8.1f}| {gp['prec']:<7.3f}{gp['rec']:<7.3f}"
              f"{gp['fp_normal']*100:<8.1f}{gp['alarm']*100:<8.1f}")
        out["gate"][str(thg)] = {"mit": gm, "ptb": gp}

    with open(MODELS / "triage_gate_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {MODELS / 'triage_gate_eval.json'}")


if __name__ == "__main__":
    main()
