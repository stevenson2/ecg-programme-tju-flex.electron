#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一二分类评估: 所有关键模型在患者级+部署链(D3)口径下的完整指标矩阵
评估: MIT域 / PTB域 / 全量, AUC + P/R/F1 @ θ∈{0.35,0.5,0.65} + 正常拍误报率
模型: P2A, exp4c, exp5c, exp6c, exp6-SGD, KD 9组
"""
import sys
from pathlib import Path
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged, add_channel_dim, load_ptb_data
from data.patient_split import build_mit_patient_map, build_incart_patient_map, build_ptb_patient_map, patient_level_split

MODELS = Path(__file__).resolve().parent / "models"


def stats(y, prob, th):
    pred = (prob >= th).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    fp_normal = int(((pred == 1) & (y == 0)).sum()) / max(1, int((y == 0).sum()))
    alarm = float((pred == 1).mean())
    return prec, rec, f1, fp_normal, alarm


def main():
    set_npz_suffix("_deploy")
    print("=" * 100)
    print("统一二分类评估: 患者级 + 部署链(D3)")
    print("=" * 100)

    # ---- 数据 ----
    mi = load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({r + 100000: "inc_" + p for r, p in build_incart_patient_map().items()})
    tr, va, te, _ = patient_level_split(mi["record_ids"], pmap)
    x_mit, y_mit = mi["beats"][te], mi["labels"][te]

    ptb = load_ptb_data()
    pmap_ptb = build_ptb_patient_map()
    tr2, va2, te2, _ = patient_level_split(ptb["record_ids"], pmap_ptb)
    x_ptb, y_ptb = ptb["beats"][te2], ptb["labels"][te2]
    print(f"MIT test: {len(y_mit)} | PTB test: {len(y_ptb)}")

    # ---- 模型列表 ----
    models = [
        # ---- 基线 ----
        ("P2A(存档)", "archived/final_resnet_l_p2a_backup.h5"),
        ("exp4c(清洁)", "best_resnet_large_exp4_patient_clean.h5"),
        ("exp5c(清洁)", "best_resnet_large_exp5_patient_clean.h5"),
        ("exp6c(清洁)", "best_resnet_large_exp6_patient_clean.h5"),
        ("exp6-SGD", "best_resnet_large_exp6_sgd.h5"),
        # ---- KD 网格 α∈{0.3,0.5,0.7} × T∈{1,3,5} ----
        ("KD a030_t1", "kd_a030_t1.h5"),
        ("KD a030_t3", "kd_a030_t3.h5"),
        ("KD a030_t5", "kd_a030_t5.h5"),
        ("KD a050_t1", "kd_a050_t1.h5"),
        ("KD a050_t3", "kd_a050_t3.h5"),
        ("KD a050_t5", "kd_a050_t5.h5"),
        ("KD a070_t1", "kd_a070_t1.h5"),
        ("KD a070_t3", "kd_a070_t3.h5"),
        ("KD a070_t5", "kd_a070_t5.h5"),
        # ---- 未来: bal_mixed ----
        ("bal_mixed", "bal_mixed.h5"),
    ]

    # 缓存预测 (CPU, 逐模型)
    results = {}
    for name, rel in models:
        path = MODELS / rel
        if not path.exists():
            print(f"  SKIP {name}: {path} 不存在")
            continue
        print(f"\n=== {name} ===")
        m = tf.keras.models.load_model(str(path), compile=False)
        p_mit = m.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
        p_ptb = m.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]
        auc_mit = roc_auc_score(y_mit, p_mit)
        auc_ptb = roc_auc_score(y_ptb, p_ptb)
        print(f"  AUC: MIT={auc_mit:.4f} PTB={auc_ptb:.4f}")

        row = {"auc_mit": auc_mit, "auc_ptb": auc_ptb, "thr": {}}
        for th in [0.35, 0.5, 0.65]:
            pm, rm, fm, fpn_m, am = stats(y_mit, p_mit, th)
            pp, rp, fp_, fpn_p, ap = stats(y_ptb, p_ptb, th)
            row["thr"][str(th)] = {
                "mit": {"P": pm, "R": rm, "F1": fm, "误报": fpn_m, "报警": am},
                "ptb": {"P": pp, "R": rp, "F1": fp_, "误报": fpn_p, "报警": ap},
            }
            print(f"  θ={th}: MIT P={pm:.3f} R={rm:.3f} F1={fm:.3f} 误报={fpn_m*100:.1f}% "
                  f"| PTB P={pp:.3f} R={rp:.3f} F1={fp_:.3f} 误报={fpn_p*100:.1f}%")
        results[name] = row
        del m

    # ---- 汇总表 ----
    print("\n" + "=" * 100)
    print("汇总 (θ=0.5):")
    print(f"{'模型':<14}{'MIT-AUC':<9}{'MIT-R':<7}{'MIT-P':<7}{'MIT误报':<8}"
          f"{'PTB-AUC':<9}{'PTB-R':<7}{'PTB-P':<7}{'PTB误报':<8}")
    for name, row in results.items():
        t = row["thr"]["0.5"]
        print(f"{name:<14}{row['auc_mit']:<9.4f}{t['mit']['R']:<7.3f}{t['mit']['P']:<7.3f}"
              f"{t['mit']['误报']*100:<8.1f}{row['auc_ptb']:<9.4f}{t['ptb']['R']:<7.3f}"
              f"{t['ptb']['P']:<7.3f}{t['ptb']['误报']*100:<8.1f}")

    import json
    with open(MODELS / "binary_class_eval_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {MODELS / 'binary_class_eval_all.json'}")


if __name__ == "__main__":
    main()
