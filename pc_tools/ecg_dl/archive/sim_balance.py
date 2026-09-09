#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模拟平衡: 类别平衡对二分类 P/R 的影响 (2026-08-03, 零训练)

核心洞察: 训练类别配比 → 模型概率分布的决策边界偏移。
用现有模型概率 + 不同"平衡后异常占比"假设, 重算 P/R/F1。
方法: 在患者级测试集上, 对每个候选阈值 θ, 计算 P(θ), R(θ);
      然后按"平衡假设下的类别先验 π"调整决策 (贝叶斯最优决策)。
"""
import sys
from pathlib import Path
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged, add_channel_dim
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split

MODELS = Path(__file__).resolve().parent / "models"


def main():
    set_npz_suffix("_deploy")
    print("=" * 80)
    print("模拟平衡: 类别配比对二分类 P/R 的影响 (零训练)")
    print("=" * 80)

    mi = load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({r + 100000: "inc_" + p for r, p in build_incart_patient_map().items()})
    tr, va, te, _ = patient_level_split(mi["record_ids"], pmap)
    x_te, y_te = mi["beats"][te], mi["labels"][te]
    n_te = len(y_te)
    n_abn = int(y_te.sum())
    obs_prev = n_abn / n_te
    print(f"MIT test: {n_te} 拍, 异常占比(观测) = {obs_prev:.4f}")

    # 模型概率
    for name, rel in [("P2A", "archived/final_resnet_l_p2a_backup.h5"),
                      ("exp6c", "best_resnet_large_exp6_patient_clean.h5"),
                      ("exp6-SGD", "best_resnet_large_exp6_sgd.h5")]:
        path = MODELS / rel
        if not path.exists():
            print(f"  SKIP {name}")
            continue
        m = tf.keras.models.load_model(str(path), compile=False)
        p = m.predict(add_channel_dim(x_te), verbose=0, batch_size=1024)[:, 1]

        print(f"\n{'='*80}\n{name} (AUC={roc_auc_score(y_te, p):.4f})\n{'='*80}")

        # 阈值网格
        ths = np.arange(0.05, 0.96, 0.01)
        P = []
        R = []
        for th in ths:
            pred = (p >= th).astype(int)
            tp = int(((pred == 1) & (y_te == 1)).sum())
            fp = int(((pred == 1) & (y_te == 0)).sum())
            fn = int(((pred == 0) & (y_te == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            P.append(prec)
            R.append(rec)
        P = np.array(P)
        R = np.array(R)

        print(f"{'θ':<6}{'P':<8}{'R':<8}{'F1':<8}{'误报率':<9}")
        for i, th in enumerate(ths):
            if i % 10 == 0:
                f1 = 2 * P[i] * R[i] / max(1e-9, P[i] + R[i])
                fp_rate = fp = 0
                pred = (p >= th).astype(int)
                fp = int(((pred == 1) & (y_te == 0)).sum())
                fpr = fp / max(1, int((y_te == 0).sum()))
                print(f"{th:<6.2f}{P[i]:<8.3f}{R[i]:<8.3f}{f1:<8.3f}{fpr*100:<9.1f}")

        # 目标: 若要求 P>=0.8, 能达到的 R? 若要求 R>=0.9, 能达到的 P?
        print(f"\n  目标分析:")
        # P>=0.8
        idx = np.where(P >= 0.8)[0]
        if len(idx):
            best_r = R[idx].max()
            best_th = ths[idx[np.argmax(R[idx])]]
            print(f"  P≥0.80: 最高R={best_r:.3f} @θ={best_th:.2f} (当前实测P={P[ths==0.5][0]:.3f})")
        else:
            print(f"  P≥0.80: 无法达到 (最高P={P.max():.3f})")
        # R>=0.9
        idx = np.where(R >= 0.9)[0]
        if len(idx):
            best_p = P[idx].max()
            best_th = ths[idx[np.argmax(P[idx])]]
            print(f"  R≥0.90: 最高P={best_p:.3f} @θ={best_th:.2f}")
        else:
            print(f"  R≥0.90: 无法达到 (最高R={R.max():.3f})")
        # 90/90
        idx = np.where((P >= 0.9) & (R >= 0.9))[0]
        print(f"  P≥0.90 且 R≥0.90: {'✅ 存在 θ='+str(ths[idx]) if len(idx) else '❌ 不存在 (单模型物理极限)'}")

        # 模拟平衡: 用"平衡后先验"调整——即测试集按目标异常占比重新加权
        # 平衡后异常占比 π ∈ {0.3, 0.5}: 把决策阈值向 0.5 靠 (训练平衡 → 概率居中)
        # 近似: 平衡训练使模型输出接近"后验概率", 最优阈值≈π
        print(f"\n  模拟平衡后 (平衡异常占比 π 对应最优阈值):")
        for pi in [0.3, 0.5]:
            # 找 P(θ)=π*(R+...) 的平衡点: 用 PR 曲线上 F1 最优 + 先验校正
            # 简化: 扫描 θ, 找"平衡下 F1" = 用 π 调整 precision 分母的贝叶斯修正
            best = (0, 0, 0, 0)
            for i, th in enumerate(ths):
                pred = (p >= th).astype(int)
                tp = int(((pred == 1) & (y_te == 1)).sum())
                fp = int(((pred == 1) & (y_te == 0)).sum())
                fn = int(((pred == 0) & (y_te == 1)).sum())
                # 平衡校正: 把测试集类别分布调整到 π (异常占 π)
                # 等价于: 假阳性按 (1-π)/π 与观测先验的比例缩放
                scale = ((1 - pi) / pi) / ((1 - obs_prev) / obs_prev)
                fp_bal = fp * scale
                prec_bal = tp / max(1, tp + fp_bal)
                rec_bal = tp / max(1, tp + fn)
                f1_bal = 2 * prec_bal * rec_bal / max(1e-9, prec_bal + rec_bal)
                if f1_bal > best[0]:
                    best = (f1_bal, prec_bal, rec_bal, th)
            f1b, pb, rb, thb = best
            print(f"    平衡π={pi:.1f}: 最优@θ={thb:.2f} → P={pb:.3f} R={rb:.3f} F1={f1b:.3f}")


if __name__ == "__main__":
    main()
