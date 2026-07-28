#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值调优: 在验证集上搜索最佳 Abnormal 分类阈值。

FocalLoss 会压缩预测概率分布，0.5 默认阈值往往不是最优。
在 val 集上遍历 0.1~0.9，找到最大化 Abnormal F1 的阈值。
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS_DIR, CLASS_NAMES
from data.dataset import prepare_datasets, load_mit_incart_merged, train_val_test_split

# Normal: class 0, Abnormal: class 1
ABNORMAL_IDX = 1


def main():
    import tensorflow as tf

    # Load best model (use --incart 训练的模型)
    model_path = MODELS_DIR / "best_model.h5"
    if not model_path.exists():
        model_path = MODELS_DIR / "final_model.h5"
    print(f"[调优] 加载模型: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # Load validation data (same split as training)
    print("[调优] 加载验证集...")
    data = load_mit_incart_merged()
    splits = train_val_test_split(
        data["beats"], data["labels"],
        record_ids=data.get("record_ids")
    )
    x_val, y_val = splits["val"]
    x_val = x_val[..., np.newaxis]  # (n, 250) -> (n, 250, 1)

    print(f"[调优] 验证集: {len(x_val)} 样本")
    nN = int((y_val == 0).sum())
    nA = int((y_val == 1).sum())
    print(f"[调优]   Normal: {nN}, Abnormal: {nA}")

    # Predict probabilities
    print("[调优] 预测中...")
    y_prob = model.predict(x_val, verbose=0)
    prob_abnormal = y_prob[:, ABNORMAL_IDX]

    # Grid search thresholds
    print(f"\n{'='*60}")
    print(f"{'Threshold':>10s}  {'Acc':>8s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}  {'N_Pred':>8s}")
    print(f"{'='*60}")

    best_f1 = 0
    best_thresh = 0.5
    best_metrics = None

    for thresh in np.arange(0.10, 0.91, 0.05):
        thresh = round(thresh, 2)
        y_pred = (prob_abnormal >= thresh).astype(int)

        # Metrics
        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        tn = ((y_pred == 0) & (y_val == 0)).sum()

        acc = (tp + tn) / len(y_val)
        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-8)
        n_pred_ab = tp + fp

        marker = " ← BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = (acc, prec, recall, f1)

        print(f"{thresh:>10.2f}  {acc:>8.4f}  {prec:>8.4f}  {recall:>8.4f}  {f1:>8.4f}  {n_pred_ab:>8d}{marker}")

    acc, prec, recall, f1 = best_metrics
    print(f"{'='*60}")
    print(f"\n✅ 最佳阈值: {best_thresh:.2f}")
    print(f"   Acc:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Prec:   {prec:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1:     {f1:.4f}")

    # Estimate test set performance at best threshold
    print(f"\n[调优] 测试集预估 (最佳阈值={best_thresh:.2f})...")
    x_test, y_test = splits["test"]
    x_test = x_test[..., np.newaxis]
    y_prob_test = model.predict(x_test, verbose=0)
    prob_ab_test = y_prob_test[:, ABNORMAL_IDX]
    y_pred_test = (prob_ab_test >= best_thresh).astype(int)

    tp = ((y_pred_test == 1) & (y_test == 1)).sum()
    fp = ((y_pred_test == 1) & (y_test == 0)).sum()
    fn = ((y_pred_test == 0) & (y_test == 1)).sum()
    tn = ((y_pred_test == 0) & (y_test == 0)).sum()

    print(f"   Acc:    {(tp+tn)/len(y_test):.4f} ({(tp+tn)/len(y_test)*100:.2f}%)")
    print(f"   Prec:   {tp/max(tp+fp,1):.4f}")
    print(f"   Recall: {tp/max(tp+fn,1):.4f}")
    print(f"   F1:     {2*tp/max(2*tp+fp+fn,1):.4f}")
    print(f"   Normal:  {tn} correct / {fp} false positive")
    print(f"   Abnormal: {tp} correct / {fn} false negative")

    print(f"\n[调优] 部署: ESP32 推理时若 P(Abnormal) >= {best_thresh:.2f} 则判定异常")


if __name__ == "__main__":
    main()
