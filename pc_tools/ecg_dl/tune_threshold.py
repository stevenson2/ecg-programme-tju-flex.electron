#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值调优: 在验证集上搜索最佳 Abnormal 分类阈值。

Phase 2A-1: 增强版 — F1最大化 + 目标Recall模式 + 细粒度扫描。
FocalLoss 会压缩预测概率分布，0.5 默认阈值往往不是最优。
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS_DIR, CLASS_NAMES, INFERENCE_CONFIG
from data.dataset import load_mit_incart_merged, load_3beat_merged, train_val_test_split

# Normal: class 0, Abnormal: class 1
ABNORMAL_IDX = 1


def compute_metrics(y_true, prob_ab, thresh):
    y_pred = (prob_ab >= thresh).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-8)
    spec = tn / max(tn + fp, 1)
    return {"acc": acc, "prec": prec, "recall": recall, "f1": f1,
            "spec": spec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def find_best_f1(prob_ab, y_val, step=0.02):
    best = None
    best_thresh = 0.5
    for thresh in np.arange(0.10, 0.91, step):
        thresh = round(thresh, 3)
        m = compute_metrics(y_val, prob_ab, thresh)
        if best is None or m["f1"] > best["f1"]:
            best = m
            best_thresh = thresh
    return best_thresh, best


def find_target_recall(prob_ab, y_val, target=0.85, step=0.005):
    best = None
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.96, step):
        thresh = round(thresh, 4)
        m = compute_metrics(y_val, prob_ab, thresh)
        if m["recall"] >= target:
            if best is None or m["prec"] > best["prec"]:
                best = m
                best_thresh = thresh
    return best_thresh, best


def main():
    parser = argparse.ArgumentParser(description="阈值调优工具")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径 (默认: 自动查找 best_model.h5 或 final_model.h5)")
    parser.add_argument("--target-recall", type=float, default=None,
                        help="目标 Recall (如 0.85), 查找满足条件且 Precision 最高的阈值")
    parser.add_argument("--coarse", action="store_true",
                        help="粗扫描 (0.05 步长, 快速预览)")
    parser.add_argument("--full-table", action="store_true",
                        help="打印完整阈值扫描表")
    parser.add_argument("--3beat", dest="use_3beat", action="store_true",
                        help="使用 3-beat 数据 (配合 750 点模型)")
    args = parser.parse_args()

    import tensorflow as tf

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = MODELS_DIR / "best_model.h5"
        if not model_path.exists():
            model_path = MODELS_DIR / "final_model.h5"
    print(f"[调优] 加载模型: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    print("[调优] 加载验证集...")
    if args.use_3beat:
        data = load_3beat_merged()
    else:
        data = load_mit_incart_merged()
    splits = train_val_test_split(
        data["beats"], data["labels"],
        record_ids=data.get("record_ids")
    )
    x_val, y_val = splits["val"]
    x_val = x_val[..., np.newaxis]

    print(f"[调优] 验证集: {len(x_val)} 样本")
    nN = int((y_val == 0).sum())
    nA = int((y_val == 1).sum())
    print(f"[调优]   Normal: {nN}, Abnormal: {nA} ({nA/len(y_val)*100:.1f}%)")

    print("[调优] 预测中...")
    y_prob = model.predict(x_val, verbose=0)
    prob_abnormal = y_prob[:, ABNORMAL_IDX]

    # Always compute F1-best threshold (needed for test-set estimation below)
    f1_step = 0.05 if args.coarse else 0.02
    best_f1_thresh, best_f1_metrics = find_best_f1(prob_abnormal, y_val, step=f1_step)

    # === 模式 1: 目标 Recall ==================
    if args.target_recall:
        print(f"\n{'='*70}")
        print(f"  目标 Recall 模式: ≥ {args.target_recall:.0%}")
        print(f"{'='*70}")
        step = 0.01 if args.coarse else 0.005
        best_thresh, best_m = find_target_recall(
            prob_abnormal, y_val, target=args.target_recall, step=step
        )
        if best_m is None:
            print(f"\n❌ 无法达到目标 Recall {args.target_recall:.0%}!")
            max_recall = compute_metrics(y_val, prob_abnormal, 0.05)["recall"]
            print(f"   当前模型最大可能 Recall: {max_recall:.4f} ({max_recall*100:.1f}%)")
        else:
            m = best_m
            print(f"\n✅ 满足 Recall≥{args.target_recall:.0%} 的最优阈值: {best_thresh:.4f}")
            print(f"   Acc:     {m['acc']*100:.2f}%")
            print(f"   Prec:    {m['prec']:.4f}  (报警精准度)")
            print(f"   Recall:  {m['recall']:.4f}  (异常检出率)")
            print(f"   Spec:    {m['spec']:.4f}  (正常特异性)")
            print(f"   F1:      {m['f1']:.4f}")
            print(f"   TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

    # === 模式 2: 完整扫描表 ===
    if args.full_table or not args.target_recall:
        step = 0.05 if args.coarse else 0.02
        print(f"\n{'='*70}")
        print(f"{'Thresh':>8s}  {'Acc%':>7s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  {'Spec':>7s}  {'N_Pred':>7s}")
        print(f"{'='*70}")

        for thresh in np.arange(0.05, 0.96, step):
            thresh = round(thresh, 3)
            m = compute_metrics(y_val, prob_abnormal, thresh)
            n_pred = m["tp"] + m["fp"]
            marker = " ← F1" if abs(thresh - best_f1_thresh) < 0.001 else ""
            rec_marker = " ★" if m["recall"] >= 0.85 else ""
            print(f"{thresh:>8.3f}  {m['acc']*100:>6.2f}  {m['prec']:>7.4f}  "
                  f"{m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['spec']:>7.4f}  {n_pred:>7d}{marker}{rec_marker}")

        print(f"{'='*70}")
        bm = best_f1_metrics
        print(f"\n★ F1最大化阈值: {best_f1_thresh:.3f}")
        print(f"   Acc={bm['acc']*100:.2f}%  Prec={bm['prec']:.4f}  Recall={bm['recall']:.4f}  "
              f"F1={bm['f1']:.4f}  Spec={bm['spec']:.4f}")

    # === 测试集预估 (最佳F1阈值) ===
    print(f"\n[调优] 测试集预估 (F1最优阈值={best_f1_thresh:.3f})...")
    x_test, y_test = splits["test"]
    x_test = x_test[..., np.newaxis]
    y_prob_test = model.predict(x_test, verbose=0)
    prob_ab_test = y_prob_test[:, ABNORMAL_IDX]
    m_test = compute_metrics(y_test, prob_ab_test, best_f1_thresh)

    print(f"   Acc:     {m_test['acc']*100:.2f}%")
    print(f"   Prec:    {m_test['prec']:.4f}")
    print(f"   Recall:  {m_test['recall']:.4f}")
    print(f"   F1:      {m_test['f1']:.4f}")
    print(f"   TP={m_test['tp']}  FP={m_test['fp']}  FN={m_test['fn']}  TN={m_test['tn']}")

    if args.target_recall and best_m:
        print(f"\n[调优] 测试集预估 (目标Recall阈值={best_thresh:.4f})...")
        prob_ab_test2 = model.predict(x_test, verbose=0)[:, ABNORMAL_IDX]
        m_test2 = compute_metrics(y_test, prob_ab_test2, best_thresh)
        print(f"   Acc:     {m_test2['acc']*100:.2f}%")
        print(f"   Prec:    {m_test2['prec']:.4f}")
        print(f"   Recall:  {m_test2['recall']:.4f}")
        print(f"   F1:      {m_test2['f1']:.4f}")

    print(f"\n[调优] 当前部署阈值: {INFERENCE_CONFIG['threshold']:.2f} "
          f"(config.py INFERENCE_CONFIG['threshold'])")
    print(f"[调优] 多拍确认: {INFERENCE_CONFIG['multi_beat_confirm']} 拍连续报警")


if __name__ == "__main__":
    main()
