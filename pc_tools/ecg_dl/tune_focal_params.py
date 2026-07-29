#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2A-2: FocalLoss γ/α 网格搜索

在验证集上扫描 16 组 (γ, α) 组合，用短训练 (reduced epochs)
快速筛选最优超参数。最终报告按 Recall 排序的矩阵。
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from config import MODELS_DIR, CLASS_NAMES, TRAIN_CONFIG
from data.dataset import prepare_datasets
from models.cnn_1d import (
    build_ecg_cnn_1d_v2, compile_model, model_summary_table
)

# ---- 搜索空间 ----
GAMMA_VALUES = [0.5, 1.0, 1.5, 2.0]
ALPHA_VALUES = [0.70, 0.75, 0.80, 0.85]


def run_single(use_incart, use_v3, use_balanced,
               gamma, alpha, quick_epochs, batch_size,
               seed, verbose):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Override config
    TRAIN_CONFIG["focal_loss"]["gamma"] = gamma
    TRAIN_CONFIG["focal_loss"]["alpha"] = alpha
    TRAIN_CONFIG["focal_loss"]["enabled"] = True
    TRAIN_CONFIG["random_seed"] = seed

    datasets = prepare_datasets(
        batch_size=batch_size,
        use_incart=use_incart,
        use_balanced=use_balanced,
    )

    if use_v3:
        from models.cnn_1d import build_ecg_cnn_1d_v3
        model = build_ecg_cnn_1d_v3(
            input_shape=datasets["input_shape"],
            n_classes=len(CLASS_NAMES)
        )
    else:
        model = build_ecg_cnn_1d_v2(
            input_shape=datasets["input_shape"],
            n_classes=len(CLASS_NAMES)
        )

    model = compile_model(model, learning_rate=TRAIN_CONFIG["learning_rate"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall", patience=15,
            restore_best_weights=True, verbose=0, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-7, verbose=0
        ),
    ]

    history = model.fit(
        datasets["train_ds"],
        validation_data=datasets["val_ds"],
        epochs=quick_epochs,
        callbacks=callbacks,
        verbose=2 if verbose else 0,
    )

    # Evaluate on val and test
    x_val, y_val = datasets["data"]["val"]
    x_val = x_val[..., np.newaxis]
    y_prob = model.predict(x_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    val_acc = float(np.mean(y_pred == y_val))
    val_tp = int(((y_pred == 1) & (y_val == 1)).sum())
    val_fp = int(((y_pred == 1) & (y_val == 0)).sum())
    val_fn = int(((y_pred == 0) & (y_val == 1)).sum())
    val_tn = int(((y_pred == 0) & (y_val == 0)).sum())
    val_prec = val_tp / max(val_tp + val_fp, 1)
    val_rec = val_tp / max(val_tp + val_fn, 1)
    val_f1 = 2 * val_prec * val_rec / max(val_prec + val_rec, 1e-8)

    x_test, y_test = datasets["data"]["test"]
    x_test = x_test[..., np.newaxis]
    y_prob_t = model.predict(x_test, verbose=0)
    y_pred_t = np.argmax(y_prob_t, axis=1)

    test_acc = float(np.mean(y_pred_t == y_test))
    test_tp = int(((y_pred_t == 1) & (y_test == 1)).sum())
    test_fp = int(((y_pred_t == 1) & (y_test == 0)).sum())
    test_fn = int(((y_pred_t == 0) & (y_test == 1)).sum())
    test_prec = test_tp / max(test_tp + test_fp, 1)
    test_rec = test_tp / max(test_tp + test_fn, 1)
    test_f1 = 2 * test_prec * test_rec / max(test_prec + test_rec, 1e-8)

    tf.keras.backend.clear_session()
    return {
        "gamma": gamma, "alpha": alpha,
        "val_acc": val_acc, "val_prec": val_prec, "val_recall": val_rec, "val_f1": val_f1,
        "test_acc": test_acc, "test_prec": test_prec, "test_recall": test_rec, "test_f1": test_f1,
        "epochs_ran": len(history.history["loss"]),
    }


def main():
    parser = argparse.ArgumentParser(description="FocalLoss 参数网格搜索")
    parser.add_argument("--incart", action="store_true", default=True,
                        help="使用 MIT-BIH + INCART")
    parser.add_argument("--v3", action="store_true",
                        help="使用 CNN v3 (30K), 默认 v2 (15K)")
    parser.add_argument("--balanced", action="store_true",
                        help="使用类别均衡采样")
    parser.add_argument("--epochs", type=int, default=50,
                        help="快速评估 epoch 数 (默认 50)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批次大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--gamma", type=float, default=None,
                        help="固定 gamma，仅扫描 alpha")
    parser.add_argument("--alpha", type=float, default=None,
                        help="固定 alpha，仅扫描 gamma")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="详细输出")
    args = parser.parse_args()

    gammas = [args.gamma] if args.gamma else GAMMA_VALUES
    alphas = [args.alpha] if args.alpha else ALPHA_VALUES

    total = len(gammas) * len(alphas)
    print(f"\n{'='*70}")
    print(f" Phase 2A-2: FocalLoss γ/α 网格搜索")
    print(f" 模型: {'CNN v3' if args.v3 else 'CNN v2'}")
    print(f" 数据集: {'MIT-BIH+INCART +4.32%Acc' if args.incart else 'MIT-BIH'}")
    print(f" 均衡采样: {'是' if args.balanced else '否'}")
    print(f" 搜索: γ ∈ {gammas}, α ∈ {alphas} ({total} 组)")
    print(f" 快速评估: {args.epochs} epochs/组")
    print(f"{'='*70}")

    results = []
    for i, gamma in enumerate(gammas):
        for j, alpha in enumerate(alphas):
            n = i * len(alphas) + j + 1
            print(f"\n[{n}/{total}] γ={gamma:.1f}, α={alpha:.2f} ...", end=" ", flush=True)
            r = run_single(
                use_incart=args.incart,
                use_v3=args.v3,
                use_balanced=args.balanced,
                gamma=gamma, alpha=alpha,
                quick_epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                verbose=args.verbose,
            )
            results.append(r)
            print(f"Val Acc={r['val_acc']*100:.1f}% "
                  f"Recall={r['val_recall']:.3f} "
                  f"F1={r['val_f1']:.3f} "
                  f"Epochs={r['epochs_ran']}")

    # ---- 结果矩阵 ----
    print(f"\n{'='*70}")
    print(f" 结果矩阵 (按 Valid Recall 降序)")
    print(f"{'='*70}")
    print(f"{'#':>3s}  {'γ':>5s}  {'α':>5s}  "
          f"{'V_Acc%':>7s}  {'V_Prec':>7s}  {'V_Rec':>7s}  {'V_F1':>7s}  "
          f"{'T_Acc%':>7s}  {'T_Prec':>7s}  {'T_Rec':>7s}  {'T_F1':>7s}  {'Ep':>4s}")
    print(f"{'-'*70}")

    results_sorted = sorted(results, key=lambda r: r["val_recall"], reverse=True)
    for i, r in enumerate(results_sorted):
        tag = " ← BEST" if i == 0 else ""
        print(f"{i+1:>3d}  {r['gamma']:>5.1f}  {r['alpha']:>5.2f}  "
              f"{r['val_acc']*100:>6.1f}  {r['val_prec']:>7.4f}  "
              f"{r['val_recall']:>7.4f}  {r['val_f1']:>7.4f}  "
              f"{r['test_acc']*100:>6.1f}  {r['test_prec']:>7.4f}  "
              f"{r['test_recall']:>7.4f}  {r['test_f1']:>7.4f}  "
              f"{r['epochs_ran']:>4d}{tag}")

    # Top-3 by Recall
    top3_rec = results_sorted[:3]
    print(f"\n{'='*70}")
    print(" Top-3 by Valid Recall:")
    print(f"{'='*70}")
    for i, r in enumerate(top3_rec):
        print(f"  {i+1}. γ={r['gamma']:.1f}, α={r['alpha']:.2f}  "
              f"→ Test Acc={r['test_acc']*100:.2f}%  "
              f"Recall={r['test_recall']:.4f}  F1={r['test_f1']:.4f}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = MODELS_DIR / f"focal_grid_{ts}.json"
    out.write_text(json.dumps(results_sorted, indent=2))
    print(f"\n[调优] 结果已保存: {out}")

    # 更新 config 建议
    best = results_sorted[0]
    print(f"\n[调优] 建议 config.py 更新:")
    print(f"  TRAIN_CONFIG['focal_loss']['gamma'] = {best['gamma']:.1f}")
    print(f"  TRAIN_CONFIG['focal_loss']['alpha'] = {best['alpha']:.2f}")


if __name__ == "__main__":
    main()
