#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cross_arch.py — 跨架构部署链失配对照：训练外部模型
================================================================
为每个外部架构在两种数据域各训练一个模型：
  baseline : 训练链（filtfilt/FFT，默认 npz）
  deploy   : 部署链（因果 biquad + comb + 抽取，*_deploy.npz）

训练协议对齐 exp6c：
  --incart --ptb-beat --ptb-abn-max 10000 --domain-balanced --patient-split
  FocalLoss + AdamW(lr=5e-4, wd=1e-4)

用法：
  python3 train_cross_arch.py --arch lstm_cnn --chain baseline
  python3 train_cross_arch.py --arch lstm_cnn --chain deploy
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_CONFIG, MODELS_DIR
import data.dataset as dataset
from models.external_architectures import ARCHITECTURES, print_summary


def compile_model(model, learning_rate=0.0005, optimizer="adamw"):
    """与项目 ResNet 编译协议一致：FocalLoss + AdamW."""
    from losses.focal_loss import FocalLoss
    fl_cfg = TRAIN_CONFIG.get("focal_loss", {})
    loss = FocalLoss(
        gamma=fl_cfg.get("gamma", 1.0),
        alpha=fl_cfg.get("alpha", 0.75),
        label_smoothing=fl_cfg.get("label_smoothing", 0.0),
        from_logits=False,
    )
    if optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9,
                                      nesterov=True, weight_decay=1e-4)
    else:
        opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    model.compile(optimizer=opt, loss=loss, metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ])
    return model


def main():
    ap = argparse.ArgumentParser(description="跨架构模型训练（baseline/deploy 链）")
    ap.add_argument("--arch", required=True, choices=sorted(ARCHITECTURES.keys()))
    ap.add_argument("--chain", required=True, choices=["baseline", "deploy"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="每个 epoch 训练步数；0=跑完整数据集（默认，但 LSTM 很慢）")
    ap.add_argument("--val-steps", type=int, default=0,
                    help="每个 epoch 验证步数；0=完整验证集")
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--out-dir", type=str, default=str(MODELS_DIR / "cross_arch"))
    ap.add_argument("--quick", action="store_true", help="小数据 smoke 测试")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # chain suffix for dataset loader
    if args.chain == "deploy":
        dataset.set_npz_suffix("_deploy")
    else:
        dataset.set_npz_suffix("")

    print("=" * 70, flush=True)
    print(f"[CrossArch] arch={args.arch} chain={args.chain} "
          f"epochs={args.epochs} patience={args.patience}", flush=True)
    print("=" * 70, flush=True)

    datasets = dataset.prepare_datasets(
        batch_size=args.batch_size,
        use_incart=True,
        use_ptb_beat=True,
        ptb_abn_max=10000,
        domain_balanced=True,
        patient_split=True,
    )

    # quick smoke: 截断 epoch? Keras datasets have .take; easier to build tiny dataset below.
    if args.quick:
        print("[quick] 仅使用 2 个 batch 训练 2 epochs")
        train_smoke = datasets["train_ds"].take(2)
        val_smoke = datasets["val_ds"].take(2)

    builder = ARCHITECTURES[args.arch]
    model = builder(input_shape=datasets["input_shape"], n_classes=2)
    print_summary(model)
    model = compile_model(model, learning_rate=args.learning_rate, optimizer=args.optimizer)

    model_path = out_dir / f"{args.arch}_{args.chain}.h5"
    history_csv = out_dir / f"{args.arch}_{args.chain}_history.csv"
    callbacks = [
        # 统一以 val_auc 作为早停/保存标准；避免 val_loss 与 val_auc 最优 epoch
        # 不一致时，restore_best_weights 把高 val_auc 权重覆盖掉。
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=args.patience,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(8, args.patience // 4),
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(history_csv), append=False),
    ]

    t0 = time.time()
    train_ds = datasets["train_ds"].take(2) if args.quick else datasets["train_ds"]
    # steps_per_epoch 模式下必须 repeat()：域平衡 dataset 是有限长度，
    # 不 repeat 会在第 N 个 epoch 耗尽（实测 lstm_cnn 第 13 epoch 中断）。
    if not args.quick and args.steps_per_epoch and args.steps_per_epoch > 0:
        train_ds = train_ds.repeat()
    steps_per_epoch = 2 if args.quick else (args.steps_per_epoch or None)
    val_steps = 2 if args.quick else (args.val_steps or None)
    history = model.fit(
        train_ds,
        validation_data=datasets["val_ds"].take(2) if args.quick else datasets["val_ds"],
        epochs=2 if args.quick else args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2,
    )

    # 保存最终模型（best checkpoint 由 ModelCheckpoint 写入同一路径）
    model.save(model_path)
    final_val_auc = max(history.history.get("val_auc", [0.0]))
    print(f"[CrossArch] 完成 arch={args.arch} chain={args.chain} "
          f"best_val_auc={final_val_auc:.4f} elapsed={time.time()-t0:.0f}s")
    meta = {
        "arch": args.arch,
        "chain": args.chain,
        "epochs_run": len(history.history.get("loss", [])),
        "best_val_auc": float(final_val_auc),
        "best_val_loss": float(min(history.history.get("val_loss", [0.0]))),
        "model_path": str(model_path),
        "history_csv": str(history_csv),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / f"{args.arch}_{args.chain}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()