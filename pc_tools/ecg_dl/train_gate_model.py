#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gate_model.py — 双专家前置关卡训练（A1）
================================================
按 dual_expert_deployment_plan.md 阶段 A1 训练“正常 vs 异常”关卡：
  - 数据：MIT+INCART deploy 链 + PTB deploy 拍（PTB 正常全量 + 异常限量 10000）
  - 划分：患者级 seed=42
  - 批采样：domain-balanced，每 batch 20% PTB，PTB loss weight=0.5
  - 架构：ResNet-Lite Medium（默认）或 Large
  - 协议：FocalLoss + SGD/AdamW；输出 models/gate_model_<arch>.h5 + meta/history

用法：
  python3 train_gate_model.py --arch resnet_medium --epochs 80 --patience 20 \
      --batch-size 256 --steps-per-epoch 500 --optimizer sgd --lr 0.01
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS_DIR, TRAIN_CONFIG
import data.dataset as dataset
from models.resnet_lite_1d import (
    build_ecg_resnet_lite_medium,
    build_ecg_resnet_lite_large,
    compile_model,
    model_summary_table,
)

_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass

ARCH_BUILDERS = {
    "resnet_medium": build_ecg_resnet_lite_medium,
    "resnet_large": build_ecg_resnet_lite_large,
}


def main():
    ap = argparse.ArgumentParser(description="Train dual-expert gate model")
    ap.add_argument("--arch", choices=sorted(ARCH_BUILDERS.keys()),
                    default="resnet_medium")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps-per-epoch", type=int, default=500,
                    help="每 epoch 训练步数；0=完整 epoch")
    ap.add_argument("--val-steps", type=int, default=0)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="sgd")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--quick", action="store_true",
                    help="2 epochs / 2 batches smoke")
    ap.add_argument("--out-dir", type=str, default=str(MODELS_DIR / "gate"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.set_npz_suffix("_deploy")

    print("=" * 70, flush=True)
    print(f"[GateTrain] arch={args.arch} epochs={args.epochs} "
          f"patience={args.patience} optimizer={args.optimizer} lr={args.lr}",
          flush=True)
    print("=" * 70, flush=True)

    datasets = dataset.prepare_datasets(
        batch_size=args.batch_size,
        use_incart=True,
        use_ptb_beat=True,
        ptb_abn_max=10000,
        domain_balanced=True,
        patient_split=True,
    )

    builder = ARCH_BUILDERS[args.arch]
    model = builder(input_shape=datasets["input_shape"])
    model_summary_table(model)
    compile_model(model, learning_rate=args.lr, optimizer=args.optimizer)

    model_path = out_dir / f"gate_model_{args.arch}.h5"
    history_csv = out_dir / f"gate_model_{args.arch}_history.csv"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=args.patience,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=max(6, args.patience // 3), min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(history_csv), append=False),
    ]

    train_ds = datasets["train_ds"].take(2) if args.quick else datasets["train_ds"]
    if not args.quick and args.steps_per_epoch > 0:
        train_ds = train_ds.repeat()
    steps_per_epoch = 2 if args.quick else (args.steps_per_epoch or None)
    val_steps = 2 if args.quick else (args.val_steps or None)

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=(datasets["val_ds"].take(2) if args.quick
                         else datasets["val_ds"]),
        epochs=2 if args.quick else args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(model_path)

    meta = {
        "task": "dual_expert_gate_A1",
        "arch": args.arch,
        "epochs_run": len(history.history.get("loss", [])),
        "best_val_auc": float(max(history.history.get("val_auc", [0.0]))),
        "best_val_loss": float(min(history.history.get("val_loss", [0.0]))),
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "model_path": str(model_path),
        "history_csv": str(history_csv),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / f"gate_model_{args.arch}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GateTrain] done best_val_auc={meta['best_val_auc']:.4f} "
          f"elapsed={meta['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
