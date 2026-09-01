#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_exp7c_ecgfounder.py — exp7c + ECGFounder 代理硬负样本弱标签微调
================================================================================
在 exp7c 基础上，加入：
  - 真实 AFE 正常拍（311，label 0）
  - ECGFounder 1-lead 距离挖掘出的 PTB-XL 全局节律异常拍（841，label 1，弱标签）
  - 因果部署链 MIT/INCART/PTB 混合金标准数据（防灾难性遗忘）

训练只解冻 fc1/out，Adam lr=1e-5，class_weight 正常类加权，便于观察
真实域抑制与基准 AUC 的权衡。

输出：
  models/best_resnet_large_exp7c_ecgfounder.h5
  models/train_history_exp7c_ecgfounder.csv
  models/deploy_match/finetune_exp7c_ecgfounder.json
"""
import sys, json, os, time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from models.resnet_lite_1d import build_ecg_resnet_lite_large

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
CACHE = MODELS / "deploy_match"
ECG_DATA = Path(os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data"))
EGF_DIR = MODELS / "ecgfounder"

OUT_H5 = MODELS / "best_resnet_large_exp7c_ecgfounder.h5"
OUT_CSV = MODELS / "train_history_exp7c_ecgfounder.csv"
OUT_JSON = CACHE / "finetune_exp7c_ecgfounder.json"

SEED = 42
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

def load_domain(tag, n_abn, n_norm):
    b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
    l = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r")
    ia = np.where(l == 1)[0]
    inn = np.where(l == 0)[0]
    sa = rng.choice(ia, min(n_abn, len(ia)), replace=False)
    sn = rng.choice(inn, min(n_norm, len(inn)), replace=False)
    # 患者级泄漏守卫 (2026-09 审计: 本脚本历史抽样混入测试患者, 见
    # models/deploy_match/provenance_leakage_audit.json); 再次运行将直接失败。
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from data.split_guard import get_guard
    import os as _os
    _ecg = _os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data")
    r = np.load(str(_ecg) + "/" + tag + "_processed_deploy_causal_record_ids.npy")
    get_guard(tag).assert_train_only(np.concatenate([r[sa], r[sn]]),
                                     context="load_domain")
    return (np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
            np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32))

def main():
    print("[FT-EGF] loading data ...")
    # 真实 AFE 正常
    real = np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32)
    extra = np.load(DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32)
    real = np.concatenate([real, extra], axis=0)

    # ECGFounder 硬负样本（弱标签 abnormal）
    hardneg = np.load(EGF_DIR / "hardneg_beats.npy").astype(np.float32)
    print(f"[FT-EGF] real normal={len(real)}, ecgfounder weak abn={len(hardneg)}")

    # 金标准混合
    mit_a, mit_al, mit_n, mit_nl = load_domain("mit_bih", 800, 200)
    inc_a, inc_al, inc_n, inc_nl = load_domain("incart", 200, 100)
    ptb_a, ptb_al, ptb_n, ptb_nl = load_domain("ptb", 300, 100)
    x_mix = np.concatenate([mit_a, inc_a, ptb_a, mit_n, inc_n, ptb_n])[..., np.newaxis]
    y_mix = np.concatenate([mit_al, inc_al, ptb_al, mit_nl, inc_nl, ptb_nl])
    print(f"[FT-EGF] gold mix={len(x_mix)} (abn={int((y_mix==1).sum())})")

    # 真实拍留出 40 验证
    val_idx = rng.choice(len(real), 40, replace=False)
    trn_idx = np.setdiff1d(np.arange(len(real)), val_idx)
    x_rtr, y_rtr = real[trn_idx][..., np.newaxis], np.zeros(len(trn_idx), dtype=np.int32)
    x_rva, y_rva = real[val_idx][..., np.newaxis], np.zeros(len(val_idx), dtype=np.int32)
    x_hard, y_hard = hardneg[..., np.newaxis], np.ones(len(hardneg), dtype=np.int32)

    # 验证集：从金标准混合中抽 200 A + 200 N + 40 真实正常，确保 AUC 可算且不过拟合看训练集
    val_abn_idx = rng.choice(np.where(y_mix == 1)[0], size=min(200, int((y_mix == 1).sum())), replace=False)
    val_norm_idx = rng.choice(np.where(y_mix == 0)[0], size=min(200, int((y_mix == 0).sum())), replace=False)
    val_idx_mix = np.concatenate([val_abn_idx, val_norm_idx])
    x_val = np.concatenate([x_mix[val_idx_mix], x_rva])
    y_val = np.concatenate([y_mix[val_idx_mix], y_rva])
    train_mask = np.ones(len(x_mix), dtype=bool)
    train_mask[val_idx_mix] = False
    x_mix_train = x_mix[train_mask]
    y_mix_train = y_mix[train_mask]

    # 训练集（打乱）
    x_train = np.concatenate([x_mix_train, x_rtr, x_hard])
    y_train = np.concatenate([y_mix_train, y_rtr, y_hard])
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[FT-EGF] train={len(x_train)}, val={len(x_val)} "
          f"(val abn={int((y_val==1).sum())}, norm={int((y_val==0).sum())})")

    # 模型
    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    model.load_weights(str(MODELS / "best_resnet_large_exp7c.h5"))
    for layer in model.layers:
        layer.trainable = layer.name in ("fc1", "out")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    def _auc(y_true, p):
        y = np.asarray(y_true).ravel()
        p = np.asarray(p).ravel()
        order = np.argsort(p)
        y = y[order]
        n1 = int((y == 1).sum())
        n0 = int((y == 0).sum())
        if n1 == 0 or n0 == 0:
            return 0.5
        ranks = np.arange(1, len(y) + 1)[y == 1].sum()
        return float((ranks - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    class ValAucCB(tf.keras.callbacks.Callback):
        def __init__(self, xv, yv):
            super().__init__()
            self.xv, self.yv = xv, yv
            self.best = 0.0
        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(self.xv, batch_size=128, verbose=0)[:, 1]
            auc = _auc(self.yv, p)
            logs = logs or {}
            logs["val_auc"] = auc
            if auc > self.best:
                self.best = auc
                self.model.save(str(OUT_H5))
                print(f"  * saved best (val_auc={auc:.4f})", flush=True)

    # 微调前基线
    m0 = tf.keras.models.load_model(str(MODELS / "best_resnet_large_exp7c.h5"), compile=False)
    p0 = m0.predict(real[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
    print(f"[FT-EGF] BEFORE (exp7c on real): mean={p0.mean():.4f} frac>0.5={float((p0>0.5).mean()):.4f}")

    cbs = [ValAucCB(x_val, y_val),
           tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                            patience=12, restore_best_weights=True,
                                            verbose=1),
           tf.keras.callbacks.CSVLogger(str(OUT_CSV))]
    t0 = time.time()
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                     batch_size=32, epochs=40, callbacks=cbs,
                     class_weight={0: 2.0, 1: 1.0}, verbose=2)
    print(f"[FT-EGF] training done in {time.time()-t0:.0f}s", flush=True)

    model.load_weights(str(OUT_H5))
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p1 = model.predict(real[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
    p1_hard = model.predict(hardneg[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
    p1_va = model.predict(x_rva, batch_size=64, verbose=0)[:, 1]
    print(f"[FT-EGF] AFTER on real: mean={p1.mean():.4f} frac>0.5={float((p1>0.5).mean()):.4f}")
    print(f"[FT-EGF] AFTER on ecgfounder weak abn: mean={p1_hard.mean():.4f} frac>0.5={float((p1_hard>0.5).mean()):.4f}")
    print(f"[FT-EGF] heldout real: mean={p1_va.mean():.4f}")

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "exp7c + ECGFounder proxy hard negative weak-label finetune",
        "data": {
            "real_normal_beats": int(len(real)),
            "ecgfounder_weak_abn_beats": int(len(hardneg)),
            "gold_mix_abn": int((y_mix == 1).sum()),
            "gold_mix_norm": int((y_mix == 0).sum()),
            "held_out_real": int(len(val_idx)),
        },
        "config": {
            "base": "best_resnet_large_exp7c.h5",
            "freeze": "backbone except fc1/out",
            "lr": 1e-5,
            "optimizer": "adam",
            "loss": "sparse CE",
            "class_weight": {0: 2.0, 1: 1.0},
            "epochs": int(len(hist.epoch)),
        },
        "confidence_real_normal": {
            "before_exp7c": {"mean": float(p0.mean()), "frac_gt_0.5": float((p0 > 0.5).mean())},
            "after_all": {"mean": float(p1.mean()), "frac_gt_0.5": float((p1 > 0.5).mean())},
            "after_heldout40": {"mean": float(p1_va.mean())},
            "after_ecgfounder_weak_abn": {
                "mean": float(p1_hard.mean()),
                "frac_gt_0.5": float((p1_hard > 0.5).mean()),
            },
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[FT-EGF] saved {OUT_H5.name} + {OUT_JSON.name}")

if __name__ == "__main__":
    main()
