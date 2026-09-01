#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_clean_baseline.py — 从零重训干净基线 (clean-split baseline, TH §99)
================================================================================
铁律 (§97/§98 固化):
  1. 一切公共库取数经 data/split_guard.py (assert_train_only / sample_train_beats),
     seed=42, 患者级划分; 测试/验证患者数据不得进入训练。
  2. 随机初始化从零训练; 不加载任何现有 checkpoint/.h5/权重 (谱系即泄漏污染物)。
  3. 全部数字落盘: 训练历史 CSV + 实验记录 JSON。

配方 (沿用 finetune_exp7c_v4.py 的数据配比, 改为从零训练):
  - 主数据: MIT (1500,500) / INCART (400,150) / PTB (600,200) (abn,norm),
    全部仅取自训练患者 (SplitGuard 断言)。
  - 真实 AFE 正常拍: 训练 ~271 拍 (seed=42 留出 40 拍做 holdout)。
  - 合成硬负样本: 仅由真实 AFE 训练拍生成 (~600 个)。
  - 不含 exp7c 概率挖出的公共库硬负样本 (依赖旧模型, 与"从零"矛盾)。
  - 验证集: MIT+INCART val 患者 (800/800) + PTB val 患者 (400/400)
    + 真实 AFE 留出 40 拍; 按最佳 val AUC 保存。

优化: Adam, lr=1e-3 + cosine decay; epochs=80; batch=32;
class_weight {0: 2.0, 1: 1.0}; EarlyStopping(val_auc, patience=20)。

输出:
  models/best_resnet_large_clean_baseline.h5
  models/train_history_clean_baseline.csv
  models/deploy_match/train_clean_baseline.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from models.resnet_lite_1d import build_ecg_resnet_lite_large
from data.split_guard import get_guard

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
CACHE = MODELS / "deploy_match"

OUT_H5 = MODELS / "best_resnet_large_clean_baseline.h5"
OUT_CSV = MODELS / "train_history_clean_baseline.csv"
OUT_JSON = CACHE / "train_clean_baseline.json"

SEED = 42
MAIN_RATIO = {
    "mit_bih": (1500, 500),
    "incart": (400, 150),
    "ptb": (600, 200),
}
N_REAL_HOLDOUT = 40
REAL_REPEAT = 2
CLASS_WEIGHT_NORMAL = 2.0
EPOCHS = 80
PATIENCE = 20
# v1 用 1e-3 → 早停于 epoch 23, cosine 未衰减, 全程近峰值学习率导致过拟合与
# θ=0.5 "恒报警"退化 (MIT 拍级 prec=0.11)。降至 3e-4 使学习率轨迹平缓、改善校准。
BASE_LR = 3e-4
BATCH = 32
VAL_MI_PER_CLASS = 800
VAL_PTB_PER_CLASS = 400


def sample_domain(tag, n_abn, n_norm, rng):
    """仅从训练患者抽 (SplitGuard), 并断言抽样结果全部属训练患者。"""
    g = get_guard(tag)
    sa, sn = g.sample_train_beats(n_abn, n_norm, rng)
    from data.split_guard import load_arrays
    b, _l, r = load_arrays(tag)
    sa, sn = np.asarray(sa), np.asarray(sn)
    g.assert_train_only(np.concatenate([np.asarray(r)[sa], np.asarray(r)[sn]]),
                        context=f"train_clean_baseline.sample_domain({tag})")
    return (
        np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
        np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32),
    )


def sample_val_domain(tag, n_abn, n_norm, rng):
    """验证集: 仅从 val 患者抽 (绝不含训练/测试患者)。"""
    from data.split_guard import load_arrays
    g = get_guard(tag)
    b, l, _r = load_arrays(tag)
    m = g.val_mask[:len(l)]
    ia = np.where(m & (l == 1))[0]
    inn = np.where(m & (l == 0))[0]
    if len(ia) < n_abn or len(inn) < n_norm:
        raise RuntimeError(f"{tag}: val 患者心拍不足 "
                           f"(abn {len(ia)}<{n_abn}, norm {len(inn)}<{n_norm})")
    sa = rng.choice(ia, n_abn, replace=False)
    sn = rng.choice(inn, n_norm, replace=False)
    return (
        np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
        np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32),
    )


def synth_hard(real_train, rng):
    """对真实 AFE 训练拍做少量多形态合成 (与 finetune_exp7c_v4.synth_hard 同源)。"""
    n_source = min(100, len(real_train))
    idx = rng.choice(len(real_train), n_source, replace=False)
    base = real_train[idx]
    outs = []
    t = np.arange(BEAT_WINDOW_SAMPLES, dtype=np.float32) / 250.0

    outs.append(base + rng.normal(0, 10 ** (-20 / 20.0), base.shape).astype(np.float32))
    outs.append(base + rng.normal(0, 10 ** (-10 / 20.0), base.shape).astype(np.float32))
    outs.append(base + (0.3 * np.sin(2 * np.pi * 0.3 * t)).astype(np.float32)[None, :])
    outs.append(base + (0.5 * np.sin(2 * np.pi * 0.8 * t)).astype(np.float32)[None, :])
    y = base.copy()
    for i in range(n_source):
        for _ in range(5):
            pos = rng.integers(0, BEAT_WINDOW_SAMPLES)
            amp = float(rng.uniform(0.5, 2.0) * rng.choice([-1, 1]))
            y[i, pos] += amp
    outs.append(y.astype(np.float32))
    mask = rng.random(base.shape) < 0.2
    outs.append((base * (1 - mask)).astype(np.float32))
    return np.concatenate(outs, axis=0)


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 按 README §GPU训练(WSL2) 配置后重试, "
                           "禁止静默退回 CPU 训练")
    print(f"[CB] GPU: {gpu}", flush=True)

    # ---------- 主数据 (仅训练患者, SplitGuard 断言) ----------
    mit_a, mit_al, mit_n, mit_nl = sample_domain("mit_bih", *MAIN_RATIO["mit_bih"], rng=rng)
    inc_a, inc_al, inc_n, inc_nl = sample_domain("incart", *MAIN_RATIO["incart"], rng=rng)
    ptb_a, ptb_al, ptb_n, ptb_nl = sample_domain("ptb", *MAIN_RATIO["ptb"], rng=rng)
    x_main = np.concatenate([mit_a, inc_a, ptb_a, mit_n, inc_n, ptb_n])[..., np.newaxis]
    y_main = np.concatenate([mit_al, inc_al, ptb_al, mit_nl, inc_nl, ptb_nl])
    print(f"[CB] main gold: {len(x_main)} "
          f"(abn={int((y_main == 1).sum())}, norm={int((y_main == 0).sum())})", flush=True)

    # ---------- 真实 AFE 正常拍: 训练/留出 ----------
    real = np.concatenate([
        np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32),
        np.load(DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32),
    ])
    holdout_idx = rng.choice(len(real), N_REAL_HOLDOUT, replace=False)
    train_real_idx = np.setdiff1d(np.arange(len(real)), holdout_idx)
    real_train = real[train_real_idx]
    x_real = np.concatenate([real_train] * REAL_REPEAT)[..., np.newaxis]
    y_real = np.zeros(len(x_real), dtype=np.int32)
    x_real_ho = real[holdout_idx][..., np.newaxis]
    y_real_ho = np.zeros(len(holdout_idx), dtype=np.int32)
    print(f"[CB] real AFE: train_unique={len(train_real_idx)} "
          f"repeated={len(x_real)}, holdout={len(x_real_ho)}", flush=True)

    # ---------- 合成硬负样本 (仅源自真实 AFE 训练拍) ----------
    hard = synth_hard(real_train, rng)
    x_hard = hard[..., np.newaxis]
    y_hard = np.zeros(len(hard), dtype=np.int32)
    print(f"[CB] synthetic hard neg: {len(x_hard)}", flush=True)

    x_train = np.concatenate([x_main, x_real, x_hard])
    y_train = np.concatenate([y_main, y_real, y_hard])
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[CB] train total={len(x_train)} abn={int((y_train == 1).sum())} "
          f"norm={int((y_train == 0).sum())}", flush=True)

    # ---------- 验证集: val 患者 + 真实 AFE 留出 ----------
    va, val, vn, vnl = sample_val_domain("mit_bih", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = sample_val_domain("incart", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = sample_val_domain("ptb", VAL_PTB_PER_CLASS, VAL_PTB_PER_CLASS, rng)
    x_val = np.concatenate([
        np.concatenate([va, vn, ia, inv, pa, pn])[..., np.newaxis],
        x_real_ho,
    ]).astype(np.float32)
    y_val = np.concatenate([val, vnl, ial, inl, pal, pnl, y_real_ho])
    print(f"[CB] val total={len(x_val)} abn={int((y_val == 1).sum())} "
          f"norm={int((y_val == 0).sum())} (incl real holdout {len(holdout_idx)})", flush=True)

    # ---------- 模型: 随机初始化从零训练 ----------
    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    print(f"[CB] params={model.count_params():,} (随机初始化, 未加载任何权重)", flush=True)

    steps_per_epoch = max(1, len(x_train) // BATCH)
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=BASE_LR, decay_steps=EPOCHS * steps_per_epoch)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    class ValAucCB(tf.keras.callbacks.Callback):
        def __init__(self, xv, yv):
            super().__init__()
            self.xv, self.yv = xv, yv
            self.best = -1.0
            self.best_epoch = -1

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(self.xv, batch_size=256, verbose=0)[:, 1]
            auc = float(roc_auc_score(self.yv, p))
            logs = logs or {}
            logs["val_auc"] = auc
            if auc > self.best:
                self.best = auc
                self.best_epoch = epoch + 1
                self.model.save(str(OUT_H5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})", flush=True)

    cbs = [
        ValAucCB(x_val, y_val),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=PATIENCE, restore_best_weights=True,
                                         verbose=1),
        tf.keras.callbacks.CSVLogger(str(OUT_CSV)),
    ]

    print(f"[CB] start from-scratch training: epochs={EPOCHS} batch={BATCH} "
          f"lr={BASE_LR} cosine, class_weight={{0:{CLASS_WEIGHT_NORMAL},1:1.0}}", flush=True)
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=cbs,
        class_weight={0: CLASS_WEIGHT_NORMAL, 1: 1.0},
        verbose=2,
    )
    print(f"[CB] training done in {time.time() - t0:.0f}s", flush=True)

    # ---------- 最佳模型自检 ----------
    best = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_real_ho, batch_size=64, verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val))
    print(f"[CB] BEST: val AUC={auc_val:.4f}; real holdout mean={p_real.mean():.4f} "
          f"frac>0.5={float((p_real > 0.5).mean()):.4f}", flush=True)

    split_stats = {
        tag: {k: get_guard(tag).stats[k]
              for k in ("n_patients", "n_train", "n_val", "n_test",
                        "beats_train", "beats_val", "beats_test")}
        for tag in ("mit_bih", "incart", "ptb")
    }
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §99 从零重训干净基线 (随机初始化 + SplitGuard 取数)",
        "provenance": "from scratch; no checkpoint loaded",
        "gpu": [str(g) for g in gpu],
        "output_model": str(OUT_H5.relative_to(BASE)),
        "patient_split_seed": SEED,
        "split_stats": split_stats,
        "data": {
            "main": {
                "mit_abn": len(mit_a), "mit_norm": len(mit_n),
                "incart_abn": len(inc_a), "incart_norm": len(inc_n),
                "ptb_abn": len(ptb_a), "ptb_norm": len(ptb_n),
            },
            "real_afe_train_unique": int(len(train_real_idx)),
            "real_afe_train_repeated": int(len(x_real)),
            "real_afe_holdout": int(len(x_real_ho)),
            "synthetic_hard_negative": int(len(x_hard)),
            "train_total": int(len(x_train)),
            "train_abn": int((y_train == 1).sum()),
            "train_norm": int((y_train == 0).sum()),
            "val_total": int(len(x_val)),
            "val_abn": int((y_val == 1).sum()),
            "val_norm": int((y_val == 0).sum()),
            "real_holdout_indices": [int(i) for i in holdout_idx],
        },
        "config": {
            "architecture": "build_ecg_resnet_lite_large",
            "params": int(model.count_params()),
            "optimizer": "adam",
            "lr_schedule": f"cosine decay from {BASE_LR}",
            "epochs_requested": EPOCHS,
            "epochs_run": int(len(hist.epoch)),
            "batch_size": BATCH,
            "early_stopping_patience": PATIENCE,
            "val_monitor": "val_auc",
            "class_weight": {0: CLASS_WEIGHT_NORMAL, 1: 1.0},
            "loss": "sparse_categorical_crossentropy",
        },
        "results": {
            "best_val_auc": auc_val,
            "real_holdout_mean_prob": float(p_real.mean()),
            "real_holdout_frac_gt_0.5": float((p_real > 0.5).mean()),
            "train_time_s": round(time.time() - t0, 1),
        },
        "anchors": {
            "v4_clean_mit_incart": {"auc": 0.848, "event_f1": 0.697},
            "ptb_clean_deployed_int8": {"auc": 0.900, "event_f1": 0.898},
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[CB] saved {OUT_H5.name}, {OUT_CSV.name}, {OUT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
