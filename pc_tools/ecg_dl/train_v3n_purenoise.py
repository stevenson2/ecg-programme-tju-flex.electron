#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_v3n_purenoise.py — Round-D1: 纯噪声硬负例重训候选 v3-N (TH §110)
================================================================================
动机 (§109 勘误后 FP 清单): 纯运动伪迹 (无 ECG 载波) 是唯一的大 FP 源
(板上 raw 0.88-0.90, 擎住 77%)。既有训练集中"正常"类从不含无 ECG 结构的
纯噪声窗 (v3-B 增强全带载波; §99 合成硬负例虽含 20% 置零掩码但基底仍是
真实 AFE 拍)。本脚本在 v3-A 配方上追加 label=0 的链后纯噪声窗。

配方: 与 v3-A 完全一致 (同主流 rng → 同抽样/同验证集/同初始化序列), 追加:
  - 1200 个纯噪声窗 (500Hz 合成 → chain_shape → 250Hz 窗 → 逐窗 z-score):
    motion 600 / emg 240 / mains 240 / baseline_wander 60 / respiratory 60
  - 在独立 rng_aug (seed=SEED+2000) 上生成, 不消耗主流。
铁律: 公共库取数经 SplitGuard (同 §100); 从零训练; 全数字落盘。
门槛 (§110): 干净 test 不劣 (§98 口径, 容差 0.02) + 语料 pure_motion FP
显著降 & abnormal 检出/normal+noise 不回退。任一不过保留 v3-A。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from data.split_guard import get_guard
from models.resnet_lite_1d import build_ecg_resnet_lite_large
from config import BEAT_WINDOW_SAMPLES
from noise_lib import synth_noise, chain_shape
from train_clean_baseline_v3 import (
    sample_domain, sample_val_domain, synth_hard, train_normal_beat_stats,
    build_model, MAIN_RATIO, N_REAL_HOLDOUT, REAL_REPEAT, EPOCHS, PATIENCE,
    BASE_LR, BATCH, VAL_MI_PER_CLASS, VAL_PTB_PER_CLASS, SEED,
    DATA_REAL, MODELS, CACHE,
)

OUT_H5 = MODELS / "best_resnet_large_v3n_purenoise.h5"
OUT_CSV = MODELS / "train_history_v3n_purenoise.csv"
OUT_JSON = CACHE / "train_v3n_purenoise.json"
RNG_NOISE_SEED = SEED + 2000

NOISE_MIX = [("motion", 600), ("emg", 240), ("mains", 240),
             ("baseline_wander", 60), ("respiratory", 60)]


def make_pure_noise_windows(rng):
    """无 ECG 结构的链后纯噪声窗 (label=0 候选)。独立 rng, 不碰主流。"""
    win = BEAT_WINDOW_SAMPLES
    outs = []
    types = []
    for ntype, n in NOISE_MIX:
        for _ in range(n):
            raw = synth_noise(ntype, 2 * win, 500, rng)
            shaped = chain_shape(raw).astype(np.float32)
            assert len(shaped) == win
            mu, sd = shaped.mean(), shaped.std()
            outs.append((shaped - mu) / max(sd, 1e-6))
            types.append(ntype)
    return np.asarray(outs, dtype=np.float32), types


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)              # 主流: 与 v3-A 一致
    rng_noise = np.random.default_rng(RNG_NOISE_SEED)
    tf.random.set_seed(SEED)

    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[V3N] GPU: {gpu}", flush=True)

    # ---------- 主数据 (与 v3-A 同主流) ----------
    avail = train_normal_beat_stats()
    domains = {}
    audits = {}
    for tag in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = MAIN_RATIO[tag]
        if n_norm > avail[tag]["train_norm_available"]:
            raise RuntimeError(f"{tag}: 正常拍不足")
        domains[tag] = sample_domain(tag, n_abn, n_norm, rng)
        audits[tag] = domains[tag][4]
    x_main = np.concatenate([
        domains["mit_bih"][0], domains["incart"][0], domains["ptb"][0],
        domains["mit_bih"][2], domains["incart"][2], domains["ptb"][2],
    ])
    y_main = np.concatenate([
        domains["mit_bih"][1], domains["incart"][1], domains["ptb"][1],
        domains["mit_bih"][3], domains["incart"][3], domains["ptb"][3],
    ])
    print(f"[V3N] main gold: {len(x_main)} "
          f"(abn={int((y_main == 1).sum())}, norm={int((y_main == 0).sum())})", flush=True)

    # ---------- 纯噪声硬负例 (Round-D1 核心) ----------
    x_noise, noise_types = make_pure_noise_windows(rng_noise)
    y_noise = np.zeros(len(x_noise), dtype=np.int32)
    print(f"[V3N] pure-noise negatives: {len(x_noise)} "
          f"({ {t: noise_types.count(t) for t in set(noise_types)} })", flush=True)

    # ---------- 真实 AFE + 合成硬负例 (与 v3-A 同主流) ----------
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
    hard = synth_hard(real_train, rng)
    x_hard = hard[..., np.newaxis]
    y_hard = np.zeros(len(hard), dtype=np.int32)

    x_train = np.concatenate([
        x_main[..., np.newaxis], x_noise[..., np.newaxis],
        x_real, x_hard]).astype(np.float32)
    y_train = np.concatenate([y_main, y_noise, y_real, y_hard])
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[V3N] train total={len(x_train)} abn={int((y_train == 1).sum())} "
          f"norm={int((y_train == 0).sum())} (纯噪声负例 {len(x_noise)})", flush=True)

    # ---------- 干净验证集 (与 v3-A 同主流) ----------
    va, val, vn, vnl = sample_val_domain("mit_bih", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = sample_val_domain("incart", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = sample_val_domain("ptb", VAL_PTB_PER_CLASS, VAL_PTB_PER_CLASS, rng)
    x_val = np.concatenate([
        np.concatenate([va, vn, ia, inv, pa, pn])[..., np.newaxis],
        x_real_ho,
    ]).astype(np.float32)
    y_val = np.concatenate([val, vnl, ial, inl, pal, pnl, y_real_ho])
    print(f"[V3N] val total={len(x_val)} (干净)", flush=True)

    # ---------- 训练 ----------
    model = build_model(0.4)
    print(f"[V3N] params={model.count_params():,} (随机初始化, dropout=0.4)", flush=True)
    steps_per_epoch = max(1, len(x_train) // BATCH)
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=BASE_LR, decay_steps=EPOCHS * steps_per_epoch)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    class ValAucCB(tf.keras.callbacks.Callback):
        def __init__(self, xv, yv):
            super().__init__()
            self.xv, self.yv = xv, yv
            self.best = -1.0

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(self.xv, batch_size=256, verbose=0)[:, 1]
            auc = float(roc_auc_score(self.yv, p))
            logs = logs or {}
            logs["val_auc"] = auc
            if auc > self.best:
                self.best = auc
                self.model.save(str(OUT_H5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})", flush=True)

    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                     batch_size=BATCH, epochs=EPOCHS,
                     callbacks=[ValAucCB(x_val, y_val),
                                tf.keras.callbacks.EarlyStopping(
                                    monitor="val_auc", mode="max",
                                    patience=PATIENCE, restore_best_weights=True,
                                    verbose=1),
                                tf.keras.callbacks.CSVLogger(str(OUT_CSV))],
                     class_weight={0: 1.0, 1: 1.0}, verbose=2)
    print(f"[V3N] training done in {time.time() - t0:.0f}s", flush=True)

    best = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_real_ho, batch_size=64, verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val))
    print(f"[V3N] BEST: val AUC={auc_val:.4f}; real holdout mean={p_real.mean():.4f} "
          f"frac>0.5={float((p_real > 0.5).mean()):.4f}", flush=True)

    split_stats = {
        tag: {k: get_guard(tag).stats[k]
              for k in ("n_patients", "n_train", "n_val", "n_test",
                        "beats_train", "beats_val", "beats_test")}
        for tag in ("mit_bih", "incart", "ptb")
    }
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Round-D1 v3-N: v3-A 配方 + 纯噪声硬负例 (TH §110)",
        "provenance": "from scratch; main rng identical to v3-A; "
                      "pure-noise windows on rng seed=%d (no patient data)" % RNG_NOISE_SEED,
        "gpu": [str(g) for g in gpu],
        "output_model": str(OUT_H5.relative_to(BASE)),
        "patient_split_seed": SEED,
        "noise_mix": {t: n for t, n in NOISE_MIX},
        "data": {"main_gold": len(x_main), "pure_noise_neg": len(x_noise),
                 "real_afe_train_unique": len(train_real_idx),
                 "real_afe_holdout": len(x_real_ho),
                 "synth_hard": len(x_hard), "train_total": len(x_train)},
        "audits": audits,
        "split_stats": split_stats,
        "best_val_auc": auc_val,
        "real_holdout": {"mean": float(p_real.mean()),
                         "frac_gt_05": float((p_real > 0.5).mean())},
        "wall_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[V3N] report -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
