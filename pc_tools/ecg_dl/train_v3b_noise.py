#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_v3b_noise.py — 抗噪重训候选 v3-B (M3, TH §107)
================================================================================
基线: v3-A = train_clean_baseline_v3.py 配置 A (MIT 1500/1500, INCART 400/400,
PTB 600/400; class_weight 1:1; large 架构 dropout 0.4; 真实 AFE 拍×2 + 合成
硬负例)。本脚本 = 同配方 + 主数据 (公共库拍) 的逐拍独立噪声增强变体
(augment_noise.augment_beats: 5 类型 × 5 SNR 梯度, 设备链整形后叠加)。

同口径 A/B 纪律:
  1. 主 rng 流与 v3-A 完全一致 (增强用独立 rng_aug, seed=SEED+1000,
     不消耗主 rng) → 抽样/验证集/初始权重序列与 v3-A 同源可比。
  2. 验证集保持干净 (不增强), val AUC 与 v3-A 可直接对比。
  3. 双门槛 (M3 验收): 干净 test 不劣于 v3-A (eval_clean_test 口径) 且
     带噪语料显著优于 v3-A (board_corpus_eval / eval_corpus_pc 口径);
     任一不过保留 v3-A (负结果入档)。

铁律: 一切公共库取数经 data/split_guard.py; 随机初始化从零训练; 全数字落盘。
输出: models/best_resnet_large_v3b_noise.h5, train_history_v3b_noise.csv,
models/deploy_match/train_v3b_noise.json
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
from data.split_guard import get_guard, load_arrays
from augment_noise import augment_beats, AUG_TYPES, AUG_SNRS
from train_clean_baseline_v3 import (
    sample_domain, sample_val_domain, synth_hard, train_normal_beat_stats,
    build_model, MAIN_RATIO, N_REAL_HOLDOUT, REAL_REPEAT, EPOCHS, PATIENCE,
    BASE_LR, BATCH, VAL_MI_PER_CLASS, VAL_PTB_PER_CLASS, SEED,
    DATA_REAL, MODELS, CACHE,
)

OUT_H5 = MODELS / "best_resnet_large_v3b_noise.h5"
OUT_CSV = MODELS / "train_history_v3b_noise.csv"
OUT_JSON = CACHE / "train_v3b_noise.json"
RNG_AUG_SEED = SEED + 1000
VARIANTS_PER_BEAT = 1
# v2 (v1 负结果见 train_v3b_noise_v1_fullsnr_ladder.json): 训练侧 SNR 梯度收敛到
# 30/20/10dB。v1 用全梯度 (30..0) 时 0/5dB 变体把载波淹没, 同类噪声窗携带
# 相反标签 -> 模型记忆化 (train acc 98.9%, val AUC epoch1 见顶 0.744 后劣化,
# 真实 AFE frac>0.5=1.0 全异常坍缩)。评估语料保持全梯度 (含 0/5dB) 作硬考题。
TRAIN_SNRS = (30, 20, 10)


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)          # 主流: 与 v3-A 完全一致
    rng_aug = np.random.default_rng(RNG_AUG_SEED)   # 增强流: 独立, 不扰主流
    tf.random.set_seed(SEED)

    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[V3B] GPU: {gpu}", flush=True)

    # ---------- 主数据 (与 v3-A 同 rng → 同抽样; SplitGuard 断言) ----------
    avail = train_normal_beat_stats()
    audits = {}
    domains = {}
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
    print(f"[V3B] main gold: {len(x_main)} "
          f"(abn={int((y_main == 1).sum())}, norm={int((y_main == 0).sum())})", flush=True)

    # ---------- 噪声增强变体 (M3 核心; 逐拍独立 type/SNR/实现) ----------
    x_noisy, aug_prov = augment_beats(x_main, rng_aug,
                                      variants_per_beat=VARIANTS_PER_BEAT,
                                      snrs=TRAIN_SNRS)
    y_noisy = y_main.copy()
    print(f"[V3B] noisy variants: {len(x_noisy)} "
          f"({len(aug_prov['counts'])} 种 type x SNR 组合)", flush=True)

    x_main_all = np.concatenate([x_main, x_noisy])[..., np.newaxis]
    y_main_all = np.concatenate([y_main, y_noisy])

    # ---------- 真实 AFE 正常拍: 训练/留出 (与 v3-A 同主流) ----------
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
    print(f"[V3B] real AFE: train_unique={len(train_real_idx)} holdout={len(x_real_ho)}",
          flush=True)

    # ---------- 合成硬负样本 (与 §99/v3 同源) ----------
    hard = synth_hard(real_train, rng)
    x_hard = hard[..., np.newaxis]
    y_hard = np.zeros(len(hard), dtype=np.int32)

    x_train = np.concatenate([x_main_all, x_real, x_hard]).astype(np.float32)
    y_train = np.concatenate([y_main_all, y_real, y_hard])
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[V3B] train total={len(x_train)} abn={int((y_train == 1).sum())} "
          f"norm={int((y_train == 0).sum())} (含噪声变体 {len(x_noisy)})", flush=True)

    # ---------- 验证集: 干净 (与 v3-A 同主流) ----------
    va, val, vn, vnl = sample_val_domain("mit_bih", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = sample_val_domain("incart", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = sample_val_domain("ptb", VAL_PTB_PER_CLASS, VAL_PTB_PER_CLASS, rng)
    x_val = np.concatenate([
        np.concatenate([va, vn, ia, inv, pa, pn])[..., np.newaxis],
        x_real_ho,
    ]).astype(np.float32)
    y_val = np.concatenate([val, vnl, ial, inl, pal, pnl, y_real_ho])
    print(f"[V3B] val total={len(x_val)} abn={int((y_val == 1).sum())} "
          f"norm={int((y_val == 0).sum())} (干净, 不增强)", flush=True)

    # ---------- 训练 (配置 A 同参) ----------
    model = build_model(0.4)
    print(f"[V3B] params={model.count_params():,} (随机初始化, dropout=0.4)", flush=True)
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

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(self.xv, batch_size=256, verbose=0)[:, 1]
            auc = float(roc_auc_score(self.yv, p))
            logs = logs or {}
            logs["val_auc"] = auc
            if auc > self.best:
                self.best = auc
                self.model.save(str(OUT_H5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})", flush=True)

    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=[
            ValAucCB(x_val, y_val),
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                             patience=PATIENCE,
                                             restore_best_weights=True, verbose=1),
            tf.keras.callbacks.CSVLogger(str(OUT_CSV)),
        ],
        class_weight={0: 1.0, 1: 1.0},
        verbose=2,
    )
    print(f"[V3B] training done in {time.time() - t0:.0f}s", flush=True)

    best = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_real_ho, batch_size=64, verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val))
    print(f"[V3B] BEST: val AUC={auc_val:.4f}; real holdout "
          f"mean={p_real.mean():.4f} frac>0.5={float((p_real > 0.5).mean()):.4f}", flush=True)

    split_stats = {
        tag: {k: get_guard(tag).stats[k]
              for k in ("n_patients", "n_train", "n_val", "n_test",
                        "beats_train", "beats_val", "beats_test")}
        for tag in ("mit_bih", "incart", "ptb")
    }
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "M3 v3-B: v3-A 配方 + 逐拍独立链整形噪声增强 (TH §107)",
        "provenance": "from scratch; no checkpoint loaded; "
                      "main rng stream identical to v3-A (aug on rng_aug)",
        "gpu": [str(g) for g in gpu],
        "output_model": str(OUT_H5.relative_to(BASE := Path(__file__).resolve().parent)),
        "patient_split_seed": SEED,
        "rng_aug_seed": RNG_AUG_SEED,
        "variants_per_beat": VARIANTS_PER_BEAT,
        "augmentation": {
            "types": list(AUG_TYPES), "train_snrs": list(TRAIN_SNRS),
            "eval_snrs_note": "corpus keeps full ladder (30..0) as hard eval",
            "chain_shaped": True, "prov": aug_prov,
            "note": "noise shaped by PC deploy-chain replica before mixing; "
                    "val kept clean",
        },
        "data": {
            "main_gold": len(x_main), "noisy_variants": len(x_noisy),
            "real_afe_train_unique": len(train_real_idx),
            "real_afe_holdout": len(x_real_ho), "synth_hard": len(x_hard),
            "train_total": len(x_train),
        },
        "audits": audits,
        "split_stats": split_stats,
        "best_val_auc": auc_val,
        "real_holdout": {"mean": float(p_real.mean()),
                         "frac_gt_05": float((p_real > 0.5).mean())},
        "history_best_epoch": int(np.argmin(hist.history.get("loss", [1])) + 1),
        "wall_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[V3B] report -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
