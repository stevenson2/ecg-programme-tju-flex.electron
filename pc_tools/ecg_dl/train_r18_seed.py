#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_r18_seed.py — B0 基线种子扩展 (R18 预注册 §1-B0, TH §118)
================================================================================
目的: v3-A 配方 (train_clean_baseline_v3.py 配置 a) 以 seed 43/44 重训,
建立 3-seed 配对基线 (seed42 = 现存 v3-A, 复测已锚定)。配方逐行同源:
  MIT (1500,1500) / INCART (400,400) / PTB (600,400) + 真实 AFE 271×2 +
  合成硬负例 ~600; Adam 3e-4 + cosine, batch 32, epochs 80, patience 20,
  按 val AUC 存档, class_weight 1:1。

本脚本不改任何历史脚本; 从 train_clean_baseline_v3 导入采样/合成函数,
仅参数化 SEED 与输出后缀。铁律同 §97/§99: 全部公共库取数经 SplitGuard。

用法: python3 train_r18_seed.py 43 44   (或单 seed)
输出: models/best_resnet_large_clean_baseline_v3_seed{N}.h5
      models/train_history_clean_baseline_v3_seed{N}.csv
      models/deploy_match/train_clean_baseline_v3_seed{N}.json
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
import train_clean_baseline_v3 as V3

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"


def run_seed(seed: int):
    out_h5 = MODELS / f"best_resnet_large_clean_baseline_v3_seed{seed}.h5"
    out_csv = MODELS / f"train_history_clean_baseline_v3_seed{seed}.csv"
    out_json = CACHE / f"train_clean_baseline_v3_seed{seed}.json"
    if out_h5.exists():
        print(f"[R18B0:{seed}] 已存在, 跳过: {out_h5.name}", flush=True)
        return json.loads(out_json.read_text(encoding="utf-8"))

    t0 = time.time()
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[R18B0:{seed}] GPU: {gpu}", flush=True)

    # ---------- 主数据 (rng 调用序与 v3-A run_config 逐行一致) ----------
    avail = V3.train_normal_beat_stats()
    audits, domains = {}, {}
    for tag in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = V3.MAIN_RATIO[tag]
        a_cap = avail[tag]["train_norm_available"]
        if n_norm > a_cap:
            raise RuntimeError(f"{tag}: 目标正常拍 {n_norm} 超过训练患者可用量 {a_cap}")
        dom = V3.sample_domain(tag, n_abn, n_norm, rng)
        domains[tag] = dom
        audits[tag] = dom[4]
    x_main = np.concatenate([
        domains["mit_bih"][0], domains["incart"][0], domains["ptb"][0],
        domains["mit_bih"][2], domains["incart"][2], domains["ptb"][2],
    ])[..., np.newaxis]
    y_main = np.concatenate([
        domains["mit_bih"][1], domains["incart"][1], domains["ptb"][1],
        domains["mit_bih"][3], domains["incart"][3], domains["ptb"][3],
    ])

    real = np.concatenate([
        np.load(V3.DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32),
        np.load(V3.DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32),
    ])
    holdout_idx = rng.choice(len(real), V3.N_REAL_HOLDOUT, replace=False)
    train_real_idx = np.setdiff1d(np.arange(len(real)), holdout_idx)
    real_train = real[train_real_idx]
    x_real = np.concatenate([real_train] * V3.REAL_REPEAT)[..., np.newaxis]
    y_real = np.zeros(len(x_real), dtype=np.int32)
    x_real_ho = real[holdout_idx][..., np.newaxis]
    y_real_ho = np.zeros(len(holdout_idx), dtype=np.int32)

    hard = V3.synth_hard(real_train, rng)
    x_hard = hard[..., np.newaxis]
    y_hard = np.zeros(len(hard), dtype=np.int32)

    x_train = np.concatenate([x_main, x_real, x_hard])
    y_train = np.concatenate([y_main, y_real, y_hard])
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[R18B0:{seed}] train total={len(x_train)} "
          f"abn={int((y_train == 1).sum())}", flush=True)

    va, val, vn, vnl = V3.sample_val_domain("mit_bih", V3.VAL_MI_PER_CLASS,
                                            V3.VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = V3.sample_val_domain("incart", V3.VAL_MI_PER_CLASS,
                                             V3.VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = V3.sample_val_domain("ptb", V3.VAL_PTB_PER_CLASS,
                                            V3.VAL_PTB_PER_CLASS, rng)
    x_val = np.concatenate([
        np.concatenate([va, vn, ia, inv, pa, pn])[..., np.newaxis],
        x_real_ho,
    ]).astype(np.float32)
    y_val = np.concatenate([val, vnl, ial, inl, pal, pnl, y_real_ho])

    # ---------- 模型与训练 (与 v3-A 配置 a 完全一致) ----------
    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    print(f"[R18B0:{seed}] params={model.count_params():,}", flush=True)

    steps = max(1, len(x_train) // V3.BATCH)
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=V3.BASE_LR, decay_steps=V3.EPOCHS * steps)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    class ValAucCB(tf.keras.callbacks.Callback):
        def __init__(self, xv, yv):
            super().__init__()
            self.xv, self.yv = xv, yv
            self.best, self.best_epoch = -1.0, -1

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(self.xv, batch_size=256, verbose=0)[:, 1]
            auc = float(roc_auc_score(self.yv, p))
            (logs or {}).__setitem__("val_auc", auc)
            if auc > self.best:
                self.best, self.best_epoch = auc, epoch + 1
                self.model.save(str(out_h5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})",
                      flush=True)

    hist = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        batch_size=V3.BATCH, epochs=V3.EPOCHS,
        callbacks=[ValAucCB(x_val, y_val),
                   tf.keras.callbacks.EarlyStopping(
                       monitor="val_auc", mode="max", patience=V3.PATIENCE,
                       restore_best_weights=True, verbose=1),
                   tf.keras.callbacks.CSVLogger(str(out_csv))],
        class_weight={0: 1.0, 1: 1.0}, verbose=2)

    best = tf.keras.models.load_model(str(out_h5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_real_ho, batch_size=64, verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val))
    print(f"[R18B0:{seed}] BEST val AUC={auc_val:.4f}; real holdout "
          f"frac>0.5={float((p_real > 0.5).mean()):.4f}", flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": f"R18 B0 基线种子扩展: v3-A 配方 seed={seed} (预注册 §1-B0)",
        "provenance": "from scratch; recipe == train_clean_baseline_v3 config a",
        "gpu": [str(g) for g in gpu],
        "seed": seed,
        "output_model": str(out_h5.relative_to(BASE)),
        "data": {
            "main": {t: {"abn": len(domains[t][0]), "norm": len(domains[t][2])}
                     for t in domains},
            "sampling_audit": audits,
            "real_afe_holdout": int(len(x_real_ho)),
            "synthetic_hard_negative": int(len(x_hard)),
            "train_total": int(len(x_train)),
            "train_abn": int((y_train == 1).sum()),
        },
        "config": {"architecture": "build_ecg_resnet_lite_large",
                   "params": int(model.count_params()),
                   "lr": V3.BASE_LR, "epochs_requested": V3.EPOCHS,
                   "epochs_run": int(len(hist.epoch)), "batch": V3.BATCH,
                   "patience": V3.PATIENCE},
        "results": {"best_val_auc": auc_val,
                    "real_holdout_frac_gt_0.5": float((p_real > 0.5).mean()),
                    "train_time_s": round(time.time() - t0, 1)},
        "anchors": {"v3a_seed42_gate1": {"mit_incart_auc": 0.8651,
                                         "mit_incart_evf1": 0.7094,
                                         "ptb_auc": 0.7662, "ptb_evf1": 0.8527}},
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[R18B0:{seed}] saved {out_h5.name}, {out_json.name}", flush=True)
    return result


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [43, 44]
    for s in seeds:
        run_seed(s)
