#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""finetune_exp7c_hardneg.py — exp7c_v2 真实 AFE 数据 + 合成硬负样本后训练
================================================================================
在 exp7b 基础上继续做后训练：
  - 真实 AFE 正常拍：ecg_real_052 (210) + rec_latest (101) = 311 拍
  - 合成“硬负样本”：对真实正常拍加高斯噪声、基线漂移、随机脉冲/脱落，仍标为正常，
    迫使模型对真实设备常见伪影输出低异常分
  - 混合因果部署链训练数据（MIT/INCART/PTB），防止灾难性遗忘
  - 冻结骨干，仅训练 fc1/out，Adam lr=1e-5，class_weight 正常类加权

产出：
  models/best_resnet_large_exp7c_v2.h5
  models/train_history_exp7c_v2.csv
  models/deploy_match/finetune_exp7c_v2.json
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

OUT_H5 = MODELS / "best_resnet_large_exp7c_v2.h5"
OUT_CSV = MODELS / "train_history_exp7c_v2.csv"
OUT_JSON = CACHE / "finetune_exp7c_v2.json"

SEED = 42
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

# ---------- 1. 真实 AFE 正常拍 ----------
real = np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32)
extra = np.load(DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32)
real = np.concatenate([real, extra], axis=0)
print(f"[FTv2] real AFE normal beats: {real.shape[0]} (old {real.shape[0]-len(extra)} + extra {len(extra)})")

# ---------- 2. 合成硬负样本（仍标 0） ----------
def synth_hard(x):
    n = len(x)
    out = []
    # Gaussian noise
    for db in (10, 20):
        sigma = 10 ** (-db / 20.0) * 1.0
        out.append(x + rng.normal(0, sigma, x.shape).astype(np.float32))
    # baseline wander: 0.3-1.0 Hz sine + slow drift
    t = np.arange(BEAT_WINDOW_SAMPLES, dtype=np.float32) / 250.0
    for freq, amp in ((0.3, 0.3), (0.8, 0.5), (1.2, 0.8)):
        out.append(x + (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :])
    # random impulses (electrode contact noise)
    for n_imp in (2, 5, 10):
        y = x.copy()
        for _ in range(n_imp):
            pos = rng.integers(0, BEAT_WINDOW_SAMPLES)
            amp = float(rng.uniform(0.5, 2.0) * rng.choice([-1, 1]))
            y[:, pos] += amp
        out.append(y.astype(np.float32))
    # dropout / flat segments
    for frac in (0.1, 0.25):
        y = x.copy()
        mask = rng.random((n, BEAT_WINDOW_SAMPLES)) < frac
        y = y * (1 - mask)
        out.append(y.astype(np.float32))
    arr = np.concatenate(out, axis=0)
    return arr

hard = synth_hard(real)
print(f"[FTv2] synthetic hard negatives: {hard.shape[0]}")

# ---------- 3. 混合因果部署链训练数据 ----------
def load_domain(tag, n_abn, n_norm):
    b = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r")
    l = np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r")
    ia = np.where(l == 1)[0]; inn = np.where(l == 0)[0]
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

mit_a, mit_al, mit_n, mit_nl = load_domain("mit_bih", 1200, 400)
inc_a, inc_al, inc_n, inc_nl = load_domain("incart", 300, 100)
ptb_a, ptb_al, ptb_n, ptb_nl = load_domain("ptb", 500, 100)
print(f"[FTv2] mix: MIT abn={len(mit_a)} norm={len(mit_n)} | INCART abn={len(inc_a)} norm={len(inc_n)} | PTB abn={len(ptb_a)} norm={len(ptb_n)}")

x_mix = np.concatenate([mit_a, inc_a, ptb_a, mit_n, inc_n, ptb_n])[..., np.newaxis]
y_mix = np.concatenate([mit_al, inc_al, ptb_al, mit_nl, inc_nl, ptb_nl])
perm = rng.permutation(len(x_mix)); x_mix, y_mix = x_mix[perm], y_mix[perm]

# 真实拍划分：40 验证
val_idx = rng.choice(len(real), 40, replace=False)
trn_idx = np.setdiff1d(np.arange(len(real)), val_idx)
x_rtr, y_rtr = real[trn_idx][..., np.newaxis], np.zeros(len(trn_idx), dtype=np.int32)
x_rva, y_rva = real[val_idx][..., np.newaxis], np.zeros(len(val_idx), dtype=np.int32)
x_hard, y_hard = hard[..., np.newaxis], np.zeros(len(hard), dtype=np.int32)

x_train = np.concatenate([x_mix, x_rtr, x_hard])
y_train = np.concatenate([y_mix, y_rtr, y_hard])
x_val = np.concatenate([x_mix[-400:], x_rva])
y_val = np.concatenate([y_mix[-400:], y_rva])
print(f"[FTv2] train={len(x_train)} val={len(x_val)}")

# ---------- 4. 模型: exp7b + 冻结骨干 ----------
model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
model.load_weights(str(MODELS / "best_resnet_large_exp7b.h5"))
for layer in model.layers:
    layer.trainable = layer.name in ("fc1", "out")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

def _auc(y_true, p):
    y = np.asarray(y_true).ravel(); p = np.asarray(p).ravel()
    order = np.argsort(p); y = y[order]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return 0.5
    ranks = np.arange(1, len(y) + 1)[y == 1].sum()
    return float((ranks - n1 * (n1 + 1) / 2.0) / (n1 * n0))

class ValAucCB(tf.keras.callbacks.Callback):
    def __init__(self, xv, yv):
        super().__init__(); self.xv, self.yv = xv, yv; self.best = 0.0
    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.xv, batch_size=128, verbose=0)[:, 1]
        auc = _auc(self.yv, p)
        logs = logs or {}; logs["val_auc"] = auc
        if auc > self.best:
            self.best = auc
            self.model.save(str(OUT_H5))
            print(f"  * saved best (val_auc={auc:.4f})")

# 微调前基线
m0 = tf.keras.models.load_model(str(MODELS / "best_resnet_large_exp7b.h5"), compile=False)
p0 = m0.predict(real[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
print(f"[FTv2] BEFORE (exp7b on real): mean={p0.mean():.4f} frac>0.5={float((p0>0.5).mean()):.4f}")

cbs = [ValAucCB(x_val, y_val),
       tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15,
                                        restore_best_weights=True, verbose=1),
       tf.keras.callbacks.CSVLogger(str(OUT_CSV))]
t0 = time.time()
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                 batch_size=32, epochs=60, callbacks=cbs,
                 class_weight={0: 4.0, 1: 1.0}, verbose=2)
print(f"[FTv2] training done in {time.time()-t0:.0f}s")

model.load_weights(str(OUT_H5))
tf.keras.backend.clear_session()
model = tf.keras.models.load_model(str(OUT_H5), compile=False)
p1 = model.predict(real[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
p1_hard = model.predict(hard[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
p1_va = model.predict(x_rva, batch_size=64, verbose=0)[:, 1]
print(f"[FTv2] AFTER on real: mean={p1.mean():.4f} frac>0.5={float((p1>0.5).mean()):.4f}")
print(f"[FTv2] AFTER on synthetic hard: mean={p1_hard.mean():.4f} frac>0.5={float((p1_hard>0.5).mean()):.4f}")
print(f"[FTv2] heldout real: mean={p1_va.mean():.4f}")

result = {
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "purpose": "exp7c_v2 后训练: real AFE normal + synthetic hard negatives",
    "data": {
        "real_normal_beats": int(len(real)),
        "synthetic_hard_negatives": int(len(hard)),
        "mix_abn": int((y_mix == 1).sum()),
        "mix_norm": int((y_mix == 0).sum()),
        "held_out_real": int(len(val_idx)),
    },
    "config": {"freeze": "backbone except fc1/out", "lr": 1e-5, "optimizer": "adam",
               "loss": "sparse CE", "class_weight": {0: 4.0, 1: 1.0},
               "epochs": int(len(hist.epoch))},
    "confidence_real_normal": {
        "before_exp7b": {"mean": float(p0.mean()), "frac_gt_0.5": float((p0 > 0.5).mean())},
        "after_v2_all": {"mean": float(p1.mean()), "frac_gt_0.5": float((p1 > 0.5).mean())},
        "after_v2_heldout40": {"mean": float(p1_va.mean())},
        "after_v2_synth_hard": {"mean": float(p1_hard.mean()),
                                "frac_gt_0.5": float((p1_hard > 0.5).mean())},
    },
}
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"[FTv2] saved {OUT_H5.name} + {OUT_JSON.name}")
