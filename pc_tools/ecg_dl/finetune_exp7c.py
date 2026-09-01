#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""finetune_exp7c.py — exp7b 真实数据域微调 (TH §40 B 方案)
=================================================================
策略: 加载 exp7b 权重 → 冻结骨干 (仅 fc1/out 可训练) → lr=1e-5 (Adam)
      混合数据: 210 真实正常拍 + 2000 原始训练异常拍 (防遗忘) + 600 原始正常拍
      损失: CE + class_weight {0:4, 1:1} (推低真实正常拍置信度)
产出: models/best_resnet_large_exp7c.h5 + train_history_exp7c.csv
      + models/deploy_match/finetune_exp7c.json (微调前后置信度对比)
"""
import sys, json, os, time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGET_FS, BEAT_WINDOW_SAMPLES
from models.resnet_lite_1d import build_ecg_resnet_lite_large

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
CACHE = MODELS / "deploy_match"
ECG_DATA = Path(os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data"))

OUT_H5 = MODELS / "best_resnet_large_exp7c.h5"
OUT_CSV = MODELS / "train_history_exp7c.csv"
OUT_JSON = CACHE / "finetune_exp7c.json"

SEED = 42
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

# ---------- 1. 数据 ----------
real = np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32)
print(f"[FT] real normal beats: {real.shape}")

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
print(f"[FT] mix: MIT abn={len(mit_a)} norm={len(mit_n)} | INCART abn={len(inc_a)} norm={len(inc_n)} | PTB abn={len(ptb_a)} norm={len(ptb_n)}")

x_mix = np.concatenate([mit_a, inc_a, ptb_a, mit_n, inc_n, ptb_n])[..., np.newaxis]
y_mix = np.concatenate([mit_al, inc_al, ptb_al, mit_nl, inc_nl, ptb_nl])
perm = rng.permutation(len(x_mix)); x_mix, y_mix = x_mix[perm], y_mix[perm]
print(f"[FT] mix total: {len(x_mix)} (abn={int((y_mix==1).sum())}, norm={int((y_mix==0).sum())})")

# 真实拍划分: 30 验证 / 其余训练
val_idx = rng.choice(len(real), 30, replace=False)
trn_idx = np.setdiff1d(np.arange(len(real)), val_idx)
x_rtr, y_rtr = real[trn_idx][..., np.newaxis], np.zeros(len(trn_idx), dtype=np.int32)
x_rva, y_rva = real[val_idx][..., np.newaxis], np.zeros(len(val_idx), dtype=np.int32)

x_train = np.concatenate([x_mix, x_rtr]); y_train = np.concatenate([y_mix, y_rtr])
x_val = np.concatenate([x_mix[-400:], x_rva]); y_val = np.concatenate([y_mix[-400:], y_rva])
print(f"[FT] train={len(x_train)} val={len(x_val)} (val 含 {len(val_idx)} 真实拍)")

# ---------- 2. 模型: exp7b 权重 + 冻结骨干 ----------
model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
model.load_weights(str(MODELS / "best_resnet_large_exp7b.h5"))
for layer in model.layers:
    if layer.name in ("fc1", "out"):
        layer.trainable = True
    else:
        layer.trainable = False
nt = sum(1 for l in model.layers if l.trainable)
print(f"[FT] trainable layers: {nt} (fc1/out); backbone frozen")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# 手写 AUC (避免 Keras AUC metric + class_weight 的 XLA 兼容 bug)
def _auc(y_true, p):
    y = np.asarray(y_true).ravel(); p = np.asarray(p).ravel()
    order = np.argsort(p); y = y[order]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)[y == 1].sum()
    return float((ranks - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class ValAucCB(tf.keras.callbacks.Callback):
    def __init__(self, xv, yv):
        super().__init__()
        self.xv, self.yv = xv, yv
        self.best = 0.0
        self.epoch_aucs = []

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.xv, batch_size=128, verbose=0)[:, 1]
        auc = _auc(self.yv, p)
        self.epoch_aucs.append(auc)
        logs = logs or {}
        logs["val_auc"] = auc
        if auc > self.best:
            self.best = auc
            self.model.save(str(OUT_H5))
            print(f"  * saved best (val_auc={auc:.4f})")

# ---------- 3. 微调前 exp7b 基线置信度 ----------
m0 = tf.keras.models.load_model(str(MODELS / "best_resnet_large_exp7b.h5"), compile=False)
p0 = m0.predict(real[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
print(f"[FT] BEFORE (exp7b on real): mean={p0.mean():.4f} median={np.median(p0):.4f} frac>0.5={float((p0>0.5).mean()):.4f}")

# ---------- 4. 训练 ----------
cbs = [
    ValAucCB(x_val, y_val),
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.CSVLogger(str(OUT_CSV)),
]
t0 = time.time()
hist = model.fit(
    x_train, y_train, validation_data=(x_val, y_val),
    batch_size=32, epochs=60, callbacks=cbs,
    class_weight={0: 4.0, 1: 1.0}, verbose=2,
)
print(f"[FT] training done in {time.time()-t0:.0f}s")

# ---------- 5. 微调后置信度 ----------
model.load_weights(str(OUT_H5))
tf.keras.backend.clear_session()
model = tf.keras.models.load_model(str(OUT_H5), compile=False)
p1 = model.predict(real[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
p1_tr = model.predict(x_rtr, batch_size=64, verbose=0)[:, 1]
p1_va = model.predict(x_rva, batch_size=64, verbose=0)[:, 1]
print(f"[FT] AFTER (exp7c on all real): mean={p1.mean():.4f} median={np.median(p1):.4f} frac>0.5={float((p1>0.5).mean()):.4f}")
print(f"[FT] AFTER trained-part: mean={p1_tr.mean():.4f} | held-out-30: mean={p1_va.mean():.4f} frac>0.5={float((p1_va>0.5).mean()):.4f}")

result = {
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "purpose": "exp7b 真实数据域微调 (TH §40 B) → exp7c",
    "data": {
        "real_normal_beats": int(len(real)), "mix_abn": int((y_mix==1).sum()),
        "mix_norm": int((y_mix==0).sum()), "held_out_real": int(len(val_idx)),
    },
    "config": {"freeze": "backbone except fc1/out", "lr": 1e-5, "optimizer": "adam",
               "loss": "sparse CE", "class_weight": {0: 4.0, 1: 1.0},
               "epochs": int(len(hist.epoch))},
    "confidence_real_normal": {
        "before_exp7b": {"mean": float(p0.mean()), "median": float(np.median(p0)),
                         "frac_gt_0.5": float((p0 > 0.5).mean()),
                         "frac_gt_0.8": float((p0 > 0.8).mean())},
        "after_exp7c_all": {"mean": float(p1.mean()), "median": float(np.median(p1)),
                            "frac_gt_0.5": float((p1 > 0.5).mean()),
                            "frac_gt_0.8": float((p1 > 0.8).mean())},
        "after_exp7c_heldout30": {"mean": float(p1_va.mean()),
                                  "frac_gt_0.5": float((p1_va > 0.5).mean())},
    },
}
CACHE.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"[FT] saved {OUT_H5.name} + {OUT_JSON.name}")
