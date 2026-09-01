#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_exp7c_ecgfounder_v4.py — 仅加入 ECGFounder 筛选的真实相似正常拍 v4 (top50, 更低权重)
================================================================================
与 v1/v2 不同：不加入公共异常弱标签，而是把 ECGFounder 距离筛选出的
“最像真实 AFE”的 PTB-XL 正常记录拍作为额外正常/硬负补充，期望降低真实 AFE
误报且尽量不损伤 MIT/PTB AUC。

输出：
  models/best_resnet_large_exp7c_ecgfounder_v4.h5
  models/train_history_exp7c_ecgfounder_v4.csv
  models/deploy_match/finetune_exp7c_ecgfounder_v4.json
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

OUT_H5 = MODELS / "best_resnet_large_exp7c_ecgfounder_v4.h5"
OUT_CSV = MODELS / "train_history_exp7c_ecgfounder_v4.csv"
OUT_JSON = CACHE / "finetune_exp7c_ecgfounder_v4.json"

SEED = 42
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

REAL_WEIGHT = 2.5
PUBLIC_NORMAL_WEIGHT = 1.0

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
    # 真实 AFE 正常
    real = np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32)
    extra = np.load(DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32)
    real = np.concatenate([real, extra], axis=0)

    # ECGFounder 选出的真实相似公共正常拍
    pub_normal = np.load(EGF_DIR / "real_like_normal_beats_top50.npy").astype(np.float32)

    # 金标准混合
    mit_a, mit_al, mit_n, mit_nl = load_domain("mit_bih", 800, 200)
    inc_a, inc_al, inc_n, inc_nl = load_domain("incart", 200, 100)
    ptb_a, ptb_al, ptb_n, ptb_nl = load_domain("ptb", 300, 100)
    x_mix = np.concatenate([mit_a, inc_a, ptb_a, mit_n, inc_n, ptb_n])[..., np.newaxis]
    y_mix = np.concatenate([mit_al, inc_al, ptb_al, mit_nl, inc_nl, ptb_nl])

    # 验证集：200 A + 200 N + 40 real
    val_abn_idx = rng.choice(np.where(y_mix == 1)[0], 200, replace=False)
    val_norm_idx = rng.choice(np.where(y_mix == 0)[0], 200, replace=False)
    val_idx_mix = np.concatenate([val_abn_idx, val_norm_idx])
    val_real_idx = rng.choice(len(real), 40, replace=False)
    x_val = np.concatenate([x_mix[val_idx_mix], real[val_real_idx][..., np.newaxis]])
    y_val = np.concatenate([y_mix[val_idx_mix], np.zeros(40, dtype=np.int32)])
    train_mask = np.ones(len(x_mix), dtype=bool)
    train_mask[val_idx_mix] = False
    x_mix_train = x_mix[train_mask]
    y_mix_train = y_mix[train_mask]
    real_train_mask = np.ones(len(real), dtype=bool)
    real_train_mask[val_real_idx] = False
    x_real_train = real[real_train_mask][..., np.newaxis]
    y_real_train = np.zeros(len(real_train_mask), dtype=np.int32)

    x_train = np.concatenate([x_mix_train, x_real_train, pub_normal[..., np.newaxis]])
    y_train = np.concatenate([y_mix_train, y_real_train, np.zeros(len(pub_normal), dtype=np.int32)])
    w_train = np.concatenate([
        np.ones(len(x_mix_train), dtype=np.float32),
        np.full(len(x_real_train), REAL_WEIGHT, dtype=np.float32),
        np.full(len(pub_normal), PUBLIC_NORMAL_WEIGHT, dtype=np.float32),
    ])
    perm = rng.permutation(len(x_train))
    x_train, y_train, w_train = x_train[perm], y_train[perm], w_train[perm]
    print(f"[FT-EGFv3] train={len(x_train)} real={len(real)} pub_normal={len(pub_normal)} "
          f"val={len(x_val)} (val_abn={(y_val==1).sum()})")

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

    m0 = tf.keras.models.load_model(str(MODELS / "best_resnet_large_exp7c.h5"), compile=False)
    p0 = m0.predict(real[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
    print(f"[FT-EGFv3] BEFORE real mean={p0.mean():.4f} frac>0.5={(p0>0.5).mean():.4f}")

    cbs = [ValAucCB(x_val, y_val),
           tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                            patience=12, restore_best_weights=True,
                                            verbose=1),
           tf.keras.callbacks.CSVLogger(str(OUT_CSV))]
    t0 = time.time()
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                     batch_size=32, epochs=40, callbacks=cbs,
                     sample_weight=w_train, verbose=2)
    print(f"[FT-EGFv3] done in {time.time()-t0:.0f}s", flush=True)

    model.load_weights(str(OUT_H5))
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p1 = model.predict(real[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
    p1_pub = model.predict(pub_normal[..., np.newaxis], batch_size=64, verbose=0)[:, 1]
    print(f"[FT-EGFv3] AFTER real mean={p1.mean():.4f} frac>0.5={(p1>0.5).mean():.4f}")
    print(f"[FT-EGFv3] pub_normal mean={p1_pub.mean():.4f} frac>0.5={(p1_pub>0.5).mean():.4f}")

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "exp7c + ECGFounder real-like public normal only",
        "data": {
            "real_normal_beats": int(len(real)),
            "real_like_public_normal_beats": int(len(pub_normal)),
            "gold_mix": int(len(x_mix)),
            "val_beats": int(len(x_val)),
        },
        "config": {
            "base": "best_resnet_large_exp7c.h5",
            "freeze": "backbone except fc1/out",
            "lr": 1e-5,
            "optimizer": "adam",
            "sample_weight": {
                "real_normal": REAL_WEIGHT,
                "public_normal": PUBLIC_NORMAL_WEIGHT,
            },
            "epochs": int(len(hist.epoch)),
        },
        "confidence_real_normal": {
            "before_exp7c": {"mean": float(p0.mean()), "frac_gt_0.5": float((p0 > 0.5).mean())},
            "after_all": {"mean": float(p1.mean()), "frac_gt_0.5": float((p1 > 0.5).mean())},
            "after_public_normal": {"mean": float(p1_pub.mean()), "frac_gt_0.5": float((p1_pub > 0.5).mean())},
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[FT-EGFv3] saved {OUT_H5.name} + {OUT_JSON.name}")

if __name__ == "__main__":
    main()
