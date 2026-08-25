#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""finetune_exp7c_v4.py — exp7c_v4 多域平衡后训练（公共库保持 + 真实 AFE 正常抑制）
================================================================================
目标：
  从 exp7c 基线开始，仅解冻 fc1/out，使用患者级 train 患者数据混合微调：
    - MIT/INCART/PTB 因果部署链金标准（保持公共库事件级能力）
    - 真实 AFE 正常拍（仅训练划入的约 271 拍）
    - 合成硬负样本（仅由真实 AFE 训练拍生成，约 300 个）
    - exp7c INT8 当前误报最高的公共库正常拍（MIT+INCART train 患者，约 150 个）
  验证集包含公共库患者级 val + 真实 AFE 留出 40 拍，按 val_auc 保存 best。

输出：
  models/best_resnet_large_exp7c_v4.h5
  models/train_history_exp7c_v4.csv
  models/deploy_match/finetune_exp7c_v4.json
"""
import sys, json, time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from models.resnet_lite_1d import build_ecg_resnet_lite_large
from data.dataset import set_npz_suffix, load_mit_incart_merged
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map, build_ptb_patient_map,
    patient_level_split,
)
from eval_exp7c_policy_sweep import reduce_mit_augmentation

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
CACHE = MODELS / "deploy_match"
ECG_DATA = Path(__import__("os").environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data"))

OUT_H5 = MODELS / "best_resnet_large_exp7c_v4.h5"
OUT_CSV = MODELS / "train_history_exp7c_v4.csv"
OUT_JSON = CACHE / "finetune_exp7c_v4.json"

SEED = 42
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

# 数据配比（可按需微调）
MAIN_RATIO = {
    "mit_bih": (1500, 500),
    "incart": (400, 150),
    "ptb": (600, 200),
}
N_SYNTH = 600
N_PUBLIC_HARD = 200
N_REAL_HOLDOUT = 40
CLASS_WEIGHT_NORMAL = 2.0
EPOCHS = 40
PATIENCE = 10
REAL_REPEAT = 2


def load_arrays(tag):
    return (
        np.load(ECG_DATA / f"{tag}_processed_deploy_causal_beats.npy", mmap_mode="r"),
        np.load(ECG_DATA / f"{tag}_processed_deploy_causal_labels.npy", mmap_mode="r"),
        np.load(ECG_DATA / f"{tag}_processed_deploy_causal_record_ids.npy", mmap_mode="r"),
    )


def add_channel(x):
    return x[..., np.newaxis]


def sample_domain(tag, n_abn, n_norm, train_mask):
    b, l, r = load_arrays(tag)
    mask = train_mask[:len(l)]
    ia = np.where(mask & (l == 1))[0]
    inn = np.where(mask & (l == 0))[0]
    sa = rng.choice(ia, min(n_abn, len(ia)), replace=False)
    sn = rng.choice(inn, min(n_norm, len(inn)), replace=False)
    return (
        np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
        np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32),
    )


def get_mit_incart_masks():
    mit_b, mit_l, mit_r = load_arrays("mit_bih")
    inc_b, inc_l, inc_r = load_arrays("incart")
    rids = np.concatenate([mit_r, inc_r + 100000])
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    return patient_level_split(rids, pmap, seed=SEED)


def synth_hard(real_train):
    """对真实 AFE 训练拍做少量多形态合成，共 N_SYNTH 个正常硬负样本。"""
    n_source = min(100, len(real_train))
    idx = rng.choice(len(real_train), n_source, replace=False)
    base = real_train[idx]
    outs = []
    t = np.arange(BEAT_WINDOW_SAMPLES, dtype=np.float32) / 250.0

    # 1) 高斯噪声 20dB
    outs.append(base + rng.normal(0, 10 ** (-20 / 20.0), base.shape).astype(np.float32))
    # 2) 高斯噪声 10dB
    outs.append(base + rng.normal(0, 10 ** (-10 / 20.0), base.shape).astype(np.float32))
    # 3) 基线漂移 0.3Hz
    outs.append(base + (0.3 * np.sin(2 * np.pi * 0.3 * t)).astype(np.float32)[None, :])
    # 4) 基线漂移 0.8Hz
    outs.append(base + (0.5 * np.sin(2 * np.pi * 0.8 * t)).astype(np.float32)[None, :])
    # 5) 随机电极脉冲
    y = base.copy()
    for _ in range(n_source):
        for _ in range(5):
            pos = rng.integers(0, BEAT_WINDOW_SAMPLES)
            amp = float(rng.uniform(0.5, 2.0) * rng.choice([-1, 1]))
            y[_, pos] += amp
    outs.append(y.astype(np.float32))
    # 6) 局部 dropout
    y2 = base.copy()
    mask = rng.random(y2.shape) < 0.2
    y2 = y2 * (1 - mask)
    outs.append(y2.astype(np.float32))

    return np.concatenate(outs, axis=0)


def select_public_hard_normal(n_select=150):
    """用 exp7c INT8 全量因果链概率，在 MIT+INCART train 正常拍中选高分硬负样本。"""
    set_npz_suffix("_deploy_causal")
    data = load_mit_incart_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    beats, labels, rids, kept_idx = reduce_mit_augmentation(beats, labels, rids)
    tr_m, _, _, _ = get_mit_incart_masks()
    tr_red = tr_m[kept_idx]
    probs = np.load(CACHE / "exp7c_causal_probs_full.npy")
    if len(probs) != len(labels):
        raise RuntimeError(f"exp7c cache mismatch: {len(probs)} vs {len(labels)}")
    norm_train = np.where(tr_red & (labels == 0))[0]
    scores = probs[norm_train]
    order = np.argsort(scores)[::-1]
    selected = norm_train[order[:n_select]]
    print(f"[FTv4] public hard normal: selected {len(selected)} from {len(norm_train)} train normal beats, "
          f"top score={scores[order[0]]:.4f}", flush=True)
    return np.asarray(beats[selected], dtype=np.float32)


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---------- 患者级划分 ----------
    tr_m, va_m, te_m, mi_stats = get_mit_incart_masks()
    ptb_b, ptb_l, ptb_r = load_arrays("ptb")
    ptr, pva, pte, ptb_stats = patient_level_split(ptb_r, build_ptb_patient_map(), seed=SEED)
    print(f"[FTv4] MIT+INCART patients: tr={mi_stats['n_train']} va={mi_stats['n_val']} te={mi_stats['n_test']}", flush=True)
    print(f"[FTv4] PTB patients: tr={ptb_stats['n_train']} va={ptb_stats['n_val']} te={ptb_stats['n_test']}", flush=True)

    # ---------- 主数据（train 患者） ----------
    mit_a, mit_al, mit_n, mit_nl = sample_domain("mit_bih", *MAIN_RATIO["mit_bih"], tr_m)
    inc_a, inc_al, inc_n, inc_nl = sample_domain("incart", *MAIN_RATIO["incart"], tr_m)
    ptb_a, ptb_al, ptb_n, ptb_nl = sample_domain("ptb", *MAIN_RATIO["ptb"], ptr)
    x_main = np.concatenate([mit_a, inc_a, ptb_a, mit_n, inc_n, ptb_n])[..., np.newaxis]
    y_main = np.concatenate([mit_al, inc_al, ptb_al, mit_nl, inc_nl, ptb_nl])
    print(f"[FTv4] main gold: {len(x_main)} (abn={int((y_main==1).sum())}, norm={int((y_main==0).sum())})", flush=True)

    # ---------- 真实 AFE 正常拍：训练/留出 ----------
    real = np.concatenate([
        np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32),
        np.load(DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32),
    ])
    split_rng = np.random.default_rng(SEED)
    holdout_idx = split_rng.choice(len(real), N_REAL_HOLDOUT, replace=False)
    train_real_idx = np.setdiff1d(np.arange(len(real)), holdout_idx)
    real_train_feats = real[train_real_idx]
    x_real_train = np.concatenate([real_train_feats] * REAL_REPEAT)[..., np.newaxis]
    y_real_train = np.zeros(len(x_real_train), dtype=np.int32)
    x_real_holdout = real[holdout_idx][..., np.newaxis]
    y_real_holdout = np.zeros(len(holdout_idx), dtype=np.int32)
    print(f"[FTv4] real AFE: train_unique={len(train_real_idx)}, train_after_repeat={len(x_real_train)}, holdout={len(x_real_holdout)}", flush=True)

    # ---------- 合成硬负样本（仅源自真实 AFE 训练拍） ----------
    hard = synth_hard(real_train_feats)
    x_hard = hard[..., np.newaxis]
    y_hard = np.zeros(len(hard), dtype=np.int32)
    print(f"[FTv4] synthetic hard neg: {len(hard)}", flush=True)

    # ---------- 公共库 exp7c 当前误报最高的 normal 硬负样本 ----------
    pub_hard = select_public_hard_normal(N_PUBLIC_HARD)
    x_pub = pub_hard[..., np.newaxis]
    y_pub = np.zeros(len(pub_hard), dtype=np.int32)
    print(f"[FTv4] public hard normal: {len(x_pub)}", flush=True)

    # ---------- 训练集 ----------
    x_train = np.concatenate([x_main, x_real_train, x_hard, x_pub])
    y_train = np.concatenate([y_main, y_real_train, y_hard, y_pub])
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[FTv4] train total={len(x_train)} abn={int((y_train==1).sum())} norm={int((y_train==0).sum())}", flush=True)

    # ---------- 验证集：公共库 val 患者 + 真实 AFE 留出 ----------
    set_npz_suffix("_deploy_causal")
    data = load_mit_incart_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    beats, labels, rids, kept_idx = reduce_mit_augmentation(beats, labels, rids)
    va_red = va_m[kept_idx]
    # MIT+INCART val 采样
    va_abn = np.where(va_red & (labels == 1))[0]
    va_norm = np.where(va_red & (labels == 0))[0]
    sa = rng.choice(va_abn, min(800, len(va_abn)), replace=False)
    sn = rng.choice(va_norm, min(800, len(va_norm)), replace=False)
    x_vmi = np.concatenate([beats[sa], beats[sn]])[..., np.newaxis].astype(np.float32)
    y_vmi = np.concatenate([np.ones(len(sa), dtype=np.int32), np.zeros(len(sn), dtype=np.int32)])
    # PTB val 采样
    pva_abn = np.where(pva & (ptb_l == 1))[0]
    pva_norm = np.where(pva & (ptb_l == 0))[0]
    spa = rng.choice(pva_abn, min(400, len(pva_abn)), replace=False)
    spn = rng.choice(pva_norm, min(400, len(pva_norm)), replace=False)
    x_vptb = np.concatenate([ptb_b[spa], ptb_b[spn]])[..., np.newaxis].astype(np.float32)
    y_vptb = np.concatenate([np.ones(len(spa), dtype=np.int32), np.zeros(len(spn), dtype=np.int32)])
    x_val = np.concatenate([x_vmi, x_vptb, x_real_holdout])
    y_val = np.concatenate([y_vmi, y_vptb, y_real_holdout])
    print(f"[FTv4] val total={len(x_val)} abn={int((y_val==1).sum())} norm={int((y_val==0).sum())} "
          f"(incl real holdout {len(holdout_idx)})", flush=True)

    # ---------- 模型 ----------
    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    model.load_weights(str(MODELS / "best_resnet_large_exp7c.h5"))
    for layer in model.layers:
        layer.trainable = layer.name in ("fc1", "out")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # 微调前基线
    base_model = tf.keras.models.load_model(str(MODELS / "best_resnet_large_exp7c.h5"), compile=False)
    p0_real = base_model.predict(x_real_holdout, batch_size=64, verbose=0)[:, 1]
    p0_val = base_model.predict(x_val, batch_size=128, verbose=0)[:, 1]
    auc0_val = float(roc_auc_score(y_val, p0_val)) if len(np.unique(y_val)) > 1 else float("nan")
    print(f"[FTv4] BEFORE exp7c: real holdout mean={p0_real.mean():.4f} "
          f"frac>0.5={float((p0_real>0.5).mean()):.4f}; val AUC={auc0_val:.4f}", flush=True)
    tf.keras.backend.clear_session()

    class ValAucCB(tf.keras.callbacks.Callback):
        def __init__(self, xv, yv):
            super().__init__()
            self.xv, self.yv = xv, yv
            self.best = -1.0
            self.best_epoch = -1
        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(self.xv, batch_size=128, verbose=0)[:, 1]
            auc = float(roc_auc_score(self.yv, p))
            logs = logs or {}
            logs["val_auc"] = auc
            if auc > self.best:
                self.best = auc
                self.best_epoch = epoch + 1
                self.model.save(str(OUT_H5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch+1})", flush=True)

    cbs = [
        ValAucCB(x_val, y_val),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(OUT_CSV)),
    ]

    print("[FTv4] start training...", flush=True)
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=32,
        epochs=EPOCHS,
        callbacks=cbs,
        class_weight={0: CLASS_WEIGHT_NORMAL, 1: 1.0},
        verbose=2,
    )
    print(f"[FTv4] training done in {time.time()-t0:.0f}s", flush=True)

    # ---------- 保存最佳模型评估 ----------
    model.load_weights(str(OUT_H5))
    tf.keras.backend.clear_session()
    best = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p1_real = best.predict(x_real_holdout, batch_size=64, verbose=0)[:, 1]
    p1_val = best.predict(x_val, batch_size=128, verbose=0)[:, 1]
    p1_train = best.predict(x_train, batch_size=128, verbose=0)[:, 1]
    auc1_val = float(roc_auc_score(y_val, p1_val))
    print(f"[FTv4] AFTER best: real holdout mean={p1_real.mean():.4f} "
          f"frac>0.5={float((p1_real>0.5).mean()):.4f} frac>0.75={float((p1_real>0.75).mean()):.4f}; "
          f"val AUC={auc1_val:.4f}", flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "exp7c_v4 multi-domain balanced finetune with real AFE normal suppression",
        "base_model": "best_resnet_large_exp7c.h5",
        "output_model": str(OUT_H5.relative_to(BASE)),
        "patient_split": {
            "mit_incart": {k: mi_stats[k] for k in ("n_patients", "n_train", "n_val", "n_test")},
            "ptb": {k: ptb_stats[k] for k in ("n_patients", "n_train", "n_val", "n_test")},
        },
        "data": {
            "main": {
                "mit_abn": len(mit_a), "mit_norm": len(mit_n),
                "incart_abn": len(inc_a), "incart_norm": len(inc_n),
                "ptb_abn": len(ptb_a), "ptb_norm": len(ptb_n),
            },
            "real_afe_train_unique": len(train_real_idx), "real_afe_train_repeated": len(x_real_train),
            "real_afe_holdout": len(x_real_holdout),
            "synthetic_hard_negative": len(x_hard),
            "public_hard_normal": len(x_pub),
            "train_total": len(x_train),
            "train_abn": int((y_train == 1).sum()),
            "train_norm": int((y_train == 0).sum()),
            "val_total": len(x_val),
            "val_abn": int((y_val == 1).sum()),
            "val_norm": int((y_val == 0).sum()),
            "real_holdout_indices": holdout_idx.tolist(),
        },
        "config": {
            "freeze": "all except fc1/out",
            "optimizer": "adam",
            "lr": 1e-5,
            "loss": "sparse_categorical_crossentropy",
            "class_weight": {0: CLASS_WEIGHT_NORMAL, 1: 1.0},
            "epochs_requested": EPOCHS,
            "epochs_run": int(len(hist.epoch)),
            "early_stopping_patience": PATIENCE,
            "val_monitor": "val_auc",
        },
        "before": {
            "real_holdout_mean": float(p0_real.mean()),
            "real_holdout_frac_gt_0.5": float((p0_real > 0.5).mean()),
            "val_auc": auc0_val if np.isfinite(auc0_val) else None,
        },
        "after": {
            "real_holdout_mean": float(p1_real.mean()),
            "real_holdout_frac_gt_0.5": float((p1_real > 0.5).mean()),
            "real_holdout_frac_gt_0.75": float((p1_real > 0.75).mean()),
            "val_auc": auc1_val,
            "train_acc": float(hist.history.get("accuracy", [None])[-1]) if hist.history.get("accuracy") else None,
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[FTv4] saved {OUT_H5.name}, {OUT_CSV.name}, {OUT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
