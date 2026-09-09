#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_clean_baseline_v3.py — 配比再平衡从零重训 (TH §100)
================================================================================
背景 (§99): v2 干净基线在 θ=0.5 下过度报警 (拍级精度 0.164, 4.5 误报/记录),
根因诊断为域构成捷径——公共库训练拍 75% 异常 (2500 abn vs 850 norm)。
本章把公共库正常拍提到近似平衡, 分离"配比"与"权重"贡献。

铁律 (§97/§98/§99 固化):
  1. 一切公共库取数经 data/split_guard.py (assert_train_only / sample_train_beats),
     seed=42, 患者级划分; 禁止 replace=True 过采样公共库拍 (守卫内部已固定
     replace=False, 本脚本不做任何二次重复)。
  2. 随机初始化从零训练; 不加载任何现有 checkpoint/.h5/权重。
  3. 全部数字落盘: 训练历史 CSV + 实验记录 JSON (含各域可用正常拍统计与
     抽样记录 × 测试记录的交集自检)。

配置 (全部: cosine decay from 3e-4, batch 32, epochs 80, patience 20,
按 val AUC 存档; 各配置独立新建 rng(42) → 抽样完全一致, 只有正则/权重不同):
  A (主): MIT (1500,1500) / INCART (400,400) / PTB (600,400) (abn,norm),
          class_weight {0:1.0, 1:1.0}; large 架构 (dropout 0.4)。
  B (消融): A + 更强正则 (head dropout 0.4→0.5, 经 build_ecg_resnet_lite 传参,
          架构/参数量不变)。
  C (归因): A 的配比 + class_weight {0:2.0, 1:1.0}, 分离配比与权重贡献。
  共同: 真实 AFE 训练拍 ~271×2 + 合成硬负例 ~600 (与 §99 同源)。

输出 (suffix: a→v3, b→v3_b, c→v3_c):
  models/best_resnet_large_clean_baseline_<suffix>.h5
  models/train_history_clean_baseline_<suffix>.csv
  models/deploy_match/train_clean_baseline_<suffix>.json
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
from models.resnet_lite_1d import build_ecg_resnet_lite, build_ecg_resnet_lite_large
from data.split_guard import get_guard, load_arrays

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
CACHE = MODELS / "deploy_match"

SEED = 42
# 配比再平衡: 公共库正常拍提到近似平衡 (以 SplitGuard 实测可用量为准)
MAIN_RATIO = {
    "mit_bih": (1500, 1500),
    "incart": (400, 400),
    "ptb": (600, 400),
}
N_REAL_HOLDOUT = 40
REAL_REPEAT = 2
EPOCHS = 80
PATIENCE = 20
BASE_LR = 3e-4
BATCH = 32
VAL_MI_PER_CLASS = 800
VAL_PTB_PER_CLASS = 400

CONFIGS = {
    "a": {"suffix": "v3", "class_weight": {0: 1.0, 1: 1.0}, "dropout": 0.4,
          "purpose": "主配置: 公共库正常拍提到近似平衡 + class_weight 1:1"},
    "b": {"suffix": "v3_b", "class_weight": {0: 1.0, 1: 1.0}, "dropout": 0.5,
          "purpose": "消融: A + 更强正则 (head dropout 0.4→0.5, 架构/参数量不变)"},
    "c": {"suffix": "v3_c", "class_weight": {0: 2.0, 1: 1.0}, "dropout": 0.4,
          "purpose": "归因: A 配比 + class_weight {0:2.0}, 分离配比与权重贡献"},
}


def build_model(dropout: float):
    """large 预设架构; dropout 0.5 时经 build_ecg_resnet_lite 显式传参
    (参数量与 dropout 无关, 架构完全一致)。"""
    if dropout == 0.4:
        return build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    return build_ecg_resnet_lite(
        input_shape=(BEAT_WINDOW_SAMPLES, 1),
        filters=(16, 32, 64, 128),
        blocks_per_stage=(2, 3, 3, 1),
        kernel_sizes=(7, 5, 3, 3),
        strides=(1, 2, 2, 1),
        dropout_rate=dropout,
    )


def train_normal_beat_stats():
    """各域训练患者可用心拍统计 (SplitGuard 掩码实测, 写进 JSON)。"""
    out = {}
    for tag in ("mit_bih", "incart", "ptb"):
        g = get_guard(tag)
        _b, l, _r = load_arrays(tag)
        l = np.asarray(l)
        m = g.train_mask[:len(l)]
        out[tag] = {
            "train_abn_available": int((m & (l == 1)).sum()),
            "train_norm_available": int((m & (l == 0)).sum()),
            "val_abn_available": int((g.val_mask[:len(l)] & (l == 1)).sum()),
            "val_norm_available": int((g.val_mask[:len(l)] & (l == 0)).sum()),
        }
    return out


def sample_domain(tag, n_abn, n_norm, rng):
    """仅从训练患者抽 (SplitGuard, replace=False 内置), 断言 + 测试交集自检。
    返回 (abn_beats, abn_labels, norm_beats, norm_labels, audit)。"""
    g = get_guard(tag)
    sa, sn = g.sample_train_beats(n_abn, n_norm, rng)
    b, _l, r = load_arrays(tag)
    sa, sn = np.asarray(sa), np.asarray(sn)
    sampled_rids = np.concatenate([np.asarray(r)[sa], np.asarray(r)[sn]])
    g.assert_train_only(sampled_rids, context=f"train_clean_baseline_v3({tag})")
    inter_test = np.intersect1d(np.unique(sampled_rids), g.test_record_ids())
    assert len(inter_test) == 0, \
        f"{tag}: 抽样记录与测试记录交集非空: {inter_test.tolist()}"
    audit = {
        "assert_train_only": "passed",
        "sampled_records_vs_test_intersection": 0,
        "sampled_unique_records": int(len(np.unique(sampled_rids))),
    }
    return (
        np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
        np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32),
        audit,
    )


def sample_val_domain(tag, n_abn, n_norm, rng):
    """验证集: 仅从 val 患者抽 (绝不含训练/测试患者)。"""
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
    """对真实 AFE 训练拍做少量多形态合成 (与 §99 train_clean_baseline.py 同源)。"""
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


def run_config(cfg_name):
    cfg = CONFIGS[cfg_name]
    suffix = cfg["suffix"]
    out_h5 = MODELS / f"best_resnet_large_clean_baseline_{suffix}.h5"
    out_csv = MODELS / f"train_history_clean_baseline_{suffix}.csv"
    out_json = CACHE / f"train_clean_baseline_{suffix}.json"

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    # 每组独立新建同种子 rng → A/B/C 抽样完全一致 (干净消融)
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[CB3:{cfg_name}] GPU: {gpu}", flush=True)

    # ---------- 主数据 (仅训练患者, SplitGuard 断言 + 测试交集自检) ----------
    avail = train_normal_beat_stats()
    audits = {}
    domains = {}
    for tag in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = MAIN_RATIO[tag]
        a_cap = avail[tag]["train_norm_available"]
        if n_norm > a_cap:
            raise RuntimeError(f"{tag}: 目标正常拍 {n_norm} 超过训练患者可用量 {a_cap}")
        dom = sample_domain(tag, n_abn, n_norm, rng)
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
    print(f"[CB3:{cfg_name}] main gold: {len(x_main)} "
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
    print(f"[CB3:{cfg_name}] real AFE: train_unique={len(train_real_idx)} "
          f"repeated={len(x_real)}, holdout={len(x_real_ho)}", flush=True)

    # ---------- 合成硬负样本 (仅源自真实 AFE 训练拍) ----------
    hard = synth_hard(real_train, rng)
    x_hard = hard[..., np.newaxis]
    y_hard = np.zeros(len(hard), dtype=np.int32)
    print(f"[CB3:{cfg_name}] synthetic hard neg: {len(x_hard)}", flush=True)

    x_train = np.concatenate([x_main, x_real, x_hard])
    y_train = np.concatenate([y_main, y_real, y_hard])
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[CB3:{cfg_name}] train total={len(x_train)} abn={int((y_train == 1).sum())} "
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
    print(f"[CB3:{cfg_name}] val total={len(x_val)} abn={int((y_val == 1).sum())} "
          f"norm={int((y_val == 0).sum())} (incl real holdout {len(holdout_idx)})", flush=True)

    # ---------- 模型: 随机初始化从零训练 ----------
    model = build_model(cfg["dropout"])
    print(f"[CB3:{cfg_name}] params={model.count_params():,} "
          f"(随机初始化, 未加载任何权重; dropout={cfg['dropout']})", flush=True)

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
                self.model.save(str(out_h5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})", flush=True)

    cbs = [
        ValAucCB(x_val, y_val),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=PATIENCE, restore_best_weights=True,
                                         verbose=1),
        tf.keras.callbacks.CSVLogger(str(out_csv)),
    ]

    cw = cfg["class_weight"]
    print(f"[CB3:{cfg_name}] start from-scratch training: epochs={EPOCHS} batch={BATCH} "
          f"lr={BASE_LR} cosine, class_weight={{0:{cw[0]},1:{cw[1]}}}", flush=True)
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=cbs,
        class_weight=cw,
        verbose=2,
    )
    print(f"[CB3:{cfg_name}] training done in {time.time() - t0:.0f}s", flush=True)

    # ---------- 最佳模型自检 ----------
    best = tf.keras.models.load_model(str(out_h5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_real_ho, batch_size=64, verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val))
    print(f"[CB3:{cfg_name}] BEST: val AUC={auc_val:.4f}; real holdout "
          f"mean={p_real.mean():.4f} frac>0.5={float((p_real > 0.5).mean()):.4f}", flush=True)

    split_stats = {
        tag: {k: get_guard(tag).stats[k]
              for k in ("n_patients", "n_train", "n_val", "n_test",
                        "beats_train", "beats_val", "beats_test")}
        for tag in ("mit_bih", "incart", "ptb")
    }
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": f"TH §100 clean_baseline_v3 配置 {cfg_name}: {cfg['purpose']}",
        "provenance": "from scratch; no checkpoint loaded",
        "gpu": [str(g) for g in gpu],
        "config_name": cfg_name,
        "output_model": str(out_h5.relative_to(BASE)),
        "patient_split_seed": SEED,
        "split_stats": split_stats,
        "train_beat_availability": avail,
        "data": {
            "main": {
                "mit_bih_abn": int(len(domains["mit_bih"][0])),
                "mit_bih_norm": int(len(domains["mit_bih"][2])),
                "incart_abn": int(len(domains["incart"][0])),
                "incart_norm": int(len(domains["incart"][2])),
                "ptb_abn": int(len(domains["ptb"][0])),
                "ptb_norm": int(len(domains["ptb"][2])),
            },
            "sampling_audit": audits,
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
            "architecture": "build_ecg_resnet_lite_large"
                            if cfg["dropout"] == 0.4
                            else "build_ecg_resnet_lite(large preset, dropout=0.5)",
            "params": int(model.count_params()),
            "dropout": cfg["dropout"],
            "optimizer": "adam",
            "lr_schedule": f"cosine decay from {BASE_LR}",
            "epochs_requested": EPOCHS,
            "epochs_run": int(len(hist.epoch)),
            "batch_size": BATCH,
            "early_stopping_patience": PATIENCE,
            "val_monitor": "val_auc",
            "class_weight": cw,
            "loss": "sparse_categorical_crossentropy",
        },
        "results": {
            "best_val_auc": auc_val,
            "real_holdout_mean_prob": float(p_real.mean()),
            "real_holdout_frac_gt_0.5": float((p_real > 0.5).mean()),
            "train_time_s": round(time.time() - t0, 1),
        },
        "anchors": {
            "v2_clean_baseline_mit_incart": {"auc": 0.851, "event_f1": 0.643, "beat_f1": 0.281},
            "v4_clean_mit_incart": {"auc": 0.848, "event_f1": 0.697},
            "ptb_clean_deployed_int8": {"auc": 0.900, "event_f1": 0.898},
        },
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[CB3:{cfg_name}] saved {out_h5.name}, {out_csv.name}, {out_json.name}", flush=True)
    return result


if __name__ == "__main__":
    names = sys.argv[1:] or ["a"]
    for n in names:
        if n not in CONFIGS:
            raise SystemExit(f"unknown config {n!r}, expected one of {list(CONFIGS)}")
    for n in names:
        run_config(n)
