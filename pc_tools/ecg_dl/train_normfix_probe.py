#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_normfix_probe.py — §105 Phase A-2 归一化修复探针训练
================================================================================
单变量隔离: 配方逐位 = v3 A (train_clean_baseline_v3.py 配置 a), 唯一差异是
MIT/INCART 用 normfix 数组 (build_normfix_arrays.py 产物) 取代旧 z-score 数组;
PTB 仍用旧数组 (刻意单变量隔离); 真实 AFE 正常拍用 normfix 归一化重算
(链逐行同 preprocess_real_exp7c.py, 保留 210+101=311 拍, keep 掩码与旧口径一致)。

rng(42) 消耗顺序与 v3 A 完全一致:
  mit 1500/1500 → incart 400/400 → ptb 600/400 → 真实 AFE holdout choice(40)
  → synth_hard 600 → perm → val (mit/incart 800/800, ptb 400/400)。
断言入日志: base=5942, val=4040。

探针双闸 (两者同时满足才进 Phase B):
  (i)  best val AUC ≥ 0.80
  (ii) 真实 AFE holdout mean<0.10 且 frac_gt_0.5=0 (口径同 acceptance_theta95_v4.py)

铁律: 公共库取数经 data/split_guard.py; 随机初始化从零训练不加载任何权重;
全部数字落盘。测试集零接触。
产物:
  models/best_resnet_large_clean_baseline_normfix_probe.h5
  models/train_history_clean_baseline_normfix_probe.csv
  models/deploy_match/train_normfix_probe.json
  models/deploy_match/normfix_probe_metrics.json
  models/deploy_match/probe.log
"""
import json
import os
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES, TARGET_FS
from models.resnet_lite_1d import build_ecg_resnet_lite_large
from data.split_guard import get_guard, load_arrays
from eval_deploy_match import (
    _comb_filter, _hp_lp_filter, causal_hp_05_fs250, extract_beats_deploy,
)
from normfix import load_spec, normalize_stream, window_beats_normfix

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
CACHE = MODELS / "deploy_match"
REPO_ROOT = BASE.parents[1]
ECG_DATA = Path(os.environ.get("ECG_PROCESSED_DIR", "/home/devcontainers/ecg_data"))

SEED = 42
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

# 单变量隔离: 只有 mit_bih/incart 换 normfix 数组; ptb 保持旧数组
NORMFIX_TAGS = ("mit_bih", "incart")

REAL_SOURCES = [
    ("exp7c", DATA_REAL / "ecg_real_052.ecgr", 210),
    ("rec_latest", REPO_ROOT / "rec_latest.ecgr", 101),
]

OPTION = "A"      # A / B
ARRAY_SUFFIX = "_normfix"          # "_normfix" | "_normfix_b"
OUT_TAG = "normfix_probe"          # "normfix_probe" | "normfix_b_probe"

# 注意: 以下路径在 main() 内根据 OPTION 重算 (模块顶层 OUT_TAG 是默认 A,
# 若在此处 f-string 求值, B 运行会写错文件名覆盖 A 产物)。
OUT_H5 = None
OUT_CSV = None
OUT_JSON = None
OUT_METRICS = None


def _set_output_paths():
    global OUT_H5, OUT_CSV, OUT_JSON, OUT_METRICS
    OUT_H5 = MODELS / f"best_resnet_large_clean_baseline_{OUT_TAG}.h5"
    OUT_CSV = MODELS / f"train_history_clean_baseline_{OUT_TAG}.csv"
    OUT_JSON = CACHE / f"train_{OUT_TAG}.json"
    OUT_METRICS = CACHE / f"{OUT_TAG}_metrics.json"


def load_beats(tag):
    """mit_bih/incart 用 normfix 数组 (mmap), ptb 用旧数组。"""
    if tag in NORMFIX_TAGS:
        return np.load(ECG_DATA / f"{tag}_processed_deploy_causal{ARRAY_SUFFIX}_beats.npy",
                       mmap_mode="r")
    return load_arrays(tag)[0]


def sample_domain(tag, n_abn, n_norm, rng):
    """与 v3 A sample_domain 逐位同 rng 消耗; 仅拍值取自 normfix 数组。
    返回 (abn_beats, abn_labels, norm_beats, norm_labels, audit)。"""
    g = get_guard(tag)
    sa, sn = g.sample_train_beats(n_abn, n_norm, rng)
    b = load_beats(tag)
    _b, _l, r = load_arrays(tag)
    sa, sn = np.asarray(sa), np.asarray(sn)
    sampled_rids = np.concatenate([np.asarray(r)[sa], np.asarray(r)[sn]])
    g.assert_train_only(sampled_rids, context=f"train_normfix_probe({tag})")
    inter_test = np.intersect1d(np.unique(sampled_rids), g.test_record_ids())
    assert len(inter_test) == 0, \
        f"{tag}: 抽样记录与测试记录交集非空: {inter_test.tolist()}"
    audit = {
        "assert_train_only": "passed",
        "sampled_records_vs_test_intersection": 0,
        "sampled_unique_records": int(len(np.unique(sampled_rids))),
        "beats_source": "normfix" if tag in NORMFIX_TAGS else "old_deploy_causal",
    }
    return (
        np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
        np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32),
        audit,
    )


def sample_val_domain(tag, n_abn, n_norm, rng):
    """与 v3 A sample_val_domain 逐位同 rng 消耗; mit/incart 拍值取 normfix。"""
    g = get_guard(tag)
    _b, l, _r = load_arrays(tag)
    b = load_beats(tag)
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
    """与 v3 A synth_hard 逐位一致 (6 变换 × min(100, len) 源拍)。"""
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


def real_chain_stream(ecgr_path):
    """与 preprocess_real_exp7c.py / normfix_rms_measure.py 逐行同链。"""
    raw = ecgr_path.read_bytes()
    n = struct.unpack_from("<I", raw, 18)[0]
    dur = struct.unpack_from("<I", raw, 14)[0]
    x = np.frombuffer(raw, dtype="<i2", count=n, offset=32).astype(np.float64) / 8000.0
    fs_eff = n / dur
    ratio = Fraction(int(round(500.0 / fs_eff * 10000)), 10000).limit_denominator(100000)
    s500 = resample_poly(x, ratio.numerator, ratio.denominator)
    dc = s500 - np.mean(s500)
    combed = _comb_filter(dc)
    filt = _hp_lp_filter(combed)
    dec = filt[0::2]
    return causal_hp_05_fs250(dec), fs_eff


def real_normfix_beats(ecgr_path, spec):
    """真实 AFE 记录 → normfix 归一化拍 (链同 preprocess_real_exp7c; XQRS 峰;
    incart 严格跳边窗口; 无窗口 z-score, 归一化已在流级完成)。
    返回 (beats, n_peaks, n_extracted, fs_eff)。"""
    from wfdb.processing import xqrs_detect
    chain250, fs_eff = real_chain_stream(ecgr_path)
    r_idx = xqrs_detect(chain250.astype(np.float64), fs=TARGET_FS, verbose=False)
    y250 = normalize_stream(chain250, spec)
    beats = window_beats_normfix(y250, np.array(r_idx, dtype=np.int64),
                                 "incart", BEAT_WINDOW_SAMPLES)
    return beats.astype(np.float32), int(len(r_idx)), int(len(beats)), float(fs_eff)


def build_model():
    return build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="normfix 探针训练 (A/B 规格, 配方=v3 A)")
    ap.add_argument("--spec", default=None,
                    help="规格 JSON 路径; 默认 A; B 用 models/deploy_match/normfix_spec_v1_b.json")
    args = ap.parse_args()

    global OPTION, ARRAY_SUFFIX, OUT_TAG
    spec = load_spec(args.spec) if args.spec else load_spec()
    OPTION = spec["option"]
    if OPTION == "B":
        ARRAY_SUFFIX = "_normfix_b"
        OUT_TAG = "normfix_b_probe"
    _set_output_paths()

    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[NFP] GPU: {gpu}", flush=True)

    # ---------- 主数据 (仅训练患者; mit/incart normfix, ptb 旧数组) ----------
    audits = {}
    domains = {}
    for tag in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = MAIN_RATIO[tag]
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
    print(f"[NFP] main gold: {len(x_main)} "
          f"(abn={int((y_main == 1).sum())}, norm={int((y_main == 0).sum())})", flush=True)

    # ---------- 真实 AFE 正常拍: normfix 重算 (210+101=311), 训练/留出 ----------
    real_parts = []
    real_meta = {}
    for name, path, expect_n in REAL_SOURCES:
        beats, n_peaks, n_extracted, fs_eff = real_normfix_beats(path, spec)
        if int(len(beats)) != expect_n:
            raise RuntimeError(f"real {name}: normfix 拍数 {len(beats)} != 期望 {expect_n}")
        real_parts.append(beats)
        real_meta[name] = {"n_peaks": n_peaks, "n_extracted": n_extracted,
                           "n_beats": int(len(beats)), "fs_eff_hz": fs_eff}
    real = np.concatenate(real_parts)
    assert len(real) == 311, f"real AFE 总拍数 {len(real)} != 311"

    holdout_idx = rng.choice(len(real), N_REAL_HOLDOUT, replace=False)
    train_real_idx = np.setdiff1d(np.arange(len(real)), holdout_idx)
    real_train = real[train_real_idx]
    x_real = np.concatenate([real_train] * REAL_REPEAT)[..., np.newaxis]
    y_real = np.zeros(len(x_real), dtype=np.int32)
    x_real_ho = real[holdout_idx][..., np.newaxis]
    y_real_ho = np.zeros(len(holdout_idx), dtype=np.int32)
    print(f"[NFP] real AFE (normfix): train_unique={len(train_real_idx)} "
          f"repeated={len(x_real)}, holdout={len(x_real_ho)}", flush=True)

    # ---------- 合成硬负样本 (仅源自真实 AFE 训练拍) ----------
    hard = synth_hard(real_train, rng)
    x_hard = hard[..., np.newaxis]
    y_hard = np.zeros(len(hard), dtype=np.int32)
    print(f"[NFP] synthetic hard neg: {len(x_hard)}", flush=True)

    x_train = np.concatenate([x_main, x_real, x_hard])
    y_train = np.concatenate([y_main, y_real, y_hard])
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[NFP] train total={len(x_train)} abn={int((y_train == 1).sum())} "
          f"norm={int((y_train == 0).sum())}", flush=True)
    assert len(x_train) == 5942, f"base 断言失败: {len(x_train)} != 5942"

    # ---------- 验证集: val 患者 (mit/incart normfix, ptb 旧) + 真实 AFE 留出 ----------
    va, val, vn, vnl = sample_val_domain("mit_bih", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = sample_val_domain("incart", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = sample_val_domain("ptb", VAL_PTB_PER_CLASS, VAL_PTB_PER_CLASS, rng)
    x_val = np.concatenate([
        np.concatenate([va, vn, ia, inv, pa, pn])[..., np.newaxis],
        x_real_ho,
    ]).astype(np.float32)
    y_val = np.concatenate([val, vnl, ial, inl, pal, pnl, y_real_ho])
    print(f"[NFP] val total={len(x_val)} abn={int((y_val == 1).sum())} "
          f"norm={int((y_val == 0).sum())} (incl real holdout {len(holdout_idx)})", flush=True)
    assert len(x_val) == 4040, f"val 断言失败: {len(x_val)} != 4040"

    # ---------- 模型: 随机初始化从零训练 ----------
    model = build_model()
    print(f"[NFP] params={model.count_params():,} (随机初始化, 未加载任何权重)", flush=True)

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

    cw = {0: 1.0, 1: 1.0}
    print(f"[NFP] start from-scratch training: epochs={EPOCHS} batch={BATCH} "
          f"lr={BASE_LR} cosine, class_weight={cw}", flush=True)
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=cbs,
        class_weight=cw,
        verbose=2,
    )
    print(f"[NFP] training done in {time.time() - t0:.0f}s", flush=True)

    # ---------- 最佳模型自检 ----------
    best = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_real_ho, batch_size=64, verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val))
    hold_mean = float(p_real.mean())
    hold_frac = float((p_real > 0.5).mean())
    print(f"[NFP] BEST: val AUC={auc_val:.4f}; real holdout "
          f"mean={hold_mean:.4f} frac>0.5={hold_frac:.4f}", flush=True)

    # ---------- 双闸判定 ----------
    gate1 = auc_val >= 0.80
    gate2 = (hold_mean < 0.10) and (hold_frac == 0.0)
    verdict = "PASS" if (gate1 and gate2) else "FAIL"
    print(f"[NFP] gate1(val_auc>=0.80)={gate1} gate2(holdout mean<0.10 & frac=0)={gate2} "
          f"=> {verdict}", flush=True)

    split_stats = {
        tag: {k: get_guard(tag).stats[k]
              for k in ("n_patients", "n_train", "n_val", "n_test",
                        "beats_train", "beats_val", "beats_test")}
        for tag in ("mit_bih", "incart", "ptb")
    }
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §105 Phase A-2: normfix 探针 (单变量隔离, 配方=v3 A)",
        "provenance": "from scratch; no checkpoint loaded",
        "gpu": [str(g) for g in gpu],
        "patient_split_seed": SEED,
        "split_stats": split_stats,
        "normfix_spec": spec["spec_id"],
        "single_variable_isolation": {
            "mit_bih": f"normfix arrays (mit_bih_processed_deploy_causal{ARRAY_SUFFIX})",
            "incart": f"normfix arrays (incart_processed_deploy_causal{ARRAY_SUFFIX})",
            "ptb": "old deploy_causal arrays (unchanged)",
            "real_afe": f"recomputed with {OPTION} normalization (210+101=311)",
        },
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
            "real_afe_normfix": real_meta,
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
            "dropout": 0.4,
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
            "real_holdout_mean_prob": hold_mean,
            "real_holdout_frac_gt_0.5": hold_frac,
            "train_time_s": round(time.time() - t0, 1),
        },
        "probe_gates": {
            "gate1_best_val_auc_ge_0.80": {"passed": bool(gate1), "value": auc_val},
            "gate2_real_holdout_mean_lt_0.10_and_frac_eq_0": {
                "passed": bool(gate2),
                "value": {"mean_prob": hold_mean, "frac_gt_0.5": hold_frac},
            },
            "verdict": verdict,
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    metrics = {
        "date": result["date"],
        "chapter": "TH §105 Phase A-2",
        "model": str(OUT_H5.relative_to(BASE)),
        "best_val_auc": auc_val,
        "real_holdout_mean_prob": hold_mean,
        "real_holdout_frac_gt_0.5": hold_frac,
        "gate1_pass": bool(gate1),
        "gate2_pass": bool(gate2),
        "verdict": verdict,
        "decision": "both pass → Phase B; else stop and report",
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"[NFP] saved {OUT_H5.name}, {OUT_CSV.name}, {OUT_JSON.name}, "
          f"{OUT_METRICS.name}", flush=True)
    return result


if __name__ == "__main__":
    main()
