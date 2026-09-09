#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_clean_baseline_v5.py — v3 A 配方 + SVDB 扩训从零重训 (TH §104)
================================================================================
规格 (本次会话冻结):
  * v3 A 配方不动: 基集 (MIT 1500/1500 + INCART 400/400 + PTB 600/400,
    rng(42) 同序消耗) + 真 AFE 271×2 + 合成硬负例 600 + 验证集构成,
    全部与 train_clean_baseline_v3.py 配置 A / v4 逐拍一致 (base=5942,
    val=4040, 日志断言 + real holdout 索引与 v4 JSON 对拍);
  * 新增: 独立流 rng(5501) 经守卫从 SVDB 训练患者采 1500 异常拍 +
    2000 正常拍 (SVDB = §104 前置基建: 184,407 拍因果链数组,
    独立患者划分 48/15/15, 注释真值 78/78 验证), 并入 base;
    新总量 9442 (abn 4000 / norm 5442, abn 占比 0.424, 与 v3 A 的
    0.421 近似, 无需再平衡);
  * 验证集 = 纯 v3 A 验证集 (不含 SVDB), 保证与 v3 A / v4 选模可比;
  * 最终全量洗牌沿用同一条 rng(5501) 流;
  * class_weight 维持 {0:1.0, 1:1.0}; 其余超参与 v3 A 完全一致;
  * 从零训练: 随机初始化, 不加载任何旧权重。

铁律:
  * 一切公共库取数经 data/split_guard.py; SVDB 采样后经
    assert_train_only + 与测试记录交集自检, 回执写进训练 JSON;
  * rng(42) 主流消耗顺序与 v4 逐行一致, 不插入任何额外随机调用;
  * SVDB 测试/验证患者零接触。

输出:
  models/best_resnet_large_clean_baseline_v5.h5
  models/train_history_clean_baseline_v5.csv
  models/deploy_match/train_clean_baseline_v5.json
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

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA_REAL = BASE / "data" / "real"
CACHE = MODELS / "deploy_match"
V4_TRAIN_JSON = CACHE / "train_clean_baseline_v4.json"

SUFFIX = "v5"
SEED = 42
SVDB_SEED = 5501
MAIN_RATIO = {"mit_bih": (1500, 1500), "incart": (400, 400), "ptb": (600, 400)}
SVDB_RATIO = {"n_abn": 1500, "n_norm": 2000}
N_REAL_HOLDOUT = 40
REAL_REPEAT = 2
EPOCHS = 80
PATIENCE = 20
BASE_LR = 3e-4
BATCH = 32
VAL_MI_PER_CLASS = 800
VAL_PTB_PER_CLASS = 400
EXPECTED_BASE = 5942
EXPECTED_VAL = 4040
EXPECTED_TRAIN_TOTAL = 9442


def synth_hard(real_train, rng):
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


def sample_domain_idx(tag, n_abn, n_norm, rng):
    """与 v3 sample_domain 同语义、同 rng 消耗; 额外返回抽样索引。"""
    g = get_guard(tag)
    sa, sn = g.sample_train_beats(n_abn, n_norm, rng)
    b, _l, r = load_arrays(tag)
    sa, sn = np.asarray(sa), np.asarray(sn)
    sampled_rids = np.concatenate([np.asarray(r)[sa], np.asarray(r)[sn]])
    g.assert_train_only(sampled_rids, context=f"train_clean_baseline_v5({tag})")
    inter_test = np.intersect1d(np.unique(sampled_rids), g.test_record_ids())
    assert len(inter_test) == 0, f"{tag}: 抽样记录与测试记录交集非空"
    audit = {"assert_train_only": "passed",
             "sampled_records_vs_test_intersection": 0,
             "sampled_unique_records": int(len(np.unique(sampled_rids)))}
    return (np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
            np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32),
            audit, sa, sn)


def sample_val_domain(tag, n_abn, n_norm, rng):
    g = get_guard(tag)
    b, l, _r = load_arrays(tag)
    m = g.val_mask[:len(l)]
    ia = np.where(m & (l == 1))[0]
    inn = np.where(m & (l == 0))[0]
    if len(ia) < n_abn or len(inn) < n_norm:
        raise RuntimeError(f"{tag}: val 患者心拍不足")
    sa = rng.choice(ia, n_abn, replace=False)
    sn = rng.choice(inn, n_norm, replace=False)
    return (np.asarray(b[sa], dtype=np.float32), np.ones(len(sa), dtype=np.int32),
            np.asarray(b[sn], dtype=np.float32), np.zeros(len(sn), dtype=np.int32))


def sample_svdb(rng_sv):
    """独立流 rng(5501): SVDB 训练患者 1500 异常拍 + 2000 正常拍。"""
    g = get_guard("svdb")
    sa, sn = g.sample_train_beats(SVDB_RATIO["n_abn"], SVDB_RATIO["n_norm"], rng_sv)
    b, l, r = load_arrays("svdb")
    sa, sn = np.asarray(sa), np.asarray(sn)
    l = np.asarray(l)
    assert int(l[sa].sum()) == len(sa), "svdb: 异常拍抽样标签不为 1"
    assert int(l[sn].sum()) == 0, "svdb: 正常拍抽样标签不为 0"
    sampled_rids = np.concatenate([np.asarray(r)[sa], np.asarray(r)[sn]])
    g.assert_train_only(sampled_rids, context="train_clean_baseline_v5(svdb)")
    inter_test = np.intersect1d(np.unique(sampled_rids), g.test_record_ids())
    assert len(inter_test) == 0, "svdb: 抽样记录与测试记录交集非空"
    audit = {"tag": "svdb",
             "seed": SVDB_SEED,
             "requested": dict(SVDB_RATIO),
             "assert_train_only": "passed",
             "sampled_records_vs_test_intersection": 0,
             "sampled_unique_records": int(len(np.unique(sampled_rids))),
             "sampled_records": sorted(int(x) for x in np.unique(sampled_rids)),
             "source_arrays": "/home/devcontainers/ecg_data/"
                              "svdb_processed_deploy_causal_{beats,labels,record_ids}.npy",
             "build_log": "pc_tools/ecg_dl/svdb_deploy_build.log",
             "guard_split": "独立患者级划分 78 = train 48 / val 15 / test 15"}
    x_abn = np.asarray(b[sa], dtype=np.float32)
    x_norm = np.asarray(b[sn], dtype=np.float32)
    return x_abn, x_norm, audit


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    out_h5 = MODELS / f"best_resnet_large_clean_baseline_{SUFFIX}.h5"
    out_csv = MODELS / f"train_history_clean_baseline_{SUFFIX}.csv"
    out_json = CACHE / f"train_clean_baseline_{SUFFIX}.json"

    v4_report = json.loads(V4_TRAIN_JSON.read_text(encoding="utf-8"))

    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)
    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[CB5] GPU: {gpu}", flush=True)

    # ---------- 基集: 与 v3 A / v4 完全同序的 rng(42) 消耗 ----------
    audits, domains = {}, {}
    for tag in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = MAIN_RATIO[tag]
        ab, al, nb, nl, audit, _sa, _sn = sample_domain_idx(tag, n_abn, n_norm, rng)
        domains[tag] = (ab, al, nb, nl)
        audits[tag] = audit
    x_main = np.concatenate([
        domains["mit_bih"][0], domains["incart"][0], domains["ptb"][0],
        domains["mit_bih"][2], domains["incart"][2], domains["ptb"][2],
    ])[..., np.newaxis]
    y_main = np.concatenate([
        domains["mit_bih"][1], domains["incart"][1], domains["ptb"][1],
        domains["mit_bih"][3], domains["incart"][3], domains["ptb"][3],
    ])

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

    x_base = np.concatenate([x_main, x_real, x_hard])
    y_base = np.concatenate([y_main, y_real, y_hard])
    perm = rng.permutation(len(x_base))          # 与 v3 A 同一消耗
    x_base, y_base = x_base[perm], y_base[perm]

    # ---------- 验证集: 与 v3 A / v4 完全一致 ----------
    va, val, vn, vnl = sample_val_domain("mit_bih", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = sample_val_domain("incart", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = sample_val_domain("ptb", VAL_PTB_PER_CLASS, VAL_PTB_PER_CLASS, rng)
    x_val = np.concatenate([
        np.concatenate([va, vn, ia, inv, pa, pn])[..., np.newaxis],
        x_real_ho,
    ]).astype(np.float32)
    y_val = np.concatenate([val, vnl, ial, inl, pal, pnl, y_real_ho])

    assert len(x_base) == EXPECTED_BASE, \
        f"base={len(x_base)} != {EXPECTED_BASE}: rng(42) 消耗链与 v3 A/v4 不一致"
    assert len(x_val) == EXPECTED_VAL, \
        f"val={len(x_val)} != {EXPECTED_VAL}: 验证集与 v3 A/v4 不一致"
    base_abn, base_norm = int((y_base == 1).sum()), int((y_base == 0).sum())
    assert (base_abn, base_norm) == (2500, 3442), \
        f"base 配比异常 abn={base_abn} norm={base_norm} (期望 2500/3442)"
    assert np.array_equal(holdout_idx,
                          np.asarray(v4_report["data"]["real_holdout_indices"])), \
        "real holdout 索引与 v4 JSON 不一致: rng(42) 流被污染"
    print(f"[CB5] base={len(x_base)} (abn={base_abn}, norm={base_norm}); "
          f"val={len(x_val)} — 与 v3 A/v4 逐位一致 (holdout 索引对拍通过)",
          flush=True)

    # ---------- SVDB 扩训 (独立流 rng(5501)) ----------
    rng_sv = np.random.default_rng(SVDB_SEED)
    x_sv_abn, x_sv_norm, svdb_audit = sample_svdb(rng_sv)
    x_svdb = np.concatenate([x_sv_abn, x_sv_norm])[..., np.newaxis]
    y_svdb = np.concatenate([np.ones(len(x_sv_abn), dtype=np.int32),
                             np.zeros(len(x_sv_norm), dtype=np.int32)])
    print(f"[CB5] svdb added={len(x_svdb)} (abn={len(x_sv_abn)}, "
          f"norm={len(x_sv_norm)}), records={svdb_audit['sampled_unique_records']}",
          flush=True)

    x_train = np.concatenate([x_base, x_svdb])
    y_train = np.concatenate([y_base, y_svdb])
    perm2 = rng_sv.permutation(len(x_train))     # 同一条 5501 流全量洗牌
    x_train, y_train = x_train[perm2], y_train[perm2]
    n_abn_tr, n_norm_tr = int((y_train == 1).sum()), int((y_train == 0).sum())
    assert len(x_train) == EXPECTED_TRAIN_TOTAL, \
        f"train total={len(x_train)} != {EXPECTED_TRAIN_TOTAL}"
    assert (n_abn_tr, n_norm_tr) == (4000, 5442), \
        f"train 配比异常 abn={n_abn_tr} norm={n_norm_tr} (期望 4000/5442)"
    print(f"[CB5] train total={len(x_train)} abn={n_abn_tr} norm={n_norm_tr} "
          f"(abn ratio {n_abn_tr / len(x_train):.3f} vs v3 A 0.421)", flush=True)

    # ---------- 从零训练 (与 v3 A 同超参) ----------
    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    print(f"[CB5] params={model.count_params():,} (随机初始化, 从零训练)",
          flush=True)
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
            self.best, self.best_epoch = -1.0, -1

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(self.xv, batch_size=256, verbose=0)[:, 1]
            auc = float(roc_auc_score(self.yv, p))
            logs = logs or {}
            logs["val_auc"] = auc
            if auc > self.best:
                self.best, self.best_epoch = auc, epoch + 1
                self.model.save(str(out_h5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})",
                      flush=True)

    cbs = [ValAucCB(x_val, y_val),
           tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                            patience=PATIENCE,
                                            restore_best_weights=True, verbose=1),
           tf.keras.callbacks.CSVLogger(str(out_csv))]
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                     batch_size=BATCH, epochs=EPOCHS, callbacks=cbs,
                     class_weight={0: 1.0, 1: 1.0}, verbose=2)
    print(f"[CB5] training done in {time.time() - t0:.0f}s", flush=True)

    best = tf.keras.models.load_model(str(out_h5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_real_ho, batch_size=64, verbose=0)[:, 1]
    print(f"[CB5] BEST val AUC={roc_auc_score(y_val, p_val):.4f}; real holdout "
          f"mean={p_real.mean():.4f} frac>0.5={float((p_real > 0.5).mean()):.4f}",
          flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §104 clean_baseline_v5: v3 A 配方 + SVDB 训练患者扩训 "
                   "(1500 abn / 2000 norm), 从零重训",
        "provenance": "from scratch; no checkpoint loaded; SVDB 基建来自 §104 前置 "
                      "(184,407 拍因果链数组, 注释真值 78/78 验证, "
                      "日志 pc_tools/ecg_dl/svdb_deploy_build.log)",
        "gpu": [str(g) for g in gpu],
        "output_model": str(out_h5.relative_to(BASE)),
        "patient_split_seed": SEED,
        "svdb_seed": SVDB_SEED,
        "base_and_val_identical_to_v3_a": True,
        "replication_checks": {
            "base_total": int(len(x_base)),
            "base_abn": base_abn,
            "base_norm": base_norm,
            "val_total": int(len(x_val)),
            "real_holdout_indices_match_v4_json": True,
        },
        "data": {
            "base_main": {f"{t}_abn": MAIN_RATIO[t][0] for t in MAIN_RATIO} |
                         {f"{t}_norm": MAIN_RATIO[t][1] for t in MAIN_RATIO},
            "sampling_audit": audits,
            "real_afe_train_unique": int(len(train_real_idx)),
            "real_afe_train_repeated": int(len(x_real)),
            "real_afe_holdout": int(len(x_real_ho)),
            "synthetic_hard_negative": int(len(x_hard)),
            "svdb_addition": svdb_audit,
            "train_total": int(len(x_train)),
            "train_abn": n_abn_tr,
            "train_norm": n_norm_tr,
            "val_total": int(len(x_val)),
            "val_abn": int((y_val == 1).sum()),
            "val_norm": int((y_val == 0).sum()),
            "val_contains_svdb": False,
            "real_holdout_indices": [int(i) for i in holdout_idx],
        },
        "config": {
            "architecture": "build_ecg_resnet_lite_large",
            "params": int(model.count_params()),
            "dropout": 0.4,
            "optimizer": "adam",
            "lr_schedule": "cosine decay from 0.0003",
            "epochs_requested": EPOCHS,
            "epochs_run": len(hist.history["loss"]),
            "batch_size": BATCH,
            "early_stopping_patience": PATIENCE,
            "val_monitor": "val_auc",
            "class_weight": {0: 1.0, 1: 1.0},
            "loss": "sparse_categorical_crossentropy",
        },
        "results": {
            "best_val_auc": float(max(hist.history["val_auc"])),
            "best_epoch": int(np.argmax(hist.history["val_auc"]) + 1),
            "real_holdout_mean_prob": float(p_real.mean()),
            "real_holdout_frac_gt_0.5": float((p_real > 0.5).mean()),
            "train_time_s": round(time.time() - t0, 1),
        },
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[CB5] saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
