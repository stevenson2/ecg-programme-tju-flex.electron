#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_clean_baseline_v4.py — v3 A 配方 + 训练划分内硬负例从零重训 (TH §103)
================================================================================
规格 (本次会话冻结):
  * v3 A 配方不动: 基集 (MIT 1500/1500 + INCART 400/400 + PTB 600/400,
    rng(42) 同序消耗) + 真 AFE 271×2 + 合成硬负例 600 + 验证集构成,
    全部与 train_clean_baseline_v3.py 配置 A 逐拍一致;
  * 追加 mine_hard_negatives_v4.py 挖出的自信误报正常拍 (label=0, p>=0.90,
    MIT+INCART 训练患者, 单记录<=150, 总量<=900);
  * 等量上调异常拍 (按挖掘拍分域数量 1:1 补抽训练患者异常拍),
    使公共库部分维持 abn:norm = 1:1;
  * class_weight 维持 {0:1.0, 1:1.0}; 其余超参与 v3 A 完全一致;
  * 从零训练: 随机初始化, 不加载任何旧权重 (v3 A 只用于挖掘打分)。

铁律:
  * 一切公共库取数经 data/split_guard.py; assert_train_only + 测试交集自检;
  * 挖掘拍与基集抽样拍去重 (重复拍剔除并计数);
  * 验证集 = v3 A 验证集 (rng(42) 消耗顺序逐行复刻);
  * 添加拍与最终洗牌的随机性用独立种子 (4242/4243), 不扰动基集/验证集流。

输出:
  models/best_resnet_large_clean_baseline_v4.h5
  models/train_history_clean_baseline_v4.csv
  models/deploy_match/train_clean_baseline_v4.json
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
MINE_JSON = CACHE / "hard_negatives_v4.json"

SUFFIX = "v4"
SEED = 42
EXTRA_ABN_SEED = 4243
FINAL_SHUFFLE_SEED = 4242
MAIN_RATIO = {"mit_bih": (1500, 1500), "incart": (400, 400), "ptb": (600, 400)}
N_REAL_HOLDOUT = 40
REAL_REPEAT = 2
EPOCHS = 80
PATIENCE = 20
BASE_LR = 3e-4
BATCH = 32
VAL_MI_PER_CLASS = 800
VAL_PTB_PER_CLASS = 400
MINE_TAGS = ("mit_bih", "incart")


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
    g.assert_train_only(sampled_rids, context=f"train_clean_baseline_v4({tag})")
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


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    out_h5 = MODELS / f"best_resnet_large_clean_baseline_{SUFFIX}.h5"
    out_csv = MODELS / f"train_history_clean_baseline_{SUFFIX}.csv"
    out_json = CACHE / f"train_clean_baseline_{SUFFIX}.json"

    mine_report = json.loads(MINE_JSON.read_text(encoding="utf-8"))
    assert mine_report["spec"]["p_min"] == 0.90, "挖掘规格不符 (需 p>=0.90)"

    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)
    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[CB4] GPU: {gpu}", flush=True)

    # ---------- 基集: 与 v3 A 完全同序的 rng(42) 消耗 ----------
    audits, domains, base_sa, base_sn = {}, {}, {}, {}
    for tag in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = MAIN_RATIO[tag]
        ab, al, nb, nl, audit, sa, sn = sample_domain_idx(tag, n_abn, n_norm, rng)
        domains[tag] = (ab, al, nb, nl)
        audits[tag] = audit
        if tag in MINE_TAGS:
            base_sa[tag], base_sn[tag] = sa, sn
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

    # ---------- 验证集: 与 v3 A 完全一致 ----------
    va, val, vn, vnl = sample_val_domain("mit_bih", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = sample_val_domain("incart", VAL_MI_PER_CLASS, VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = sample_val_domain("ptb", VAL_PTB_PER_CLASS, VAL_PTB_PER_CLASS, rng)
    x_val = np.concatenate([
        np.concatenate([va, vn, ia, inv, pa, pn])[..., np.newaxis],
        x_real_ho,
    ]).astype(np.float32)
    y_val = np.concatenate([val, vnl, ial, inl, pal, pnl, y_real_ho])
    print(f"[CB4] base={len(x_base)} (abn={int((y_base == 1).sum())}, "
          f"norm={int((y_base == 0).sum())}); val={len(x_val)} (与 v3 A 一致)",
          flush=True)

    # ---------- 挖掘硬负例 (去重) + 等量异常拍回补 ----------
    rng_e = np.random.default_rng(EXTRA_ABN_SEED)
    mined_beats, extra_abn_beats = [], []
    add_audit = {}
    for tag in MINE_TAGS:
        m = mine_report["mined"][tag]
        if m.get("mined", 0) == 0:
            add_audit[tag] = {"mined_added": 0, "extra_abn_added": 0}
            continue
        idx = np.asarray(m["indices"], dtype=np.int64)
        rids = np.asarray(m["record_ids"], dtype=np.int64)
        base_set = np.concatenate([base_sa[tag], base_sn[tag]])
        dup = np.isin(idx, base_set)
        idx, rids = idx[~dup], rids[~dup]
        b, _l, r = load_arrays(tag)
        g = get_guard(tag)
        assert np.array_equal(np.asarray(r)[idx], rids), \
            f"{tag}: 挖掘索引与记录号不一致"
        g.assert_train_only(rids, context=f"v4_mined_beats({tag})")
        assert len(np.intersect1d(np.unique(rids), g.test_record_ids())) == 0
        mined_beats.append(np.asarray(b[idx], dtype=np.float32))

        # 等量异常拍回补: 排除基集已抽异常拍, replace=False
        l = np.asarray(load_arrays(tag)[1])
        pool = np.where(g.train_mask[:len(l)] & (l == 1))[0]
        pool = np.setdiff1d(pool, base_sa[tag])
        n_extra = int(len(idx))
        if len(pool) < n_extra:
            raise RuntimeError(f"{tag}: 异常拍回补池不足 {len(pool)}<{n_extra}")
        ea = rng_e.choice(pool, n_extra, replace=False)
        ea_rids = np.asarray(r)[ea]
        g.assert_train_only(ea_rids, context=f"v4_extra_abn({tag})")
        assert len(np.intersect1d(np.unique(ea_rids), g.test_record_ids())) == 0
        extra_abn_beats.append(np.asarray(b[ea], dtype=np.float32))
        add_audit[tag] = {
            "mined_in_json": int(m["mined"]),
            "mined_dup_with_base_dropped": int(dup.sum()),
            "mined_added": int(len(idx)),
            "mined_records": sorted(int(x) for x in np.unique(rids)),
            "extra_abn_added": int(n_extra),
            "extra_abn_records": sorted(int(x) for x in np.unique(ea_rids)),
            "guard_assert_train_only": "passed (mined + extra_abn)",
        }
        print(f"[CB4] {tag}: mined_added={len(idx)} (dup_dropped={int(dup.sum())}) "
              f"extra_abn={n_extra}", flush=True)

    x_mined = np.concatenate(mined_beats) if mined_beats else np.zeros(
        (0, BEAT_WINDOW_SAMPLES), dtype=np.float32)
    x_extra = np.concatenate(extra_abn_beats) if extra_abn_beats else np.zeros(
        (0, BEAT_WINDOW_SAMPLES), dtype=np.float32)
    y_mined = np.zeros(len(x_mined), dtype=np.int32)
    y_extra = np.ones(len(x_extra), dtype=np.int32)
    assert len(x_mined) == len(x_extra), "1:1 回补失衡"

    x_train = np.concatenate([x_base, x_mined[..., np.newaxis],
                              x_extra[..., np.newaxis]])
    y_train = np.concatenate([y_base, y_mined, y_extra])
    perm2 = np.random.default_rng(FINAL_SHUFFLE_SEED).permutation(len(x_train))
    x_train, y_train = x_train[perm2], y_train[perm2]
    n_abn_tr, n_norm_tr = int((y_train == 1).sum()), int((y_train == 0).sum())
    print(f"[CB4] train total={len(x_train)} abn={n_abn_tr} norm={n_norm_tr} "
          f"(公共库主域 abn:norm = {2500 + len(x_extra)}:{2500 + len(x_mined)})",
          flush=True)

    # ---------- 从零训练 (与 v3 A 同超参) ----------
    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    print(f"[CB4] params={model.count_params():,} (随机初始化, 从零训练)",
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
    print(f"[CB4] training done in {time.time() - t0:.0f}s", flush=True)

    best = tf.keras.models.load_model(str(out_h5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_real_ho, batch_size=64, verbose=0)[:, 1]
    print(f"[CB4] BEST val AUC={roc_auc_score(y_val, p_val):.4f}; real holdout "
          f"mean={p_real.mean():.4f} frac>0.5={float((p_real > 0.5).mean()):.4f}",
          flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §103 clean_baseline_v4: v3 A 配方 + 训练划分内硬负例 "
                   "(p>=0.90) 与等量异常拍 1:1 回补, 从零重训",
        "provenance": "from scratch; no checkpoint loaded; v3 A 仅用于挖掘打分",
        "gpu": [str(g) for g in gpu],
        "output_model": str(out_h5.relative_to(BASE)),
        "patient_split_seed": SEED,
        "base_and_val_identical_to_v3_a": True,
        "data": {
            "base_main": {f"{t}_abn": MAIN_RATIO[t][0] for t in MAIN_RATIO} |
                         {f"{t}_norm": MAIN_RATIO[t][1] for t in MAIN_RATIO},
            "sampling_audit": audits,
            "real_afe_train_unique": int(len(train_real_idx)),
            "real_afe_train_repeated": int(len(x_real)),
            "real_afe_holdout": int(len(x_real_ho)),
            "synthetic_hard_negative": int(len(x_hard)),
            "hard_negative_mining": {
                "source": str(MINE_JSON.relative_to(BASE)),
                "spec": mine_report["spec"],
                "mine_seed": mine_report["mine_seed"],
                "already_in_base_audit": mine_report["already_in_base_audit"],
                "per_tag_addition": add_audit,
                "mined_total_added": int(len(x_mined)),
                "extra_abn_total_added": int(len(x_extra)),
            },
            "train_total": int(len(x_train)),
            "train_abn": n_abn_tr,
            "train_norm": n_norm_tr,
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
    print(f"[CB4] saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
