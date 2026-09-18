#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_r19m_valid.py — R19M: 冻结 v3-A 骨干, 只训 valid 载波有效性头
================================================================================
预注册: runs/R19M_PREREG.md (2026-09-19 冻结, 任何训练之前)。承接 R18 §118
机制发现: valid 头推理门控 6/6 压制 pure_motion, 失败仅在共享骨干联合训练。

设计 (冻结):
  - v3-A (best_resnet_large_clean_baseline_v3.h5) 全部权重 trainable=False;
  - 头变体: V1 fc1(64d)→Dense(1)  |  V2 fc1→Dense(32,relu)→Dropout(0.2)→
    Dense(1)  |  V3 gap(128d)→Dense(32,relu)→Dropout(0.2)→Dense(1);
  - 选择规则: seed42 val valid AUC 最高 (平手<0.005 取参数少者 V1>V2>V3);
  - 训练素材: v3-A 训练池特征 5942 (valid=1, 与 v3-A 逐位同源) + 纯噪声窗
    1200 (valid=0, 全设备链复刻, rng=20260919+seed) + val 噪声 200;
    arm B 再加 400 mains@20dB 腐蚀 mit 训练患者载波 (valid=0);
  - 组合导出模型: v3-A 原图 (不动) + 头 → Concatenate → (3,);
    C0 断言: abn 通道 vs v3-A 逐位相等 (max|Δ|==0)。

用法:
  python3 train_r19m_valid.py --stage select              # V1/V2/V3 @seed42 选主
  python3 train_r19m_valid.py --stage full --variant V1   # 主选 ×3 seeds arm A
  python3 train_r19m_valid.py --stage armb --variant V1   # +mains 腐蚀载波 ×3
  (--smoke: 每头 5 epochs 冒烟)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BEAT_WINDOW_SAMPLES
from data.split_guard import get_guard, load_arrays
from augment_noise import augment_beats
import train_clean_baseline_v3 as V3
from train_r18_dualhead import gen_pure_noise_windows

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
V3A_H5 = MODELS / "best_resnet_large_clean_baseline_v3.h5"

SEED_NOISE = 20260919
NOISE_TRAIN_COUNTS = {"motion": 600, "emg": 240, "mains": 240,
                      "baseline_wander": 60, "respiratory": 60}
NOISE_VAL_N = 200
ARMB_CARRIER_N = 400
TAP = {"V1": "fc1", "V2": "fc1", "V3": "gap"}


def assemble_v3a_pool(seed: int):
    """v3-A 训练池 (rng 调用序与 v3-A 逐行一致, R18 已验证逐位同源)。"""
    rng = np.random.default_rng(seed)
    avail = V3.train_normal_beat_stats()
    domains = {}
    for t in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = V3.MAIN_RATIO[t]
        dom = V3.sample_domain(t, n_abn, n_norm, rng)
        domains[t] = dom
    x_parts = [np.concatenate([domains[t][0] for t in ("mit_bih", "incart", "ptb")]),
               np.concatenate([domains[t][2] for t in ("mit_bih", "incart", "ptb")])]
    real = np.concatenate([
        np.load(V3.DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32),
        np.load(V3.DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32),
    ])
    holdout_idx = rng.choice(len(real), V3.N_REAL_HOLDOUT, replace=False)
    train_real_idx = np.setdiff1d(np.arange(len(real)), holdout_idx)
    real_train = real[train_real_idx]
    x_parts.append(np.concatenate([real_train] * V3.REAL_REPEAT))
    x_parts.append(V3.synth_hard(real_train, rng))
    return np.concatenate(x_parts)[..., np.newaxis].astype(np.float32)


def build_head(variant: str, tap_dim: int, name_prefix="valid"):
    """头结构 (冻结规则): 返回 Keras 头模型 features->p_valid。"""
    inp = tf.keras.Input(shape=(tap_dim,), name="feat_in")
    if variant == "V1":
        out = tf.keras.layers.Dense(1, activation="sigmoid",
                                    name=f"{name_prefix}_out")(inp)
    else:
        x = tf.keras.layers.Dense(32, activation="relu",
                                  name=f"{name_prefix}_h1")(inp)
        x = tf.keras.layers.Dropout(0.2, name=f"{name_prefix}_do")(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid",
                                    name=f"{name_prefix}_out")(x)
    return tf.keras.Model(inp, out, name=f"valid_head_{variant}")


def train_head(variant, feats_pos, feats_neg, feats_val_neg, feats_val_pos,
               seed, epochs):
    """训头: BCE + 类平衡权重; 选模/早停监控 val valid AUC。"""
    tf.random.set_seed(seed)
    head = build_head(variant, feats_pos.shape[1])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="binary_crossentropy", metrics=["accuracy"])
    x = np.concatenate([feats_pos, feats_neg])
    y = np.concatenate([np.ones(len(feats_pos)), np.zeros(len(feats_neg))])
    perm = np.random.default_rng(seed).permutation(len(x))
    x, y = x[perm], y[perm]
    xv = np.concatenate([feats_val_pos, feats_val_neg])
    yv = np.concatenate([np.ones(len(feats_val_pos)),
                         np.zeros(len(feats_val_neg))])
    cw = {0: len(feats_pos) / max(1, len(feats_neg)), 1: 1.0}

    class ValAuc(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.best_epoch = -1.0, -1
            self.hist = []

        def on_epoch_end(self, epoch, logs=None):
            p = np.asarray(self.model(xv, training=False))[:, 0]
            auc = float(roc_auc_score(yv, p))
            self.hist.append(auc)
            (logs or {}).__setitem__("val_valid_auc", auc)
            if auc > self.best:
                self.best, self.best_epoch = auc, epoch + 1
                self.model.save_weights(str(MODELS / f"r19m_head_{variant}_tmp.weights.h5"))

    cb = ValAuc()
    head.fit(x, y, validation_data=(xv, yv), batch_size=64, epochs=epochs,
             class_weight=cw, verbose=0,
             callbacks=[cb,
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_valid_auc", mode="max", patience=20,
                            restore_best_weights=True, verbose=0)])
    return head, cb.best, cb.best_epoch, cb.hist


def assemble_combined(variant, head, c0_samples):
    """v3-A 冻结原图 + 头 → (3,) 组合模型; C0 断言 abn 逐位不变。"""
    base = tf.keras.models.load_model(str(V3A_H5), compile=False)
    base.trainable = False
    tap_layer = TAP[variant]
    tap_out = base.get_layer(tap_layer).output
    # 重建头结构并载入训练权重 (同名层)
    dim = {"fc1": 64, "gap": 128}[tap_layer]
    head2 = build_head(variant, dim)
    # 载入训练好的头权重
    for layer in head.layers:
        if layer.name.startswith("valid_"):
            head2.get_layer(layer.name).set_weights(layer.get_weights())
    v = head2(tap_out)
    combined = tf.keras.Model(base.input,
                              tf.keras.layers.Concatenate(name="out_r19m")(
                                  [base.output, v]),
                              name=f"v3a_frozen_valid_{variant}")
    for layer in combined.layers:
        layer.trainable = layer.name.startswith("valid_")
    # C0: abn 通道逐位相等
    pa = base.predict(c0_samples, batch_size=256, verbose=0)
    pc = combined.predict(c0_samples, batch_size=256, verbose=0)
    dmax = float(np.max(np.abs(pa[:, 1] - pc[:, 1])))
    return combined, dmax


def run(seed, variant, arm, epochs, tag_suffix="", noise_scale=1.0):
    tag = f"r19m{'B' if arm == 'B' else ''}_valid_{variant}_s{seed}{tag_suffix}"
    out_h5 = MODELS / f"{tag}.h5"
    out_json = CACHE / f"train_{tag}.json"
    t0 = time.time()

    base = tf.keras.models.load_model(str(V3A_H5), compile=False)
    base.trainable = False
    feat_model = {"fc1": None, "gap": None}
    feat_model["fc1"] = tf.keras.Model(base.input, base.get_layer("fc1").output)
    feat_model["gap"] = tf.keras.Model(base.input, base.get_layer("gap").output)

    # ---------- 素材 ----------
    x_pool = assemble_v3a_pool(seed)                     # v3-A 训练池 (5942)
    rng_noise = np.random.default_rng(SEED_NOISE + seed)
    counts = {k: max(1, int(round(v * noise_scale)))
              for k, v in NOISE_TRAIN_COUNTS.items()}
    noise_train = gen_pure_noise_windows(counts, rng_noise)
    noise_val = gen_pure_noise_windows({"motion": NOISE_VAL_N // 2,
                                        "mains": NOISE_VAL_N // 2}, rng_noise)
    armB_prov = None
    if arm == "B":
        g = get_guard("mit_bih")
        _s0, s1 = g.sample_train_beats(0, ARMB_CARRIER_N, rng_noise)
        carriers = np.asarray(load_arrays("mit_bih")[0])[s1].astype(np.float32)
        corrupt, armB_prov = augment_beats(carriers, rng_noise,
                                           variants_per_beat=1,
                                           types=("mains",), snrs=(20,))
        noise_train = np.concatenate([noise_train, corrupt])

    print(f"[{tag}] pool={len(x_pool)} noise_train={len(noise_train)} "
          f"noise_val={len(noise_val)}", flush=True)

    feats = {}
    for key, model in feat_model.items():
        feats[key] = {}
        feats[key]["pos"] = model.predict(x_pool, batch_size=512, verbose=0)
        feats[key]["neg"] = model.predict(noise_train[..., np.newaxis],
                                          batch_size=512, verbose=0)
        feats[key]["neg_val"] = model.predict(noise_val[..., np.newaxis],
                                              batch_size=512, verbose=0)
        feats[key]["pos_val"] = feats[key]["pos"][:500]   # 干净 val 抽样

    results = {}
    heads = {}
    variants = [variant] if variant else ["V1", "V2", "V3"]
    for vname in variants:
        dim = {"V1": 64, "V2": 64, "V3": 128}[vname]
        head, best_auc, best_ep, hist = train_head(
            vname, feats[TAP[vname]]["pos"], feats[TAP[vname]]["neg"],
            feats[TAP[vname]]["neg_val"], feats[TAP[vname]]["pos_val"],
            seed, epochs)
        heads[vname] = head
        results[vname] = {"val_valid_auc": round(best_auc, 4),
                          "best_epoch": best_ep,
                          "params": int(head.count_params())}
        print(f"[{tag}] {vname}: val_valid_auc={best_auc:.4f} "
              f"@ep{best_ep} params={head.count_params()}", flush=True)

    # ---------- 组装 + C0 ----------
    chosen = variant
    if not chosen:
        order = sorted(results, key=lambda k: (-results[k]["val_valid_auc"],
                                               ["V1", "V2", "V3"].index(k)))
        chosen = order[0]
        # 平手 <0.005 取参数少: 上面的排序已按 V1>V2>V3 优先序处理平手
        top = results[order[0]]["val_valid_auc"]
        for k in ("V1", "V2", "V3"):
            if top - results[k]["val_valid_auc"] < 0.005:
                chosen = k
                break
        print(f"[{tag}] selection -> {chosen} (rule: best val AUC, "
              f"tie<0.005 -> fewest params)", flush=True)
    c0_samples = x_pool[:512]
    combined, dmax = assemble_combined(chosen, heads[chosen], c0_samples)
    c0_pass = (dmax == 0.0)
    print(f"[{tag}] C0 abn max|Δ|={dmax} -> {'PASS' if c0_pass else 'FAIL'}",
          flush=True)
    combined.save(str(out_h5))

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prereg": "runs/R19M_PREREG.md",
        "arm": arm, "seed": seed, "chosen_variant": chosen,
        "noise_rng_seed": SEED_NOISE + seed,
        "data": {"pool_windows": int(len(x_pool)),
                 "noise_train": int(len(noise_train)),
                 "noise_val": int(len(noise_val)),
                 "armB_mains_corrupted": (ARMB_CARRIER_N if arm == "B" else 0),
                 "armB_prov": armB_prov,
                 "noise_counts": counts,
                 "noise_scale": noise_scale},
        "variants": results,
        "c0_abn_max_absdiff": dmax, "c0_pass": bool(c0_pass),
        "output_model": str(out_h5.relative_to(BASE)),
        "train_time_s": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[{tag}] saved {out_h5.name}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["select", "full", "armb"])
    ap.add_argument("--variant", default="", help="full/armb 必填: V1/V2/V3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="valid=0 纯噪声配比缩放 (预注册迭代旋钮 ±50%%)")
    a = ap.parse_args()
    epochs = 5 if a.smoke else 200
    suffix = "" if a.noise_scale == 1.0 else "_ns%g" % a.noise_scale
    if a.stage == "select":
        run(a.seed, "", "A", epochs)
    elif a.stage == "full":
        assert a.variant in ("V1", "V2", "V3")
        for s in (42, 43, 44):
            run(s, a.variant, "A", epochs, suffix, a.noise_scale)
    else:
        assert a.variant in ("V1", "V2", "V3")
        for s in (42, 43, 44):
            run(s, a.variant, "B", epochs)
