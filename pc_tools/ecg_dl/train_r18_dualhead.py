#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_r18_dualhead.py — R18 假设臂训练入口 (预注册 runs/R18_PREREG.md §1)
================================================================================
臂 (--arm):
  h2a   双头 (abnormal + valid 载波有效性); 主数据流与 v3-A 逐位同源
        (同 seed rng 调用序), 增补 1200 纯噪声窗 (valid=0, 无 abn 标签,
        mask=0) + 200 valid-val 噪声窗; 损失 = mask*CE_abn + λ*BCE_valid。
  h2b   = h2a + 400 个 mains@20dB 腐蚀正常载波窗 (valid=0; mit 训练患者
        载波, noise_rng 流; 仅 H2a 过门槛1 后启动)。
  h4    单头 v3-A 配方 + cinc2017 N 类正常拍 (--cinc-n, 默认 4000) 入正常池。
  h2h4  双头 + 纯噪声 + cinc2017 正常拍 (组合臂, 两线各自过线后)。
  h3    单头 v3-A 配方 + 批内 mixup α=0.2 (β~Beta(0.2,0.2), 软标签 CCE;
        可选臂, 预注册 §1-H3; 门槛1-only)。

冻结设计 (预注册 §1 H2a):
  - 架构 build_ecg_resnet_lite_large_dualhead: 输出 (3,)=[p_norm,p_abn,p_valid]
  - 噪声素材独立 rng 流 seed=20260918 (与语料/v3-N 流不相交)
  - 选模 = 干净 val AUC (abn 头) —— 与 v3-A 同口径; valid val AUC 仅披露
  - λ (valid 损失权重) 默认 1.0; 迭代旋钮 {1.0, 0.5} (预登记)
泄漏纪律: 公共库取数全部经 SplitGuard; cinc2017 只取 train 患者拍
(其 test 永不进入门槛1)。y 打包 (N,4)=[1-y_abn, y_abn, y_valid, mask_abn]。

用法:
  python3 train_r18_dualhead.py --arm h2a --seed 42
  python3 train_r18_dualhead.py --arm h2a --seed 42 --smoke   # 5-epoch 冒烟
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
from models.resnet_lite_1d import (build_ecg_resnet_lite_large,
                                   build_ecg_resnet_lite_large_dualhead)
from data.split_guard import get_guard, load_arrays
from noise_lib import synth_noise, chain_shape
from eval_corpus_pc import ai_hp_250
from augment_noise import augment_beats
import train_clean_baseline_v3 as V3

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"

SEED_NOISE = 20260918          # 预注册冻结: 独立噪声 rng 流
NOISE_TRAIN_COUNTS = {"motion": 600, "emg": 240, "mains": 240,
                      "baseline_wander": 60, "respiratory": 60}   # =1200
NOISE_VAL_N = 200
H2B_CARRIER_N = 400
EPS = 1e-6


def gen_pure_noise_windows(counts, rng):
    """全设备链复刻纯噪声窗: synth@500Hz → chain_shape → ai_hp_250 → z-score。

    与语料评估路径 (eval_corpus_pc.corpus_windows) 同链, 与 augment_noise
    的链整形同源; 每窗独立实现 (逐窗合成, 非连续流切片)。"""
    outs = []
    for ntype, n in counts.items():
        for _ in range(n):
            shaped = chain_shape(synth_noise(ntype, 500, 500, rng))
            w = ai_hp_250(np.asarray(shaped, dtype=np.float64))
            w = (w - w.mean()) / (w.std() + EPS)
            outs.append(w.astype(np.float32))
    return np.stack(outs)


def r18_dual_loss(lambda_v=1.0):
    """packed y_true (N,4)=[1-y_abn,y_abn,y_valid,mask_abn] vs y_pred (N,3)。

    abn 头: 二值 CE via p_abn (2 类 softmax 的等价形式); 仅 mask=1 样本。
    valid 头: BCE; 全部样本。"""
    def _loss(y_true, y_pred):
        p_abn = tf.clip_by_value(y_pred[:, 1], 1e-7, 1.0 - 1e-7)
        p_val = tf.clip_by_value(y_pred[:, 2], 1e-7, 1.0 - 1e-7)
        y_abn = y_true[:, 1]
        y_val = y_true[:, 2]
        m = y_true[:, 3]
        ce_abn = -(y_abn * tf.math.log(p_abn)
                   + (1.0 - y_abn) * tf.math.log(1.0 - p_abn))
        bce_v = -(y_val * tf.math.log(p_val)
                  + (1.0 - y_val) * tf.math.log(1.0 - p_val))
        return m * ce_abn + lambda_v * bce_v
    _loss.__name__ = "r18_dual_loss"
    return _loss


def run_arm(arm: str, seed: int, smoke: bool = False, lambda_v: float = 1.0,
            cinc_n: int = 4000, epochs: int = None, patience: int = None):
    assert arm in ("h2a", "h2b", "h4", "h2h4", "h3"), arm
    dual = arm in ("h2a", "h2b", "h2h4")
    use_cinc = arm in ("h4", "h2h4")
    use_mixup = arm == "h3"
    epochs = epochs or (5 if smoke else V3.EPOCHS)
    patience = patience or (5 if smoke else V3.PATIENCE)

    tag = f"r18_{arm}_seed{seed}" + ("_smoke" if smoke else "")
    if dual and lambda_v != 1.0:
        tag += f"_lam{lambda_v:g}"   # 迭代旋钮 λ: 产物名隔离, 不覆盖默认迭代
    out_h5 = MODELS / f"{tag}.h5"
    out_csv = MODELS / f"train_history_{tag}.csv"
    out_json = CACHE / f"train_{tag}.json"

    t0 = time.time()
    rng = np.random.default_rng(seed)          # 主流: 与 v3-A 同调用序
    rng_noise = np.random.default_rng(SEED_NOISE)   # 噪声流: 预注册冻结
    tf.random.set_seed(seed)

    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[{tag}] GPU: {gpu}", flush=True)

    # ---------- 主数据 (调用序与 v3-A 逐行一致 → 样本流逐位同源) ----------
    avail = V3.train_normal_beat_stats()
    audits, domains = {}, {}
    for t in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = V3.MAIN_RATIO[t]
        if n_norm > avail[t]["train_norm_available"]:
            raise RuntimeError(f"{t}: 正常拍不足")
        dom = V3.sample_domain(t, n_abn, n_norm, rng)
        domains[t] = dom
        audits[t] = dom[4]
    x_parts = [np.concatenate([domains[t][0] for t in ("mit_bih", "incart", "ptb")]),
               np.concatenate([domains[t][2] for t in ("mit_bih", "incart", "ptb")])]
    y_abn_parts = [np.concatenate([np.ones(len(domains[t][0]), dtype=np.int32)
                                   for t in ("mit_bih", "incart", "ptb")]),
                   np.concatenate([np.zeros(len(domains[t][2]), dtype=np.int32)
                                   for t in ("mit_bih", "incart", "ptb")])]

    real = np.concatenate([
        np.load(V3.DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32),
        np.load(V3.DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32),
    ])
    holdout_idx = rng.choice(len(real), V3.N_REAL_HOLDOUT, replace=False)
    train_real_idx = np.setdiff1d(np.arange(len(real)), holdout_idx)
    real_train = real[train_real_idx]
    x_parts.append(np.concatenate([real_train] * V3.REAL_REPEAT))
    y_abn_parts.append(np.zeros(len(real_train) * V3.REAL_REPEAT, dtype=np.int32))
    x_real_ho = real[holdout_idx]
    x_parts.append(V3.synth_hard(real_train, rng))
    y_abn_parts.append(np.zeros(len(x_parts[-1]), dtype=np.int32))

    # ---------- H4: cinc2017 N 类正常拍 (仅 train 患者) ----------
    cinc_audit = None
    if use_cinc:
        g = get_guard("cinc2017")
        l_c = np.asarray(load_arrays("cinc2017")[1])
        m_c = g.train_mask[:len(l_c)]
        inn = np.where(m_c & (l_c == 0))[0]
        if len(inn) < cinc_n:
            raise RuntimeError(f"cinc2017 train 正常拍不足: {len(inn)}<{cinc_n}")
        sc = rng.choice(inn, cinc_n, replace=False)
        b_c, _l, r_c = load_arrays("cinc2017")
        g.assert_train_only(np.asarray(r_c)[sc], context=f"r18:{arm}:cinc2017")
        inter = np.intersect1d(np.unique(np.asarray(r_c)[sc]),
                               g.test_record_ids())
        assert len(inter) == 0
        x_parts.append(np.asarray(b_c)[sc].astype(np.float32))
        y_abn_parts.append(np.zeros(cinc_n, dtype=np.int32))
        cinc_audit = {"n": cinc_n, "unique_records": int(len(np.unique(np.asarray(r_c)[sc]))),
                      "assert_train_only": "passed",
                      "sampled_vs_test_intersection": 0}

    # ---------- H2: 纯噪声 (valid=0, 无 abn 标签) ----------
    noise_train = noise_val = None
    h2b_prov = None
    if dual:
        n_all = gen_pure_noise_windows(NOISE_TRAIN_COUNTS, rng_noise)
        extra_val = gen_pure_noise_windows({"motion": NOISE_VAL_N // 2,
                                            "mains": NOISE_VAL_N // 2}, rng_noise)
        noise_train, noise_val = n_all, extra_val
        if arm == "h2b":
            g_mit = get_guard("mit_bih")
            s0, s1 = g_mit.sample_train_beats(0, H2B_CARRIER_N, rng_noise)
            carriers = np.asarray(load_arrays("mit_bih")[0])[s1].astype(np.float32)
            corrupt, h2b_prov = augment_beats(carriers, rng_noise,
                                              variants_per_beat=1,
                                              types=("mains",), snrs=(20,))
            noise_train = np.concatenate([noise_train, corrupt])

    # ---------- 组装训练集 ----------
    x_core = np.concatenate(x_parts)[..., np.newaxis].astype(np.float32)
    y_core = np.concatenate(y_abn_parts)
    if dual:
        x_noise = noise_train[..., np.newaxis].astype(np.float32)
        x_train = np.concatenate([x_core, x_noise])
        # packed y: [1-y, y, valid, mask_abn]
        y_core_pack = np.stack([1 - y_core, y_core,
                                np.ones(len(y_core), dtype=np.float32),
                                np.ones(len(y_core), dtype=np.float32)], axis=1)
        y_noise_pack = np.zeros((len(noise_train), 4), dtype=np.float32)
        y_train = np.concatenate([y_core_pack, y_noise_pack])
    else:
        x_train = x_core
        y_train = y_core
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    print(f"[{tag}] train total={len(x_train)} abn_labeled={int((y_train[:, 1] > 0).sum()) if dual else int((y_train == 1).sum())} "
          f"noise_only={len(noise_train) if dual else 0}", flush=True)

    # ---------- 验证集 (与 v3-A 同 rng 调用序 → 同源) ----------
    va, val, vn, vnl = V3.sample_val_domain("mit_bih", V3.VAL_MI_PER_CLASS,
                                            V3.VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = V3.sample_val_domain("incart", V3.VAL_MI_PER_CLASS,
                                             V3.VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = V3.sample_val_domain("ptb", V3.VAL_PTB_PER_CLASS,
                                            V3.VAL_PTB_PER_CLASS, rng)
    x_val_core = np.concatenate([va, vn, ia, inv, pa, pn]).astype(np.float32)
    y_val_core = np.concatenate([val, vnl, ial, inl, pal, pnl])
    x_val = np.concatenate([x_val_core, x_real_ho]).astype(np.float32)
    y_val = np.concatenate([y_val_core,
                            np.zeros(len(x_real_ho), dtype=np.int32)])

    # ---------- 模型 / 损失 ----------
    if dual:
        model = build_ecg_resnet_lite_large_dualhead(
            input_shape=(BEAT_WINDOW_SAMPLES, 1))
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=V3.BASE_LR,
                decay_steps=epochs * max(1, len(x_train) // V3.BATCH))),
            loss=r18_dual_loss(lambda_v))
    else:
        model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=V3.BASE_LR,
                decay_steps=epochs * max(1, len(x_train) // V3.BATCH))),
            loss=(tf.keras.losses.CategoricalCrossentropy() if use_mixup
                  else tf.keras.losses.SparseCategoricalCrossentropy()),
            metrics=["accuracy"])
    print(f"[{tag}] params={model.count_params():,} dual={dual} "
          f"mixup={use_mixup}", flush=True)

    # ---------- 选模回调: 干净 val AUC (abn 头) ----------
    class ValAucCB(tf.keras.callbacks.Callback):
        """回调内一律用直接前向 (model(x, training=False)) 而非 predict:
        GPU/XLA 下 fit 回调中 predict 会触发 unknown TensorShape (冒烟实测),
        直接调用无数据管线, 等价且稳。"""

        def __init__(self, xv, yv, xv2=None, nv=None):
            super().__init__()
            self.xv, self.yv = xv, yv
            self.xv2, self.nv = xv2, nv   # valid-val: 干净 val 抽样 + 噪声 val
            self.best, self.best_epoch = -1.0, -1

        def _fwd(self, x):
            return np.asarray(self.model(x, training=False))

        def on_epoch_end(self, epoch, logs=None):
            pred = self._fwd(self.xv)
            p = pred[:, 1] if pred.ndim > 1 and pred.shape[-1] >= 2 else pred
            auc = float(roc_auc_score(self.yv, p))
            (logs or {}).__setitem__("val_auc", auc)
            line = f"  val_auc={auc:.4f}"
            if self.nv is not None:
                k = min(500, len(self.xv))
                pv_clean = self._fwd(self.xv[:k])[:, 2]
                pv_noise = self._fwd(self.nv[..., np.newaxis])[:, 2]
                auc_v = float(roc_auc_score(
                    np.r_[np.ones(k), np.zeros(len(pv_noise))],
                    np.r_[pv_clean, pv_noise]))
                line += f" val_valid_auc={auc_v:.4f}"
                (logs or {}).__setitem__("val_valid_auc", auc_v)
            if auc > self.best:
                self.best, self.best_epoch = auc, epoch + 1
                self.model.save(str(out_h5))
                line = "  *" + line + f" saved (epoch {epoch + 1})"
            print(line, flush=True)

    cbs = [ValAucCB(x_val, y_val,
                    xv2=x_val_core,
                    nv=noise_val if dual else None),
           tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                            patience=patience,
                                            restore_best_weights=True, verbose=1),
           tf.keras.callbacks.CSVLogger(str(out_csv))]
    if dual:
        cw = None   # packed y 不支持 class_weight; v3-A 配方 a 权重 1:1 为恒等
        train_arg = x_train
    elif use_mixup:
        # 批内 mixup (α=0.2): 洗牌→成批→凸组合输入与软标签 (CCE 接受软标签)
        ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
              .shuffle(len(x_train), seed=seed, reshuffle_each_iteration=True)
              .batch(V3.BATCH))
        alpha_mix = 0.2

        def _mixup(xb, yb):
            # Beta(0.2,0.2) 采样: TF 无原生 beta, 用两独立 Gamma(0.2) 比值
            # (与 Beta 同分布)。
            g1 = tf.random.gamma([], alpha_mix)
            g2 = tf.random.gamma([], alpha_mix)
            lam = g1 / (g1 + g2 + 1e-12)
            idx = tf.random.shuffle(tf.range(tf.shape(xb)[0]), seed=seed)
            xb2 = tf.gather(xb, idx)
            yb2 = tf.gather(yb, idx)
            x_mix = lam * xb + (1.0 - lam) * xb2
            oh1 = tf.one_hot(tf.cast(yb, tf.int32), 2)
            oh2 = tf.one_hot(tf.cast(yb2, tf.int32), 2)
            return x_mix, lam * oh1 + (1.0 - lam) * oh2

        train_arg = ds.map(_mixup).prefetch(tf.data.AUTOTUNE)
        cw = None
    else:
        train_arg = x_train
        cw = {0: 1.0, 1: 1.0}
    hist = model.fit(train_arg, y=(None if (dual or use_mixup) else y_train),
        validation_data=(
        (x_val, tf.one_hot(y_val, 2)) if use_mixup else (
        (x_val, np.stack([1 - y_val, y_val,
                          np.ones(len(y_val)), np.ones(len(y_val))], axis=1))
        if dual else (x_val, y_val))),
        batch_size=V3.BATCH, epochs=epochs, callbacks=cbs,
        class_weight=cw, verbose=2)

    # ---------- 最佳模型自检 (直接前向分块, 规避 Keras3 GPU 小批量 predict bug) ----------
    def _fwd_all(model, x, bs=2048):
        outs = [np.asarray(model(x[i:i + bs], training=False))
                for i in range(0, len(x), bs)]
        return np.concatenate(outs, axis=0)

    best = tf.keras.models.load_model(str(out_h5), compile=False)
    pv = _fwd_all(best, x_val)
    p_val = pv[:, 1] if pv.ndim > 1 and pv.shape[-1] >= 2 else pv
    auc_val = float(roc_auc_score(y_val, p_val))
    p_real = _fwd_all(best, x_real_ho[..., np.newaxis], bs=64)[:, 1]
    extra = {}
    if dual:
        pn_tr = _fwd_all(best, noise_train[..., np.newaxis], bs=2048)
        extra = {
            "noise_train_valid_mean": float(pn_tr[:, 2].mean()),
            "noise_train_abn_frac_gt_05": float((pn_tr[:, 1] > 0.5).mean()),
            "lambda_v": lambda_v,
        }
    print(f"[{tag}] BEST val AUC={auc_val:.4f}; real holdout frac>0.5="
          f"{float((p_real > 0.5).mean()):.4f}; {extra}", flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": f"R18 {arm} (seed {seed})" + (" [SMOKE]" if smoke else ""),
        "prereg": "runs/R18_PREREG.md §1",
        "gpu": [str(g) for g in gpu],
        "arm": arm, "seed": seed, "smoke": bool(smoke),
        "output_model": str(out_h5.relative_to(BASE)),
        "data": {
            "main": {t: {"abn": len(domains[t][0]), "norm": len(domains[t][2])}
                     for t in domains},
            "sampling_audit": audits,
            "real_afe_holdout": int(len(x_real_ho)),
            "cinc2017": cinc_audit,
            "noise_train": ({"total": int(len(noise_train)),
                             "counts": NOISE_TRAIN_COUNTS,
                             "h2b_corrupted": int(H2B_CARRIER_N if arm == "h2b" else 0),
                             "h2b_prov": h2b_prov} if dual else None),
            "noise_val": (int(len(noise_val)) if dual else 0),
            "train_total": int(len(x_train)),
        },
        "config": {"architecture": ("build_ecg_resnet_lite_large_dualhead"
                                    if dual else "build_ecg_resnet_lite_large"),
                   "params": int(model.count_params()),
                   "lambda_v": (lambda_v if dual else None),
                   "lr": V3.BASE_LR, "epochs_requested": epochs,
                   "epochs_run": int(len(hist.epoch)), "batch": V3.BATCH,
                   "patience": patience,
                   "noise_rng_seed": (SEED_NOISE if dual else None)},
        "results": {"best_val_auc": auc_val,
                    "real_holdout_frac_gt_0.5": float((p_real > 0.5).mean()),
                    **extra,
                    "train_time_s": round(time.time() - t0, 1)},
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[{tag}] saved {out_h5.name}, {out_json.name}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["h2a", "h2b", "h4", "h2h4", "h3"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lambda-v", type=float, default=1.0)
    ap.add_argument("--cinc-n", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    a = ap.parse_args()
    run_arm(a.arm, a.seed, smoke=a.smoke, lambda_v=a.lambda_v,
            cinc_n=a.cinc_n, epochs=a.epochs, patience=a.patience)
