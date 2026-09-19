#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_r21_kd.py — R21 H7/H8: KD 锚定双头联合训练 (预注册 runs/R21M_PREREG.md §1-§2)
================================================================================
机制 (对 §118/§119 三线负结果图谱的对症设计):
  骨干可塑 (保泛化, 反 R19M/R20M 冻结封顶) + abn 头对冻结 v3-A 的概率 KL 锚
  (保 θ=0.5 工作点校准, 反 R18 无锚漂移)。

臂 (--arm):
  kd    H7: R18 H2a 数据配方 (v3-A 主流逐位同源 + 1200 纯噪声 valid=0)
        + KD 锚; 损失 = mask·CE_abn + λ_v·BCE_valid + λ_kd·mask·KL(p_abn‖p_t)。
  h8    H8 (条件, H7 过线后): kd + 800 轻噪腐蚀载波窗 valid=1 (SNR 30/20dB,
        {bw,emg,motion,mains}×100/类·SNR; 只进 valid 头, abn mask=0;
        10dB 不入 valid 监督 —— LIT_REVIEW A3 接受线论证)。

冻结设计 (R21M_PREREG §1):
  - 噪声 rng 流 = 20260921 + seed (R21 新流, 与 20260918/19/20 不相交)
  - 教师 = v3-A (best_resnet_large_clean_baseline_v3.h5) 冻结前向, KD 只作用
    mask=1 样本 (纯噪声/腐蚀窗无 abn 标签, 不锚)
  - λ_kd 默认 1.0 (LwF λ_o=1 惯例), λ_v 默认 1.0; 迭代旋钮全部进产物名
  - 选模 = 干净 val AUC (abn 头, 与 v3-A/B0 同口径); valid val AUC 披露
packed y (N,5) = [1-y_abn, y_abn, y_valid, mask_abn, p_teacher]。

用法:
  python3 train_r21_kd.py --selftest                 # 冒烟: 单测+KD 数值+教师一致性
  python3 train_r21_kd.py --arm kd --seed 42 --smoke # 5-epoch 快训冒烟
  python3 train_r21_kd.py --arm kd --seed 42
  python3 train_r21_kd.py --arm kd --seed 42 --lambda-kd 3.0   # 迭代2
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
from models.resnet_lite_1d import build_ecg_resnet_lite_large_dualhead
from data.split_guard import get_guard, load_arrays
from noise_lib import synth_noise, chain_shape
from eval_corpus_pc import ai_hp_250
from augment_noise import augment_beats
import train_clean_baseline_v3 as V3

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"

SEED_NOISE_BASE = 20260921          # 预注册冻结: R21 新噪声流 = base + seed
TEACHER_H5 = "best_resnet_large_clean_baseline_v3.h5"
NOISE_TRAIN_COUNTS = {"motion": 600, "emg": 240, "mains": 240,
                      "baseline_wander": 60, "respiratory": 60}   # =1200
NOISE_VAL_N = 200
H8_CARRIER_TOTAL = 800              # {bw,emg,motion,mains} × {30,20}dB × 100
H8_TYPES = ("baseline_wander", "emg", "motion", "mains")
H8_SNRS = (30, 20)
EPS = 1e-6
EXPECTED_PARAMS = 62899
EXPECTED_TEACHER_PARAMS = 62834


def gen_pure_noise_windows(counts, rng):
    """全设备链复刻纯噪声窗 (与 train_r18_dualhead.py 同实现)。"""
    outs = []
    for ntype, n in counts.items():
        for _ in range(n):
            shaped = chain_shape(synth_noise(ntype, 500, 500, rng))
            w = ai_hp_250(np.asarray(shaped, dtype=np.float64))
            w = (w - w.mean()) / (w.std() + EPS)
            outs.append(w.astype(np.float32))
    return np.stack(outs)


def r21_kd_loss(lambda_v=1.0, lambda_kd=1.0):
    """packed y_true (N,5)=[1-y_abn,y_abn,y_valid,mask_abn,p_teacher] vs (N,3)。

    KD 项为二元概率 KL = p_t·log(p_t/p_s)+(1-p_t)·log((1-p_t)/(1-p_s))
    (软目标 BCE + 教师熵常数; proper scoring rule, 只作用 mask=1)。"""
    def _loss(y_true, y_pred):
        p_abn = tf.clip_by_value(y_pred[:, 1], 1e-7, 1.0 - 1e-7)
        p_val = tf.clip_by_value(y_pred[:, 2], 1e-7, 1.0 - 1e-7)
        p_t = tf.clip_by_value(y_true[:, 4], 1e-7, 1.0 - 1e-7)
        y_abn, y_val, m = y_true[:, 1], y_true[:, 2], y_true[:, 3]
        ce_abn = -(y_abn * tf.math.log(p_abn)
                   + (1.0 - y_abn) * tf.math.log(1.0 - p_abn))
        bce_v = -(y_val * tf.math.log(p_val)
                  + (1.0 - y_val) * tf.math.log(1.0 - p_val))
        kl = (p_t * (tf.math.log(p_t) - tf.math.log(p_abn))
              + (1.0 - p_t) * (tf.math.log(1.0 - p_t)
                               - tf.math.log(1.0 - p_abn)))
        return m * ce_abn + lambda_v * bce_v + lambda_kd * m * kl
    _loss.__name__ = "r21_kd_loss"
    return _loss


def load_teacher():
    t = tf.keras.models.load_model(str(MODELS / TEACHER_H5), compile=False)
    assert t.count_params() == EXPECTED_TEACHER_PARAMS, \
        f"教师参数 {t.count_params()} != {EXPECTED_TEACHER_PARAMS} (v3-A)"
    return t


def teacher_forward(teacher, x, bs=2048):
    """直接前向分块 (Keras3 GPU 小批量 predict 规避, §118.6)。返回 p_abn (N,)。"""
    outs = [np.asarray(teacher(x[i:i + bs], training=False))[:, 1]
            for i in range(0, len(x), bs)]
    return np.concatenate(outs, axis=0).astype(np.float32)


# ---------------------------------------------------------------- 冒烟自检
def selftest():
    """R21M_PREREG §6 冒烟 3/4: 模型单测 + 教师一致性 + KD 数值。"""
    print("== [selftest] 模型单测 ==")
    m = build_ecg_resnet_lite_large_dualhead(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    assert m.count_params() == EXPECTED_PARAMS, \
        f"参数 {m.count_params()} != {EXPECTED_PARAMS}"
    x = np.random.default_rng(0).normal(size=(8, BEAT_WINDOW_SAMPLES, 1)
                                         ).astype(np.float32)
    y = m(x, training=False).numpy()
    assert y.shape == (8, 3), y.shape
    assert np.allclose(y[:, 0] + y[:, 1], 1.0, atol=1e-5), "abn 两通道非 softmax"
    assert np.all((y[:, 2] >= 0) & (y[:, 2] <= 1)), "valid 非 sigmoid 域"
    print(f"  params={m.count_params():,} out={y.shape} softmax/sigmoid OK")

    print("== [selftest] 教师前向一致性 ==")
    t = load_teacher()
    a = teacher_forward(t, x)
    b = teacher_forward(t, x)
    assert np.array_equal(a, b), "教师前向非确定性"
    assert len(a) == 8 and np.all((a >= 0) & (a <= 1))
    print(f"  确定性 max|Δa-b|={np.abs(a - b).max():.1e}; 探针 p_t[:4]="
          f"{np.round(a[:4], 4).tolist()}")

    print("== [selftest] KD 数值 ==")
    # 解析手算: p_t=0.7, p_s=0.5 → KL = 0.7·ln(1.4)+0.3·ln(0.6) = 0.08224
    pt, ps = 0.7, 0.5
    expect = pt * np.log(pt / ps) + (1 - pt) * np.log((1 - pt) / (1 - ps))
    yt = tf.constant([[0.3, 0.7, 1.0, 1.0, 0.7]], dtype=tf.float32)
    yp = tf.constant([[0.5, 0.5, 0.5, 0.9, 0.5]], dtype=tf.float32)
    for lk in (1.0, 3.0):
        v = r21_kd_loss(1.0, lk)(yt, yp).numpy()
        # KD 分量隔离: 同 mask=1 样本上 λ_kd=lk 与 λ_kd=0 的差分
        v_l0 = r21_kd_loss(1.0, 0.0)(yt, yp).numpy()
        kd_component = float((v - v_l0).item())
        assert abs(kd_component - lk * expect) < 1e-6, (kd_component, expect)
        print(f"  λ_kd={lk}: KD 分量={kd_component:.6f} (期望 {lk * expect:.6f}) OK")

    # 梯度非零 (KD 通道)
    with tf.GradientTape() as tape:
        yp_v = m(x[:4], training=True)
        loss = r21_kd_loss(1.0, 1.0)(
            tf.constant(np.tile(np.array([[0.3, 0.7, 1.0, 1.0, 0.7]],
                                         np.float32), (4, 1))), yp_v)
    grads = tape.gradient(loss, m.trainable_variables)
    n_nz = sum(1 for g in grads if g is not None and float(tf.reduce_sum(
        tf.abs(g))) > 0)
    assert n_nz == len(m.trainable_variables), \
        f"{n_nz}/{len(m.trainable_variables)} 变量梯度为零"
    loss_v = float(tf.reduce_mean(loss).numpy())
    print(f"  梯度非零 {n_nz}/{len(m.trainable_variables)} 变量; "
          f"loss={loss_v:.4f} 有限无 NaN: {np.isfinite(loss_v)}")
    assert np.isfinite(loss_v)
    print("== [selftest] ALL PASS ==")


# ---------------------------------------------------------------- 训练
def run_arm(arm: str, seed: int, smoke: bool = False, lambda_v: float = 1.0,
            lambda_kd: float = 1.0, epochs: int = None, patience: int = None):
    assert arm in ("kd", "h8"), arm
    epochs = epochs or (5 if smoke else V3.EPOCHS)
    patience = patience or (5 if smoke else V3.PATIENCE)

    tag = f"r21_{arm}_seed{seed}"
    if smoke:
        tag += "_smoke"
    if lambda_kd != 1.0:
        tag += f"_lkd{lambda_kd:g}"      # 旋钮进产物名 (§118.6)
    if lambda_v != 1.0:
        tag += f"_lv{lambda_v:g}"
    out_h5 = MODELS / f"{tag}.h5"
    out_csv = MODELS / f"train_history_{tag}.csv"
    out_json = CACHE / f"train_{tag}.json"

    t0 = time.time()
    rng = np.random.default_rng(seed)               # 主流: 与 v3-A 同调用序
    seed_noise = SEED_NOISE_BASE + seed             # R21 新流 (预注册冻结)
    rng_noise = np.random.default_rng(seed_noise)
    tf.random.set_seed(seed)

    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[{tag}] GPU: {gpu}; noise_rng={seed_noise}", flush=True)

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

    # ---------- 纯噪声 (valid=0, 无 abn 标签) ----------
    noise_train = gen_pure_noise_windows(NOISE_TRAIN_COUNTS, rng_noise)
    noise_val = gen_pure_noise_windows({"motion": NOISE_VAL_N // 2,
                                        "mains": NOISE_VAL_N // 2}, rng_noise)

    # ---------- H8: 轻噪腐蚀载波 valid=1 (只进 valid 头) ----------
    h8_prov = None
    if arm == "h8":
        g_mit = get_guard("mit_bih")
        s0, s1 = g_mit.sample_train_beats(0, H8_CARRIER_TOTAL, rng_noise)
        carriers = np.asarray(load_arrays("mit_bih")[0])[s1].astype(np.float32)
        corrupt, h8_prov = augment_beats(
            carriers, rng_noise, variants_per_beat=1,
            types=H8_TYPES, snrs=H8_SNRS)
        noise_train = np.concatenate([noise_train, corrupt])

    # ---------- 组装 (packed y 5 列) ----------
    x_core = np.concatenate(x_parts)[..., np.newaxis].astype(np.float32)
    y_core = np.concatenate(y_abn_parts)
    x_noise = noise_train[..., np.newaxis].astype(np.float32)
    x_train = np.concatenate([x_core, x_noise])
    # 核心样本: valid=1, mask=1; 纯噪声: 全 0 (valid=0, mask=0);
    # H8 腐蚀载波: valid=1, mask=0 (从 noise_train 尾部按构造顺序已知数量)
    y_core_pack = np.stack([1 - y_core, y_core,
                            np.ones(len(y_core), dtype=np.float32),
                            np.ones(len(y_core), dtype=np.float32)], axis=1)
    y_noise_pack = np.zeros((len(noise_train), 4), dtype=np.float32)
    if arm == "h8":
        y_noise_pack[len(noise_train) - H8_CARRIER_TOTAL:, 2] = 1.0  # valid=1
    y_train4 = np.concatenate([y_core_pack, y_noise_pack])
    perm = rng.permutation(len(x_train))
    x_train, y_train4 = x_train[perm], y_train4[perm]
    print(f"[{tag}] train total={len(x_train)} "
          f"abn_labeled={int((y_train4[:, 3] > 0).sum())} "
          f"valid_pos={int((y_train4[:, 2] > 0).sum())} "
          f"noise_valid0={int(((y_train4[:, 3] == 0) & (y_train4[:, 2] == 0)).sum())}",
          flush=True)

    # ---------- 教师 KD 目标 (v3-A 冻结前向, 全样本计算, 损失按 mask 生效) ----------
    teacher = load_teacher()
    p_teacher = teacher_forward(teacher, x_train)
    y_train = np.concatenate([y_train4, p_teacher[:, None]], axis=1).astype(np.float32)
    probe = {"n": int(len(p_teacher)),
             "p_t_mean_mask1": round(float(p_teacher[y_train4[:, 3] > 0].mean()), 4),
             "p_t_frac_gt05_mask1": round(float(
                 (p_teacher[y_train4[:, 3] > 0] > 0.5).mean()), 4)}
    print(f"[{tag}] teacher probe: {probe}", flush=True)

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
    x_val_in = x_val[..., np.newaxis].astype(np.float32)
    y_val_pack = np.stack([1 - y_val, y_val,
                           np.ones(len(y_val), dtype=np.float32),
                           np.ones(len(y_val), dtype=np.float32)], axis=1)
    p_t_val = teacher_forward(teacher, x_val_in)
    y_val_full = np.concatenate([y_val_pack, p_t_val[:, None]],
                                axis=1).astype(np.float32)

    # ---------- 模型 / 损失 ----------
    model = build_ecg_resnet_lite_large_dualhead(
        input_shape=(BEAT_WINDOW_SAMPLES, 1))
    assert model.count_params() == EXPECTED_PARAMS
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=V3.BASE_LR,
            decay_steps=epochs * max(1, len(x_train) // V3.BATCH))),
        loss=r21_kd_loss(lambda_v, lambda_kd))
    print(f"[{tag}] params={model.count_params():,} "
          f"λ_v={lambda_v} λ_kd={lambda_kd}", flush=True)

    # ---------- 选模回调: 干净 val AUC (abn 头), valid val AUC 披露 ----------
    class ValAucCB(tf.keras.callbacks.Callback):
        def __init__(self, xv, yv, xvc, nv):
            super().__init__()
            self.xv, self.yv, self.xvc, self.nv = xv, yv, xvc, nv
            self.best, self.best_epoch = -1.0, -1

        def _fwd(self, x):
            return np.asarray(self.model(x, training=False))

        def on_epoch_end(self, epoch, logs=None):
            pred = self._fwd(self.xv)
            auc = float(roc_auc_score(self.yv, pred[:, 1]))
            (logs or {}).__setitem__("val_auc", auc)
            line = f"  val_auc={auc:.4f}"
            k = min(500, len(self.xvc))
            pv_clean = self._fwd(self.xvc[:k])[:, 2]
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

    cbs = [ValAucCB(x_val_in, y_val, x_val_core, noise_val),
           tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                            patience=patience,
                                            restore_best_weights=True, verbose=1),
           tf.keras.callbacks.CSVLogger(str(out_csv))]
    hist = model.fit(x_train, y=y_train,
                     validation_data=(x_val_in, y_val_full),
                     batch_size=V3.BATCH, epochs=epochs, callbacks=cbs,
                     verbose=2)

    # ---------- 最佳模型自检 (直接前向分块) ----------
    def _fwd_all(mm, x, bs=2048):
        outs = [np.asarray(mm(x[i:i + bs], training=False))
                for i in range(0, len(x), bs)]
        return np.concatenate(outs, axis=0)

    best = tf.keras.models.load_model(str(out_h5), compile=False)
    pv = _fwd_all(best, x_val_in)
    auc_val = float(roc_auc_score(y_val, pv[:, 1]))
    p_real = _fwd_all(best, x_real_ho[..., np.newaxis].astype(np.float32),
                      bs=64)[:, 1]
    pn_tr = _fwd_all(best, noise_train[..., np.newaxis])
    # KD 锚距离披露: mask=1 训练样本上 student vs teacher
    m1 = y_train4[:, 3] > 0
    ps_m1 = _fwd_all(best, x_train[m1])[:, 1]
    kd_mean = float(np.abs(ps_m1 - y_train[m1, 4]).mean())
    extra = {
        "noise_train_valid_mean": float(pn_tr[:, 2].mean()),
        "noise_train_abn_frac_gt_05": float((pn_tr[:, 1] > 0.5).mean()),
        "kd_abs_dist_mask1_mean": kd_mean,
        "lambda_v": lambda_v, "lambda_kd": lambda_kd,
    }
    print(f"[{tag}] BEST val AUC={auc_val:.4f}; real holdout frac>0.5="
          f"{float((p_real > 0.5).mean()):.4f}; kd|Δp|mask1={kd_mean:.4f}; "
          f"{extra}", flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": f"R21 {arm} (seed {seed})" + (" [SMOKE]" if smoke else ""),
        "prereg": "runs/R21M_PREREG.md §1-§2",
        "gpu": [str(g) for g in gpu],
        "arm": arm, "seed": seed, "smoke": bool(smoke),
        "output_model": str(out_h5.relative_to(BASE)),
        "data": {
            "main": {t: {"abn": len(domains[t][0]), "norm": len(domains[t][2])}
                     for t in domains},
            "sampling_audit": audits,
            "real_afe_holdout": int(len(x_real_ho)),
            "noise_train": {"total": int(len(noise_train)),
                            "counts": NOISE_TRAIN_COUNTS,
                            "h8_corrupted": int(H8_CARRIER_TOTAL if arm == "h8"
                                                else 0),
                            "h8_prov": h8_prov},
            "noise_val": int(len(noise_val)),
            "train_total": int(len(x_train)),
            "teacher_probe": probe,
        },
        "config": {"architecture": "build_ecg_resnet_lite_large_dualhead",
                   "params": int(model.count_params()),
                   "lambda_v": lambda_v, "lambda_kd": lambda_kd,
                   "lr": V3.BASE_LR, "epochs_requested": epochs,
                   "epochs_run": int(len(hist.epoch)), "batch": V3.BATCH,
                   "patience": patience, "noise_rng_seed": seed_noise,
                   "teacher_h5": TEACHER_H5,
                   "kd_form": "binary prob KL (soft-target BCE), mask=1 only"},
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
    ap.add_argument("--arm", default="kd", choices=["kd", "h8"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lambda-v", type=float, default=1.0)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run_arm(a.arm, a.seed, smoke=a.smoke, lambda_v=a.lambda_v,
                lambda_kd=a.lambda_kd, epochs=a.epochs, patience=a.patience)
