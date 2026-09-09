#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_distill_pilot.py — 蒸馏试点 Stage 2: KD 重训 (TH §106)
================================================================================
载入 Stage 1 一次性数据包 (distill_pilot_targets.py 产出, 本脚本零 rng),
以教师软伪概率 + 硬标签混合目标 (α=0.5) 从零重训与 v3 配置 a 完全相同的
62,834 参数架构。

打包决定 (预注册): 伪 logits [log(1−p), log(p)] + temperature=1.0 ——
softmax(log[1−p, p]) 严格还原教师概率; T²=1 使损失
  L = (1−α)·CE + α·KL(p_teacher ‖ y_pred)
可直接对账。拒绝 T>1: 教师输出是已校准概率, 除 T 压平置信度无依据。

与 v3 逐字一致项: build_model(0.4) / Adam + CosineDecay(3e-4, 80×185) /
batch 32 / epochs 80 / EarlyStopping(val_auc, patience 20, restore_best) /
ValAucCB (仅输出路径不同) / CSVLogger。不传 class_weight (v3 配置 a 的
{0:1,1:1} 是恒等, 且与打包 y_true 不兼容)。

铁律: 测试集零接触; 门与裁定见 distill_pilot_prereg.json (val 侧)。

输出:
  models/best_resnet_large_distill_pilot.h5
  models/train_history_distill_pilot.csv
  models/deploy_match/train_distill_pilot.json
运行环境: WSL + GPU (禁止静默退回 CPU)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses.kd_loss import make_kd_loss, SlicedAUC
from train_clean_baseline_v3 import (
    SEED, EPOCHS, PATIENCE, BASE_LR, BATCH, build_model,
)

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
TARGETS_NPZ = CACHE / "distill_pilot_teacher_targets.npz"
TARGETS_JSON = CACHE / "distill_pilot_teacher_targets.json"
OUT_H5 = MODELS / "best_resnet_large_distill_pilot.h5"
OUT_CSV = MODELS / "train_history_distill_pilot.csv"
OUT_JSON = CACHE / "train_distill_pilot.json"

KD_ALPHA = 0.5
KD_TEMPERATURE = 1.0
EPS = 1e-7
N_TRAIN, N_VAL, N_HOLDOUT = 5942, 4040, 40
N_PARAMS = 62834


def pack_kd(y, p):
    """y_true = concat([onehot(2), teacher_logits(2)], axis=1) → (n, 4)。
    teacher_logits = [log(clip(1−p)), log(clip(p))]; T=1 时 softmax 严格还原
    [1−p, p]。"""
    onehot = np.eye(2, dtype=np.float32)[np.asarray(y).astype(np.int64)]
    logits = np.stack([
        np.log(np.clip(1.0 - p, EPS, 1.0)),
        np.log(np.clip(p, EPS, 1.0)),
    ], axis=1).astype(np.float32)
    return np.concatenate([onehot, logits], axis=1).astype(np.float32)


def main():
    t0 = time.time()
    if not TARGETS_NPZ.exists() or not TARGETS_JSON.exists():
        raise SystemExit(f"[DP2] Stage 1 产物缺失: {TARGETS_NPZ.name} / "
                         f"{TARGETS_JSON.name} (先跑 distill_pilot_targets.py)")
    stage1 = json.loads(TARGETS_JSON.read_text(encoding="utf-8"))
    if stage1.get("aborted", True):
        raise SystemExit("[DP2] Stage 1 已中止 (aborted=true); 拒绝训练")

    d = np.load(TARGETS_NPZ)
    x_train = np.asarray(d["x_train_perm"], dtype=np.float32)
    y_train = np.asarray(d["y_train"]).astype(np.int32)
    p_train = np.asarray(d["p_train"], dtype=np.float32)
    x_val = np.asarray(d["x_val"], dtype=np.float32)
    y_val = np.asarray(d["y_val"]).astype(np.int32)
    p_val = np.asarray(d["p_val"], dtype=np.float32)
    x_hold = np.asarray(d["x_real_holdout"], dtype=np.float32)
    assert x_train.shape == (N_TRAIN, 250), f"x_train {x_train.shape}"
    assert x_val.shape == (N_VAL, 250), f"x_val {x_val.shape}"
    assert x_hold.shape == (N_HOLDOUT, 250), f"x_real_holdout {x_hold.shape}"
    assert len(p_train) == N_TRAIN and len(p_val) == N_VAL
    assert float(p_train.min()) >= 0.0 and float(p_train.max()) <= 1.0
    assert float(p_val.min()) >= 0.0 and float(p_val.max()) <= 1.0
    print(f"[DP2] npz loaded: train={x_train.shape} val={x_val.shape} "
          f"holdout={x_hold.shape}; p_train mean={p_train.mean():.4f} "
          f"p_val mean={p_val.mean():.4f}", flush=True)

    tf.random.set_seed(SEED)
    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[DP2] GPU: {gpu}", flush=True)

    xt = x_train[..., np.newaxis]
    xv = x_val[..., np.newaxis]
    yv = y_val
    y_kd_train = pack_kd(y_train, p_train)
    y_kd_val = pack_kd(y_val, p_val)

    model = build_model(0.4)
    n_params = model.count_params()
    if n_params != N_PARAMS:
        raise SystemExit(f"[DP2] 参数量 {n_params} != {N_PARAMS}; 架构漂移")
    print(f"[DP2] params={n_params:,} (随机初始化, 未加载任何权重; "
          f"架构与 v3 配置 a 完全相同)", flush=True)

    steps_per_epoch = max(1, len(xt) // BATCH)
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=BASE_LR, decay_steps=EPOCHS * steps_per_epoch)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=make_kd_loss(alpha=KD_ALPHA, temperature=KD_TEMPERATURE),
        metrics=[SlicedAUC(name="sliced_auc")],
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
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})",
                      flush=True)

    cbs = [
        ValAucCB(xv, yv),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(OUT_CSV)),
    ]

    print(f"[DP2] start KD training: epochs={EPOCHS} batch={BATCH} "
          f"lr={BASE_LR} cosine, alpha={KD_ALPHA} T={KD_TEMPERATURE}, "
          f"no class_weight", flush=True)
    hist = model.fit(
        xt, y_kd_train,
        validation_data=(xv, y_kd_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=2,
    )
    print(f"[DP2] training done in {time.time() - t0:.0f}s", flush=True)

    best = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p_val_hat = best.predict(xv, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_hold[..., np.newaxis], batch_size=64,
                          verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val_hat))
    print(f"[DP2] BEST: val AUC={auc_val:.4f}; real holdout "
          f"mean={p_real.mean():.4f} frac>0.5={float((p_real > 0.5).mean()):.4f}",
          flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §106 蒸馏试点 Stage 2: 教师 tile LORO 软伪概率 KD 重训 "
                   "(与 v3 配置 a 同架构同优化器); 全部裁定门在 val 侧, "
                   "测试集零接触",
        "provenance": "from scratch; no checkpoint loaded",
        "gpu": [str(g) for g in gpu],
        "output_model": str(OUT_H5.relative_to(BASE)),
        "teacher_targets": {
            "source_npz": str(TARGETS_NPZ.relative_to(BASE)),
            "source_json": str(TARGETS_JSON.relative_to(BASE)),
            "loro_auc_public": stage1["loro"]["auc_public"],
            "real_synth_teacher_target_swap":
                stage1["real_synth"]["teacher_target_swap"],
            "real_synth_mean_p_before_swap":
                stage1["real_synth"]["mean_p_before_swap"],
            "identity_anchors": "见 distill_pilot_teacher_targets.json "
                                "identity_checks (全部 PASS, 对照 "
                                "train_clean_baseline_v3.json)",
        },
        "kd": {
            "alpha": KD_ALPHA,
            "temperature": KD_TEMPERATURE,
            "packing": "y_kd = concat([onehot(2), [log(clip(1-p,1e-7)), "
                       "log(clip(p,1e-7))]], axis=1); T=1 时 softmax 严格还原 "
                       "[1-p, p]",
            "packing_rationale": "教师输出是已校准概率; T=1 使损失 "
                                 "(1-α)CE + α·KL(p_teacher||y_pred) 可直接对账; "
                                 "拒绝 T>1 (无依据压平置信度)",
            "eps": EPS,
            "class_weight": None,
        },
        "data": {
            "train_total": int(len(x_train)),
            "train_abn": int((y_train == 1).sum()),
            "train_norm": int((y_train == 0).sum()),
            "val_total": int(len(x_val)),
            "val_abn": int((y_val == 1).sum()),
            "val_norm": int((y_val == 0).sum()),
            "real_holdout": int(len(x_hold)),
            "p_train_mean": float(p_train.mean()),
            "p_val_mean": float(p_val.mean()),
        },
        "config": {
            "architecture": "build_ecg_resnet_lite_large (dropout 0.4, "
                            "与 v3 配置 a 相同)",
            "params": int(n_params),
            "optimizer": "adam",
            "lr_schedule": f"cosine decay from {BASE_LR}",
            "epochs_requested": EPOCHS,
            "epochs_run": int(len(hist.epoch)),
            "batch_size": BATCH,
            "early_stopping_patience": PATIENCE,
            "val_monitor": "val_auc (ValAucCB predict 口径, 与 v3 相同)",
            "loss": f"kd_loss(alpha={KD_ALPHA}, temperature={KD_TEMPERATURE})",
        },
        "results": {
            "best_val_auc": auc_val,
            "real_holdout_mean_prob": float(p_real.mean()),
            "real_holdout_frac_gt_0.5": float((p_real > 0.5).mean()),
            "train_time_s": round(time.time() - t0, 1),
        },
        "known_limitations": [
            "val 双重使用: 早停监控与试点门共用 val 患者 → 一切数字是相对指标",
            "教师域偏移事实: 真 10s 读出 0.7794 未过 0.80 GO 线; 本试点教的是 "
            "tile 视图 (0.8893)",
            "伪目标含噪 (LORO AUC≈0.89 非 1): α=0.5 保留硬标签锚",
            "单种子运行; 多种子复跑为后续工作",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[DP2] saved {OUT_H5.name}, {OUT_CSV.name}, {OUT_JSON.name}",
          flush=True)


if __name__ == "__main__":
    main()
