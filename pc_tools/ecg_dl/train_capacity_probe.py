#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_capacity_probe.py — 容量/架构探针 Stage 2: 从零重训更大架构 (TH §109)
================================================================================
按 capacity_probe_prereg.json 锁定的变体, 从零训练更大/更深/更宽架构,
检验"v3 A (62,834 参数) 从零容量不足"是否为 fc1 嵌入不可分离与 val 侧
正常拍高置信误报的结构性原因。

铁律:
  - 数据只来自 distill_pilot_teacher_targets.npz 的 x_train_perm / y_train /
    x_val / y_val / x_real_holdout; 不读取 p_train/p_val (教师软目标禁用);
    测试集零接触。
  - 随机初始化从零训练; 不加载任何 checkpoint/.h5/权重; 无 KD。
  - 训练配置与 v3 A 对齐: Adam + CosineDecay(3e-4, 80 epochs), batch 32,
    patience 20, monitor val_auc, restore best, 无 class_weight, dropout 0.4。
  - 全部产物落 models/deploy_match/; 日志标记 [CAP]。

运行 (WSL, stdout 重定向):
  python3 train_capacity_probe.py cap_hybrid:42 cap_w2:42 cap_d2:42 \
      > models/deploy_match/capacity_probe_train.log 2>&1
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
from models.resnet_lite_1d import build_ecg_resnet_lite

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
TARGETS_NPZ = CACHE / "distill_pilot_teacher_targets.npz"
PREREG = CACHE / "capacity_probe_prereg.json"
ANCHOR_JSON = CACHE / "train_clean_baseline_v3.json"

EPOCHS = 80
PATIENCE = 20
BASE_LR = 3e-4
BATCH = 32

# v3 A 锚点恒等检查 (来自 train_clean_baseline_v3.json, 盘上 artifact)
ANCHOR = {
    "train_total": 5942, "train_abn": 2500, "train_norm": 3442,
    "val_total": 4040, "holdout": 40,
}


def load_variant_spec(name):
    reg = json.loads(PREREG.read_text(encoding="utf-8"))
    if name not in reg["variants"]:
        raise SystemExit(f"[CAP] unknown variant {name!r}, "
                         f"expected one of {list(reg['variants'])}")
    return reg["variants"][name]


def build_model(spec):
    return build_ecg_resnet_lite(
        input_shape=(BEAT_WINDOW_SAMPLES, 1),
        filters=tuple(spec["filters"]),
        blocks_per_stage=tuple(spec["blocks_per_stage"]),
        kernel_sizes=tuple(spec["kernel_sizes"]),
        strides=tuple(spec["strides"]),
        dropout_rate=spec["dropout"],
    )


def run_variant(variant, seed):
    spec = load_variant_spec(variant)
    out_h5 = CACHE / f"best_resnet_capacity_{variant}_seed{seed}.h5"
    out_csv = CACHE / f"train_history_capacity_{variant}_seed{seed}.csv"
    out_json = CACHE / f"train_capacity_{variant}_seed{seed}.json"
    if out_h5.exists():
        print(f"[CAP] {variant} seed{seed}: h5 已存在, 跳过 (防覆盖)", flush=True)
        return

    t0 = time.time()
    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("[CAP] 无 GPU 可见; 禁止静默退回 CPU 训练")
    print(f"[CAP] {variant} seed{seed} GPU: {gpu}", flush=True)

    # ---------- 数据: 只认 npz x/y/holdout, 不读教师 p_* ----------
    d = np.load(TARGETS_NPZ)
    x_train = np.asarray(d["x_train_perm"], dtype=np.float32)
    y_train = np.asarray(d["y_train"], dtype=np.int32)
    x_val = np.asarray(d["x_val"], dtype=np.float32)
    y_val = np.asarray(d["y_val"], dtype=np.int32)
    x_ho = np.asarray(d["x_real_holdout"], dtype=np.float32)
    # 教师目标 p_train/p_val 存在于 npz 但本脚本显式不读取 (禁 KD);
    # 只允许消费下列键:
    used = {"x_train_perm", "y_train", "x_val", "y_val", "x_real_holdout"}
    assert used.issubset(set(d.files)) and \
        set(d.files) - used == {"p_train", "p_val", "group_ids", "train_domain"}, \
        f"[CAP] npz 键集异常: {d.files}"
    assert x_train.shape == (ANCHOR["train_total"], BEAT_WINDOW_SAMPLES)
    assert x_val.shape == (ANCHOR["val_total"], BEAT_WINDOW_SAMPLES)
    assert x_ho.shape == (ANCHOR["holdout"], BEAT_WINDOW_SAMPLES)
    assert int((y_train == 1).sum()) == ANCHOR["train_abn"]
    assert int((y_train == 0).sum()) == ANCHOR["train_norm"]
    print(f"[CAP] {variant} seed{seed} data identity PASS: "
          f"train={len(x_train)} (abn={int((y_train==1).sum())}, "
          f"norm={int((y_train==0).sum())}), val={len(x_val)}, "
          f"real_holdout={len(x_ho)}", flush=True)

    # ---------- 模型: 随机初始化, 无预训练/KD ----------
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = build_model(spec)
    params = int(model.count_params())
    assert params == spec["params_frozen"], \
        f"[CAP] 参数量 {params} 与预注册 {spec['params_frozen']} 不一致"
    print(f"[CAP] {variant} seed{seed} params={params:,} "
          f"(随机初始化, 未加载任何权重, 无 KD; 预注册一致)", flush=True)

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
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})",
                      flush=True)

    cbs = [
        ValAucCB(x_val, y_val),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out_csv)),
    ]

    print(f"[CAP] {variant} seed{seed} start from-scratch training: "
          f"epochs={EPOCHS} batch={BATCH} lr={BASE_LR} cosine, "
          f"no class_weight", flush=True)
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=2,
    )
    print(f"[CAP] {variant} seed{seed} training done in {time.time()-t0:.0f}s",
          flush=True)

    # ---------- 最佳模型自检 (重载盘上 h5) ----------
    best = tf.keras.models.load_model(str(out_h5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    p_ho = best.predict(x_ho, batch_size=64, verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val))
    print(f"[CAP] {variant} seed{seed} BEST: val AUC={auc_val:.4f} "
          f"(epoch {cbs[0].best_epoch}); real holdout mean={p_ho.mean():.4f} "
          f"frac>0.5={float((p_ho > 0.5).mean()):.4f}", flush=True)

    anchor = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": f"TH §109 容量探针: {spec['description']} (从零, 无 KD)",
        "provenance": "from scratch; no checkpoint loaded; no teacher targets",
        "gpu": [str(g) for g in gpu],
        "variant": variant,
        "variant_spec": spec,
        "seed": seed,
        "output_model": str(out_h5.relative_to(BASE)),
        "data_source": str(TARGETS_NPZ.relative_to(BASE)),
        "identity_anchor_vs_v3A": {
            "train_total": int(len(x_train)), "train_abn": int((y_train == 1).sum()),
            "train_norm": int((y_train == 0).sum()), "val_total": int(len(x_val)),
            "real_holdout": int(len(x_ho)),
            "v3A_anchor": ANCHOR, "match": True,
        },
        "config": {
            "architecture": "build_ecg_resnet_lite",
            "params": params,
            "dropout": spec["dropout"],
            "optimizer": "adam",
            "lr_schedule": f"cosine decay from {BASE_LR}",
            "epochs_requested": EPOCHS,
            "epochs_run": int(len(hist.epoch)),
            "batch_size": BATCH,
            "early_stopping_patience": PATIENCE,
            "val_monitor": "val_auc",
            "class_weight": None,
            "loss": "sparse_categorical_crossentropy",
            "v3A_baseline_params": int(anchor["config"]["params"]),
        },
        "results": {
            "best_val_auc": auc_val,
            "best_epoch": int(cbs[0].best_epoch),
            "real_holdout_mean_prob": float(p_ho.mean()),
            "real_holdout_frac_gt_0.5": float((p_ho > 0.5).mean()),
            "train_time_s": round(time.time() - t0, 1),
        },
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[CAP] {variant} seed{seed} saved {out_h5.name}, {out_csv.name}, "
          f"{out_json.name}", flush=True)


if __name__ == "__main__":
    specs = sys.argv[1:]
    if not specs:
        raise SystemExit("usage: train_capacity_probe.py <variant>:<seed> ...")
    for s in specs:
        v, sd = s.split(":")
        run_variant(v, int(sd))
