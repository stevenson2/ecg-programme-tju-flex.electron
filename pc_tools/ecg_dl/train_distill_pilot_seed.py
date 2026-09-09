#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_distill_pilot_seed.py -- Stage 2 multi-seed / variant training for TH 106+.

This script intentionally does NOT modify the frozen distill-pilot scripts.
It reuses the Stage 1 data package (distill_pilot_teacher_targets.npz) and the
frozen v3-A architecture/optimizer recipe, but lets the caller change only:

  * the Stage-2 random seed (model initialisation / shuffle state used by TF)
  * the KD alpha / temperature
  * an optional teacher-probability tail-preservation transform

Outputs land under models/deploy_match/ so all new artifacts are in one place.

Usage:
  python3 train_distill_pilot_seed.py --seed 42
  python3 train_distill_pilot_seed.py --seed 43 --alpha 0.2 --temperature 0.7 \
      --tail-mode sharpen --suffix v2_a020_t07
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
from losses.kd_loss import make_kd_loss, SlicedAUC
from train_clean_baseline_v3 import (
    EPOCHS, PATIENCE, BASE_LR, BATCH, build_model,
)

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
TARGETS_NPZ = CACHE / "distill_pilot_teacher_targets.npz"
TARGETS_JSON = CACHE / "distill_pilot_teacher_targets.json"
EPS = 1e-7
N_TRAIN, N_VAL, N_HOLDOUT = 5942, 4040, 40
N_PARAMS = 62834


def tail_sharpen(p, temperature):
    """Monotone tail-preserving transform of teacher probabilities.

    With temperature < 1 this is a sharpening transform applied to the
    teacher probability before the standard KL packing: the abnormal tail
    moves further toward 1.0 and the normal tail further toward 0.0.  The
    transform is strictly monotone in p, so it does not change ranking and
    therefore cannot improve AUC by itself; it only changes how strongly the
    student is pulled toward the extreme ends of the calibrated scale.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    logit = np.log(p / (1.0 - p))
    logit_sharp = logit / max(float(temperature), 1e-6)
    p_new = 1.0 / (1.0 + np.exp(-logit_sharp))
    return p_new.astype(np.float32)


def isotonic_calibrate(p, y):
    """One-dimensional isotonic calibration fit on the training split only.

    Used only as an explicit alternative in the pre-registered v2 variant;
    never touches val labels.  The fit is monotone, so ordering (AUC) is
    unchanged; it changes only the probability scale.
    """
    from sklearn.isotonic import IsotonicRegression
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso.transform(p).astype(np.float32)


def pack_kd(y, p):
    """Same packing as the frozen pilot: onehot(2) + teacher logits (2)."""
    onehot = np.eye(2, dtype=np.float32)[np.asarray(y).astype(np.int64)]
    logits = np.stack([
        np.log(np.clip(1.0 - p, EPS, 1.0)),
        np.log(np.clip(p, EPS, 1.0)),
    ], axis=1).astype(np.float32)
    return np.concatenate([onehot, logits], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kd-temperature", type=float, default=None,
                    help="KD loss temperature; defaults to --temperature")
    ap.add_argument("--tail-mode", default="none",
                    choices=["none", "sharpen", "isotonic"])
    ap.add_argument("--suffix", default=None)
    args = ap.parse_args()

    seed = int(args.seed)
    alpha = float(args.alpha)
    temperature = float(args.temperature)
    kd_temperature = float(args.kd_temperature) if args.kd_temperature is not None else temperature
    tail_mode = args.tail_mode
    suffix = args.suffix if args.suffix else f"seed{seed}"

    t0 = time.time()
    if not TARGETS_NPZ.exists() or not TARGETS_JSON.exists():
        raise SystemExit(f"[DPS] Stage 1 products missing: {TARGETS_NPZ.name} / "
                         f"{TARGETS_JSON.name}")
    stage1 = json.loads(TARGETS_JSON.read_text(encoding="utf-8"))
    if stage1.get("aborted", True):
        raise SystemExit("[DPS] Stage 1 aborted (aborted=true); refuse training")

    d = np.load(TARGETS_NPZ)
    x_train = np.asarray(d["x_train_perm"], dtype=np.float32)
    y_train = np.asarray(d["y_train"]).astype(np.int32)
    p_train_src = np.asarray(d["p_train"], dtype=np.float32)
    x_val = np.asarray(d["x_val"], dtype=np.float32)
    y_val = np.asarray(d["y_val"]).astype(np.int32)
    p_val_src = np.asarray(d["p_val"], dtype=np.float32)
    x_hold = np.asarray(d["x_real_holdout"], dtype=np.float32)

    assert x_train.shape == (N_TRAIN, 250), f"x_train {x_train.shape}"
    assert x_val.shape == (N_VAL, 250), f"x_val {x_val.shape}"
    assert x_hold.shape == (N_HOLDOUT, 250), f"x_real_holdout {x_hold.shape}"
    assert len(p_train_src) == N_TRAIN and len(p_val_src) == N_VAL
    assert float(p_train_src.min()) >= 0.0 and float(p_train_src.max()) <= 1.0
    assert float(p_val_src.min()) >= 0.0 and float(p_val_src.max()) <= 1.0

    p_train = p_train_src
    p_val = p_val_src
    transform_desc = "none"
    if tail_mode == "sharpen":
        if temperature >= 1.0:
            raise SystemExit("[DPS] sharpen requires temperature < 1.0")
        p_train = tail_sharpen(p_train_src, temperature)
        p_val = tail_sharpen(p_val_src, temperature)
        transform_desc = f"logit / {temperature}"
    elif tail_mode == "isotonic":
        # Fit only on training labels.  This is a Stage-2 target transform;
        # it is monotone and never uses val/test labels to fit.
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_train_src, y_train)
        p_train = iso.transform(p_train_src).astype(np.float32)
        p_val = iso.transform(p_val_src).astype(np.float32)
        transform_desc = "isotonic fitted on train only"

    print(f"[DPS] npz loaded: train={x_train.shape} val={x_val.shape} "
          f"holdout={x_hold.shape}; p_train mean={p_train.mean():.4f} "
          f"p_val mean={p_val.mean():.4f}", flush=True)
    print(f"[DPS] seed={seed} alpha={alpha} temperature={temperature} "
          f"kd_temperature={kd_temperature} tail_mode={tail_mode} "
          f"transform={transform_desc}", flush=True)

    tf.random.set_seed(seed)
    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("[DPS] no GPU visible; refuse default CPU training")
    print(f"[DPS] GPU: {gpu}", flush=True)

    xt = x_train[..., np.newaxis]
    xv = x_val[..., np.newaxis]
    yv = y_val
    y_kd_train = pack_kd(y_train, p_train)
    y_kd_val = pack_kd(y_val, p_val)

    model = build_model(0.4)
    n_params = model.count_params()
    if n_params != N_PARAMS:
        raise SystemExit(f"[DPS] params {n_params} != {N_PARAMS}; architecture drift")
    print(f"[DPS] params={n_params:,} (random init; architecture identical to v3 A)",
          flush=True)

    out_h5 = CACHE / f"best_resnet_large_distill_pilot_{suffix}.h5"
    out_csv = CACHE / f"train_history_distill_pilot_{suffix}.csv"
    out_json = CACHE / f"train_distill_pilot_{suffix}.json"

    steps_per_epoch = max(1, len(xt) // BATCH)
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=BASE_LR, decay_steps=EPOCHS * steps_per_epoch)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=make_kd_loss(alpha=alpha, temperature=kd_temperature),
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
                self.model.save(str(out_h5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})",
                      flush=True)

    cbs = [
        ValAucCB(xv, yv),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out_csv)),
    ]

    print(f"[DPS] start KD training: epochs={EPOCHS} batch={BATCH} "
          f"lr={BASE_LR} cosine, alpha={alpha} T={kd_temperature}, "
          f"tail_mode={tail_mode}, no class_weight", flush=True)
    hist = model.fit(
        xt, y_kd_train,
        validation_data=(xv, y_kd_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=2,
    )
    print(f"[DPS] training done in {time.time() - t0:.0f}s", flush=True)

    best = tf.keras.models.load_model(str(out_h5), compile=False)
    p_val_hat = best.predict(xv, batch_size=256, verbose=0)[:, 1]
    p_real = best.predict(x_hold[..., np.newaxis], batch_size=64,
                          verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val_hat))
    print(f"[DPS] BEST: val AUC={auc_val:.4f}; real holdout "
          f"mean={p_real.mean():.4f} frac>0.5={float((p_real > 0.5).mean()):.4f}",
          flush=True)

    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH106+ Stage 2 multi-seed/variant training (new script, no frozen edit)",
        "seed": seed,
        "provenance": "from scratch; no checkpoint loaded; Stage1 package unchanged",
        "gpu": [str(g) for g in gpu],
        "output_model": str(out_h5.relative_to(BASE)),
        "teacher_targets": {
            "source_npz": str(TARGETS_NPZ.relative_to(BASE)),
            "source_json": str(TARGETS_JSON.relative_to(BASE)),
            "loro_auc_public": stage1["loro"]["auc_public"],
            "real_synth_teacher_target_swap": stage1["real_synth"]["teacher_target_swap"],
            "real_synth_mean_p_before_swap": stage1["real_synth"]["mean_p_before_swap"],
        },
        "kd": {
            "alpha": alpha,
            "temperature": kd_temperature,
            "teacher_tail_transform_temperature": temperature,
            "tail_mode": tail_mode,
            "transform": transform_desc,
            "packing": "y_kd = concat([onehot(2), [log(clip(1-p,1e-7)), "
                       "log(clip(p,1e-7))]], axis=1)",
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
            "p_train_mean_raw": float(p_train_src.mean()),
            "p_train_mean_transformed": float(p_train.mean()),
            "p_val_mean_raw": float(p_val_src.mean()),
            "p_val_mean_transformed": float(p_val.mean()),
        },
        "config": {
            "architecture": "build_ecg_resnet_lite_large (dropout 0.4, v3 A same)",
            "params": int(n_params),
            "optimizer": "adam",
            "lr_schedule": f"cosine decay from {BASE_LR}",
            "epochs_requested": EPOCHS,
            "epochs_run": int(len(hist.epoch)),
            "batch_size": BATCH,
            "early_stopping_patience": PATIENCE,
            "val_monitor": "val_auc (ValAucCB predict path, same as v3)",
            "loss": f"kd_loss(alpha={alpha}, temperature={kd_temperature})",
            "teacher_tail_transform": transform_desc,
        },
        "results": {
            "best_val_auc": auc_val,
            "real_holdout_mean_prob": float(p_real.mean()),
            "real_holdout_frac_gt_0.5": float((p_real > 0.5).mean()),
            "train_time_s": round(time.time() - t0, 1),
        },
        "known_limitations": [
            "val double-use: early stopping + evaluation share val; all numbers are relative indicators",
            "teacher domain offset: tile view 0.8893 vs true deploy 10s 0.7794",
            "pseudo-target contains noise (LORO AUC 0.88, not oracle)",
            "multi-seed only changes Stage-2 init; Stage-1 data/teacher package is fixed",
        ],
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[DPS] saved {out_h5.name}, {out_csv.name}, {out_json.name}",
          flush=True)


if __name__ == "__main__":
    main()
