#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_v6_recipe.py -- v6a fine-tune recipe.

Loads the 100k balanced mined dataset and fine-tunes the frozen v3-A baseline
(all layers trainable) with hard-label CE.  Validation/early stopping uses the
existing v3-A val split (distill_pilot_teacher_targets.npz x_val/y_val).

No test contact.
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

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
MODEL_BASE = MODELS / "best_resnet_large_clean_baseline_v3.h5"
DATA_NPZ = CACHE / "v6a_train_100k.npz"
VAL_NPZ = CACHE / "distill_pilot_teacher_targets.npz"

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--anchor", action="store_true",
                    help="append the original v3-A train split as an anchor")
    ap.add_argument("--scratch", action="store_true",
                    help="train from scratch (do not load v3 A weights)")
    ap.add_argument("--dataset", default="v6a_train_100k.npz")
    ap.add_argument("--suffix", default="v6a_finetune_v3a_100k")
    args = ap.parse_args()

    DATA_NPZ = CACHE / args.dataset

    BASE_LR = float(args.lr)
    EPOCHS = int(args.epochs)
    PATIENCE = int(args.patience)
    BATCH = int(args.batch)
    OUT_H5 = CACHE / f"best_{args.suffix}.h5"
    OUT_JSON = CACHE / f"train_{args.suffix}.json"

    t0 = time.time()
    tf.random.set_seed(SEED)
    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("[V6] no GPU visible; refuse CPU training")
    print(f"[V6] GPU: {gpu}", flush=True)

    d = np.load(DATA_NPZ)
    x_tr = np.asarray(d["x"], dtype=np.float32)[..., np.newaxis]
    y_tr = np.asarray(d["y"], dtype=np.int32)
    del d
    if args.anchor:
        v2 = np.load(VAL_NPZ)
        x_anchor = np.asarray(v2["x_train_perm"], dtype=np.float32)[..., np.newaxis]
        y_anchor = np.asarray(v2["y_train"], dtype=np.int32)
        x_tr = np.concatenate([x_tr, x_anchor], axis=0)
        y_tr = np.concatenate([y_tr, y_anchor], axis=0)
        print(f"[V6] anchor: added {len(y_anchor)} original v3-A train beats", flush=True)
    print(f"[V6] train loaded: x={x_tr.shape} n={len(y_tr)} "
          f"norm={int((y_tr==0).sum())} abn={int((y_tr==1).sum())}", flush=True)

    v = np.load(VAL_NPZ)
    x_val = np.asarray(v["x_val"], dtype=np.float32)[..., np.newaxis]
    y_val = np.asarray(v["y_val"], dtype=np.int32)
    print(f"[V6] val loaded: x={x_val.shape} n={len(y_val)}", flush=True)

    if args.scratch:
        from train_clean_baseline_v3 import build_model
        model = build_model(0.4)
        print(f"[V6] built v3 A architecture from scratch, params={model.count_params():,}", flush=True)
    else:
        model = tf.keras.models.load_model(str(MODEL_BASE), compile=False)
        print(f"[V6] loaded v3 A baseline, params={model.count_params():,}", flush=True)
    model.trainable = True

    steps_per_epoch = max(1, len(x_tr) // BATCH)
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=BASE_LR, decay_steps=EPOCHS * steps_per_epoch)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class ValAucCB(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = -1.0
            self.best_epoch = -1

        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(x_val, batch_size=256, verbose=0)[:, 1]
            auc = float(roc_auc_score(y_val, p))
            logs = logs or {}
            logs["val_auc"] = auc
            if auc > self.best:
                self.best = auc
                self.best_epoch = epoch + 1
                self.model.save(str(OUT_H5))
                print(f"  * saved best val_auc={auc:.4f} (epoch {epoch + 1})",
                      flush=True)

    cbs = [
        ValAucCB(),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=PATIENCE,
                                         restore_best_weights=True, verbose=1),
    ]
    print(f"[V6] start fine-tune: epochs={EPOCHS} batch={BATCH} "
          f"lr={BASE_LR} cosine, CE, no aug", flush=True)
    hist = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=2,
    )
    print(f"[V6] training done in {time.time()-t0:.0f}s", flush=True)

    best = tf.keras.models.load_model(str(OUT_H5), compile=False)
    p_val = best.predict(x_val, batch_size=256, verbose=0)[:, 1]
    auc_val = float(roc_auc_score(y_val, p_val))
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "v6a fast fine-tune recipe",
        "recipe": "fine-tune v3 A on 100k balanced mined hard-normal data",
        "seed": SEED,
        "gpu": [str(g) for g in gpu],
        "data": {
            "npz": str(DATA_NPZ.relative_to(BASE)),
            "train_total": int(len(y_tr)),
            "train_normal": int((y_tr == 0).sum()),
            "train_abnormal": int((y_tr == 1).sum()),
            "val_total": int(len(y_val)),
        },
        "config": {
            "batch": BATCH,
            "epochs_requested": EPOCHS,
            "epochs_run": int(len(hist.epoch)),
            "patience": PATIENCE,
            "lr": BASE_LR,
            "loss": "sparse_categorical_crossentropy",
            "augmentation": None,
        },
        "results": {
            "best_val_auc": auc_val,
            "best_epoch_auc": float(np.max([logs.get("val_auc", 0) for logs in hist.history.values()]) if False else 0.0),
            "train_time_s": round(time.time() - t0, 1),
        },
        "model": str(OUT_H5.relative_to(BASE)),
        "zero_test_contact": True,
    }
    # Put simple history summary in results.
    result["results"]["val_auc_history"] = [float(v) for v in hist.history.get("val_auc", [])]
    result["results"]["epochs"] = len(hist.history.get("val_auc", []))
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[V6] saved {OUT_H5.name} + {OUT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
