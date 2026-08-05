#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双专家模型 TFLite INT8 导出 + 量化损失评估
  1. P2A (心律失常) + exp5 (MI) -> INT8 TFLite
  2. 校准集: MIT+INCART train + PTB 拍 (双域覆盖)
  3. FP32 vs INT8 对比: MIT 测试集 + PTB 独立测试集 (AUC/R@0.5/R@0.35)
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
from data.dataset import (
    load_mit_incart_merged, load_ptb_data, train_val_test_split, add_channel_dim,
)

MODELS = Path(__file__).resolve().parent / "models"
PTB_NPZ = MODELS.parent / "data" / "processed" / "ptb_processed.npz"

# ---------- 1. 校准集 (双域混合) ----------
data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_train = splits["train"][0]
rng = np.random.default_rng(0)
idx = rng.choice(len(x_train), 700, replace=False)
calib = [x_train[i:i + 1][..., np.newaxis].astype(np.float32) for i in idx]

ptb = load_ptb_data()
idx_ptb = rng.choice(len(ptb["beats"]), 300, replace=False)
calib += [ptb["beats"][i:i + 1][..., np.newaxis].astype(np.float32) for i in idx_ptb]


def rep_ds():
    for s in calib:
        yield [s]


# ---------- 2. 导出 INT8 ----------
MODELS = [
    ("P2A", MODELS_DIR / "archived" / "final_resnet_l_p2a_backup.h5",
     MODELS_DIR / "ecg_model_p2a_int8.tflite"),
    # ★ exp5 用 val_auc 最优 checkpoint (双域评估一致的模型, 非最终权重)
    ("exp5", MODELS_DIR / "best_resnet_large_exp5_ptb_capped.h5",
     MODELS_DIR / "ecg_model_exp5_int8.tflite"),
]

for name, h5, out in MODELS:
    print(f"\n=== 导出 {name}: {h5.name} ===")
    if not h5.exists():
        print(f"  跳过 (不存在): {h5}")
        continue
    model = tf.keras.models.load_model(str(h5), compile=False)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_ds
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()
    out.write_bytes(tflite)
    print(f"  saved: {out.name} ({out.stat().st_size/1024:.1f} KB)")


# ---------- 3. FP32 vs INT8 评估 (双域) ----------
def load_tflite(path):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    in_d = it.get_input_details()[0]
    out_d = it.get_output_details()[0]
    return it, in_d, out_d


def tflite_predict(it, in_d, out_d, x):
    xs = (x / float(in_d['quantization_parameters']['scales'].flatten()[0])
          + int(in_d['quantization_parameters']['zero_points'].flatten()[0]))
    xs = xs.astype(np.int8)
    out = np.zeros((len(x), 2), dtype=np.float32)
    for i in range(len(x)):
        it.set_tensor(in_d['index'], xs[i:i + 1])
        it.invoke()
        o = it.get_tensor(out_d['index'])
        o = (o.astype(np.float32) - float(out_d['quantization_parameters']['zero_points'].flatten()[0])) \
            * float(out_d['quantization_parameters']['scales'].flatten()[0])
        out[i] = o
    return out[:, 1]


# PTB 独立测试集 (患者级留出, 与 eval_ptb_holdout 一致)
RECORDS = next((Path(c) for c in [
    r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\ECG-Database\RECORDS",
    "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/ECG-Database/RECORDS",
] if Path(c).exists()), None)
recs = [l.strip() for l in open(RECORDS) if l.strip()]
d = np.load(PTB_NPZ)
x_ptb, y_ptb, rids = d["beats"], d["labels"], d["record_ids"]
rec_to_patient = {}
for rec in recs:
    rec_to_patient.setdefault(rec.split("/")[0], []).append(rec)
patients = sorted(rec_to_patient.keys())
rng = np.random.default_rng(42)
n_test = max(1, int(len(patients) * 0.2))
test_pats = set(rng.choice(patients, n_test, replace=False))
test_recs = set()
for p in test_pats:
    test_recs.update(rec_to_patient[p])
test_mask = np.array([recs[int(r) - 400000] in test_recs for r in rids])
x_ptb, y_ptb = x_ptb[test_mask], y_ptb[test_mask]

x_mit, y_mit = splits["test"][0], splits["test"][1]
x_mit_in = add_channel_dim(x_mit)
x_ptb_in = add_channel_dim(x_ptb)


def at(y, prob, th):
    pred = (prob >= th).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    return tp / max(1, tp + fn), tp / max(1, tp + fp)


for name, h5, out in MODELS:
    if not out.exists():
        continue
    print(f"\n=== {name}: FP32 vs INT8 ===")
    m32 = tf.keras.models.load_model(str(h5), compile=False)
    it, in_d, out_d = load_tflite(out)
    for tag, x, y in [("MIT", x_mit_in, y_mit), ("PTB", x_ptb_in, y_ptb)]:
        p32 = m32.predict(x, verbose=0)[:, 1]
        p8 = tflite_predict(it, in_d, out_d, x)
        auc32 = roc_auc_score(y, p32)
        auc8 = roc_auc_score(y, p8)
        r32, pr32 = at(y, p32, 0.5)
        r8, pr8 = at(y, p8, 0.5)
        r35_8, pr35_8 = at(y, p8, 0.35)
        print(f"  {tag}: FP32 AUC {auc32:.4f} R@.5 {r32:.3f} P {pr32:.3f} | "
              f"INT8 AUC {auc8:.4f} R@.5 {r8:.3f} P {pr8:.3f} R@.35 {r35_8:.3f} | "
              f"AUC 损失 {auc32-auc8:+.4f}")
