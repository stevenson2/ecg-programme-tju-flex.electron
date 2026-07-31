#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp5 vs P2A: MIT测试集 + PTB独立测试的多阈值工作点"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import load_mit_incart_merged, train_val_test_split, add_channel_dim

MODELS = Path(__file__).resolve().parent / "models"

# MIT+INCART 测试集
data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_mit, y_mit = splits["test"]

# PTB 独立测试集 (患者级留出)
PTB_NPZ = Path(__file__).resolve().parent / "data" / "processed" / "ptb_processed.npz"
RECORDS = next((Path(c) for c in [
    r"C:\Users\cai\OneDrive\Desktop\ecg-programme-tju-flex.electron-master\ECG-Database\RECORDS",
    "/mnt/c/Users/cai/OneDrive/Desktop/ecg-programme-tju-flex.electron-master/ECG-Database/RECORDS",
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

cands = [
    ("exp5 PTB受控", MODELS / "best_resnet_large.h5"),
    ("P2A 部署", MODELS / "archived" / "final_resnet_l_p2a_backup.h5"),
]
thresholds = [0.35, 0.50, 0.65, 0.80]


def metrics(y, prob, th):
    pred = (prob >= th).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1


for name, p in cands:
    m = tf.keras.models.load_model(str(p), compile=False)
    p_mit = m.predict(add_channel_dim(x_mit), verbose=0)[:, 1]
    p_ptb = m.predict(add_channel_dim(x_ptb), verbose=0)[:, 1]
    print(f"\n=== {name} ===")
    print(f"  MIT测试 AUC {roc_auc_score(y_mit, p_mit):.4f} | "
          f"PTB测试 AUC {roc_auc_score(y_ptb, p_ptb):.4f}")
    print(f"  {'θ':<6} {'MIT-P':<7} {'MIT-R':<7} {'MIT-F1':<7} | {'PTB-P':<7} {'PTB-R':<7} {'PTB-F1':<7}")
    for th in thresholds:
        pm, rm, fm = metrics(y_mit, p_mit, th)
        pp, rp, fp_ = metrics(y_ptb, p_ptb, th)
        print(f"  {th:<6} {pm:<7.3f} {rm:<7.3f} {fm:<7.3f} | {pp:<7.3f} {rp:<7.3f} {fp_:<7.3f}")
