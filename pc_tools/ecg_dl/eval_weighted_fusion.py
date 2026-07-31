#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加权分数融合: P = α·P2A + (1-α)·exp5
网格搜索 α, 目标: 双域 Recall 高 + MIT 误报率低 (治报警疲劳)
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import load_mit_incart_merged, train_val_test_split, add_channel_dim

MODELS = Path(__file__).resolve().parent / "models"

data = load_mit_incart_merged()
splits = train_val_test_split(data["beats"], data["labels"], record_ids=data["record_ids"])
x_mit, y_mit = splits["test"]

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

m_p2a = tf.keras.models.load_model(str(MODELS / "archived" / "final_resnet_l_p2a_backup.h5"), compile=False)
m_exp5 = tf.keras.models.load_model(str(MODELS / "best_resnet_large.h5"), compile=False)
p2a_mit = m_p2a.predict(add_channel_dim(x_mit), verbose=0)[:, 1]
exp5_mit = m_exp5.predict(add_channel_dim(x_mit), verbose=0)[:, 1]
p2a_ptb = m_p2a.predict(add_channel_dim(x_ptb), verbose=0)[:, 1]
exp5_ptb = m_exp5.predict(add_channel_dim(x_ptb), verbose=0)[:, 1]


def metrics(y, prob, th):
    pred = (prob >= th).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    fp_rate = fp / max(1, (y == 0).sum())
    return prec, rec, f1, fp_rate


print("α(P2A权重) 网格: 每个 α 取最优阈值 (MIT F1 优先)")
print(f"{'α':<6} {'θ*':<6} | {'MIT-P':<7} {'MIT-R':<7} {'MIT-F1':<7} {'MIT误报':<7} | "
      f"{'PTB-P':<7} {'PTB-R':<7} {'PTB-F1':<7} {'PTB误报':<7}")
best = None
for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    f_mit = alpha * p2a_mit + (1 - alpha) * exp5_mit
    f_ptb = alpha * p2a_ptb + (1 - alpha) * exp5_ptb
    best_row = None
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        pm, rm, fm, fpm = metrics(y_mit, f_mit, th)
        if best_row is None or fm > best_row[0]:
            pp, rp, fp_, fpp = metrics(y_ptb, f_ptb, th)
            best_row = (fm, th, pm, rm, fpm, pp, rp, fp_, fpp)
    fm, th, pm, rm, fpm, pp, rp, fp_, fpp = best_row
    print(f"{alpha:<6} {th:<6} | {pm:<7.3f} {rm:<7.3f} {fm:<7.3f} {fpm:<7.3f} | "
          f"{pp:<7.3f} {rp:<7.3f} {fp_:<7.3f} {fpp:<7.3f}")
    if best is None or fm > best[0]:
        best = (fm, alpha, th)
print(f"\n最优 (按 MIT F1): α={best[1]}, θ={best[2]}, MIT F1={best[0]:.3f}")
