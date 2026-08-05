#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB 独立测试: exp5(见过PTB) vs P2A(未见过PTB) 在 PTB 拍上的检测能力
按患者级留出 20% 患者作为 PTB 测试集 (同一患者的多条记录不跨划分)
注意: exp5 训练时按拍随机抽了 10K PTB 异常, 测试患者的部分拍可能见过
      (seed42 抽 10000/59066≈17%), 结果存在轻微泄漏, 但方向性结论可靠
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import add_channel_dim

MODELS = Path(__file__).resolve().parent / "models"
PTB_NPZ = Path(__file__).resolve().parent / "data" / "processed" / "ptb_processed.npz"
RECORDS = next((Path(c) for c in [
    r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\ECG-Database\RECORDS",
    "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/ECG-Database/RECORDS",
] if Path(c).exists()), None)
if RECORDS is None:
    raise RuntimeError("RECORDS 文件未找到")

# 1. 构建 patient -> record_ids 映射 (RECORDS[i] = patientXXX/record)
recs = [l.strip() for l in open(RECORDS) if l.strip()]
d = np.load(PTB_NPZ)
x, y, rids = d["beats"], d["labels"], d["record_ids"]
rid_to_rec = {400000 + i: recs[i] for i in range(len(recs))}
rec_to_patient = {}
for rec in recs:
    pat = rec.split("/")[0]
    rec_to_patient.setdefault(pat, []).append(rec)

# 2. 患者级留出 20% 测试患者
patients = sorted(rec_to_patient.keys())
rng = np.random.default_rng(42)
n_test = max(1, int(len(patients) * 0.2))
test_pats = set(rng.choice(patients, n_test, replace=False))
test_recs = set()
for p in test_pats:
    test_recs.update(rec_to_patient[p])
print(f"患者总数 {len(patients)}, 测试患者 {len(test_pats)}, 测试记录 {len(test_recs)}")

test_mask = np.array([rid_to_rec.get(int(r), "") in test_recs for r in rids])
x_test, y_test = x[test_mask], y[test_mask]
print(f"PTB 测试拍: {len(x_test)} (N={(y_test==0).sum()}, A={(y_test==1).sum()})")

x_in = add_channel_dim(x_test)
models = [
    # 注意: 通用名 best_resnet_large.h5 现为 exp6 权重 (4.4-4 权重对比证实),
    # exp5 真权重 = best_resnet_large_exp5_ptb_capped.h5
    ("exp5 PTB受控", MODELS / "best_resnet_large_exp5_ptb_capped.h5"),
    ("P2A 部署", MODELS / "archived" / "final_resnet_l_p2a_backup.h5"),
]
for name, p in models:
    if not p.exists():
        print(f"{name}: 不存在 ({p})")
        continue
    m = tf.keras.models.load_model(str(p), compile=False)
    prob = m.predict(x_in, verbose=0)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(f"\n{name}: PTB测试 AUC {auc:.4f}")
    print(f"  P(abn|PTB正常)={prob[y_test==0].mean():.4f}  "
          f"P(abn|PTB异常)={prob[y_test==1].mean():.4f}")
