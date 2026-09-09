#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_leakage.py — T3-6/M10: 记录级划分泄漏审计 (PTB 训练/测试患者交集)
======================================================================
任务: 必做清单 T3-6 ② / solutions.md M10
目标: 统计**历史记录级划分** (rng.choice 无放回, 旧 eval 脚本语义) 下,
      PTB 测试拍中来自训练患者的比例 — 论文 §4.2 方法句的数据支撑
方法:
  1. 历史语义: seed=42, 从 PTB 记录中 choice 20% 作测试 (记录级, 无患者隔离)
  2. 患者映射: RECORDS 文件名 patientXXX/ 前缀
  3. 泄漏统计: 测试记录的患者 ∩ 训练/验证患者 → 测试拍数中泄漏拍比例
  4. 复算 solutions.md L52 引用的 ~17% 数字
输出: models/leakage_audit.json
用法 (WSL): python3 audit_leakage.py
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR
from data.patient_split import build_ptb_patient_map, patient_level_split

MODELS_DIR = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS_DIR / "leakage_audit.json"
SEED = 42
TEST_FRAC = 0.2

RECORDS = next((Path(c) for c in [
    r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\ECG-Database\RECORDS",
    "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/ECG-Database/RECORDS",
] if Path(c).exists()), None)
if RECORDS is None:
    raise RuntimeError("RECORDS 文件未找到")

recs = [l.strip() for l in open(RECORDS) if l.strip()]
n_rec = len(recs)
print(f"PTB 记录总数: {n_rec}")

# 历史泄漏语义: 记录级 choice (旧 eval 脚本)
rng = np.random.default_rng(SEED)
n_test_rec = max(1, int(n_rec * TEST_FRAC))
test_recs = set(rng.choice(recs, n_test_rec, replace=False))
train_val_recs = set(recs) - test_recs
print(f"记录级划分: test {len(test_recs)} 条, train+val {len(train_val_recs)} 条")

# 患者映射 (文件名前缀)
def patient_of(rec):
    return rec.split("/")[0]

test_pats = {patient_of(r) for r in test_recs}
train_val_pats = {patient_of(r) for r in train_val_recs}
leak_pats = test_pats & train_val_pats
print(f"测试患者 {len(test_pats)} 个, 其中与训练患者交集 {len(leak_pats)} 个")

# 拍级泄漏: PTB 测试拍中来自"训练患者"的记录
d = np.load(PROCESSED_DIR / "ptb_processed.npz")
beats, labels, rids = d["beats"], d["labels"], d["record_ids"]
rec_to_idx = {rec: i for i, rec in enumerate(recs)}
leak_mask = np.zeros(len(rids), dtype=bool)
test_mask = np.zeros(len(rids), dtype=bool)
for rid in np.unique(rids):
    rec = recs[int(rid) - 400000]
    m = rids == rid
    if rec in test_recs:
        test_mask |= m
        if patient_of(rec) in leak_pats:
            leak_mask |= m

n_test = int(test_mask.sum())
n_leak = int((leak_mask & test_mask).sum())
frac = n_leak / n_test if n_test else 0.0
print(f"测试拍: {n_test:,} | 其中来自训练患者(泄漏): {n_leak:,} ({frac*100:.1f}%)")

# 患者级划分对照 (清洁版: 泄漏应为 0)
pmap = build_ptb_patient_map()
tr, va, te, stats = patient_level_split(rids, pmap)
clean_leak = int(((te) & (np.array([pmap.get(int(r), "") in set(stats["train_patients"]) | set(stats["val_patients"]) for r in rids]))).sum())
print(f"患者级划分 (清洁版): 训练/验证患者拍泄漏进测试 = {clean_leak} 拍 (应为 0)")

output = {
    "meta": {
        "date": "2026-08-05",
        "task": "T3-6/M10 泄漏审计",
        "method": "历史记录级划分 (seed42, choice 20% 记录作测试, 无患者隔离) vs "
                  "患者级划分 (patient_level_split, 60/20/20)",
        "source": "RECORDS 患者前缀 + ptb_processed.npz 拍级映射",
    },
    "record_level": {
        "n_records": n_rec, "n_test_records": len(test_recs),
        "n_test_patients": len(test_pats),
        "n_leak_patients": len(leak_pats),
        "n_test_beats": int(n_test), "n_leak_beats": int(n_leak),
        "leak_fraction": round(frac, 4),
        "verdict": "与 solutions.md L52 引用的 ~17% 一致" if abs(frac - 0.17) < 0.05 else "需复核",
    },
    "patient_level": {
        "n_leak_beats_clean": int(clean_leak),
        "verdict": "清洁版无泄漏" if clean_leak == 0 else "存在泄漏!",
    },
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n✅ 已保存: {OUT_JSON}")
