#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者级数据划分 (4.4-4 发表级严谨性)
====================================
为 MIT-BIH / INCART / PTB 三个数据集构建 record_id -> patient_id 映射,
并按患者分组划分 train/val/test, 保证同一患者的所有记录/心拍
不跨划分 (消除记录级划分的泄漏)。

关键事实 (2026-08 确认):
  - MIT-BIH: 48 记录 / 47 患者; 仅 201/202 同患者 (官方记录202注明
    "taken from the same analog tape as record 201")
  - INCART: 75 记录 / 32 患者 (.hea 头文件 "# patient N" 字段, 患者1~32;
    部分患者多条记录, 如 I29-I32 同属患者14)
  - PTB: 549 记录 / 290 患者 (RECORDS 文件名 patientXXX/ 前缀)

划分: seed=42, 患者级 60/20/20 (train/val/test), 与历史 eval 脚本同 seed。
用法:
  python data/patient_split.py --save          # 生成并保存划分
  python data/patient_split.py --print-maps    # 打印映射
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR

SPLIT_FILE = PROCESSED_DIR / "patient_split.json"
SEED = 42
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

# ---------------- 路径解析 ----------------
ROOT = Path(__file__).resolve().parents[3]   # 项目根
ECG_DIR = ROOT / "ECG-Database"
INCART_DIR = ROOT / "st-petersburg-incart-12-lead-arrhythmia-database-1.0.0" / "files"
MIT_DIR = PROCESSED_DIR  # 使用预处理后的 npz 中 record_ids


def build_mit_patient_map() -> dict:
    """MIT-BIH: record_id -> patient_id. 47 患者, 201/202 同患者."""
    from data.dataset import load_processed_data
    d = load_processed_data()
    rec_ids = np.unique(d["record_ids"])
    pmap = {}
    for rid in rec_ids:
        if int(rid) in (201, 202):
            pmap[int(rid)] = "mit_201_202"
        else:
            pmap[int(rid)] = "mit_%d" % int(rid)
    return pmap


def build_incart_patient_map() -> dict:
    """INCART: record_id -> patient_id. 解析 .hea '# patient N'."""
    pmap = {}
    if not INCART_DIR.exists():
        return pmap
    for hea in sorted(INCART_DIR.glob("*.hea")):
        rid = int(hea.stem[1:])          # I01 -> 1
        content = hea.read_text(encoding="latin1")
        m = re.search(r"#\s*patient\s+([^\r\n]+)", content)
        pat = m.group(1).strip() if m else "unknown"
        pmap[rid] = "inc_%s" % pat
    return pmap


def build_ptb_patient_map() -> dict:
    """PTB: record_id(400000+i) -> patientXXX 前缀 (RECORDS 文件第 i 行)."""
    recs_file = ECG_DIR / "RECORDS"
    if not recs_file.exists():
        return {}
    recs = [l.strip() for l in open(recs_file) if l.strip()]
    pmap = {}
    for i, rec in enumerate(recs):
        pat = rec.split("/")[0]
        pmap[400000 + i] = "ptb_%s" % pat
    return pmap


def patient_level_split(record_ids: np.ndarray, pmap: dict, seed: int = SEED):
    """按患者分组划分. 返回 (train_mask, val_mask, test_mask)."""
    rid_to_pat = {}
    for rid in np.unique(record_ids):
        rid = int(rid)
        pat = pmap.get(rid, "unknown_%d" % rid)
        rid_to_pat[rid] = pat

    patients = sorted(set(rid_to_pat.values()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patients))
    patients = [patients[i] for i in perm]

    n = len(patients)
    n_test = max(1, int(n * TEST_FRAC))
    n_val = max(1, int(n * VAL_FRAC))
    n_train = n - n_test - n_val

    test_pats = set(patients[:n_test])
    val_pats = set(patients[n_test:n_test + n_val])
    train_pats = set(patients[n_test + n_val:])

    pat_of_beat = np.array([rid_to_pat.get(int(r), "unknown_%d" % int(r))
                            for r in record_ids])
    train_mask = np.array([p in train_pats for p in pat_of_beat])
    val_mask = np.array([p in val_pats for p in pat_of_beat])
    test_mask = np.array([p in test_pats for p in pat_of_beat])

    stats = {
        "n_patients": n,
        "n_train": len(train_pats), "n_val": len(val_pats), "n_test": len(test_pats),
        "beats_train": int(train_mask.sum()),
        "beats_val": int(val_mask.sum()),
        "beats_test": int(test_mask.sum()),
        "test_patients": sorted(test_pats),
        "val_patients": sorted(val_pats),
        "train_patients": sorted(train_pats),
    }
    return train_mask, val_mask, test_mask, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="保存划分 JSON")
    ap.add_argument("--print-maps", action="store_true", help="打印患者映射")
    args = ap.parse_args()

    mit_map = build_mit_patient_map()
    inc_map = build_incart_patient_map()
    ptb_map = build_ptb_patient_map()
    print(f"MIT-BIH 患者: {len(set(mit_map.values()))} (记录 {len(mit_map)})")
    print(f"INCART 患者: {len(set(inc_map.values()))} (记录 {len(inc_map)})")
    print(f"PTB 患者: {len(set(ptb_map.values()))} (记录 {len(ptb_map)})")

    if args.print_maps:
        print("\nINCART 映射 (患者 -> 记录):")
        rev = {}
        for rid, pat in inc_map.items():
            rev.setdefault(pat, []).append("I%02d" % rid)
        for pat, recs in sorted(rev.items()):
            print("  %-8s %s" % (pat, recs))
        print("\nMIT-BIH 共享患者:", [r for r in mit_map if r in (201, 202)])

    # 对每个数据集构建患者级划分
    result = {}
    for name, pmap in [("mit", mit_map), ("incart", inc_map), ("ptb", ptb_map)]:
        if not pmap:
            continue
        # 用记录 ID 集合构造"每条记录一拍"的假数组, 只统计患者分组
        rec_ids = np.array(sorted(pmap.keys()), dtype=np.int64)
        tr, va, te, stats = patient_level_split(rec_ids, pmap, seed=SEED)
        result[name] = stats
        print(f"\n[{name}] 患者划分: train {stats['n_train']} / val {stats['n_val']}"
              f" / test {stats['n_test']} (总 {stats['n_patients']})")
        print(f"        拍数(全量): train {stats['beats_train']} / "
              f"val {stats['beats_val']} / test {stats['beats_test']}")

    if args.save:
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SPLIT_FILE, "w", encoding="utf-8") as f:
            json.dump({"seed": SEED, "frac": {"train": TRAIN_FRAC, "val": VAL_FRAC,
                                              "test": TEST_FRAC},
                       "datasets": result}, f, indent=2, ensure_ascii=False)
        print(f"\n划分已保存: {SPLIT_FILE}")


if __name__ == "__main__":
    main()
