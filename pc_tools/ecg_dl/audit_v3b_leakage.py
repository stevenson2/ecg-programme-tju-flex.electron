#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_v3b_leakage.py — v3b QAT 训练数据患者级泄漏审计
================================================================================
核查 qat_exp7c_v3b.py 中 load_domain 的随机抽样是否混入了 MIT/INCART/PTB
测试/验证患者。这是 AGENTS.md §8 强制的患者级泄漏检查。
"""
import sys, json, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map, build_ptb_patient_map,
    patient_level_split,
)
from config import BEAT_WINDOW_SAMPLES

ECG_DATA = Path("/home/devcontainers/ecg_data")
OUT = Path(__file__).resolve().parent / "models" / "deploy_match" / "v3b_leakage_audit.json"
SEED = 42

# 与 qat_exp7c_v3b.py 完全一致的抽样参数
SPEC = {
    "mit_bih": (1200, 400),
    "incart": (300, 100),
    "ptb": (500, 150),
}
# 注意 v3b 原脚本是 (1200,400),(300,100),(500,150)
# 这里以 v3b 为准

def load_arrays(tag):
    suffix = "_processed_deploy_causal"
    b = np.load(ECG_DATA / f"{tag}{suffix}_beats.npy", mmap_mode="r")
    l = np.load(ECG_DATA / f"{tag}{suffix}_labels.npy", mmap_mode="r")
    r = np.load(ECG_DATA / f"{tag}{suffix}_record_ids.npy", mmap_mode="r")
    return b, l, r

def sample_with_rng(tag, n_abn, n_norm, rng):
    b, l, r = load_arrays(tag)
    ia = np.where(l == 1)[0]
    inn = np.where(l == 0)[0]
    sa = rng.choice(ia, min(n_abn, len(ia)), replace=False)
    sn = rng.choice(inn, min(n_norm, len(inn)), replace=False)
    return r[sa], r[sn]

def main():
    rng = np.random.default_rng(SEED)
    leakage = {}
    # MIT/INCART 患者划分 (合并同事件脚本口径)
    mit_map = build_mit_patient_map()
    inc_map = build_incart_patient_map()

    # 构建合并 record_ids 用于 split，使用 npy 实际记录编号
    mit_b, mit_l, mit_r = load_arrays("mit_bih")
    inc_b, inc_l, inc_r = load_arrays("incart")
    merged_rids = np.concatenate([mit_r, inc_r + 100000])
    pmap = {}
    pmap.update(mit_map)
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in inc_map.items()})
    tr_m, va_m, te_m, stats = patient_level_split(merged_rids, pmap, seed=SEED)
    test_rids = set(np.unique(merged_rids[te_m]).tolist())
    val_rids = set(np.unique(merged_rids[va_m]).tolist())
    train_rids = set(np.unique(merged_rids[tr_m]).tolist())

    print("[LEAK] MIT+INCART patients:", stats["n_patients"],
          "train/val/test:", stats["n_train"], stats["n_val"], stats["n_test"])
    print("[LEAK] test records:", len(test_rids), "val records:", len(val_rids))

    for tag in ["mit_bih", "incart"]:
        r_abn, r_norm = sample_with_rng(tag, *SPEC[tag], rng)
        all_sel = np.concatenate([r_abn, r_norm])
        rid_key = all_sel if tag == "mit_bih" else all_sel + 100000
        n_test = int(np.isin(rid_key, list(test_rids)).sum())
        n_val = int(np.isin(rid_key, list(val_rids)).sum())
        n_train = int(np.isin(rid_key, list(train_rids)).sum())
        print(f"[LEAK] {tag}: sampled={len(all_sel)} train={n_train} val={n_val} test={n_test}")
        leakage[tag] = {
            "sampled": int(len(all_sel)),
            "train": n_train, "val": n_val, "test": n_test,
            "test_ratio": float(n_test / len(all_sel)),
            "leaked": n_test > 0,
        }

    # PTB 患者划分
    ptb_b, ptb_l, ptb_r = load_arrays("ptb")
    ptb_map = build_ptb_patient_map()
    tr_p, va_p, te_p, pstats = patient_level_split(ptb_r, ptb_map, seed=SEED)
    ptb_test_rids = set(np.unique(ptb_r[te_p]).tolist())
    ptb_val_rids = set(np.unique(ptb_r[va_p]).tolist())
    ptb_train_rids = set(np.unique(ptb_r[tr_p]).tolist())
    r_abn, r_norm = sample_with_rng("ptb", *SPEC["ptb"], rng)
    all_sel = np.concatenate([r_abn, r_norm])
    n_test = int(np.isin(all_sel, list(ptb_test_rids)).sum())
    n_val = int(np.isin(all_sel, list(ptb_val_rids)).sum())
    n_train = int(np.isin(all_sel, list(ptb_train_rids)).sum())
    print(f"[LEAK] ptb: sampled={len(all_sel)} train={n_train} val={n_val} test={n_test}")
    leakage["ptb"] = {
        "sampled": int(len(all_sel)), "train": n_train, "val": n_val, "test": n_test,
        "test_ratio": float(n_test / len(all_sel)), "leaked": n_test > 0,
    }

    result = {
        "audit_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Replicate qat_exp7c_v3b load_domain random sampling with seed=42, map sampled record_ids to patient-level split",
        "patient_split": stats,
        "ptb_patient_split": {
            "n_patients": pstats["n_patients"],
            "n_train": pstats["n_train"], "n_val": pstats["n_val"], "n_test": pstats["n_test"],
            "test_patients": pstats["test_patients"],
        },
        "leakage": leakage,
        "verdict": "LEAKED" if any(v.get("leaked") for v in leakage.values()) else "CLEAN",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("[LEAK] verdict:", result["verdict"], "saved", OUT)

if __name__ == "__main__":
    main()
