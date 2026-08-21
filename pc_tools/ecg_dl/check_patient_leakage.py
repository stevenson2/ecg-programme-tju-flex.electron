#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_patient_leakage.py — 跨架构训练患者级无泄漏检查
================================================================
用与 train_cross_arch.py 完全相同的 patient_level_split (seed=42)
验证：
  1. MIT+INCART 合并数据的 train/val/test 患者集合两两不相交；
  2. PTB 训练侧只取 train 患者，train 与 PTB val/test 患者不相交；
  3. baseline / deploy 链的 record_ids 一致（同患者分区）。

用法:
  python3 check_patient_leakage.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data.dataset as dataset
from data.patient_split import (
    build_mit_patient_map,
    build_incart_patient_map,
    build_ptb_patient_map,
    patient_level_split,
)


def patient_set(record_ids, indices, pmap):
    pats = set()
    for rid in np.unique(record_ids[indices]):
        rid_i = int(rid)
        pats.add(pmap.get(rid_i, f"unknown_{rid_i}"))
    return pats


def check_split(name, tr_idx, va_idx, te_idx, rids, pmap):
    tr_pats = patient_set(rids, tr_idx, pmap)
    va_pats = patient_set(rids, va_idx, pmap)
    te_pats = patient_set(rids, te_idx, pmap)
    print(f"\n[{name}]")
    print(f"  患者数: train={len(tr_pats)} val={len(va_pats)} test={len(te_pats)}")
    print(f"  train∩val = {sorted(tr_pats & va_pats)}")
    print(f"  train∩test= {sorted(tr_pats & te_pats)}")
    print(f"  val∩test  = {sorted(va_pats & te_pats)}")
    if tr_pats & va_pats or tr_pats & te_pats or va_pats & te_pats:
        raise SystemExit(f"[FAIL] {name} 患者集合存在重叠")
    return tr_pats, va_pats, te_pats


def main():
    chain = "baseline"
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        chain = "deploy"
        dataset.set_npz_suffix("_deploy")
    else:
        dataset.set_npz_suffix("")

    print("=" * 70)
    print(f"患者级无泄漏检查 (chain={chain})")
    print("=" * 70)

    # ---- MIT+INCART 合并 ----
    data = dataset.load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    tr, va, te, stats = patient_level_split(data["record_ids"], pmap)
    print(f"\n患者级划分: {stats}")
    tr_pats, va_pats, te_pats = check_split(
        "MIT+INCART", tr, va, te, data["record_ids"], pmap)

    # ---- PTB ----
    ptb = dataset.load_ptb_data()
    ptb_pmap = build_ptb_patient_map()
    ptr, pva, pte, pstats = patient_level_split(ptb["record_ids"], ptb_pmap)
    print(f"\nPTB 患者级划分: {pstats}")
    check_split("PTB", ptr, pva, pte, ptb["record_ids"], ptb_pmap)

    # ---- deploy chain record_ids 一致性（只对比 record_ids，不重复加载大数组）----
    if chain == "baseline":
        dataset.set_npz_suffix("_deploy")
        data_d = dataset.load_mit_incart_merged()
        ptb_d = dataset.load_ptb_data()
        same_mit = np.array_equal(np.sort(data_d["record_ids"]), np.sort(data["record_ids"]))
        same_ptb = np.array_equal(np.sort(ptb_d["record_ids"]), np.sort(ptb["record_ids"]))
        print(f"\n[chain 一致性] MIT+INCART record_ids baseline==deploy: {same_mit}")
        print(f"[chain 一致性] PTB record_ids baseline==deploy: {same_ptb}")
        if not same_mit or not same_ptb:
            raise SystemExit("[FAIL] baseline/deploy record_ids 不一致，患者分区可能错位")
        dataset.set_npz_suffix("")

    print("\n[PASS] 患者级无泄漏检查通过")
    print("train/val/test 患者集合两两不相交；PTB 训练只使用 train 患者。")


if __name__ == "__main__":
    main()