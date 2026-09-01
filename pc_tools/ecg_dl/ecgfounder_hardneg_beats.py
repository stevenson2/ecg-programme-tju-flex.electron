#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecgfounder_hardneg_beats.py — 将 ECGFounder 硬负样本候选映射回单导联 250 点拍
================================================================================
从 hard_negative_candidates.json 中读取 top PTB-XL 异常记录，使用 exp7c 一致的
corrected_deployment_chain + XQRS + extract_beats_deploy 提取 Lead II 拍窗口，
保存为可供后训练使用的弱标签异常拍数据。

注意：
  这些拍来自记录级全局节律标签，不能视为逐拍金标准；只作为弱标签/辅助数据。
  禁止直接将本输出冒充真实 AFE 异常数据。

输出：
  pc_tools/ecg_dl/models/ecgfounder/hardneg_beats.npy
  pc_tools/ecg_dl/models/ecgfounder/hardneg_beats_meta.json
"""
import sys, json, time, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import corrected_deployment_chain, extract_beats_deploy
from config import BEAT_WINDOW_SAMPLES

ROOT = Path(__file__).resolve().parents[2]
PTBXL_DIR = ROOT / "PTB-XL_ECG"
OUT_DIR = Path(__file__).resolve().parent / "models" / "ecgfounder"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=60, help="使用前 N 个异常候选")
    args = ap.parse_args()

    with open(OUT_DIR / "hard_negative_candidates.json", "r", encoding="utf-8") as f:
        cand = json.load(f)
    records = cand["hard_negative_top"][:args.topk]
    print(f"[HARDNEG] top {len(records)} records", flush=True)

    import wfdb
    from wfdb.processing import xqrs_detect

    all_beats, all_meta = [], []
    failed = 0
    for i, rec in enumerate(records):
        fn = rec["filename_hr"]
        try:
            rd = wfdb.rdrecord(str(PTBXL_DIR / fn))
            lead = rd.p_signal[:, 1].astype(np.float64)
            chain = corrected_deployment_chain(lead, rd.fs)
            r_idx = xqrs_detect(chain.astype(np.float64), fs=250, verbose=False)
            beats = extract_beats_deploy(chain, np.asarray(r_idx, dtype=np.int64), "ptb")
            if len(beats) == 0:
                failed += 1
                continue
            all_beats.append(beats)
            all_meta.append({
                "ecg_id": rec["ecg_id"],
                "patient_id": rec["patient_id"],
                "filename_hr": fn,
                "n_beats": int(len(beats)),
                "dist_min": rec["dist_min"],
                "dist_mean": rec["dist_mean"],
                "scp_codes": rec["scp_codes"],
                "label": "abnormal_weak",
            })
        except Exception as e:
            failed += 1
            print(f"  [skip] {fn}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(records)}] {time.time():.0f}s", flush=True)

    beats = np.concatenate(all_beats, axis=0).astype(np.float32)
    np.save(OUT_DIR / "hardneg_beats.npy", beats)
    with open(OUT_DIR / "hardneg_beats_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_beats": int(len(beats)),
            "n_records": len(all_meta),
            "failed": failed,
            "chain": "corrected_deployment_chain + XQRS @250 + extract_beats_deploy(ptb)",
            "label_meaning": "record-level global rhythm abnormal -> weak beat-level abnormal; not gold",
            "records": all_meta,
        }, f, ensure_ascii=False, indent=2)
    print(f"[HARDNEG] saved {beats.shape} -> {OUT_DIR / 'hardneg_beats.npy'}, failed={failed}")


if __name__ == "__main__":
    main()
