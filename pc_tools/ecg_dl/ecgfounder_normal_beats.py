#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecgfounder_normal_beats.py — 提取与真实 AFE 最相似的 PTB-XL 正常记录拍
================================================================================
作用：从 ECGFounder 距离表中选出距真实 AFE 最近的公共正常记录，
按 exp7c 部署链提取 Lead II 250 点拍，作为真实域正常/硬负补充候选。

输出：
  pc_tools/ecg_dl/models/ecgfounder/real_like_normal_beats.npy
  pc_tools/ecg_dl/models/ecgfounder/real_like_normal_beats_meta.json
"""
import sys, json, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import corrected_deployment_chain, extract_beats_deploy

ROOT = Path(__file__).resolve().parents[2]
PTBXL_DIR = ROOT / "PTB-XL_ECG"
OUT_DIR = Path(__file__).resolve().parent / "models" / "ecgfounder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=100, help="取距离最近的正常记录数")
    args = ap.parse_args()

    df = pd.read_csv(OUT_DIR / "hard_negative_candidates.csv")
    norm = df[df["label"] == "normal"].sort_values("dist_min").head(args.topk)
    print(f"[NORMAL] top {len(norm)} normal candidates", flush=True)

    import wfdb
    from wfdb.processing import xqrs_detect

    all_beats, all_meta, failed = [], [], 0
    for i, (_, rec) in enumerate(norm.iterrows()):
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
                "ecg_id": int(rec["ecg_id"]),
                "patient_id": str(rec["patient_id"]),
                "filename_hr": fn,
                "n_beats": int(len(beats)),
                "dist_min": float(rec["dist_min"]),
                "dist_mean": float(rec["dist_mean"]),
                "scp_codes": rec["scp_codes"],
                "label": "normal_public",
            })
        except Exception as e:
            failed += 1
            print(f"  [skip] {fn}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(norm)}] {time.time():.0f}s", flush=True)

    beats = np.concatenate(all_beats, axis=0).astype(np.float32)
    np.save(OUT_DIR / "real_like_normal_beats.npy", beats)
    with open(OUT_DIR / "real_like_normal_beats_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_beats": int(len(beats)),
            "n_records": len(all_meta),
            "failed": failed,
            "chain": "corrected_deployment_chain + XQRS @250 + extract_beats_deploy(ptb)",
            "records": all_meta,
        }, f, ensure_ascii=False, indent=2)
    print(f"[NORMAL] saved {beats.shape} -> {OUT_DIR / 'real_like_normal_beats.npy'}")


if __name__ == "__main__":
    main()
