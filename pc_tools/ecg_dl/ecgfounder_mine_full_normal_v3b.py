#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ecgfounder_mine_full_normal_v3b.py — v3b 全量 PTB-XL 正常拍 hard normal 挖掘
================================================================================
对 PTB-XL human-validated 且仅含正常节律标签的记录，用 exp7c 部署链提取拍，
再用 v3b QAT INT8 评分，挑出高异常分的“误报正常拍”作为 hard normal。

输出：
  models/ecgfounder/full_normal_scores_v3b.npy
  models/ecgfounder/full_normal_hard_v3b.npy
  models/ecgfounder/full_normal_hard_meta.json
"""
import sys, json, time, argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import OpResolverType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import corrected_deployment_chain, extract_beats_deploy

ROOT = Path(__file__).resolve().parents[2]
PTBXL_DIR = ROOT / "PTB-XL_ECG"
OUT_DIR = Path(__file__).resolve().parent / "models" / "ecgfounder"
MODEL = Path(__file__).resolve().parent / "models" / "ecg_model_exp7c_ecgfounder_v3b_qat_int8.tflite"
NORMAL_CODES = {"NORM", "SR", "SBRAD", "STACH", "SARRH"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-threshold", type=float, default=0.7)
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd, ast, wfdb
    from wfdb.processing import xqrs_detect

    db = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
    db = db[db["validated_by_human"] == True].copy()
    records = []
    for _, r in db.iterrows():
        codes = set(ast.literal_eval(r["scp_codes"]).keys())
        if codes <= NORMAL_CODES:
            records.append(r)
    if args.max_records:
        records = records[:args.max_records]
    print(f"[MINE] full normal records: {len(records)}", flush=True)

    # TFLite
    it = tf.lite.Interpreter(model_path=str(MODEL),
                             experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                             num_threads=1)
    it.allocate_tensors()
    in_d = it.get_input_details()[0]
    out_d = it.get_output_details()[0]
    in_scale = float(in_d["quantization_parameters"]["scales"].flatten()[0])
    in_zp = int(in_d["quantization_parameters"]["zero_points"].flatten()[0])
    out_scale = float(out_d["quantization_parameters"]["scales"].flatten()[0])
    out_zp = int(out_d["quantization_parameters"]["zero_points"].flatten()[0])

    all_beats = []
    all_scores = []
    all_meta = []
    t0 = time.time()
    for i, r in enumerate(records):
        fn = r["filename_hr"]
        try:
            rd = wfdb.rdrecord(str(PTBXL_DIR / fn))
            lead = rd.p_signal[:, 1].astype(np.float64)
            chain = corrected_deployment_chain(lead, rd.fs)
            r_idx = xqrs_detect(chain.astype(np.float64), fs=250, verbose=False)
            beats = extract_beats_deploy(chain, np.asarray(r_idx, dtype=np.int64), "ptb")
            for b in beats:
                xq = np.clip(np.round(b.astype(np.float32)[None, :, None] / in_scale) + in_zp,
                             -128, 127).astype(np.int8)
                it.set_tensor(in_d["index"], xq)
                it.invoke()
                q = it.get_tensor(out_d["index"])[0]
                p = (q.astype(np.float32) - out_zp) * out_scale
                all_beats.append(b)
                all_scores.append(float(p[1]))
            all_meta.append({
                "ecg_id": int(r["ecg_id"]),
                "patient_id": str(r["patient_id"]),
                "filename_hr": fn,
                "n_beats": int(len(beats)),
            })
        except Exception as e:
            print(f"  [skip] {fn}: {e}", flush=True)
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(records)}] {time.time()-t0:.0f}s, beats={len(all_beats)}", flush=True)

    beats = np.stack(all_beats).astype(np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    np.save(OUT_DIR / "full_normal_beats_v3b.npy", beats)
    np.save(OUT_DIR / "full_normal_scores_v3b.npy", scores)
    hard_idx = np.where(scores >= args.hard_threshold)[0]
    np.save(OUT_DIR / "full_normal_hard_v3b.npy", beats[hard_idx])
    json.dump({
        "n_records": len(all_meta),
        "n_beats": int(len(scores)),
        "n_hard": int(len(hard_idx)),
        "hard_threshold": args.hard_threshold,
        "mean_score": float(scores.mean()),
        "max_score": float(scores.max()),
        "frac_hard": float(len(hard_idx) / max(1, len(scores))),
        "records": all_meta,
    }, open(OUT_DIR / "full_normal_hard_meta.json", "w"), indent=2, ensure_ascii=False)
    print(f"[MINE] saved full normal: beats={len(scores)}, hard={len(hard_idx)} "
          f"({len(hard_idx)/max(1,len(scores))*100:.2f}%), elapse={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
