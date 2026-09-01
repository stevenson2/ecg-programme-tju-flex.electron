#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_event_grid_from_cache.py — 使用缓存概率细扫 θ × alert_cooldown
================================================================================
读取 v3b/exp7c 在全量 reduced 序列上的因果链概率缓存，
在 MIT+INCART 患者级测试集上细扫 θ 和 alert_cooldown，寻找最终最优操作点。
不重新做 TFLite 推理。
"""
import sys, json, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
from data.dataset import set_npz_suffix, load_mit_incart_merged
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_aami_breakdown import recover_mit_symbols_per_record, recover_incart_symbols_per_record, align_symbols_to_npz
from eval_exp7c_policy_sweep import reduce_mit_augmentation, evaluate_sequence_set, DEFAULT_GT_GAP
import eval_exp7c_policy_sweep as pol

OUT = Path(__file__).resolve().parent / "models" / "deploy_match" / "event_grid_from_cache.json"
CACHE = Path(__file__).resolve().parent / "models" / "deploy_match"

THETAS = [0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]
COOLDOWNS = [5, 6, 8, 10]

def main():
    t0 = time.time()
    set_npz_suffix("_deploy_causal")
    data = load_mit_incart_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    beats, labels, rids, kept_idx = reduce_mit_augmentation(beats, labels, rids)

    per_rec_syms = recover_mit_symbols_per_record()
    incart_dir = Path(__file__).resolve().parent / "data" / "raw" / "incart"
    per_rec_syms.update(recover_incart_symbols_per_record(incart_dir))
    sym_full, _ = align_symbols_to_npz(per_rec_syms, data["record_ids"], 6)
    symbols = sym_full[kept_idx]
    te_red = patient_level_split(
        data["record_ids"],
        {**build_mit_patient_map(),
         **{rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()}},
    )[2][kept_idx]

    rr, yy, ss = rids[te_red], labels[te_red], symbols[te_red]
    print(f"[GRID] test beats={int(te_red.sum())}, records={len(np.unique(rr))}")

    results = {}
    for tag, prob_file, thetas in [
        ("v3b_qat", "v3b_causal_probs_full.npy", THETAS),
        ("exp7c", "exp7c_causal_probs_full.npy", [0.50, 0.52, 0.55]),
    ]:
        probs = np.load(CACHE / prob_file)
        pp = probs[te_red]
        results[tag] = []
        pol.THETAS = thetas
        pol.POLICIES = [(1, 5)]
        for cool in COOLDOWNS:
            rows = evaluate_sequence_set("test", rr, yy, pp, ss, DEFAULT_GT_GAP, cool)
            for r in rows:
                results[tag].append({
                    "theta": r["theta"], "cooldown": cool,
                    "recall": r["event_recall"], "precision": r["event_precision"],
                    "f1": r["event_f1"], "fp_per_record": r["fp_per_record"],
                    "fp_per_1000": r["fp_per_1000_beats"],
                })
        # print top 5 by F1
        top = sorted(results[tag], key=lambda x: -x["f1"])[:10]
        print(f"\n[{tag}] top F1:")
        for r in top:
            print(f"  θ={r['theta']} cool={r['cooldown']} rec={r['recall']:.4f} "
                  f"prec={r['precision']:.4f} f1={r['f1']:.4f} fp={r['fp_per_record']:.4f}")
        results[tag + "_top"] = top

    json.dump({"theta_grid": THETAS, "cooldown_grid": COOLDOWNS,
               "results": results, "elapsed_s": round(time.time() - t0, 1)},
              open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[GRID] saved {OUT}")


if __name__ == "__main__":
    main()
