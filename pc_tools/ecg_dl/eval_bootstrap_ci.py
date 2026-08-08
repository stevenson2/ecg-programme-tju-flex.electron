#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_bootstrap_ci.py — T3-6/M8: 主结果患者级 bootstrap 95% CI
======================================================================
任务: 必做清单 T3-6 ① / solutions.md M8
方法: 患者级重采样 (bootstrap, 500 reps, seed=123) — 复用 eval_deploy_match
      _patient_bootstrap_delta_auc 模式; 对每个模型预计算测试拍概率,
      再按患者重采样计算 AUC 分布 → 95% CI
口径: T1-2 未增强测试拍 (MIT) + PTB 原始拍; patient_level_split (seed 42)
模型: patient_split_eval.json 全部主结果模型 (13 个: 3 患者级清洁 + 10 历史跨域)
输出: models/bootstrap_ci_eval.json
用法 (WSL): export ECG_PROCESSED_DIR=$HOME/ecg_data; python3 eval_bootstrap_ci.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR
from data.dataset import load_incart_data, add_channel_dim
from data.patient_split import (build_mit_patient_map, build_incart_patient_map,
                                build_ptb_patient_map, patient_level_split)

MODELS = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS / "bootstrap_ci_eval.json"
N_REPS = 500
SEED = 123


def bootstrap_auc_ci(prob, labels, rec_ids, pmap, n_reps=N_REPS, seed=SEED):
    """患者级重采样 AUC bootstrap CI (预计算概率)."""
    rid_to_pat = {}
    for rid in np.unique(rec_ids):
        rid_to_pat[int(rid)] = pmap.get(int(rid), f"unknown_{int(rid)}")
    pat_of_beat = np.array([rid_to_pat[int(r)] for r in rec_ids])
    unique_pats = np.unique(pat_of_beat)
    pat_indices = {p: np.where(pat_of_beat == p)[0] for p in unique_pats}
    pat_list = list(pat_indices.keys())
    n_pats = len(pat_list)
    rng = np.random.default_rng(seed)
    aucs = np.zeros(n_reps)
    for rep in range(n_reps):
        idx = np.concatenate([pat_indices[p] for p in rng.choice(pat_list, n_pats, replace=True)])
        if len(np.unique(labels[idx])) < 2:
            aucs[rep] = 0.5
            continue
        aucs[rep] = roc_auc_score(labels[idx], prob[idx])
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)),
            float(aucs.mean()), float(aucs.std()), int(n_pats))


def main():
    t0 = time.time()
    print("=" * 70)
    print("T3-6/M8 主结果 bootstrap CI")
    print("=" * 70)

    # ---- 数据: 未增强 MIT + INCART + PTB ----
    d_mit = np.load(PROCESSED_DIR / "mit_bih_processed_noaug.npz")
    inc = load_incart_data()
    beats = np.concatenate([d_mit["beats"], inc["beats"]], axis=0)
    labels = np.concatenate([d_mit["labels"], inc["labels"]], axis=0)
    rids = np.concatenate([d_mit["record_ids"], inc["record_ids"] + 100000], axis=0)
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    tr, va, te, _ = patient_level_split(rids, pmap)
    x_mit, y_mit, r_mit = beats[te], labels[te], rids[te]

    d_ptb = np.load(PROCESSED_DIR / "ptb_processed.npz")
    pmap_ptb = build_ptb_patient_map()
    tr2, va2, te2, _ = patient_level_split(d_ptb["record_ids"], pmap_ptb)
    x_ptb, y_ptb, r_ptb = d_ptb["beats"][te2], d_ptb["labels"][te2], d_ptb["record_ids"][te2]
    print(f"test: MIT+INCART {len(x_mit)} 拍 / PTB {len(x_ptb)} 拍")

    # ---- 模型清单 (patient_split_eval.json 主结果条目, 250 点) ----
    eval_data = json.load(open(MODELS / "patient_split_eval.json", encoding="utf-8"))
    cands = []
    seen = set()
    for r in eval_data["results"]:
        if r.get("input_len", 250) != 250 or not r.get("file"):
            continue
        base = r["file"].split("/")[-1]
        if base in seen:
            continue  # 去重 (患者级清洁 vs 未增强测试条目同权重)
        seen.add(base)
        cand = MODELS / r["file"]
        if not cand.exists():
            hits = list(MODELS.glob(f"**/{base}"))
            cand = hits[0] if hits else None
        if cand is not None and cand.exists():
            cands.append((r["name"], cand))
    print(f"模型数 (250点, 去重): {len(cands)}")

    results = {}
    for name, path in cands:
        m = tf.keras.models.load_model(str(path), compile=False)
        out = {"file": path.name}
        for dom, x, y, r, pm in [("mit", x_mit, y_mit, r_mit, pmap),
                                 ("ptb", x_ptb, y_ptb, r_ptb, pmap_ptb)]:
            prob_raw = m.predict(add_channel_dim(x), batch_size=512, verbose=0)
            if isinstance(prob_raw, (list, tuple)):
                prob_raw = prob_raw[0]
            prob = prob_raw[:, 1]
            auc0 = float(roc_auc_score(y, prob))
            lo, hi, mean, std, n_pats = bootstrap_auc_ci(prob, y, r, pm)
            out[dom] = {"auc": auc0, "ci_lo": lo, "ci_hi": hi,
                        "ci_width": hi - lo, "mean": mean, "std": std,
                        "n_patients": n_pats}
            print(f"  [{name}/{dom}] AUC={auc0:.4f} 95%CI=[{lo:.4f},{hi:.4f}] "
                  f"(±{(hi-lo)/2:.4f}, n_pat={n_pats})")
        results[name] = out
        del m
        tf.keras.backend.clear_session()

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "T3-6/M8 bootstrap CI",
            "method": "患者级重采样 500 reps (seed 123), 患者级 60/20/20 (seed 42); "
                      "MIT 域未增强测试拍 (T1-2 口径); 95% CI = 2.5/97.5 百分位",
        },
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {OUT_JSON} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
