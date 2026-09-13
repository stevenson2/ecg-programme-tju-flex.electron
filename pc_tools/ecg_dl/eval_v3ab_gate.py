#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_v3ab_gate.py — M3 门槛1: 干净 test 同口径对比 v3-A vs v3-B (TH §107)
================================================================================
口径 = eval_clean_test.py (§98): 患者级 test 划分 (SplitGuard), θ=0.50,
1-of-5, cooldown=5, 事件级 + 拍级。两个模型均为从零训练 (v3-A §100 配方 A,
v3-B §107 增强配方), 无泄漏出身 → full = clean, 无需剔除子集。
判定: v3-B 在两域 AUC/事件F1 均不低于 v3-A - 0.02 (容差) → 门槛1 PASS。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from eval_clean_test import (make_predictor, beat_metrics, event_metrics,
                             MIT_NPZ, PTB_NPZ, CACHE)

MODELS = {
    "v3a": ("h5", BASE / "models" / "best_resnet_large_clean_baseline_v3.h5"),
    "v3b": ("h5", BASE / "models" / "best_resnet_large_v3b_noise.h5"),
}
TOLERANCE = 0.02


def main():
    t0 = time.time()
    dmi = np.load(MIT_NPZ)
    dpt = np.load(PTB_NPZ)
    sets = {}
    for dom, d in (("mit_incart", dmi), ("ptb", dpt)):
        sets[dom] = (np.asarray(d["beats"], dtype=np.float32),
                     np.asarray(d["labels"]).astype(np.int32),
                     np.asarray(d["record_ids"]))
        b, l, r = sets[dom]
        print(f"[GATE1] {dom}: beats={len(b)} abn={int((l==1).sum())} "
              f"records={len(np.unique(r))}", flush=True)

    results = {}
    for name, (kind, path) in MODELS.items():
        assert path.exists(), path
        predict = make_predictor(kind, path)
        entry = {}
        for dom, (beats, labels, rids) in sets.items():
            p = predict(beats)
            entry[dom] = {"beat": beat_metrics(labels, p),
                          "event": event_metrics(rids, labels, p, f"{dom}_{name}")}
            e = entry[dom]
            print(f"[GATE1] {name} {dom}: AUC={e['beat']['auc']:.4f} "
                  f"evF1={e['event'].get('event_f1')} FP/rec="
                  f"{e['event'].get('fp_per_record')}", flush=True)
        results[name] = {"file": str(path.name), "domains": entry}

    # ---- 判定 ----
    verdict = {"tolerance": TOLERANCE, "per_domain": {}}
    overall = True
    for dom in sets:
        a = results["v3a"]["domains"][dom]
        b = results["v3b"]["domains"][dom]
        dauc = b["beat"]["auc"] - a["beat"]["auc"]
        ef1a, ef1b = a["event"].get("event_f1"), b["event"].get("event_f1")
        ef1a = float(ef1a) if ef1a is not None else None
        ef1b = float(ef1b) if ef1b is not None else None
        df1 = (ef1b - ef1a) if (ef1a is not None and ef1b is not None) else None
        ok = (dauc >= -TOLERANCE) and (df1 is None or df1 >= -TOLERANCE)
        overall &= ok
        verdict["per_domain"][dom] = {
            "v3a_auc": a["beat"]["auc"], "v3b_auc": b["beat"]["auc"],
            "delta_auc": round(dauc, 4),
            "v3a_evf1": ef1a, "v3b_evf1": ef1b,
            "delta_evf1": round(df1, 4) if df1 is not None else None,
            "pass": bool(ok),
        }
    verdict["gate1_pass"] = bool(overall)

    out = {"tool": "eval_v3ab_gate.py",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "protocol": "eval_clean_test.py口径 (θ=0.5, 1-of-5, cooldown=5, "
                       "patient-level test, from-scratch models → full=clean)",
           "results": results, "verdict": verdict}
    outp = CACHE / "v3ab_gate1_clean_test.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"[GATE1] {'PASS' if overall else 'FAIL'} -> {outp} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
