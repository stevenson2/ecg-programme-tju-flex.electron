#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_v3.py — v3 配置验收失败诊断 (概率分布 + 分域分解 + 逐记录误报)
================================================================================
口径与 eval_clean_baseline_v3 完全一致 (同一测试缓存、θ=0.5), 仅追加分解视角:
  1. 预测概率分布: 每域按类别的分位数 + >0.5 占比
  2. MIT / INCART / PTB 分域拍级与事件级指标 (MIT+INCART 合并口径不变,
     此处只是拆开看)
  3. 逐记录误报警块分布 (MIT+INCART): 找到 fp/rec 的集中来源

输出: models/deploy_match/diag_<suffix>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clean_test import make_predictor, beat_metrics, event_metrics

BASE = Path(__file__).resolve().parent
CACHE = BASE / "models" / "deploy_match"
MODELS = BASE / "models"


def quantiles(p):
    if len(p) == 0:
        return None
    q = np.quantile(p, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(p)), "mean": float(p.mean()),
            "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]),
            "frac_gt_0.5": float((p > 0.5).mean())}


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "v3"
    model = MODELS / f"best_resnet_large_clean_baseline_{suffix}.h5"
    out = CACHE / f"diag_{suffix}.json"
    predict = make_predictor("h5", model)

    dmi = np.load(CACHE / "mit_deploy_causal_match.npz")
    dpt = np.load(CACHE / "ptb_deploy_causal_match.npz")
    mi_b = np.asarray(dmi["beats"], dtype=np.float32)
    mi_l = np.asarray(dmi["labels"]).astype(np.int32)
    mi_r = np.asarray(dmi["record_ids"])
    pt_b = np.asarray(dpt["beats"], dtype=np.float32)
    pt_l = np.asarray(dpt["labels"]).astype(np.int32)
    pt_r = np.asarray(dpt["record_ids"])

    p_mi = predict(mi_b)
    p_pt = predict(pt_b)

    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model.name,
              "suffix": suffix}

    # ---- 1. 概率分布 (每域 × 类别) ----
    mit_mask = mi_r < 100000
    report["prob_dist"] = {
        "mit": {"abn": quantiles(p_mi[mit_mask & (mi_l == 1)]),
                "norm": quantiles(p_mi[mit_mask & (mi_l == 0)])},
        "incart": {"abn": quantiles(p_mi[~mit_mask & (mi_l == 1)]),
                   "norm": quantiles(p_mi[~mit_mask & (mi_l == 0)])},
        "ptb": {"abn": quantiles(p_pt[pt_l == 1]),
                "norm": quantiles(p_pt[pt_l == 0])},
    }

    # ---- 2. 分域指标 ----
    report["per_domain"] = {}
    for name, m in (("mit", mit_mask), ("incart", ~mit_mask)):
        r, l, p = mi_r[m], mi_l[m], p_mi[m]
        report["per_domain"][name] = {
            "beat": beat_metrics(l, p),
            "event": event_metrics(r, l, p, f"diag_{suffix}_{name}"),
        }
    report["per_domain"]["ptb"] = {
        "beat": beat_metrics(pt_l, p_pt),
        "event": event_metrics(pt_r, pt_l, p_pt, f"diag_{suffix}_ptb"),
    }
    report["per_domain"]["mit_incart_merged"] = {
        "beat": beat_metrics(mi_l, p_mi),
        "event": event_metrics(mi_r, mi_l, p_mi, f"diag_{suffix}_merged"),
    }

    # ---- 3. 逐记录误报警块 (1-of-5, cooldown=5, 与事件口径同) ----
    import eval_exp7c_policy_sweep as pol
    from eval_exp7c_policy_sweep import DEFAULT_GT_GAP
    per_record = {}
    for rid in np.unique(mi_r):
        m = mi_r == rid
        l, p = mi_l[m], p_mi[m]
        alarms = pol.apply_k_of_n(p, 0.5, 1, 5)
        pred_blocks = pol.merge_blocks_with_gap(pol.bool_blocks(alarms), 5)
        gt_blocks = pol.merge_blocks_with_gap(pol.bool_blocks(l == 1), DEFAULT_GT_GAP)
        per_record[int(rid)] = {
            "n_beats": int(m.sum()),
            "abn_beats": int((l == 1).sum()),
            "alert_blocks": len(pred_blocks),
            "gt_events": len(gt_blocks),
        }
    fp_blocks = {k: v["alert_blocks"] for k, v in per_record.items()}
    top = sorted(fp_blocks.items(), key=lambda kv: -kv[1])[:10]
    report["per_record_fp"] = {
        "records": len(per_record),
        "records_with_alerts": int(sum(1 for v in fp_blocks.values() if v > 0)),
        "total_alert_blocks": int(sum(fp_blocks.values())),
        "top10_by_alert_blocks": [
            {"rid": rid, **per_record[rid]} for rid, _ in top],
    }

    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # 控制台摘要
    for d in ("mit", "incart", "ptb", "mit_incart_merged"):
        b = report["per_domain"][d]["beat"]
        e = report["per_domain"][d]["event"]
        print(f"[DIAG] {d}: AUC={b['auc']:.4f} F1={b['f1']:.4f} "
              f"Sens={b['sensitivity']:.4f} Prec={b['precision']:.4f} "
              f"evF1={e.get('event_f1')} FP/rec={e.get('fp_per_record')} "
              f"alerts={e.get('alert_blocks')}")
    print(f"[DIAG] top FP records: {[(t[0], per_record[t[0]]['alert_blocks'], per_record[t[0]]['abn_beats']) for t in top]}")
    print(f"[DIAG] saved {out}")


if __name__ == "__main__":
    main()
