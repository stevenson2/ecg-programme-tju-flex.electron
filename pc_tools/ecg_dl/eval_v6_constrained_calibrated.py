#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_v6_constrained_calibrated.py — v6 约束选参 + 校准重扫
================================================================================
使用已缓存的 clean v6 / exp7c 全量因果链概率：
  1. 在患者级 validation 上按 FP/record<=5、<=3、alert_rate<=10% 选参；
  2. 冻结到 test 报告；
  3. 在 validation 上拟合 Temperature / Platt 校准，并在校准分数上重扫；
  4. 输出 Pareto/对比表。
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

OUT = Path(__file__).resolve().parent / "models" / "deploy_match" / "v6_constrained_calibrated.json"
CACHE = Path(__file__).resolve().parent / "models" / "deploy_match"
THETAS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92]
POLICIES = [(1, 1), (1, 3), (1, 5), (1, 7)]
COOLDOWNS = [5, 6, 8, 10]


def select_with_constraint(rows, fp_limit, alert_limit=0.10):
    """在 validation rows 中，按 FP/record<=fp_limit 且 alert_rate<=alert_limit，
    再最大化 event_f1（若并列取更高 recall）。"""
    eligible = [r for r in rows
                if r.get("fp_per_record") is not None
                and r["fp_per_record"] <= fp_limit
                and r.get("alert_rate") is not None
                and r["alert_rate"] <= alert_limit]
    if not eligible:
        return None
    return sorted(eligible, key=lambda r: (-r["event_f1"], -r["event_recall"]))[0]


def calibrate_temperature(y, p):
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    best = None
    for T in np.arange(0.5, 6.01, 0.05):
        pc = 1.0 / (1.0 + np.exp(-logit / T))
        pc = np.clip(pc, eps, 1 - eps)
        nll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        if best is None or nll < best[1]:
            best = (float(T), float(nll))
    T = best[0]
    return T, np.clip(1.0 / (1.0 + np.exp(-logit / T)), eps, 1 - eps)


def calibrate_platt(y, p):
    from sklearn.linear_model import LogisticRegression
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    X = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(X, y)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return a, b, np.clip(lr.predict_proba(X)[:, 1], eps, 1 - eps)


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

    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(data["record_ids"], pmap)
    va_red, te_red = va_m[kept_idx], te_m[kept_idx]

    rr_v, yy_v, ss_v = rids[va_red], labels[va_red], symbols[va_red]
    rr_t, yy_t, ss_t = rids[te_red], labels[te_red], symbols[te_red]

    # 加载缓存概率
    v6_probs = np.load(CACHE / "clean_v6_causal_probs_full.npy")
    exp7c_probs = np.load(CACHE / "exp7c_causal_probs_full.npy")
    print(f"[DATA] v6 probs={v6_probs.shape}, exp7c probs={exp7c_probs.shape}", flush=True)

    pol.THETAS = THETAS
    pol.POLICIES = POLICIES

    def scan(name, probs, calibrated=False):
        """返回 val/test rows 两层 dict。"""
        val_rows, test_rows = [], []
        for cool in COOLDOWNS:
            rows_v = evaluate_sequence_set("validation", rr_v, yy_v,
                                           probs[va_red], ss_v,
                                           DEFAULT_GT_GAP, cool)
            rows_t = evaluate_sequence_set("test", rr_t, yy_t,
                                           probs[te_red], ss_t,
                                           DEFAULT_GT_GAP, cool)
            for r in rows_v:
                r["cooldown"] = cool
                r["calibrated"] = calibrated
                val_rows.append(r)
            for r in rows_t:
                r["cooldown"] = cool
                r["calibrated"] = calibrated
                test_rows.append(r)
        return val_rows, test_rows

    results = {}
    # v6 raw
    v6_val, v6_test = scan("v6_raw", v6_probs)
    results["v6_raw"] = {"val": v6_val, "test": v6_test}

    # v6 temperature
    T, v6_val_cal_t = calibrate_temperature(yy_v, v6_probs[va_red])
    v6_test_cal_t = 1.0 / (1.0 + np.exp(
        -np.log(np.clip(v6_probs[te_red], 1e-9, 1 - 1e-9) /
                (1 - np.clip(v6_probs[te_red], 1e-9, 1 - 1e-9))) / T))
    # Build full calibrated array in original full-reduced order
    full_idx_val = np.flatnonzero(va_red)
    full_idx_test = np.flatnonzero(te_red)
    full_t = v6_probs.copy()
    full_t[full_idx_val] = v6_val_cal_t
    full_t[full_idx_test] = v6_test_cal_t
    v6_val_t, v6_test_t = scan("v6_temp", full_t, True)
    results["v6_temp"] = {"val": v6_val_t, "test": v6_test_t, "temperature": T}

    # v6 platt
    a, b, v6_val_cal_p = calibrate_platt(yy_v, v6_probs[va_red])
    logit_t = np.log(np.clip(v6_probs[te_red], 1e-9, 1 - 1e-9) /
                     (1 - np.clip(v6_probs[te_red], 1e-9, 1 - 1e-9)))
    v6_test_cal_p = np.clip(1.0 / (1.0 + np.exp(-(a * logit_t + b))), 1e-9, 1 - 1e-9)
    full_p = v6_probs.copy()
    full_p[full_idx_val] = v6_val_cal_p
    full_p[full_idx_test] = v6_test_cal_p
    v6_val_p, v6_test_p = scan("v6_platt", full_p, True)
    results["v6_platt"] = {"val": v6_val_p, "test": v6_test_p, "platt_a": a, "platt_b": b}

    # 比较用 exp7c raw
    exp_val, exp_test = scan("exp7c_raw", exp7c_probs)
    results["exp7c_raw"] = {"val": exp_val, "test": exp_test}

    summary = {}
    for tag in ["v6_raw", "v6_temp", "v6_platt", "exp7c_raw"]:
        for fp_limit in [5.0, 3.0]:
            sel = select_with_constraint(results[tag]["val"], fp_limit)
            key = f"{tag}_fp{int(fp_limit)}"
            if sel is None:
                summary[key] = {"selected": None, "test": None}
                continue
            # 找对应测试行（同 theta, policy, cooldown, calibrated）
            test_row = None
            for r in results[tag]["test"]:
                if (r["theta"] == sel["theta"] and r["policy"] == sel["policy"]
                        and r["cooldown"] == sel["cooldown"]
                        and r.get("calibrated") == sel.get("calibrated")):
                    test_row = r
                    break
            summary[key] = {
                "selected_on_validation": {
                    "theta": sel["theta"], "policy": sel["policy"],
                    "cooldown": sel["cooldown"],
                    "recall": sel["event_recall"], "precision": sel["event_precision"],
                    "f1": sel["event_f1"], "fp_per_record": sel["fp_per_record"],
                    "fp_per_1000": sel["fp_per_1000_beats"],
                    "alert_rate": sel.get("alert_rate"),
                },
                "test_frozen": (None if test_row is None else {
                    "theta": test_row["theta"], "policy": test_row["policy"],
                    "cooldown": test_row["cooldown"],
                    "recall": test_row["event_recall"], "precision": test_row["event_precision"],
                    "f1": test_row["event_f1"], "fp_per_record": test_row["fp_per_record"],
                    "fp_per_1000": test_row["fp_per_1000_beats"],
                    "alert_rate": test_row.get("alert_rate"),
                }),
            }
            print(f"[{key}] val sel {sel['theta']}/{sel['policy']}/cool{sel['cooldown']} "
                  f"F1={sel['event_f1']:.4f} -> test "
                  f"F1={summary[key]['test_frozen']['f1']:.4f} "
                  f"recall={summary[key]['test_frozen']['recall']:.4f} "
                  f"prec={summary[key]['test_frozen']['precision']:.4f} "
                  f"fp={summary[key]['test_frozen']['fp_per_record']:.4f}", flush=True)

    json.dump({
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patient_stats": {k: pstats[k] for k in ["n_patients","n_train","n_val","n_test"]},
        "constraints": {"fp_per_record": [5.0, 3.0], "alert_rate_max": 0.10},
        "calibration": {"temperature": T, "platt_a": a, "platt_b": b},
        "summary": summary,
        "note": "约束选参只在 validation；test 仅为冻结报告。",
    }, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[DONE] saved {OUT}, elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
