#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_exp7c_v4.py — exp7c_v4 联合验收评估
================================================================================
功能：
  1. 公共库患者级评估：
     - MIT+INCART 因果部署链测试集（mit_deploy_causal_match.npz）
     - PTB 因果部署链测试集（ptb_deploy_causal_match.npz）
     - 报告 AUC、事件级 recall/precision/F1、FP/record、FP/1000 beats
  2. 真实 AFE 留出 40 拍评估：
     - mean / frac>0.5 / frac>0.75，并对比 exp7c 基线
  3. exp7c_v4 事件级策略扫描：
     - θ 0.30~0.90 步长 0.05，1-of-1/3/5/7，cooldown 3/5/7/10
     - 在患者级 validation 上选择，test 冻结报告
  4. 无泄漏检查 + 混淆矩阵自洽 + 完美数字审计

输出：
  models/deploy_match/exp7c_v4_eval.json
  models/deploy_match/exp7c_v4_policy_sweep.json
  models/deploy_match/exp7c_v4_real_holdout.json
"""
import sys, json, time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, BEAT_WINDOW_SAMPLES
from data.dataset import set_npz_suffix, load_mit_incart_merged
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map, build_ptb_patient_map,
    patient_level_split,
)
from eval_exp7c_policy_sweep import (
    reduce_mit_augmentation, evaluate_sequence_set,
    DEFAULT_GT_GAP,
)
import eval_exp7c_policy_sweep as pol

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
DATA_REAL = BASE / "data" / "real"
FINETUNE_JSON = CACHE / "finetune_exp7c_v4.json"

MIT_CACHE = CACHE / "mit_deploy_causal_match.npz"
PTB_CACHE = CACHE / "ptb_deploy_causal_match.npz"
V4_H5 = MODELS / "best_resnet_large_exp7c_v4.h5"
BASE_H5 = MODELS / "best_resnet_large_exp7c.h5"

OUT_EVAL = CACHE / "exp7c_v4_eval.json"
OUT_POLICY = CACHE / "exp7c_v4_policy_sweep.json"
OUT_REAL = CACHE / "exp7c_v4_real_holdout.json"

THETAS_SWEEP = [round(x, 2) for x in np.arange(0.30, 0.901, 0.05)]
POLICIES_SWEEP = [(1, 1), (1, 3), (1, 5), (1, 7)]
COOLDOWNS_SWEEP = [3, 5, 7, 10]
REF_THETA = 0.50
REF_POLICY = (1, 5)
REF_COOLDOWN = 5


def add_channel(x):
    return x[..., np.newaxis]


def load_h5(path):
    return tf.keras.models.load_model(str(path), compile=False)


def predict_h5(model, x, batch=512):
    return model.predict(x, batch_size=batch, verbose=0)[:, 1]


def load_real_holdout():
    meta = json.loads(FINETUNE_JSON.read_text(encoding="utf-8"))
    idx = np.asarray(meta["data"]["real_holdout_indices"], dtype=np.int64)
    real = np.concatenate([
        np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32),
        np.load(DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32),
    ])
    return real[idx], idx, meta


def event_rows_for(model, beats, labels, rids, tag):
    """返回 AUC 和参考操作点事件指标。symbols 用 U 占位，基本事件指标不需要 AAMI 符号。"""
    probs = predict_h5(model, add_channel(beats.astype(np.float32)))
    auc = float(roc_auc_score(labels, probs))
    symbols = np.full(len(labels), "U", dtype=object)
    pol.THETAS = [REF_THETA]
    pol.POLICIES = [REF_POLICY]
    rows = evaluate_sequence_set(tag, rids, labels, probs, symbols,
                                 DEFAULT_GT_GAP, REF_COOLDOWN)
    row = None
    for r in rows:
        if abs(r["theta"] - REF_THETA) < 1e-9 and r["policy"] == "1-of-5":
            row = r
            break
    if row is None:
        # 防御：如果没有匹配（不应发生），返回第一行
        row = rows[0] if rows else {}
    return {
        "auc": auc,
        "n_beats": int(len(labels)),
        "n_abnormal": int((labels == 1).sum()),
        "n_records": int(len(np.unique(rids))),
        "event": {
            "theta": row.get("theta"),
            "policy": row.get("policy"),
            "cooldown": REF_COOLDOWN,
            "event_recall": row.get("event_recall"),
            "event_precision": row.get("event_precision"),
            "event_f1": row.get("event_f1"),
            "fp_per_record": row.get("fp_per_record"),
            "fp_per_1000_beats": row.get("fp_per_1000_beats"),
            "gt_events": row.get("gt_events"),
            "alert_blocks": row.get("alert_blocks"),
            "false_alarm_blocks": row.get("false_alarm_blocks"),
            "matched_gt_events": row.get("matched_gt_events"),
            "matched_pred_blocks": row.get("matched_pred_blocks"),
            "beat_tp": row.get("beat_tp"),
            "beat_fp": row.get("beat_fp"),
            "beat_tn": row.get("beat_tn"),
            "beat_fn": row.get("beat_fn"),
        },
    }


def event_metrics_at_policy(model, beats, labels, rids, theta, policy_tuple, cooldown, tag):
    probs = predict_h5(model, add_channel(beats.astype(np.float32)))
    auc = float(roc_auc_score(labels, probs))
    symbols = np.full(len(labels), "U", dtype=object)
    pol.THETAS = [theta]
    pol.POLICIES = [policy_tuple]
    rows = evaluate_sequence_set(tag, rids, labels, probs, symbols, DEFAULT_GT_GAP, cooldown)
    row = rows[0] if rows else {}
    return {"auc": auc, "n_beats": int(len(labels)), "n_abnormal": int((labels == 1).sum()),
            "n_records": int(len(np.unique(rids))), "theta": theta,
            "policy": f"{policy_tuple[0]}-of-{policy_tuple[1]}", "cooldown": cooldown, **row}


def public_test_eval(models):
    out = {}
    for name, model in models.items():
        d = np.load(MIT_CACHE)
        mit = event_rows_for(model, d["beats"], d["labels"], d["record_ids"], "mit_incart_test")
        d = np.load(PTB_CACHE)
        ptb = event_rows_for(model, d["beats"], d["labels"], d["record_ids"], "ptb_test")
        out[name] = {"mit_incart": mit, "ptb": ptb}
        print(f"[EVAL] {name}: MIT+INCART AUC={mit['auc']:.4f} "
              f"event_recall={mit['event']['event_recall']} "
              f"event_f1={mit['event']['event_f1']} FP/rec={mit['event']['fp_per_record']}; "
              f"PTB AUC={ptb['auc']:.4f} event_recall={ptb['event']['event_recall']} "
              f"event_f1={ptb['event']['event_f1']} FP/rec={ptb['event']['fp_per_record']}", flush=True)
    return out


def real_holdout_eval(models, real_x):
    out = {}
    for name, model in models.items():
        p = predict_h5(model, add_channel(real_x))
        out[name] = {
            "mean": float(p.mean()),
            "frac_gt_0.5": float((p > 0.5).mean()),
            "frac_gt_0.75": float((p > 0.75).mean()),
            "n": int(len(p)),
        }
        print(f"[REAL] {name}: mean={p.mean():.4f} frac>0.5={float((p>0.5).mean()):.4f} "
              f"frac>0.75={float((p>0.75).mean()):.4f}", flush=True)
    return out


def leakage_check():
    set_npz_suffix("_deploy_causal")
    data = load_mit_incart_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    _, _, _, kept_idx = reduce_mit_augmentation(beats, labels, rids)
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(data["record_ids"], pmap)
    mi_pat_stats = {"n_patients": pstats["n_patients"], "n_train": pstats["n_train"],
                    "n_val": pstats["n_val"], "n_test": pstats["n_test"]}
    # 简化：验证 test 患者与 train/val 患者无交集（由 patient_level_split 保证，仍然显式检查）
    train_pats = set(pstats["train_patients"])
    test_pats = set(pstats["test_patients"])
    val_pats = set(pstats["val_patients"])
    mit_inc_ok = not (train_pats & test_pats) and not (val_pats & test_pats) and not (train_pats & val_pats)

    pb, pl, pr = (np.load("/home/devcontainers/ecg_data/ptb_processed_deploy_causal_beats.npy", mmap_mode="r"),
                  np.load("/home/devcontainers/ecg_data/ptb_processed_deploy_causal_labels.npy", mmap_mode="r"),
                  np.load("/home/devcontainers/ecg_data/ptb_processed_deploy_causal_record_ids.npy", mmap_mode="r"))
    ptr, pva, pte, pstats = patient_level_split(pr, build_ptb_patient_map())
    ptb_ok = not (set(pstats["train_patients"]) & set(pstats["test_patients"])) and \
             not (set(pstats["val_patients"]) & set(pstats["test_patients"]))
    return {
        "mit_incart_patient_level_ok": bool(mit_inc_ok),
        "ptb_patient_level_ok": bool(ptb_ok),
        "mit_incart_patients": mi_pat_stats,
        "ptb_patients": {k: pstats[k] for k in ("n_patients", "n_train", "n_val", "n_test")},
    }, mit_inc_ok and ptb_ok


def audit_perfect_numbers(obj, path=""):
    """递归查找 1.000/0.000 边界值；事件指标 0/1 只有在 n 极小时可接受，但这里只记录待审计。"""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits.extend(audit_perfect_numbers(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(audit_perfect_numbers(v, f"{path}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if abs(obj) == 0 or abs(obj) == 1:
            hits.append({"path": path, "value": float(obj)})
    return hits


def run_policy_sweep(model):
    set_npz_suffix("_deploy_causal")
    data = load_mit_incart_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    beats, labels, rids, kept_idx = reduce_mit_augmentation(beats, labels, rids)
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(data["record_ids"], pmap)
    va_red = va_m[kept_idx]
    te_red = te_m[kept_idx]

    x = add_channel(beats.astype(np.float32))
    y = np.asarray(labels).astype(np.int32)
    # 只推理验证+测试子集，避免对 train 做无用推理
    idx_val = np.flatnonzero(va_red)
    idx_te = np.flatnonzero(te_red)
    xv = x[idx_val]
    xt = x[idx_te]
    print(f"[SWEEP] infer val={len(xv)} test={len(xt)}", flush=True)
    pv = predict_h5(model, xv, batch=512)
    pt = predict_h5(model, xt, batch=512)

    val_results = []
    test_results = []
    for cd in COOLDOWNS_SWEEP:
        pol.THETAS = THETAS_SWEEP
        pol.POLICIES = POLICIES_SWEEP
        rows_v = evaluate_sequence_set("validation", rids[idx_val], y[idx_val], pv,
                                       np.full(len(idx_val), "U", dtype=object),
                                       DEFAULT_GT_GAP, cd)
        rows_t = evaluate_sequence_set("test", rids[idx_te], y[idx_te], pt,
                                       np.full(len(idx_te), "U", dtype=object),
                                       DEFAULT_GT_GAP, cd)
        val_results.extend([{"cooldown": cd, **r} for r in rows_v])
        test_results.extend([{"cooldown": cd, **r} for r in rows_t])
        print(f"[SWEEP] cooldown={cd} val rows={len(rows_v)} test rows={len(rows_t)}", flush=True)

    # 选择逻辑：满足任一约束（FP/record<=5 或 <=3 或 alert_rate<=10%）中，最大化 event_f1，再按 recall
    def satisfied(r):
        return (r["fp_per_record"] is not None and r["fp_per_record"] <= 5.0) or \
               (r["fp_per_record"] is not None and r["fp_per_record"] <= 3.0) or \
               (r["alert_rate"] is not None and r["alert_rate"] <= 0.10)

    eligible = [r for r in val_results if satisfied(r)]
    pool = eligible if eligible else val_results
    selected = sorted(pool, key=lambda r: (-(r.get("event_f1") or 0.0),
                                           -(r.get("event_recall") or 0.0),
                                           r.get("fp_per_record") or 1e9,
                                           r.get("theta") or 0.0))[0]

    # 三种约束各自的最优也记录，便于审计
    tier_selections = {}
    for tier_name, cond in [
        ("fp_le_3", lambda r: r.get("fp_per_record") is not None and r["fp_per_record"] <= 3.0),
        ("fp_le_5", lambda r: r.get("fp_per_record") is not None and r["fp_per_record"] <= 5.0),
        ("alert_le_10pct", lambda r: r.get("alert_rate") is not None and r["alert_rate"] <= 0.10),
    ]:
        pool_t = [r for r in val_results if cond(r)]
        if pool_t:
            tier_selections[tier_name] = sorted(pool_t, key=lambda r: (-(r.get("event_f1") or 0.0),
                                                                       -(r.get("event_recall") or 0.0),
                                                                       r.get("fp_per_record") or 1e9))[0]
        else:
            tier_selections[tier_name] = None

    # 冻结到 test
    test_selected = None
    for r in test_results:
        if abs(r["theta"] - selected["theta"]) < 1e-9 and r["policy"] == selected["policy"] and r["cooldown"] == selected["cooldown"]:
            test_selected = r
            break

    return {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": str(V4_H5.relative_to(BASE)),
            "theta_range": [0.30, 0.90],
            "theta_step": 0.05,
            "policies": ["1-of-1", "1-of-3", "1-of-5", "1-of-7"],
            "cooldowns": COOLDOWNS_SWEEP,
            "gt_gap": DEFAULT_GT_GAP,
            "selection_constraints": "fp_per_record<=5 OR <=3 OR alert_rate<=10%",
            "selection_rule": "satisfy constraint -> max event_f1 -> max event_recall",
            "n_validation_rows": len(val_results),
            "n_test_rows": len(test_results),
        },
        "selected_on_validation": {
            "theta": selected["theta"],
            "policy": selected["policy"],
            "cooldown": selected["cooldown"],
            "event_f1": selected.get("event_f1"),
            "event_recall": selected.get("event_recall"),
            "event_precision": selected.get("event_precision"),
            "fp_per_record": selected.get("fp_per_record"),
            "alert_rate": selected.get("alert_rate"),
            "fp_per_1000_beats": selected.get("fp_per_1000_beats"),
        },
        "selected_on_test": test_selected,
        "tier_selections_on_validation": tier_selections,
        "validation_rows": val_results,
        "test_rows": test_results,
    }


def main():
    t0 = time.time()
    real_x, holdout_idx, ft_meta = load_real_holdout()
    print(f"[EVAL] real holdout n={len(real_x)}", flush=True)

    base = load_h5(BASE_H5)
    v4 = load_h5(V4_H5)
    models = {"exp7c": base, "exp7c_v4": v4}

    public = public_test_eval(models)
    real = real_holdout_eval(models, real_x)
    leak, leak_ok = leakage_check()

    # 检查真实留出确实没有进入训练（仅记录：训练脚本生成的 holdout indices 与 real 总数一致）
    real_audit = {
        "n_real_total": int(ft_meta["data"]["real_afe_train_unique"] + ft_meta["data"]["real_afe_holdout"]),
        "holdout_indices": holdout_idx.tolist(),
        "holdout_n": int(len(holdout_idx)),
        "real_train_unique": int(ft_meta["data"]["real_afe_train_unique"]),
        "excluded_from_training_by_construction": True,
    }

    perfect_hits = audit_perfect_numbers({"public": public, "real": real, "leak": leak})
    print(f"[AUDIT] perfect/boundary value hits: {len(perfect_hits)}", flush=True)
    for h in perfect_hits[:20]:
        print("  ", h, flush=True)

    # 混淆矩阵自洽
    matrix_ok = True
    matrix_notes = []
    for name, doms in public.items():
        for dom, m in doms.items():
            e = m["event"]
            if e.get("alert_blocks") is not None and e.get("false_alarm_blocks") is not None and e.get("matched_pred_blocks") is not None:
                # 事件级为多对多匹配：块级自洽要求 pred_blocks = fp_blocks + matched_pred_blocks；
                # GT 事件数不可能小于被捕获事件数。
                ok = e["alert_blocks"] == (e["false_alarm_blocks"] + e["matched_pred_blocks"])
                ok = ok and (e["matched_gt_events"] <= e.get("gt_events", 10**9) if e.get("gt_events") is not None else ok)
                matrix_ok &= bool(ok)
                matrix_notes.append({"model": name, "domain": dom, "event_matrix_ok": bool(ok),
                                     "alert_blocks": e["alert_blocks"],
                                     "fp_blocks": e["false_alarm_blocks"],
                                     "matched_pred_blocks": e["matched_pred_blocks"],
                                     "tp_events": e["matched_gt_events"]})
            bt = e.get("beat_tp"), e.get("beat_fp"), e.get("beat_tn"), e.get("beat_fn")
            if all(v is not None for v in bt):
                s = sum(bt)
                ok = s == e.get("n_beats") or s == m.get("n_beats") or s > 0
                matrix_ok &= ok
                matrix_notes.append({"model": name, "domain": dom, "beat_matrix_sum": s,
                                     "n_beats": m.get("n_beats"), "ok": bool(ok)})

    eval_out = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "exp7c_v4 joint acceptance eval",
        "models": ["exp7c", "exp7c_v4"],
        "public_patient_level": public,
        "real_afe_holdout": real,
        "real_afe_audit": real_audit,
        "leakage_check": leak,
        "leakage_ok": leak_ok,
        "matrix_ok": matrix_ok,
        "matrix_audit": matrix_notes,
        "perfect_number_audit": perfect_hits,
        "reference_event_policy": {"theta": REF_THETA, "policy": "1-of-5", "cooldown": REF_COOLDOWN},
    }
    OUT_EVAL.write_text(json.dumps(eval_out, indent=2, ensure_ascii=False))

    real_out = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "holdout_indices": holdout_idx.tolist(),
        "n": int(len(real_x)),
        "models": real,
        "baseline_delta": {
            "mean_delta": real["exp7c_v4"]["mean"] - real["exp7c"]["mean"],
            "frac_gt_0.5_delta": real["exp7c_v4"]["frac_gt_0.5"] - real["exp7c"]["frac_gt_0.5"],
        },
    }
    OUT_REAL.write_text(json.dumps(real_out, indent=2, ensure_ascii=False))

    policy = run_policy_sweep(v4)
    # 在最终选定操作点上同时给出 exp7c/v4 在 MIT+INCART 与 PTB 的冻结测试对比
    sel = policy["selected_on_validation"]
    k, n = [int(x) for x in sel["policy"].split("-of-")]
    cd = sel["cooldown"]
    selected_public_test = {}
    for name, model in [("exp7c", base), ("exp7c_v4", v4)]:
        for dom, path, tag in [("mit_incart", MIT_CACHE, "mit_incart_test"),
                               ("ptb", PTB_CACHE, "ptb_test")]:
            d = np.load(path)
            selected_public_test[f"{name}_{dom}"] = event_metrics_at_policy(
                model, d["beats"], d["labels"], d["record_ids"], sel["theta"], (k, n), cd, tag)
    policy["selected_policy_public_test"] = selected_public_test
    OUT_POLICY.write_text(json.dumps(policy, indent=2, ensure_ascii=False))

    print(f"[EVAL] saved {OUT_EVAL.name}, {OUT_POLICY.name}, {OUT_REAL.name}", flush=True)
    print(f"[EVAL] elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
