#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sweep_val_policy_v3.py — v3 A 模型在 val 患者全序列上的阈值/策略扫描 (TH §101)
================================================================================
背景 (§100): v3 A 在 θ=0.5、1-of-5、cooldown=5 下 FP/rec 7.96 > 1.5 门。
本脚本在 val 患者 (模型训练从未见过) 的完整记录序列上扫描
θ × K-of-N × cooldown 网格, 按约束选操作点, 再在从未参与任何选择的
患者级测试集 (缓存 npz) 上确认。

口径: 与 eval_exp7c_policy_sweep.evaluate_sequence_set 完全一致——
按记录分组 → apply_k_of_n → episode_metrics (gt_gap=DEFAULT_GT_GAP,
pred 块 cooldown 回并) → 跨记录池化 (ev_rec=匹配GT/总GT,
fp_per_record=误报块/记录数, 拍级混淆矩阵拼接)。

选择规则 (预注册, 只看 val):
  feasible: mit_incart event_recall ≥ 0.90 且 fp_per_record ≤ 1.5
  排序: event_f1 降序, 平手取 fp_per_record 小者
  若无可行点: 报告 evRec≥0.90 中 FP/rec 最小者 + 全局 evF1 最高者

已知限制 (写入结果):
  - val 同时被用作早停监控 (抽样 4,040 拍 val AUC), 阈值选择与早停共用
    val 患者, 存在双重使用; 测试确认数字才是诚实估计。
  - val 真 AFE 留出 40 拍包含在 val 全序列中。

输出:
  models/deploy_match/val_policy_sweep_v3.json   (val 全网格)
  models/deploy_match/test_policy_confirm_v3.json (选中点 + 参考点 @ test)
运行环境: WSL (GPU 优先)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard, load_arrays
from eval_clean_test import make_predictor
import eval_exp7c_policy_sweep as pol
from eval_exp7c_policy_sweep import DEFAULT_GT_GAP

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
MODEL = MODELS / "best_resnet_large_clean_baseline_v3.h5"
MIT_NPZ = CACHE / "mit_deploy_causal_match.npz"
PTB_NPZ = CACHE / "ptb_deploy_causal_match.npz"
OUT_VAL = CACHE / "val_policy_sweep_v3.json"
OUT_TEST = CACHE / "test_policy_confirm_v3.json"

THETAS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
KONS = [(1, 1), (1, 5), (2, 3), (2, 5), (3, 5), (3, 7), (4, 7)]
COOLDOWNS = [5, 10, 20]
REF_POLICY = {"theta": 0.50, "k": 1, "n": 5, "cooldown": 5}

CONSTR = {"event_recall_min": 0.90, "fp_per_record_max": 1.5}


def r4(x):
    return float(np.round(float(x), 4))


def record_groups(rids):
    groups = []
    for rid in np.unique(rids):
        idx = np.flatnonzero(rids == rid)
        groups.append((int(rid), idx))
    return groups


def pooled_metrics(groups, labels, probs, theta, k, n, cooldown):
    """与 evaluate_sequence_set 同语义的单操作点池化指标。"""
    y_all, a_all = [], []
    tot_gt = tot_pb = tot_mg = tot_mp = tot_fp = 0
    lat = []
    for _rid, idx in groups:
        y, p = labels[idx], probs[idx]
        alarms = pol.apply_k_of_n(p, theta, k, n)
        m = pol.episode_metrics(y, alarms, None, DEFAULT_GT_GAP, cooldown)
        tot_gt += m["gt_events"]
        tot_pb += m["pred_alert_blocks"]
        tot_mg += m["matched_gt_events"]
        tot_mp += m["matched_pred_blocks"]
        tot_fp += m["false_alarm_blocks"]
        lat.extend(m["latencies"])
        y_all.append(y)
        a_all.append(alarms)
    y_all = np.concatenate(y_all)
    a_all = np.concatenate(a_all)
    tp = int(((y_all == 1) & a_all).sum())
    fp = int(((y_all == 0) & a_all).sum())
    tn = int(((y_all == 0) & ~a_all).sum())
    fn = int(((y_all == 1) & ~a_all).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    ev_prec = tot_mp / tot_pb if tot_pb else 0.0
    ev_rec = tot_mg / tot_gt if tot_gt else 0.0
    ev_f1 = 2 * ev_prec * ev_rec / (ev_prec + ev_rec) if (ev_prec + ev_rec) else 0.0
    return {
        "theta": theta, "k": k, "n": n, "cooldown": cooldown,
        "beat_tp": tp, "beat_fp": fp, "beat_tn": tn, "beat_fn": fn,
        "beat_recall": r4(rec), "beat_precision": r4(prec), "beat_f1": r4(f1),
        "gt_events": tot_gt, "alert_blocks": tot_pb,
        "matched_gt_events": tot_mg, "false_alarm_blocks": tot_fp,
        "event_recall": r4(ev_rec), "event_precision": r4(ev_prec),
        "event_f1": r4(ev_f1),
        "fp_per_record": r4(tot_fp / len(groups)),
        "median_latency_beats": r4(np.median(lat)) if lat else None,
        "n_records": len(groups),
    }


def sweep_sets(sets, thetas, kons, cooldowns):
    """sets: {name: (groups, labels, probs)}。θ 级候选掩码缓存复用。"""
    rows = {name: [] for name in sets}
    cands = {}
    for name, (groups, labels, probs) in sets.items():
        cands[name] = {t: [(idx, probs[idx] >= t) for _rid, idx in groups]
                       for t in thetas}
    for t in thetas:
        for k, n in kons:
            for cd in cooldowns:
                for name, (groups, labels, probs) in sets.items():
                    row = pooled_metrics(groups, labels, probs, t, k, n, cd)
                    rows[name].append(row)
    return rows


def val_sequences(predict):
    """val 患者全序列: mit/incart 合并 (incart rid +100000, 与测试 npz 一致)。"""
    seqs = {}
    for tag in ("mit_bih", "incart", "ptb"):
        g = get_guard(tag)
        b, l, r = load_arrays(tag)
        m = g.val_mask[:len(l)]
        b, l, r = np.asarray(b)[m], np.asarray(l)[m], np.asarray(r)[m]
        if tag == "incart":
            r = r + 100000
        seqs[tag] = (b.astype(np.float32), l.astype(np.int32), r)
        print(f"[SWEEP] val {tag}: {len(l)} beats, abn={int((l == 1).sum())}, "
              f"records={len(np.unique(r))}", flush=True)

    def merge(tags):
        beats = np.concatenate([seqs[t][0] for t in tags])
        labels = np.concatenate([seqs[t][1] for t in tags])
        rids = np.concatenate([seqs[t][2] for t in tags])
        return beats, labels, rids

    mi_b, mi_l, mi_r = merge(("mit_bih", "incart"))
    pt_b, pt_l, pt_r = seqs["ptb"]
    p_mi = predict(mi_b)
    p_pt = predict(pt_b)
    return {
        "mit_incart": (record_groups(mi_r), mi_l, p_mi),
        "ptb": (record_groups(pt_r), pt_l, p_pt),
    }, {"mit_incart": (mi_l, p_mi), "ptb": (pt_l, p_pt)}


def test_sequences(predict):
    dmi, dpt = np.load(MIT_NPZ), np.load(PTB_NPZ)
    out = {}
    for name, d in (("mit_incart", dmi), ("ptb", dpt)):
        b = np.asarray(d["beats"], dtype=np.float32)
        l = np.asarray(d["labels"]).astype(np.int32)
        r = np.asarray(d["record_ids"])
        out[name] = (record_groups(r), l, predict(b))
        print(f"[SWEEP] test {name}: {len(l)} beats, records={len(np.unique(r))}",
              flush=True)
    return out


def select(rows_mi):
    feas = [r for r in rows_mi
            if r["event_recall"] >= CONSTR["event_recall_min"]
            and r["fp_per_record"] <= CONSTR["fp_per_record_max"]]
    feas.sort(key=lambda r: (-r["event_f1"], r["fp_per_record"]))
    rec_ok = [r for r in rows_mi if r["event_recall"] >= CONSTR["event_recall_min"]]
    rec_ok.sort(key=lambda r: (r["fp_per_record"], -r["event_f1"]))
    best_any = max(rows_mi, key=lambda r: r["event_f1"])
    return {
        "constraints": CONSTR,
        "n_feasible": len(feas),
        "selected": feas[0] if feas else None,
        "top5_feasible": feas[:5],
        "fallback_min_fp_rec_at_rec_ok": rec_ok[0] if rec_ok else None,
        "fallback_best_evf1_any": best_any,
    }


def main():
    t0 = time.time()
    if not MODEL.exists():
        raise SystemExit(f"model missing: {MODEL}")
    predict = make_predictor("h5", MODEL)

    val_sets, val_flat = val_sequences(predict)
    aucs = {name: r4(roc_auc_score(l, p)) for name, (l, p) in val_flat.items()}
    print(f"[SWEEP] val AUC: {aucs}", flush=True)

    rows = sweep_sets(val_sets, THETAS, KONS, COOLDOWNS)
    ref_row = next(r for r in rows["mit_incart"]
                   if r["theta"] == REF_POLICY["theta"] and r["k"] == REF_POLICY["k"]
                   and r["n"] == REF_POLICY["n"]
                   and r["cooldown"] == REF_POLICY["cooldown"])
    print(f"[SWEEP] val reference policy row: evF1={ref_row['event_f1']} "
          f"FP/rec={ref_row['fp_per_record']}", flush=True)

    sel = select(rows["mit_incart"])
    chosen = sel["selected"] or sel["fallback_min_fp_rec_at_rec_ok"]
    print(f"[SWEEP] selected: theta={chosen['theta']} {chosen['k']}-of-{chosen['n']} "
          f"cooldown={chosen['cooldown']} | evF1={chosen['event_f1']} "
          f"evRec={chosen['event_recall']} FP/rec={chosen['fp_per_record']} "
          f"(feasible={sel['selected'] is not None})", flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §101: v3 A 在 val 患者全序列上扫描 θ×K-of-N×cooldown; "
                   "选择规则预注册, 选择只看 val",
        "model": str(MODEL.relative_to(BASE)),
        "grid": {"thetas": THETAS, "k_of_n": [list(x) for x in KONS],
                 "cooldowns": COOLDOWNS},
        "val_auc": aucs,
        "val_set_sizes": {
            name: {"beats": int(len(val_flat[name][0])),
                   "records": len(val_sets[name][0]),
                   "gt_events_ref": ref_row["gt_events"] if name == "mit_incart" else None}
            for name in val_sets
        },
        "reference_policy": REF_POLICY,
        "reference_row_val": {
            "mit_incart": ref_row,
            "ptb": next(r for r in rows["ptb"]
                        if r["theta"] == REF_POLICY["theta"]
                        and r["k"] == REF_POLICY["k"] and r["n"] == REF_POLICY["n"]
                        and r["cooldown"] == REF_POLICY["cooldown"]),
        },
        "selection": sel,
        "val_rows": rows,
        "caveats": [
            "val 同时用于早停监控 (抽样 4040 拍), 阈值选择与早停共用 val 患者; "
            "诚实估计以 test_policy_confirm_v3.json 为准",
            "val 全序列含真 AFE 留出 40 拍",
            "incart rid +100000 与测试 npz 一致",
        ],
    }
    OUT_VAL.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"[SWEEP] saved {OUT_VAL} ({time.time() - t0:.0f}s)", flush=True)

    # ---------- test 确认 ----------
    test_sets = test_sequences(predict)
    policies = [REF_POLICY,
                {"theta": chosen["theta"], "k": chosen["k"], "n": chosen["n"],
                 "cooldown": chosen["cooldown"]}]
    confirm = {}
    for p_ in policies:
        key = f"theta{p_['theta']}_{p_['k']}of{p_['n']}_cd{p_['cooldown']}"
        confirm[key] = {}
        for name, (groups, labels, probs) in test_sets.items():
            row = pooled_metrics(groups, labels, probs, p_["theta"], p_["k"],
                                 p_["n"], p_["cooldown"])
            confirm[key][name] = row
            print(f"[CONFIRM] {key} {name}: evF1={row['event_f1']} "
                  f"evRec={row['event_recall']} FP/rec={row['fp_per_record']} "
                  f"beatF1={row['beat_f1']} sens={row['beat_recall']}", flush=True)

    test_report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §101: val 选中操作点在患者级测试集 (从未参与选择) 上的确认",
        "model": str(MODEL.relative_to(BASE)),
        "selected_on_val": chosen,
        "selection_feasible": sel["selected"] is not None,
        "policies": {
            f"theta{p_['theta']}_{p_['k']}of{p_['n']}_cd{p_['cooldown']}": p_
            for p_ in policies
        },
        "results": confirm,
    }
    OUT_TEST.write_text(json.dumps(test_report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[SWEEP] saved {OUT_TEST} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
