#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_val_overalert_v3.py — P1: val 划分侧是否可见"自信正常误报"症状 (TH §105)
================================================================================
问题: 测试侧过度报警 (正常拍概率近 1, 集中在少数 INCART/MIT 记录, 如 100060 纯正常
      记录 58 误报块) 是否在 **val 划分** 上同样存在? 若存在 → val 是合法的症状代理;
      若不存在 → val 对该症状结构性失明, 只能靠最终验收一次性裁定。

口径: 与 diag_v3.py 完全一致 (θ=0.5, 1-of-5, cooldown=5, GT_GAP=5)。
输入: 不用测试缓存 npz, 而是经 SplitGuard 从 {tag}_processed_deploy_causal_*.npy
      取 **val_mask 全量拍** (逐记录结构保留); 同时用 test_mask 重建测试拍复算
      AUC, 与 diag_v3.json 对照, 自证数组/模型/口径一致。

INCART rid 偏移约定: 与 diag_v3 相同 — 合并空间中 incart rid 已 +100000
(即 mi_r<100000 = mit; 与模块头部注释一致)。

输出: models/deploy_match/p1_val_overalert_v3.json
运行环境: WSL (export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clean_test import make_predictor, beat_metrics, event_metrics
from data.split_guard import get_guard, load_arrays, INCART_RID_OFFSET

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
OUT = CACHE / "p1_val_overalert_v3.json"
MODEL = MODELS / "best_resnet_large_clean_baseline_v3.h5"
DIAG_V3 = CACHE / "diag_v3.json"

REF_THETA = 0.50
REF_POLICY = (1, 5)
REF_COOLDOWN = 5


def quantiles(p, extra=None):
    if len(p) == 0:
        return None
    q = np.quantile(p, [0.05, 0.25, 0.5, 0.75, 0.95])
    d = {"n": int(len(p)), "mean": float(p.mean()),
         "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
         "p75": float(q[3]), "p95": float(q[4]),
         "frac_gt_0.5": float((p > 0.5).mean()),
         "frac_gt_0.95": float((p > 0.95).mean())}
    if extra:
        d.update(extra)
    return d


def per_record_blocks(rids, labels, probs):
    """逐记录误报警块 (与 diag_v3 第 3 节同逻辑)。返回 {rid: {...}}。"""
    import eval_exp7c_policy_sweep as pol
    from eval_exp7c_policy_sweep import DEFAULT_GT_GAP
    per = {}
    for rid in np.unique(rids):
        m = rids == rid
        l, p = labels[m], probs[m]
        alarms = pol.apply_k_of_n(p, REF_THETA, REF_POLICY[0], REF_POLICY[1])
        pred_blocks = pol.merge_blocks_with_gap(pol.bool_blocks(alarms), REF_COOLDOWN)
        gt_blocks = pol.merge_blocks_with_gap(pol.bool_blocks(l == 1), DEFAULT_GT_GAP)
        per[int(rid)] = {
            "n_beats": int(m.sum()),
            "abn_beats": int((l == 1).sum()),
            "alert_blocks": len(pred_blocks),
            "gt_events": len(gt_blocks),
        }
    return per


def main():
    t0 = time.time()

    # ---- 自证 A: 用权威测试缓存 (npz) 复算 AUC, 必须≈diag_v3 ----
    #      (mit_deploy_causal_match.npz 是 51,883 拍截断缓存, 非全量 test_mask,
    #       故不能拿 test_mask 重建 AUC 去对; 唯一可比的基底就是 npz 本身)
    diag = json.loads(DIAG_V3.read_text(encoding="utf-8"))
    predict = make_predictor("h5", MODEL)
    dmi_cache = np.load(CACHE / "mit_deploy_causal_match.npz")
    dpt_cache = np.load(CACHE / "ptb_deploy_causal_match.npz")
    c_mi_b = np.asarray(dmi_cache["beats"], dtype=np.float32)
    c_mi_l = np.asarray(dmi_cache["labels"]).astype(np.int32)
    c_mi_r = np.asarray(dmi_cache["record_ids"])
    c_pt_b = np.asarray(dpt_cache["beats"], dtype=np.float32)
    c_pt_l = np.asarray(dpt_cache["labels"]).astype(np.int32)
    c_pt_r = np.asarray(dpt_cache["record_ids"])
    p_cmi = predict(c_mi_b)
    p_cpt = predict(c_pt_b)
    c_mi_auc = float(beat_metrics(c_mi_l, p_cmi)["auc"])
    c_pt_auc = float(beat_metrics(c_pt_l, p_cpt)["auc"])
    diag_mi_auc = diag["per_domain"]["mit_incart_merged"]["beat"]["auc"]
    diag_pt_auc = diag["per_domain"]["ptb"]["beat"]["auc"]

    def split_arrays(tag):
        b, l, r = load_arrays(tag)
        g = get_guard(tag)
        return (np.asarray(b, dtype=np.float32), np.asarray(l).astype(np.int32),
                np.asarray(r), g)

    mi_b, mi_l, mi_r, g_mit = split_arrays("mit_bih")
    in_b, in_l, in_r, g_inc = split_arrays("incart")
    pt_b, pt_l, pt_r, g_ptb = split_arrays("ptb")

    # ---- 自证 B: SplitGuard test 记录号集 == npz 记录号集 (合并空间) ----
    guard_mi_rids = sorted(set(int(x) for x in np.unique(mi_r[g_mit.test_mask])))
    guard_inc_rids = sorted(int(x) + INCART_RID_OFFSET for x in np.unique(in_r[g_inc.test_mask]))
    guard_merged = set(guard_mi_rids + guard_inc_rids)
    npz_merged = set(int(x) for x in np.unique(c_mi_r))
    rid_set_match = (guard_merged == npz_merged)

    mi_auc_match = abs(c_mi_auc - diag_mi_auc) < 1e-3
    pt_auc_match = abs(c_pt_auc - diag_pt_auc) < 1e-3
    selfcheck = {
        "cache_mit_incart_auc": c_mi_auc, "diag_v3_mit_incart_auc": diag_mi_auc,
        "cache_ptb_auc": c_pt_auc, "diag_v3_ptb_auc": diag_pt_auc,
        "mit_incart_auc_match": mi_auc_match, "ptb_auc_match": pt_auc_match,
        "guard_test_rids_merged_n": len(guard_merged),
        "npz_rids_merged_n": len(npz_merged),
        "guard_vs_npz_rid_set_match": rid_set_match,
        "note": "npz 测试缓存为截断拍子集 (51883 拍), 故 AUC 只用 npz 基底对齐; "
                "记录号集对齐则证明 SplitGuard 测试划分与权威缓存一致",
    }
    print(f"[P1] selfcheck: cache MIT+INCART AUC={c_mi_auc:.4f} "
          f"diag={diag_mi_auc:.4f} match={mi_auc_match}", flush=True)
    print(f"[P1] selfcheck: cache PTB AUC={c_pt_auc:.4f} "
          f"diag={diag_pt_auc:.4f} match={pt_auc_match}", flush=True)
    print(f"[P1] selfcheck: guard test rids(n={len(guard_merged)}) vs "
          f"npz rids(n={len(npz_merged)}) set_match={rid_set_match}", flush=True)
    if not (mi_auc_match and pt_auc_match and rid_set_match):
        print("[P1] ⚠️ 自证未通过: val 分析不可信, 中止", flush=True)
        OUT.write_text(json.dumps(
            {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
             "selfcheck": selfcheck, "aborted": True, "reason": "selfcheck failed"},
            indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # ---- val 划分全量拍 (合并空间) ----
    vmi = (mi_b[g_mit.val_mask], mi_l[g_mit.val_mask], mi_r[g_mit.val_mask])
    vin = (in_b[g_inc.val_mask], in_l[g_inc.val_mask],
           np.asarray(in_r[g_inc.val_mask]) + INCART_RID_OFFSET)
    vpt = (pt_b[g_ptb.val_mask], pt_l[g_ptb.val_mask], pt_r[g_ptb.val_mask])
    vmi_b = np.concatenate([vmi[0], vin[0]])
    vmi_l = np.concatenate([vmi[1], vin[1]])
    vmi_r = np.concatenate([vmi[2], vin[2]])
    print(f"[P1] val MIT+INCART: {len(vmi_b)} beats "
          f"(abn={int((vmi_l==1).sum())}, recs={len(np.unique(vmi_r))})", flush=True)
    print(f"[P1] val PTB: {len(vpt[0])} beats "
          f"(abn={int((vpt[1]==1).sum())}, recs={len(np.unique(vpt[2]))})", flush=True)

    p_vmi = predict(vmi_b)
    p_vpt = predict(vpt[0])

    # ---- val 概率分布 (每域×类别) ----
    mit_mask = vmi_r < 100000
    prob_dist = {
        "mit": {"abn": quantiles(p_vmi[mit_mask & (vmi_l == 1)]),
                "norm": quantiles(p_vmi[mit_mask & (vmi_l == 0)])},
        "incart": {"abn": quantiles(p_vmi[~mit_mask & (vmi_l == 1)]),
                   "norm": quantiles(p_vmi[~mit_mask & (vmi_l == 0)])},
        "ptb": {"abn": quantiles(p_vpt[vpt[1] == 1]),
                "norm": quantiles(p_vpt[vpt[1] == 0])},
    }

    # ---- val 逐记录误报警块 ----
    per_mi = per_record_blocks(vmi_r, vmi_l, p_vmi)
    per_pt = per_record_blocks(vpt[2], vpt[1], p_vpt)
    fp_mi = {k: v["alert_blocks"] for k, v in per_mi.items()}
    top_mi = sorted(fp_mi.items(), key=lambda kv: -kv[1])[:10]

    def summarize(per, domain):
        blocks = {k: v["alert_blocks"] for k, v in per.items()}
        recs = len(per)
        recs_alert = sum(1 for v in blocks.values() if v > 0)
        total = int(sum(blocks.values()))
        # 纯正常记录 (abn_beats==0): 干净误报指标
        clean_norm = [k for k, v in per.items() if v["abn_beats"] == 0]
        clean_norm_alert = {k: v["alert_blocks"] for k, v in per.items()
                            if v["abn_beats"] == 0 and v["alert_blocks"] > 0}
        return {
            "domain": domain,
            "records": recs,
            "records_with_alerts": recs_alert,
            "total_alert_blocks": total,
            "clean_normal_records": len(clean_norm),
            "clean_normal_records_with_alerts": len(clean_norm_alert),
            "clean_normal_top_by_alert_blocks": sorted(
                clean_norm_alert.items(), key=lambda kv: -kv[1])[:10],
            "top10_by_alert_blocks": [
                {"rid": rid, **per[rid]} for rid, _ in top_mi if domain == "mit_incart"
            ] if domain == "mit_incart" else [
                {"rid": rid, **per[rid]} for rid, _ in
                 sorted(blocks.items(), key=lambda kv: -kv[1])[:10]],
        }

    val_merged_beat = beat_metrics(vmi_l, p_vmi)
    val_merged_event = event_metrics(vmi_r, vmi_l, p_vmi, "p1_val_merged")

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "P1 (TH §105): val 划分侧是否可见自信正常误报症状; "
                   "口径与 diag_v3 一致 (θ=0.5,1-of-5,cooldown=5)",
        "model": str(MODEL.relative_to(BASE)),
        "reference_policy": {"theta": REF_THETA, "policy": "1-of-5",
                             "cooldown": REF_COOLDOWN},
        "scope": "val_mask 全量拍 (逐记录); test 仅用于 AUC 自证",
        "val_sets": {"mit_incart": {"beats": int(len(vmi_b)),
                                    "records": [int(r) for r in np.unique(vmi_r)]},
                     "ptb": {"beats": int(len(vpt[0])),
                             "records": [int(r) for r in np.unique(vpt[2])]}},
        "selfcheck": selfcheck,
        "test_reference_reminder": {
            "top10_rids": [t["rid"] for t in diag["per_record_fp"]["top10_by_alert_blocks"]],
            "total_alert_blocks": diag["per_record_fp"]["total_alert_blocks"],
        },
        "prob_dist": prob_dist,
        "val_metrics": {
            "mit_incart": {"beat": val_merged_beat, "event": val_merged_event},
            "ptb": {"beat": beat_metrics(vpt[1], p_vpt),
                    "event": event_metrics(vpt[2], vpt[1], p_vpt, "p1_val_ptb")},
        },
        "per_record": {"mit_incart": summarize(per_mi, "mit_incart"),
                       "ptb": summarize(per_pt, "ptb")},
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # 控制台摘要
    def fmt_quant(d, key):
        if not d:
            return "n/a"
        return (f"mean={d['mean']:.3f} p95={d['p95']:.3f} "
                f">0.5={d['frac_gt_0.5']:.3f} >0.95={d['frac_gt_0.95']:.3f}")
    print("\n[P1] val 正常拍概率分布:", flush=True)
    for dom in ("mit", "incart", "ptb"):
        print(f"  {dom:8s} norm: {fmt_quant(prob_dist[dom]['norm'], 'norm')}", flush=True)
    print("[P1] val 异常拍概率分布:", flush=True)
    for dom in ("mit", "incart", "ptb"):
        print(f"  {dom:8s} abn: {fmt_quant(prob_dist[dom]['abn'], 'abn')}", flush=True)
    s = report["per_record"]["mit_incart"]
    print(f"\n[P1] val MIT+INCART 逐记录: recs={s['records']} "
          f"with_alerts={s['records_with_alerts']} total_alert_blocks={s['total_alert_blocks']}",
          flush=True)
    print(f"     纯正常记录: {s['clean_normal_records']} 条, "
          f"其中 {s['clean_normal_records_with_alerts']} 条产生误报", flush=True)
    print(f"     top10: {[ (t['rid'], t['alert_blocks'], t['abn_beats']) for t in s['top10_by_alert_blocks'] ]}", flush=True)
    print(f"[P1] val MIT+INCART 拍级 AUC={val_merged_beat['auc']:.4f} "
          f"F1={val_merged_beat['f1']:.4f}", flush=True)
    print(f"[P1] saved {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
