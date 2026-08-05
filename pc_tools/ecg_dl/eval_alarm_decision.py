#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_alarm_decision.py — T1-4: 报警决策层 (误报率指标 + 融合粒度 + 三态输出)
======================================================================
任务: 必做清单 T1-4 / 架构计划模块 2/3/4 确认策略定参数 (论文 H2/M9)
模型: 板上 exp6-SGD (部署链) 单模型概率 (同一模型, 系统对比粒度)
数据: deploy_match 缓存测试拍 (beats_deploy, δ=0) + record_ids 分组

指标定义:
  可行动报警 (actionable alarm) = 报警事件含 ≥1 真异常拍
  每千小时误报数 FAR_1000h = 假报警事件数 / 总监测时长(h) × 1000
  总监测时长: MIT/INCART 记录 30min/条, PTB 120s/条 (测试记录数 × 时长)

粒度对比 (同一概率流):
  拍级: 单拍 p ≥ θ → 报警 (基线)
  分数加权: M 拍滑动窗分数均值 ≥ θ → 报警 (段级分数聚合)
  段级确认: M 拍窗内 ≥N 拍 ≥ θ (N-of-M)
  事件级: 报警拍聚类 (GAP=3) + 事件内 ≥K 拍 ≥ θ 且持续 ≥T 拍 → 事件报警
时延: 确认所需拍数 × 0.8s/拍 (75bpm 假设)

三态输出 (R4 equivocal, SQI 驱动等价):
  正常: p < θ_lo; 无法判定: θ_lo ≤ p ≤ θ_hi (不报警, 计数); 异常: p > θ_hi
  SQI 门控敏感度: 模拟丢弃边际概率拍 (低置信) 比例 0/5/10/15% → FAR 变化
输出: models/alarm_decision_eval.json
用法 (WSL): export ECG_PROCESSED_DIR=$HOME/ecg_data; python3 eval_alarm_decision.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_deploy_match import CACHE_DIR, _add_channel_dim

MODELS_DIR = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS_DIR / "alarm_decision_eval.json"
BEAT_S = 0.8          # 平均拍间隔 (75 bpm 假设)
MIT_MIN_PER_REC = 30  # MIT/INCART 记录时长
PTB_S_PER_REC = 120   # PTB 记录时长
GAP = 3               # 事件聚类间隔 (与 sim_temporal_agg 一致)
THETA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
N_M_GRID = [(2, 3), (3, 5), (3, 7), (5, 10)]


def load_probs():
    """exp6-SGD 概率 (缓存 npy)."""
    cache = MODELS_DIR / "deploy_match" / "exp6sgd_probs.npz"
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    print("[T1-4] 计算 exp6-SGD 概率...")
    model = tf.keras.models.load_model(str(MODELS_DIR / "best_resnet_large_exp6_sgd.h5"), compile=False)
    out = {}
    for dom in ("mit", "ptb"):
        d = np.load(CACHE_DIR / f"{dom}_deploy_match.npz")
        x = _add_channel_dim(d["beats_deploy"])
        out[f"{dom}_prob"] = model.predict(x, batch_size=512, verbose=0)[:, 1]
        out[f"{dom}_labels"] = d["labels"]
        out[f"{dom}_rids"] = d["record_ids"]
    np.savez_compressed(cache, **out)
    return out


def duration_hours(rids, dom):
    n_rec = len(np.unique(rids))
    return n_rec * (MIT_MIN_PER_REC / 60.0) if dom == "mit" else n_rec * (PTB_S_PER_REC / 3600.0)


def events_from_alarm_flags(alarm_flags, rids, labels, gap=GAP):
    """报警拍 → 事件聚类 (按记录). 返回事件列表 (start,end,n_alarm,has_true,isolated)."""
    events = []
    for rid in np.unique(rids):
        mask = rids == rid
        af = alarm_flags[mask]
        lab = labels[mask]
        idx = np.where(af)[0]
        if len(idx) == 0:
            continue
        clusters = [[idx[0]]]
        for i in idx[1:]:
            if i - clusters[-1][-1] <= gap:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        for cl in clusters:
            events.append({
                "rid": int(rid),
                "start": int(cl[0]), "end": int(cl[-1]),
                "n_alarm": len(cl),
                "n_beats": int(cl[-1] - cl[0] + 1),
                "has_true": bool(lab[cl].sum() > 0),
                "isolated": int(cl[-1] - cl[0]) <= 1,  # 孤立事件 (簇长 ≤2 拍)
            })
    return events


def gt_hit_recall(evs, gt_evs, rids, labels):
    """GT 匹配召回: GT 事件被 ≥1 报警事件覆盖 (同记录, 窗口重叠) 计命中.
    孤立/持续 GT 事件分别统计."""
    gt_by_rec = {}
    for g in gt_evs:
        rid = g["rid"]
        gt_by_rec.setdefault(rid, []).append(g)
    evs_by_rec = {}
    for e in evs:
        evs_by_rec.setdefault(e["rid"], []).append(e)
    hit_iso = hit_cont = n_iso = n_cont = 0
    for rid, glist in gt_by_rec.items():
        elist = evs_by_rec.get(rid, [])
        for g in glist:
            if g["isolated"]:
                n_iso += 1
                hit = any(e["start"] <= g["end"] and e["end"] >= g["start"] for e in elist)
                hit_iso += int(hit)
            else:
                n_cont += 1
                hit = any(e["start"] <= g["end"] and e["end"] >= g["start"] for e in elist)
                hit_cont += int(hit)
    return hit_iso / n_iso if n_iso else 1.0, hit_cont / n_cont if n_cont else 1.0, n_iso, n_cont


def evaluate_policy(prob, labels, rids, mode, theta, N=None, M=None, K=None, min_len=None,
                    lo=None, hi=None, sqi_drop=0.0, rng=None):
    """单策略评估. 返回 {FAR_1000h, event_rec, event_prec, latency_s, n_events, ...}"""
    if rng is None:
        rng = np.random.default_rng(42)
    p = prob.copy()
    y = labels.copy()
    # 三态: 无法判定拍 (不参与报警, 但保留真异常计数)
    undecided_frac = 0.0
    if lo is not None and hi is not None:
        undecided_frac = float(((p >= lo) & (p <= hi)).mean())
        p = p.copy()
        p[(p >= lo) & (p <= hi)] = -1.0  # 无法判定 → 不报警
    # SQI 门控: 随机丢弃 (模拟低质量拍)
    if sqi_drop > 0:
        n_drop = int(len(p) * sqi_drop)
        drop_idx = rng.choice(len(p), n_drop, replace=False)
        p = p.copy()
        p[drop_idx] = -1.0  # 无信号
        y = y.copy()

    valid = p >= 0
    if mode == "beat":
        af = (p >= theta) & valid
        lat = 1
    elif mode == "score_avg":
        M = M or 5
        af = np.zeros(len(p), dtype=bool)
        for i in range(len(p)):
            w = p[max(0, i - M + 1): i + 1]
            if len(w) == M and w.mean() >= theta:
                af[i] = True
        lat = M
    elif mode == "nofm":
        M = M or 5
        N = N or 3
        af = np.zeros(len(p), dtype=bool)
        for i in range(len(p)):
            w = p[max(0, i - M + 1): i + 1]
            if len(w) == M and (w >= theta).sum() >= N:
                af[i] = True
        lat = M
    elif mode == "event":
        K = K or 2
        min_len = min_len or 4
        # 事件级确认: 拍级标记 → 事件聚类 → 仅保留 n_alarm≥K 且持续≥min_len 拍的事件
        af0 = (p >= theta) & valid
        evs0 = events_from_alarm_flags(af0, rids, y, gap=GAP)
        af = np.zeros(len(p), dtype=bool)
        for rid in np.unique(rids):
            mask = rids == rid
            local = np.where(mask)[0]
            for e in evs0:
                if e["n_alarm"] >= K and e["n_beats"] >= min_len:
                    af[local[e["start"]:e["end"] + 1]] = True
        lat = K
    else:
        raise ValueError(mode)

    evs = events_from_alarm_flags(af, rids, y, gap=GAP)
    n_true = int(sum(1 for e in evs if e["has_true"]))
    n_false = len(evs) - n_true
    # Ground truth 事件 (标签聚类, 含孤立/持续分型) → 覆盖匹配召回
    gt = events_from_alarm_flags(y.astype(bool), rids, y, gap=GAP)
    se_iso, se_cont, n_gt_iso, n_gt_cont = gt_hit_recall(evs, gt, rids, y)
    n_gt = len(gt)
    hours = duration_hours(rids, "mit" if rids.max() < 400000 else "ptb")
    far = n_false / hours * 1000.0 if hours > 0 else 0.0
    evt_rec = (se_iso * n_gt_iso + se_cont * n_gt_cont) / n_gt if n_gt > 0 else 0.0
    evt_prec = n_true / len(evs) if evs else 0.0
    return {
        "mode": mode, "theta": theta, "N": N, "M": M, "K": K,
        "n_events": len(evs), "n_true_events": n_true, "n_false_events": n_false,
        "n_gt_events": n_gt, "se_iso": se_iso, "se_cont": se_cont,
        "n_gt_iso": n_gt_iso, "n_gt_cont": n_gt_cont,
        "FAR_1000h": far, "event_recall": evt_rec, "event_precision": evt_prec,
        "latency_s": round(lat * BEAT_S, 2),
        "hours": round(hours, 2),
        "sqi_drop": sqi_drop,
        "undecided_frac": undecided_frac,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("T1-4 报警决策层评估")
    print("=" * 70)
    data = load_probs()
    rng = np.random.default_rng(42)

    results = {}
    for dom in ("mit", "ptb"):
        prob, labels, rids = data[f"{dom}_prob"], data[f"{dom}_labels"], data[f"{dom}_rids"]
        dom_res = {}

        # ---- 1. 拍级基线 (θ 网格) ----
        dom_res["beat_level"] = [
            evaluate_policy(prob, labels, rids, "beat", th) for th in THETA_GRID]

        # ---- 2. 分数加权 (段平均) ----
        dom_res["score_avg"] = [
            evaluate_policy(prob, labels, rids, "score_avg", th, M=m) for th in THETA_GRID
            for m in (3, 5)]

        # ---- 3. 段级确认 N-of-M ----
        dom_res["nofm"] = [
            evaluate_policy(prob, labels, rids, "nofm", th, N=n, M=m)
            for th in THETA_GRID for (n, m) in N_M_GRID]

        # ---- 4. 事件级确认 (K=1..3, min_len=2..6) ----
        dom_res["event"] = [
            evaluate_policy(prob, labels, rids, "event", th, K=k, min_len=ml)
            for th in THETA_GRID for (k, ml) in ((1, 2), (2, 4), (3, 6))]

        # ---- 5. 三态 (概率双阈值) ----
        tri = []
        for th_lo in (0.3, 0.4):
            for th_hi in (0.6, 0.7):
                r = evaluate_policy(prob, labels, rids, "beat", th_hi, lo=th_lo, hi=th_hi)
                r["theta_lo"] = th_lo
                r["theta_hi"] = th_hi
                tri.append(r)
        dom_res["tristate"] = tri

        # ---- 6. SQI 门控敏感度 (低置信边际代理: |p−0.5| 最小 x% 拍标记无法判定, beat θ=0.5) ----
        sqi = []
        for drop in (0.0, 0.05, 0.10, 0.15, 0.20):
            margin = np.abs(prob - 0.5)
            cut = np.quantile(margin, drop) if drop > 0 else -1.0
            lo = 0.5 - cut if drop > 0 else 0.5
            hi = 0.5 + cut if drop > 0 else 0.5
            r = evaluate_policy(prob, labels, rids, "beat", 0.5, lo=lo, hi=hi)
            r["sqi_drop"] = drop
            r["sqi_margin_cut"] = float(round(float(cut), 4))
            sqi.append(r)
        dom_res["sqi_gate"] = sqi

        results[dom] = dom_res

        # 控制台摘要 (Pareto 前沿: 每个 Se 档的最小 FAR)
        print(f"\n=== {dom} ===")
        for mode in ("beat_level", "score_avg", "nofm", "event"):
            rows = dom_res[mode]
            pareto = {}
            for r in rows:
                se_key = round(r["event_recall"], 2)
                cur = pareto.get(se_key)
                if cur is None or r["FAR_1000h"] < cur["FAR_1000h"]:
                    pareto[se_key] = r
            pts = sorted(pareto.items(), reverse=True)[:4]
            for se_key, r in pts:
                print(f"  [{mode}] Se~{se_key:.2f}: FAR={r['FAR_1000h']:.1f}/千h "
                      f"P={r['event_precision']:.3f} 时延={r['latency_s']}s "
                      f"(θ={r['theta']}, N={r['N']}, M={r['M']}, K={r.get('K')})")
        # 三态摘要
        for t in dom_res["tristate"]:
            print(f"  [三态 {t['theta_lo']}/{t['theta_hi']}] FAR={t['FAR_1000h']:.1f}/千h "
                  f"Se={t['event_recall']:.3f} 无法判定占比={t['undecided_frac']*100:.1f}%")
        # SQI 摘要
        for s in dom_res["sqi_gate"]:
            print(f"  [SQI门控 {s['sqi_drop']*100:.0f}%] FAR={s['FAR_1000h']:.1f}/千h "
                  f"Se={s['event_recall']:.3f} (孤立 {s['se_iso']:.3f}/持续 {s['se_cont']:.3f}) "
                  f"无法判定={s['undecided_frac']*100:.1f}%")

    # ---- 推荐参数 (分场景: 监护 MIT / 筛查 PTB) ----
    recommendations = {}
    for dom in ("mit", "ptb"):
        dom_res = results[dom]
        all_cands = dom_res["nofm"] + dom_res["event"]
        if dom == "mit":
            # 监护: 严 FAR (≤42/千h = ≤1 次/天), Se ≥ 0.85, 时延 ≤ 60s
            target_far = 42
            cands = [r for r in all_cands
                     if r["FAR_1000h"] <= target_far and r["event_recall"] >= 0.85
                     and r["latency_s"] <= 60]
            criteria = "监护: FAR≤42/千h (≤1次/天), Se≥0.85, 时延≤60s"
        else:
            # 筛查: Se 优先 (≥0.90), FAR 尽量低, 时延 ≤ 120s
            target_far = 2000
            cands = [r for r in all_cands
                     if r["FAR_1000h"] <= target_far and r["event_recall"] >= 0.90
                     and r["latency_s"] <= 120]
            criteria = "筛查: Se≥0.90, FAR 尽量低, 时延≤120s"
        if cands:
            rec = min(cands, key=lambda r: r["FAR_1000h"])
        else:
            # 未达标的场景: 报告 Pareto 前沿最优点 (Se 最高档中 FAR 最小)
            best_se = max(r["event_recall"] for r in all_cands)
            se_cands = [r for r in all_cands if r["event_recall"] >= best_se - 0.02]
            rec = min(se_cands, key=lambda r: r["FAR_1000h"])
            criteria += " [未达标, 报告 Pareto 前沿点]"
        recommendations[dom] = {
            "policy": rec["mode"], "theta": rec["theta"], "N": rec["N"], "M": rec["M"],
            "K": rec.get("K"), "FAR_1000h": round(rec["FAR_1000h"], 2),
            "event_recall": round(rec["event_recall"], 4),
            "event_precision": round(rec["event_precision"], 4),
            "latency_s": rec["latency_s"],
            "criteria": criteria,
        }
        print(f"\n[推荐 {dom}] {json.dumps(recommendations[dom], ensure_ascii=False)}")

    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "T1-4 报警决策层 (必做清单 T1-4)",
            "model": "exp6-SGD (板上部署链模型), beats_deploy δ=0",
            "indicators": {
                "FAR_1000h": "每千小时误报事件数 = 假报警事件/监测时长(h)×1000",
                "actionable": "可行动报警 = 真异常事件 (事件含 ≥1 异常拍)",
                "duration": "MIT/INCART 30min/记录, PTB 120s/记录",
            },
            "assumptions": {"beat_interval_s": BEAT_S, "gap": GAP},
            "reference": "R4 equivocal 范式 (SQI 驱动三态); TUNING_HISTORY §8.9.9 段级逻辑; "
                         "sim_temporal_agg.py N-of-M 事件语义",
        },
        "results": results,
        "recommendations": recommendations,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存: {OUT_JSON} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
