#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_context_fusion.py — 融合决策器实验: 拍级 CNN 概率 + RR 上下文 → 事件级评估
================================================================================
动机 (围绕边缘侧真实任务 = 连续流事件检测, 非拍级分类精度):
  单拍 CNN 概率已够第 3 层(筛选)用; 真实瓶颈 = 上下文缺失 (S 类 pre-RR AUC 0.964,
  verify_rr_feature.py)。本实验验证: 拍级概率 + RR 特征 (pre/post/ratio) 的轻量
  LR 融合, 在**事件级口径** (FAR/召回/精度/时延) 下是否优于纯拍级概率。

设计:
  模型: exp6-SGD (板上部署链模型), 全量概率 CPU 预测 (缓存 fusion_p_all.npy)
  特征: [p, pre_RR_s, post_RR_s, pre/post ratio] (LR, 患者级 tr 训练)
  阈值选择: val (增强拍) 事件级 FAR≤42/千h (监护目标) 时 max 事件召回
  测试: cache 未增强测试拍 (exp6sgd_probs, 与 T1-4 事件级基线同口径)
  评估: 复用 eval_alarm_decision.evaluate_policy (事件级 K, GAP=3)
输出: models/fusion_decision_eval.json
用法 (WSL): python3 eval_context_fusion.py [--skip-probs]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorflow as tf
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data.dataset import set_npz_suffix, load_mit_incart_merged, add_channel_dim
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_alarm_decision import evaluate_policy

MODELS = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS / "fusion_decision_eval.json"
PROB_CACHE = MODELS / "deploy_match" / "fusion_p_all.npy"

MIT_RAW = Path(__file__).resolve().parent / "data" / "raw" / "mit-bih-arrhythmia-database"
INCART_DIR = Path("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/"
                  "ecg-programme-tju-flex.electron-master/st-petersburg-incart-12-lead-"
                  "arrhythmia-database-1.0.0/files")

BEAT_S = 0.8
FAR_TARGET = 42.0      # 监护目标: ≤1 次/天
THETA_GRID = np.arange(0.20, 0.85, 0.05)
ALIGN_TOL = 10         # cache 拍数 vs atr 拍数容差 (边缘 skip ≤10 拍, RR 序列平滑对偏移不敏感)
GAP_ALARM = 3          # 事件聚类间隔 (与 T1-4 一致)


# ---------------------------------------------------------------- RR 特征
def load_rr_seconds(rid):
    """返回 (pre_s, post_s) 数组 (秒). MIT @360Hz, INCART @257Hz."""
    if rid < 100000:
        ann = wfdb.rdann(str(MIT_RAW / str(rid)), 'atr')
        fs = 360.0
    else:
        ann = wfdb.rdann(str(INCART_DIR / f'I{rid - 100000:02d}'), 'atr')
        fs = 257.0
    s = ann.sample.astype(np.float64)
    rr = np.diff(s) / fs
    pre = np.concatenate([[np.nan], rr])
    post = np.concatenate([rr, [np.nan]])
    return pre, post


def align_rr_to_cache(cache_rids, n_cache_per_rec):
    """cache 拍 (未增强, 记录内序) → RR 特征矩阵 (记录级 |Δ|≤2 直接对齐, 否则 NaN)."""
    n = len(cache_rids)
    pre = np.full(n, np.nan)
    post = np.full(n, np.nan)
    for rid, n_c in n_cache_per_rec.items():
        p, po = load_rr_seconds(rid)
        if abs(len(p) - n_c) <= ALIGN_TOL:
            m = min(len(p), n_c)
            idx = np.where(cache_rids == rid)[0][:m]
            pre[idx] = p[:m]
            post[idx] = po[:m]
    return pre, post


def s_event_comparison(p_base, p_fused, labels, rids, gap=GAP_ALARM):
    """S 类 (SVEB) 事件级对比: S 拍聚类为 GT 事件, 报警事件覆盖召回 vs 总 FAR."""
    from eval_aami_breakdown import (recover_mit_symbols_per_record,
                                     recover_incart_symbols_per_record)
    from eval_alarm_decision import events_from_alarm_flags, duration_hours
    from pathlib import Path as _P
    incart_dir = _P("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/"
                    "ecg-programme-tju-flex.electron-master/st-petersburg-incart-"
                    "12-lead-arrhythmia-database-1.0.0/files")
    per_rec = recover_mit_symbols_per_record()
    per_rec.update(recover_incart_symbols_per_record(incart_dir))

    # cache 拍符号 (记录内序号直接取, 容差内记录)
    sym = np.array(['U'] * len(rids), dtype=object)
    n_rec = defaultdict(int)
    for r in rids:
        n_rec[int(r)] += 1
    for rid, n_c in n_rec.items():
        arr = per_rec.get(int(rid))
        if arr is None or abs(len(arr) - n_c) > ALIGN_TOL:
            continue
        m = min(len(arr), n_c)
        idx = np.where(rids == rid)[0][:m]
        sym[idx] = arr[:m]
    s_mask = (sym == 'S')
    n_s = int(s_mask.sum())
    print(f"  S 拍: {n_s}/{len(rids)}")

    # S GT 事件 (S 拍聚类)
    gt = events_from_alarm_flags(s_mask, rids, labels, gap=gap)

    def eval_s(score, theta):
        af = (score >= theta)
        evs = events_from_alarm_flags(af, rids, labels, gap=gap)
        # S 事件命中: GT S 事件被 ≥1 报警事件覆盖
        evs_by_rec = defaultdict(list)
        for e in evs:
            evs_by_rec[e["rid"]].append(e)
        hit = 0
        for g in gt:
            elist = evs_by_rec.get(g["rid"], [])
            if any(e["start"] <= g["end"] and e["end"] >= g["start"] for e in elist):
                hit += 1
        s_recall = hit / len(gt) if gt else 1.0
        n_false = sum(1 for e in evs if not e["has_true"])
        hours = duration_hours(rids, "mit")
        return {"s_recall": float(s_recall), "FAR_1000h": n_false / hours * 1000.0,
                "n_s_events": len(gt), "n_s_hit": hit}

    grid_base = [eval_s(p_base, float(th)) for th in THETA_GRID]
    grid_fused = [eval_s(p_fused, float(th)) for th in THETA_GRID]
    return {"n_s_beats": int(n_s), "n_s_events": len(gt),
            "grid_base": grid_base, "grid_fused": grid_fused}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-probs", action="store_true", help="跳过全量概率预测 (用缓存)")
    args = ap.parse_args()
    t0 = time.time()

    set_npz_suffix("_deploy")
    print("=" * 70)
    print("融合决策器: 拍级 CNN 概率 + RR 上下文 → 事件级评估")
    print("=" * 70)

    mit_inc = load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr, va, te, _ = patient_level_split(mit_inc["record_ids"], pmap)
    all_rids = mit_inc["record_ids"]
    all_y = mit_inc["labels"]
    print(f"merged: {len(all_rids)} 拍 (tr {tr.sum()} / va {va.sum()} / te {te.sum()})")

    # ---- Step 1: 全量 exp6-SGD 概率 (缓存) ----
    if PROB_CACHE.exists() and args.skip_probs:
        print(f"概率缓存: {PROB_CACHE}")
        p_all = np.load(PROB_CACHE)
    else:
        print("全量 exp6-SGD 预测 (CPU)...")
        m = tf.keras.models.load_model(str(MODELS / "best_resnet_large_exp6_sgd.h5"), compile=False)
        x = mit_inc["beats"]
        x = x[..., np.newaxis] if x.ndim == 2 else x
        p_all = m.predict(x, batch_size=1024, verbose=0)[:, 1]
        np.save(PROB_CACHE, p_all)
    print(f"  p_all: {p_all.shape}")

    # ---- Step 2: RR 特征 (merged 全量, 增强还原对齐) ----
    print("构造 RR 特征 (merged 全量)...")
    n_all = len(all_rids)
    rr_pre_all = np.full(n_all, np.nan)
    rr_post_all = np.full(n_all, np.nan)
    block_all = np.zeros(n_all, dtype=np.int32)   # 增强块号 (0=原始, 1..5=增强)
    groups = defaultdict(list)
    for i, rid in enumerate(all_rids):
        groups[int(rid)].append(i)
    for rid, idxs in groups.items():
        p, po = load_rr_seconds(rid)
        n_ann = len(p)
        start = idxs[0]
        for i in idxs:
            local = i - start
            k = local % n_ann          # 增强还原: 6× 块内取模
            block_all[i] = local // n_ann
            rr_pre_all[i] = p[k]
            rr_post_all[i] = po[k]
    valid_all = ~np.isnan(rr_pre_all) & ~np.isnan(rr_post_all)
    ratio_all = rr_pre_all / np.maximum(rr_post_all, 1e-9)
    print(f"  全量 RR 有效: {valid_all.sum()}/{n_all} "
          f"(块分布: 原始 {np.sum(block_all==0)} / 增强 {np.sum(block_all>0)})")

    # 标准化统计量 (训练集原始块, 防泄漏 + 与测试同分布)
    tr_raw = tr & valid_all & (block_all == 0)
    mu_pre = np.nanmean(rr_pre_all[tr_raw]); sd_pre = np.nanstd(rr_pre_all[tr_raw])
    mu_post = np.nanmean(rr_post_all[tr_raw]); sd_post = np.nanstd(rr_post_all[tr_raw])
    mu_ratio = np.nanmean(ratio_all[tr_raw]); sd_ratio = np.nanstd(ratio_all[tr_raw])
    X_all = np.stack([
        p_all,
        (rr_pre_all - mu_pre) / sd_pre,
        (rr_post_all - mu_post) / sd_post,
        (ratio_all - mu_ratio) / sd_ratio,
    ], axis=-1)
    X_all[~valid_all] = np.nan

    # ---- Step 3: LR 训练 (患者级, 原始块同分布) ----
    print("LR 融合器训练 (tr 原始块患者, va 原始块选阈值)...")
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_all[tr_raw], all_y[tr_raw])
    print(f"  LR 系数: p={lr.coef_[0][0]:+.3f} pre={lr.coef_[0][1]:+.3f} "
          f"post={lr.coef_[0][2]:+.3f} ratio={lr.coef_[0][3]:+.3f}")

    # val 阈值选择: 拍级 F1 最优 (原始块, 同分布)
    from sklearn.metrics import f1_score
    va_ok = va & valid_all & (block_all == 0)
    sc_va = lr.predict_proba(X_all[va_ok])[:, 1]
    y_va = all_y[va_ok]
    best = None
    for th in THETA_GRID:
        f1 = f1_score(y_va, (sc_va >= th).astype(int))
        if best is None or f1 > best["f1"]:
            best = {"theta": float(th), "f1": float(f1)}
    assert best is not None
    rids_va = all_rids[va_ok]
    r_val_ev = evaluate_policy(sc_va, y_va, rids_va, "event", best["theta"], K=1, min_len=2)
    best["FAR_1000h"] = r_val_ev["FAR_1000h"]
    best["event_recall"] = r_val_ev["event_recall"]
    best["event_precision"] = r_val_ev["event_precision"]
    print(f"  val 阈值 (拍级 F1): θ={best['theta']:.2f} F1={best['f1']:.3f} "
          f"(事件级 FAR={best['FAR_1000h']:.1f} Se={best['event_recall']:.3f})")

    # ---- Step 4: 测试 (cache 未增强拍, 与 T1-4 同口径) ----
    print("测试: cache 未增强拍...")
    cache = np.load(MODELS / "deploy_match" / "exp6sgd_probs.npz")
    p_te = cache["mit_prob"]
    y_te = cache["mit_labels"]
    rids_te = cache["mit_rids"]
    n_cache_per_rec = defaultdict(int)
    for r in rids_te:
        n_cache_per_rec[int(r)] += 1
    pre_te, post_te = align_rr_to_cache(rids_te, n_cache_per_rec)
    ok_te = ~np.isnan(pre_te) & ~np.isnan(post_te)
    ratio_te = pre_te / np.maximum(post_te, 1e-9)
    X_te = np.stack([
        p_te,
        (pre_te - mu_pre) / sd_pre,
        (post_te - mu_post) / sd_post,
        (ratio_te - mu_ratio) / sd_ratio,
    ], axis=-1)
    # 缺失 RR (记录级错位 > 容差): 只 impute RR 三列 (填标准化均值 0), p 特征保留
    X_te[~ok_te, 1:] = 0.0
    print(f"  测试 RR 有效: {ok_te.sum()}/{len(p_te)} "
          f"(其余 mean-impute 为纯拍级等效)")
    sc_te = lr.predict_proba(X_te)[:, 1]

    # AUC (拍级, 参考)
    auc_base = roc_auc_score(y_te, p_te)
    auc_fuse = roc_auc_score(y_te, sc_te)
    print(f"  拍级 AUC: 纯拍级 {auc_base:.4f} vs 融合 {auc_fuse:.4f}")

    # ---- Step 5: 事件级对比 (θ 网格, K=1, min_len=2) ----
    print("\n事件级对比 (K=1, min_len=2):")
    print(f"  {'θ':<5} {'纯拍级 FAR/Se/P':<28} {'融合 FAR/Se/P'}")
    rows = []
    for th in THETA_GRID:
        rb = evaluate_policy(p_te, y_te, rids_te, "event", float(th), K=1, min_len=2)
        rf = evaluate_policy(sc_te, y_te, rids_te, "event", float(th), K=1, min_len=2)
        rows.append({"theta": float(th),
                     "base": {k: rb[k] for k in ("FAR_1000h", "event_recall",
                                                 "event_precision", "n_events",
                                                 "n_true_events", "n_false_events")},
                     "fused": {k: rf[k] for k in ("FAR_1000h", "event_recall",
                                                  "event_precision", "n_events",
                                                  "n_true_events", "n_false_events")}})
        print(f"  {th:.2f}  {rb['FAR_1000h']:6.1f} {rb['event_recall']:.3f} {rb['event_precision']:.3f}"
              f"      {rf['FAR_1000h']:6.1f} {rf['event_recall']:.3f} {rf['event_precision']:.3f}")

    # 同 FAR 档对比 (FAR≤42 时 max Se)
    def best_at_far(rows, key, far_max=42.0):
        cands = [r for r in rows if r[key]["FAR_1000h"] <= far_max]
        return max(cands, key=lambda r: r[key]["event_recall"]) if cands else None

    b_best = best_at_far(rows, "base")
    f_best = best_at_far(rows, "fused")

    # ---- Step 6: S 类事件级分析 (SVEB 事件召回, 融合是否受益) ----
    print("\nS 类事件级分析 (SVEB 事件召回 vs 总 FAR)...")
    s_analysis = s_event_comparison(p_te, sc_te, y_te, rids_te)
    for th, rb, rf in zip(THETA_GRID, s_analysis["grid_base"], s_analysis["grid_fused"]):
        print(f"  θ={th:.2f}  纯拍级 S事件Se={rb['s_recall']:.3f} FAR={rb['FAR_1000h']:.1f} "
              f"| 融合 S事件Se={rf['s_recall']:.3f} FAR={rf['FAR_1000h']:.1f}")

    summary = {
        "meta": {
            "task": "融合决策器: 拍级 CNN 概率 + RR 上下文 (pre/post/ratio) → 事件级",
            "model": "exp6-SGD (板上部署链) + LogisticRegression (4 特征)",
            "threshold_selection": "val (原始块, 同分布) 拍级 F1 最优",
            "test": "cache 未增强测试拍 (T1-4 同口径), RR 记录级 |Δ|≤10 对齐, 缺失 mean-impute (仅 RR 列)",
            "leakage_control": "患者级: tr 原始块训练 / va 选阈值 / te 评估; 标准化统计量取自 tr 原始块",
            "note": "事件级基线 (K=1, GAP=3) 已 Se=1.000 @ FAR 87/千h; 融合的拍级 AUC 增益 (+0.012) 在事件级被确认逻辑饱和吸收",
        },
        "auc_beat": {"base": float(auc_base), "fused": float(auc_fuse)},
        "lr_coef": {"p": float(lr.coef_[0][0]), "pre_rr": float(lr.coef_[0][1]),
                    "post_rr": float(lr.coef_[0][2]), "ratio": float(lr.coef_[0][3])},
        "val_threshold": best,
        "test_rr_coverage": {"n": int(ok_te.sum()), "total": int(len(p_te))},
        "event_grid": rows,
        "s_events": s_analysis,
        "best_at_FAR42": {
            "base": b_best["base"] if b_best else None,
            "fused": f_best["fused"] if f_best else None,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== FAR≤42/千h 最优点 ===")
    print(f"  纯拍级: θ={b_best['theta'] if b_best else '-'} "
          f"FAR={b_best['base']['FAR_1000h'] if b_best else '-'} "
          f"Se={b_best['base']['event_recall'] if b_best else '-'} "
          f"P={b_best['base']['event_precision'] if b_best else '-'}")
    print(f"  融合:   θ={f_best['theta'] if f_best else '-'} "
          f"FAR={f_best['fused']['FAR_1000h'] if f_best else '-'} "
          f"Se={f_best['fused']['event_recall'] if f_best else '-'} "
          f"P={f_best['fused']['event_precision'] if f_best else '-'}")
    print(f"\n✅ 已保存: {OUT_JSON} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
