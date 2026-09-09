#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mine_hard_negatives_v4.py — 训练划分内自信误报正常拍挖掘 (TH §103)
================================================================================
任务规格 (本次会话冻结):
  * 只在 MIT+INCART 的 TRAIN 患者心拍内挖掘 (split_guard train_mask);
  * 教师模型 = v3 A (best_resnet_large_clean_baseline_v3.h5, 只读, 不加载进训练);
  * 候选 = label=0 且 p>=0.90 的"自信误报正常拍";
  * 按记录分组: 单记录 <=150 拍, 总量上限 900 (候选不足则全取);
  * 对入选记录跑 assert_train_only + 与测试记录交集自检 (guard 回执);
  * 审计"已入训拍"占比: 复刻 v3 A 基集 rng(42) 抽样, 计算挖掘拍与基集
    抽样拍的重叠——若几乎全是已入训拍, 说明自信误报是记忆/拟合失败,
    而非未见形态混淆, 重加这些拍预期收益有限 (如实记录)。

输出: models/deploy_match/hard_negatives_v4.json
  含: 每记录挖掘数、p 分布、guard 回执、已入训拍占比、挖掘索引清单
  (供 train_clean_baseline_v4.py 消费; 训练脚本会对与基集重复的拍去重)。

铁律: 不触碰 val/test 心拍; 不读测试缓存; 不修改任何既有产物。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard, load_arrays
from eval_clean_test import make_predictor

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
TEACHER = MODELS / "best_resnet_large_clean_baseline_v3.h5"
OUT = CACHE / "hard_negatives_v4.json"

SEED = 42           # 复刻 v3 A 基集抽样用
MINE_SEED = 4242    # 挖掘选择独立种子
P_CONF = 0.90
PER_RECORD_CAP = 150
TOTAL_CAP = 900
TAGS = ("mit_bih", "incart")
BASE_RATIO = {"mit_bih": (1500, 1500), "incart": (400, 400), "ptb": (600, 400)}
CHUNK = 4096


def prob_bins(p):
    return {
        "[0.90,0.95)": int(((p >= 0.90) & (p < 0.95)).sum()),
        "[0.95,0.99)": int(((p >= 0.95) & (p < 0.99)).sum()),
        "[0.99,1.00]": int((p >= 0.99).sum()),
    }


def quantile_block(p):
    if len(p) == 0:
        return None
    q = np.quantile(p, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(p)), "mean": float(p.mean()), "min": float(p.min()),
            "max": float(p.max()), "p05": float(q[0]), "p25": float(q[1]),
            "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    predict = make_predictor("h5", TEACHER)

    # ---- 复刻 v3 A 基集抽样 (rng(42) 同序消耗), 得到已入训拍索引 ----
    rng = np.random.default_rng(SEED)
    base_idx = {}
    for tag in ("mit_bih", "incart", "ptb"):
        g = get_guard(tag)
        n_abn, n_norm = BASE_RATIO[tag]
        sa, sn = g.sample_train_beats(n_abn, n_norm, rng)
        if tag in TAGS:
            base_idx[tag] = np.sort(np.concatenate([np.asarray(sa),
                                                    np.asarray(sn)]))
    print(f"[MINE4] v3 A 基集抽样复刻: " +
          ", ".join(f"{t}={len(v)}" for t, v in base_idx.items()), flush=True)

    # ---- 对训练患者正常拍打分, 取 p>=P_CONF ----
    per_tag = {}
    pooled = []  # (tag, beat_idx, rid, prob)
    for tag in TAGS:
        g = get_guard(tag)
        b, l, r = load_arrays(tag)
        l, r = np.asarray(l), np.asarray(r)
        cand = np.where(g.train_mask[:len(l)] & (l == 0))[0]
        probs = np.zeros(len(cand), dtype=np.float32)
        for s in range(0, len(cand), CHUNK):
            chunk = np.asarray(b[cand[s:s + CHUNK]], dtype=np.float32)
            probs[s:s + CHUNK] = predict(chunk)
        conf = probs >= P_CONF
        per_tag[tag] = {
            "train_normal_beats_total": int(len(cand)),
            "confident_fp_beats": int(conf.sum()),
            "confident_p_dist": quantile_block(probs[conf]),
            "confident_p_bins": prob_bins(probs[conf]),
        }
        for j in np.where(conf)[0]:
            pooled.append((tag, int(cand[j]), int(r[cand[j]]), float(probs[j])))
        print(f"[MINE4] {tag}: train_norm={len(cand)} "
              f"confident(>={P_CONF})={int(conf.sum())}", flush=True)

    if not pooled:
        raise RuntimeError(f"无 p>={P_CONF} 的训练正常拍, 无法挖掘")

    # ---- 按记录限额抽样: 记录内洗牌截断 -> 全局洗牌取上限 ----
    rng_m = np.random.default_rng(MINE_SEED)
    by_rec = {}
    for item in pooled:
        by_rec.setdefault((item[0], item[2]), []).append(item)
    capped = []
    for _key, lst in by_rec.items():
        order = rng_m.permutation(len(lst))
        capped.extend(lst[j] for j in order[:PER_RECORD_CAP])
    take_n = min(TOTAL_CAP, len(capped))
    sel = rng_m.choice(len(capped), take_n, replace=False)
    chosen = [capped[j] for j in np.sort(sel)]

    # ---- guard 回执 + 已入训拍占比 ----
    mined = {}
    guard_receipts = {}
    overlap_audit = {}
    for tag in TAGS:
        items = [it for it in chosen if it[0] == tag]
        if not items:
            mined[tag] = {"mined": 0}
            continue
        idx = np.array([it[1] for it in items], dtype=np.int64)
        rids = np.array([it[2] for it in items], dtype=np.int64)
        probs = np.array([it[3] for it in items], dtype=np.float32)
        g = get_guard(tag)
        g.assert_train_only(rids, context=f"mine_hard_negatives_v4({tag})")
        inter = np.intersect1d(np.unique(rids), g.test_record_ids())
        assert len(inter) == 0, f"{tag}: 挖掘记录与测试记录交集非空 {inter}"
        inter_val = np.intersect1d(np.unique(rids), g.val_record_ids())
        assert len(inter_val) == 0, f"{tag}: 挖掘记录与验证记录交集非空 {inter_val}"
        guard_receipts[tag] = {
            "assert_train_only": "passed",
            "mined_records_vs_test_intersection": 0,
            "mined_records_vs_val_intersection": 0,
            "mined_records": int(len(np.unique(rids))),
        }
        recs, cnts = np.unique(rids, return_counts=True)
        overlap = np.isin(idx, base_idx[tag])
        overlap_audit[tag] = {
            "mined_vs_base_overlap": int(overlap.sum()),
            "mined_vs_base_frac": float(overlap.mean()),
            "note": "overlap=挖掘拍中已在 v3 A 基集抽样内的拍数 "
                    "(训练脚本会去重; 高占比=记忆失败而非形态混淆)",
        }
        mined[tag] = {
            "mined": int(len(idx)),
            "per_record": {int(k): int(v) for k, v in zip(recs, cnts)},
            "p_dist": quantile_block(probs),
            "p_bins": prob_bins(probs),
            "indices": [int(x) for x in idx],
            "record_ids": [int(x) for x in rids],
            "probs": [float(x) for x in probs],
        }
        print(f"[MINE4] {tag}: mined={len(idx)} records={len(recs)} "
              f"overlap_with_base={int(overlap.sum())} "
              f"({overlap.mean():.1%})", flush=True)

    all_idx_overlap = sum(v.get("mined_vs_base_overlap", 0)
                          for v in overlap_audit.values())
    total_mined = sum(v.get("mined", 0) for v in mined.values())

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §103: 训练划分内挖掘自信误报正常拍 (v4 规格: "
                   "MIT+INCART, p>=0.90, 单记录<=150, 总量<=900)",
        "teacher_model": str(TEACHER.relative_to(BASE)),
        "patient_split_seed": SEED,
        "mine_seed": MINE_SEED,
        "spec": {"p_min": P_CONF, "per_record_cap": PER_RECORD_CAP,
                 "total_cap": TOTAL_CAP, "tags": list(TAGS)},
        "per_tag_candidates": per_tag,
        "mined": mined,
        "guard_receipts": guard_receipts,
        "already_in_base_audit": {
            **overlap_audit,
            "total_overlap": int(all_idx_overlap),
            "total_mined": int(total_mined),
            "total_frac": float(all_idx_overlap / total_mined) if total_mined else 0.0,
            "verdict_note": ("占比高→挖掘拍多为已入训拍, 自信误报属记忆/拟合失败, "
                             "重加收益预期有限; 占比低→存在未见形态混淆"),
        },
        "capped_pool_size": int(len(capped)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[MINE4] total_mined={total_mined} "
          f"already_in_base={all_idx_overlap} "
          f"({all_idx_overlap / total_mined:.1%})" if total_mined else
          "[MINE4] total_mined=0", flush=True)
    print(f"[MINE4] saved {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
