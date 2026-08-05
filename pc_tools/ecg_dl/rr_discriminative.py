#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检验 RR 特征对 SVEB 的判别力 (核心问题: SVEB 联律间期是否显著提前)

对比 N (正常) vs S (SVEB) 的 pre-RR / post-RR / RR-ratio 分布。
若 SVEB 的 pre-RR 显著短 (联律间期提前), 则 RR 特征有判别力 → 支持"单拍+RR"方向。
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_aami_breakdown import recover_mit_symbols_per_record, align_symbols_to_npz
import wfdb

set_npz_suffix("_deploy")
mi = load_mit_incart_merged()
per = recover_mit_symbols_per_record()
sym, nuk = align_symbols_to_npz(per, mi["record_ids"], 6)

# 恢复全量 RR 特征 (MIT 记录, .atr)
rec_dir = Path(__file__).resolve().parent / "data" / "raw" / "mit-bih-arrhythmia-database"
n_all = len(mi["beats"])
rr_pre = np.full(n_all, np.nan)
rr_post = np.full(n_all, np.nan)
from collections import defaultdict
groups = defaultdict(list)
for i, rid in enumerate(mi["record_ids"]):
    groups[int(rid)].append(i)

for rid, idxs in groups.items():
    rec_name = str(int(rid))
    atr_path = rec_dir / f"{rec_name}.atr"
    if atr_path.exists():
        ann = wfdb.rdann(str(atr_path.with_suffix("")), 'atr')
        samples = ann.sample.astype(np.int64)
        rr = np.diff(samples).astype(float)
        pre = np.concatenate([[np.nan], rr])
        post = np.concatenate([rr, [np.nan]])
        n_ann = len(pre)
        start = idxs[0]
        for i in idxs:
            local = i - start
            orig_k = local % n_ann
            if orig_k < len(pre):
                rr_pre[i] = pre[orig_k]
                rr_post[i] = post[orig_k]

# 测试集
pmap = {}
pmap.update(build_mit_patient_map())
pmap.update({r + 100000: "inc_" + p for r, p in build_incart_patient_map().items()})
tr, va, te, _ = patient_level_split(mi["record_ids"], pmap)
sym_te = sym[te]
valid = te & ~np.isnan(rr_pre) & ~np.isnan(rr_post)

n_mask = (sym == "N") & valid
s_mask = (sym == "S") & valid
v_mask = (sym == "V") & valid

def desc(name, mask):
    pre = rr_pre[mask]
    post = rr_post[mask]
    ratio = pre / np.maximum(post, 1e-9)
    print(f"{name:<6} n={mask.sum():>6} | pre-RR {np.nanmean(pre):7.1f}±{np.nanstd(pre):6.1f} "
          f"| post-RR {np.nanmean(post):7.1f}±{np.nanstd(post):6.1f} "
          f"| ratio {np.nanmean(ratio):.3f}±{np.nanstd(ratio):.3f}")
    return pre, post, ratio

print("RR 特征分布 (原始采样点 @360Hz; 平均 RR ≈ 75bpm ≈ 288 点):")
print("=" * 78)
p_n, po_n, r_n = desc("N", n_mask)
p_s, po_s, r_s = desc("S", s_mask)
p_v, po_v, r_v = desc("V", v_mask)

print("\n统计检验 (N vs S):")
# 删 NaN 后 Mann-Whitney U (非参数, 分布偏态)
for nm, a, b in [("pre-RR", p_n[~np.isnan(p_n)], p_s[~np.isnan(p_s)]),
                 ("post-RR", po_n[~np.isnan(po_n)], po_s[~np.isnan(po_s)]),
                 ("ratio", r_n[~np.isnan(r_n)], r_s[~np.isnan(r_s)])]:
    if len(a) > 0 and len(b) > 0:
        u, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        # 中位数
        med_a, med_b = np.median(a), np.median(b)
        print(f"  {nm:<8} N中位={med_a:7.1f} S中位={med_b:7.1f} | U={u:.0f} p={pval:.2e} "
              f"{'*** 显著' if pval < 1e-6 else '(不显著)'}")
