#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.4-4 蹊跷点1: 模型权重对比 (h5py, 无需 TF)
============================================
逐层对比 .h5 权重数组, 判定"指标完全相同"的模型对是否实为同一权重:
  1. archived/final_model.h5 ("ResNet-L(v2)") vs final_resnet_m.h5 (ResNet-M)
  2. final_resnet_l_exp4_ptb.h5 (exp4) vs final_resnet_l_exp3_focal_a075.h5 (exp3)
  3. best_resnet_large_exp4_ptb.h5 (真exp4?) vs final_resnet_l_exp3_focal_a075.h5
  4. best_resnet_large.h5 (历史脚本当 exp5 用) vs best_resnet_large_exp6_domain_balanced.h5
  5. best_resnet_large.h5 vs best_resnet_large_exp5_ptb_capped.h5 (真exp5)

运行 (WSL2 Ubuntu):
  cd /mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl
  python3 compare_model_weights.py
"""
from pathlib import Path

import h5py
import numpy as np

MODELS = Path(__file__).resolve().parent / "models"


def load_weights(path):
    """遍历 h5 中所有 dataset, 返回 {路径: ndarray} (仅数值型)。"""
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fi":
            out[name] = obj[()]

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out


def compare(pa, pb):
    if not pa.exists():
        return f"缺失: {pa.name}"
    if not pb.exists():
        return f"缺失: {pb.name}"
    wa, wb = load_weights(pa), load_weights(pb)
    ka, kb = set(wa), set(wb)
    common = ka & kb
    n_a = sum(v.size for v in wa.values())
    n_b = sum(v.size for v in wb.values())
    lines = [f"  权重张量数: {len(wa)} vs {len(wb)} | 参数总数: {n_a:,} vs {n_b:,}"]
    if ka != kb:
        lines.append(f"  结构不同: 仅A {len(ka - kb)} 个, 仅B {len(kb - ka)} 个")
    if not common:
        lines.append("  无共同权重路径, 无法逐值对比")
        return "\n".join(lines), None
    diffs = []
    for k in common:
        a, b = wa[k], wb[k]
        if a.shape != b.shape:
            diffs.append(np.inf)
            continue
        diffs.append(float(np.max(np.abs(a - b))) if a.size else 0.0)
    maxd = max(diffs) if diffs else None
    if maxd is None:
        lines.append("  无可对比权重")
        return "\n".join(lines), None
    lines.append(f"  共同路径 {len(common)} 个 | 最大逐值差: {maxd:.3e}")
    verdict = "同一权重" if maxd == 0.0 else ("近似相同" if maxd < 1e-5 else "不同权重")
    lines.append(f"  => {verdict}")
    return "\n".join(lines), maxd


PAIRS = [
    ("archived/final_model.h5", "final_resnet_m.h5",
     "ResNet-L(v2) vs ResNet-M (蹊跷点1: MIT指标完全相同)"),
    ("final_resnet_l_exp4_ptb.h5", "final_resnet_l_exp3_focal_a075.h5",
     "exp4-final vs exp3-final (sha256 已确认相同, 复核)"),
    ("best_resnet_large_exp4_ptb.h5", "final_resnet_l_exp3_focal_a075.h5",
     "exp4-best vs exp3-final (验证 best 才是真 exp4)"),
    ("best_resnet_large.h5", "best_resnet_large_exp6_domain_balanced.h5",
     "通用best vs exp6-best (时间戳同秒: 通用名是否=exp6)"),
    ("best_resnet_large.h5", "best_resnet_large_exp5_ptb_capped.h5",
     "通用best vs exp5-best (历史 eval_ptb_holdout 指向通用名)"),
]

if __name__ == "__main__":
    for a, b, desc in PAIRS:
        print("=" * 64)
        print(desc)
        print(f"  A={a}\n  B={b}")
        res = compare(MODELS / a, MODELS / b)
        print(res[0] if isinstance(res, tuple) else res)
    print("=" * 64)
