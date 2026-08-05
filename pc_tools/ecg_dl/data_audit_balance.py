#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据盘点: 三个数据集类别规模 + 平衡混合方案估算 (2026-08-03)
回答: 如果做"有问题/没问题"二分类平衡混合, 各数据集提供多少正常/异常拍?
      MIT+INCART 拍级金标准 + PTB 患者级(全部患者=异常)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged, load_ptb_data

set_npz_suffix("_deploy")

print("=" * 78)
print("数据盘点: 二分类 (有问题/没问题) 平衡混合可行性")
print("=" * 78)

# --- MIT+INCART ---
mi = load_mit_incart_merged()
y_mi = mi["labels"]
n_mi = len(y_mi)
n_mi_n = int((y_mi == 0).sum())
n_mi_a = int((y_mi == 1).sum())
print(f"\n[1] MIT+INCART (拍级金标准, 部署链):")
print(f"    总拍: {n_mi:,} | 正常: {n_mi_n:,} ({n_mi_n/n_mi*100:.1f}%) "
      f"| 异常: {n_mi_a:,} ({n_mi_a/n_mi*100:.1f}%)")

# --- PTB ---
ptb = load_ptb_data()
y_ptb = ptb["labels"]
n_ptb = len(y_ptb)
n_ptb_n = int((y_ptb == 0).sum())
n_ptb_a = int((y_ptb == 1).sum())
print(f"\n[2] PTB (患者级, 患者记录=异常, 部署链):")
print(f"    总拍: {n_ptb:,} | 正常(健康对照): {n_ptb_n:,} ({n_ptb_n/n_ptb*100:.1f}%) "
      f"| 异常(患者): {n_ptb_a:,} ({n_ptb_a/n_ptb*100:.1f}%)")

# --- 平衡方案估算 ---
print(f"\n[3] 平衡混合方案 (二分类 有问题/没问题):")
print(f"    {'方案':<36}{'正常':<12}{'异常':<12}{'异常占比':<10}")
print(f"    {'-'*70}")

# 方案 A: 全量混合 (现状, 实验22会失败)
n_n = n_mi_n + n_ptb_n
n_a = n_mi_a + n_ptb_a
print(f"    {'A. 全量硬混合':<36}{n_n:<12,}{n_a:<12,}{n_a/(n_n+n_a)*100:<10.1f}%")

# 方案 B: MIT全量 + PTB异常按需补充, 异常占比 30%
# 目标: 异常占比 30% → 异常 = 正常*0.3/0.7
for target_abn in [0.3, 0.5]:
    # 用 MIT+INCART 全部 + PTB 补充异常到目标比例
    # 假设正常全用 (MIT的83.5%正常 + PTB健康对照)
    # 先算: 用全部正常 (MIT正常 + PTB正常), 异常需要多少
    total_norm = n_mi_n + n_ptb_n
    need_abn = total_norm * target_abn / (1 - target_abn)
    from_ptb = max(0, need_abn - n_mi_a)  # MIT 异常先用, 不够 PTB 补
    print(f"    {'B. 异常占比'+str(int(target_abn*100))+'% (正常全用)':<36}"
          f"{total_norm:<12,}{need_abn:<12,.0f}{target_abn*100:<10.1f}%"
          f"  [PTB需补{from_ptb:,.0f}异常拍]")

# 方案 C: 纯平衡 50/50, 从各域采样
print(f"    {'C. 严格50/50 (各域均衡采样)':<36}{'≈均衡':<12}{'≈均衡':<12}{'50.0':<10}%")

# 方案 D: 域平衡 (每batch固定比例 PTB) - 已有 exp6 雏形
print(f"    {'D. 域平衡采样 (batch 20% PTB)':<36}{'-':<12}{'-':<12}{'-':<10}%")

# --- 关键: PTB 健康对照太少 ---
print(f"\n[4] 关键约束: PTB 健康对照(正常)仅 {n_ptb_n:,} 拍 ({n_ptb_n/n_ptb*100:.1f}%)")
print(f"    → PTB 无法提供足够的'正常'样本, 正常侧必须靠 MIT+INCART")
print(f"    → 混合后模型学的'正常'主要是 MIT 形态 (域偏移仍在)")
print(f"    → 建议: MIT/INCART 提供正常+心律失常异常, PTB 仅补充'患者异常'作为心梗域多样性")
