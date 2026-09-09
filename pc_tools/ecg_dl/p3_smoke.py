#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p3_smoke.py — P3 前置冒烟: 教师加载/输入契约/输出顺序/吞吐。"""
import sys, time
from pathlib import Path
import numpy as np

ECGFOUNDER_DIR = Path("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ECGFounder")
sys.path.insert(0, str(ECGFOUNDER_DIR))
import torch
from net1d import Net1D
from util import filter_bandpass

CKPT = ECGFOUNDER_DIR / "checkpoint" / "1_lead_ECGFounder.pth"

model = Net1D(in_channels=1, base_filters=64, ratio=1,
              filter_list=[64, 160, 160, 400, 400, 1024, 1024],
              m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
              kernel_size=16, stride=2, groups_width=16,
              verbose=False, use_bn=False, use_do=False,
              n_classes=150, return_features=True)
ck = torch.load(str(CKPT), map_location="cpu")
missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
print(f"[smoke] loaded; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
if missing:
    print("  missing[:5]", missing[:5], flush=True)
model.eval()

# 合成输入: 2 段 (2,1,5000), 走教师预处理
x = np.random.default_rng(0).standard_normal((2, 5000)) * 0.5
filt = filter_bandpass(x, 500)
z = ((filt - filt.mean(axis=-1, keepdims=True)) /
     (filt.std(axis=-1, keepdims=True) + 1e-8)).astype(np.float32)[:, None, :]
print(f"[smoke] input z shape={z.shape} dtype={z.dtype}", flush=True)

t0 = time.time()
with torch.no_grad():
    out = model(torch.from_numpy(z))
print(f"[smoke] forward types={ [type(o).__name__ for o in out] }", flush=True)
a, b = out
print(f"[smoke] out[0].shape={tuple(a.shape)} out[1].shape={tuple(b.shape)}", flush=True)
print(f"[smoke] 2-seg forward {time.time()-t0:.1f}s", flush=True)

# 吞吐: batch=8 跑 3 次
t0 = time.time()
z8 = np.repeat(z, 4, axis=0)  # (8,1,5000)
with torch.no_grad():
    for _ in range(3):
        model(torch.from_numpy(z8))
dt = time.time() - t0
print(f"[smoke] batch8 x3 = {dt:.1f}s -> {24/dt:.1f} seg/s", flush=True)
print("[smoke] OK", flush=True)
