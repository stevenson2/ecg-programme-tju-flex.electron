#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用固件同链路 HR v6 复刻跑 rec_latest.ecgr, 逐拍诊断 70 vs 100 BPM 偏差"""
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_heartrate_ludb_v5 import chain_filter_v5, HP, LP
from verify_heartrate_ludb_v6 import HRDetectorV6, HRParamsV6

root = Path(__file__).resolve().parents[2]
data = (root / 'rec_latest.ecgr').read_bytes()
samples_n = struct.unpack_from('<I', data, 18)[0]
x = np.frombuffer(data, dtype='<i2', count=samples_n, offset=32).astype(np.float64) / 8000.0

# 固件前置链: 梳状×2 + HP0.5 + LP40 (gain=1)
y = chain_filter_v5(x, gain=1.0)
print('raw: n=%d pp=%.3fV | chain: pp=%.3fV rms=%.4f' % (len(x), np.ptp(x), np.ptp(y), np.sqrt(np.mean(y**2))))

# v3 实测主循环 496.17Hz → 录制有效采样率 ≈ 248.1Hz; 检测器内部按 500Hz 计时,
# 先重采样到 500Hz 再喂 (等价固件把 248Hz 数据当 500Hz? 不——固件实时是 500Hz 节拍,
# 录音是 2:1 抽取产物; 离线把 248.1→500 恢复实时节拍)
FS_REC = 248.1
t_rec = np.arange(len(y)) / FS_REC
t500 = np.arange(int(len(y) * 500.0 / FS_REC)) / 500.0
y500 = interp1d(t_rec, y, kind='linear', bounds_error=False, fill_value='extrapolate')(t500)

p = HRParamsV6(STARTUP_BLANK_SAMP=260, MIN_CONF_FEAT=3, GATE_RF_MAX=40,
               GATE_AMP_FRAC_PREV=0.55, GATE_RR_RATIO=0.65)
det = HRDetectorV6(p)
beats = []
for i in range(len(y500)):
    res = det.process(float(y500[i]))
    if res['beatDetected']:
        rr = (i - beats[-1][0]) / 500.0 if beats else 0.0
        beats.append((i, rr, det.mwi_prev, det.get_qrs_width(), det.last_rf,
                      det.beat_count, res['bpm']))

print('beats n=%d' % len(beats))
for b in beats[::3]:
    print('  i=%5d rr=%6.3fs bpm_inst=%5.1f w=%3d rf=%5.1f peak=%.2e out_bpm=%d' %
          (b[0], b[1], 60.0/b[1] if b[1] > 0 else 0, b[3], b[4] if b[4] is not None else -1,
           b[2], b[6]))
rrs = np.array([b[1] for b in beats if b[1] > 0])
if len(rrs):
    print('RR median=%.3fs -> %.1f bpm; RR 分位 %s' %
          (np.median(rrs), 60.0/np.median(rrs),
           [round(float(np.percentile(rrs, q)), 3) for q in (10, 25, 50, 75, 90)]))
    # 短/长 RR 交替检查
    rr2 = rrs[2:]
    print('RR 序列前 40:', [round(r, 3) for r in rrs[:40]])

# 保存前 20s 波形 (500Hz 复刻) + 检测标记供查看
out = root / 'rec_latest_diag.csv'
with open(out, 'w') as f:
    f.write('idx,sec,chain,is_beat\n')
    beat_idx = {b[0] for b in beats}
    for i in range(min(len(y500), int(20 * 500))):
        f.write('%d,%.4f,%.6f,%d\n' % (i, i/500.0, y500[i], 1 if i in beat_idx else 0))
print('diag csv:', out)
