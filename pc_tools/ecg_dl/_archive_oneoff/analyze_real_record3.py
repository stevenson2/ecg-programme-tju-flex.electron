"""尖峰分析: 分布、与 R 峰对齐关系、幅度"""
import sys, struct, json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGET_FS
from eval_deploy_match import _comb_filter, _hp_lp_filter, causal_hp_05_fs250

DATA_REAL = Path(__file__).resolve().parent / "data" / "real"
raw = (DATA_REAL / "ecg_real_052.ecgr").read_bytes()
n = struct.unpack_from("<I", raw, 18)[0]
dur = struct.unpack_from("<I", raw, 14)[0]
x = np.frombuffer(raw, dtype="<i2", count=n, offset=32).astype(np.float64) / 8000.0
fs_eff = n / dur

# 尖峰位置 (|x|>1.0)
spikes = np.where(np.abs(x) > 1.0)[0]
print("spikes(|x|>1.0): n=%d rate=%.2f/s" % (len(spikes), len(spikes) / dur))
if len(spikes) > 0:
    print("spike magnitudes: min=%.2f p50=%.2f max=%.2f" % (
        np.abs(x[spikes]).min(), np.median(np.abs(x[spikes])), np.abs(x[spikes]).max()))
    # 相邻尖峰间隔分布
    gaps = np.diff(spikes) / fs_eff
    print("gap between consecutive spikes: p25=%.2fs p50=%.2fs p75=%.2fs" % (
        np.percentile(gaps, 25), np.percentile(gaps, 50), np.percentile(gaps, 75)))

# 链输出 + R 峰
ratio = Fraction(int(round(500.0 / fs_eff * 10000)), 10000).limit_denominator(100000)
s500 = resample_poly(x, ratio.numerator, ratio.denominator)
dc = s500 - np.mean(s500)
chain250 = causal_hp_05_fs250(_hp_lp_filter(_comb_filter(dc))[0::2])
from wfdb.processing import xqrs_detect
r_idx = xqrs_detect(chain250.astype(np.float64), fs=TARGET_FS, verbose=False)
r_raw = (r_idx * fs_eff / TARGET_FS).astype(int)
# 每个 R 峰前 0.15s 到后 0.45s 的窗口内是否有尖峰
has_spike = 0
at_r = 0
for ri in r_raw:
    w = x[max(0, ri - int(0.15 * fs_eff)): min(n, ri + int(0.45 * fs_eff))]
    if np.abs(w).max() > 1.0:
        has_spike += 1
        # 尖峰是否在 R 峰 ±0.1s 内
        w2 = x[max(0, ri - int(0.1 * fs_eff)): min(n, ri + int(0.1 * fs_eff))]
        if np.abs(w2).max() > 1.0:
            at_r += 1
print("beats with spike in -0.15..+0.45s: %d/%d" % (has_spike, len(r_raw)))
print("beats with spike at R-peak +-0.1s: %d/%d" % (at_r, len(r_raw)))
# 尖峰相对最近 R 峰的时间偏移分布
if len(spikes) > 0:
    rr = r_raw[:, None]
    d = np.abs(spikes[None, :] - rr).min(axis=0) / fs_eff
    print("spike distance to nearest R-peak: p25=%.3fs p50=%.3fs p75=%.3fs" % (
        np.percentile(d, 25), np.percentile(d, 50), np.percentile(d, 75)))
# 尖峰处链输出的值 (看梳状/滤波后是否仍是极端值)
spk250 = (spikes * TARGET_FS / fs_eff).astype(int)
spk250 = spk250[spk250 < len(chain250)]
print("chain output at spikes: p50=%.2f max=%.2f (z前)" % (
    np.median(np.abs(chain250[spk250])), np.abs(chain250[spk250]).max()))
