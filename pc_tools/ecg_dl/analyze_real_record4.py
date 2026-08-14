"""每拍削顶严重度分布"""
import sys, struct
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

ratio = Fraction(int(round(500.0 / fs_eff * 10000)), 10000).limit_denominator(100000)
s500 = resample_poly(x, ratio.numerator, ratio.denominator)
dc = s500 - np.mean(s500)
chain250 = causal_hp_05_fs250(_hp_lp_filter(_comb_filter(dc))[0::2])
from wfdb.processing import xqrs_detect
r_idx = xqrs_detect(chain250.astype(np.float64), fs=TARGET_FS, verbose=False)
r_raw = (r_idx * fs_eff / TARGET_FS).astype(int)

half = int(0.5 * fs_eff)
maxes = []
rail_n = []
for ri in r_raw:
    w = x[max(0, ri - half): min(n, ri + half)]
    maxes.append(np.abs(w).max())
    rail_n.append(int((np.abs(w) > 1.6).sum()))
maxes = np.array(maxes); rail_n = np.array(rail_n)
for thr in (1.55, 1.60, 1.62, 1.64):
    print("beats with max|x|<%.2f: %d/%d (keep)" % (thr, int((maxes < thr).sum()), len(maxes)))
print("beats with >=1 rail sample (>=1.6): %d" % int((rail_n >= 1).sum()))
print("beats with >=3 rail samples: %d" % int((rail_n >= 3).sum()))
print("beats with 0 samples >1.4: %d" % int(((maxes > 1.4) == False).sum()))
print("max|x| percentiles: p10=%.2f p50=%.2f p90=%.2f" % (
    np.percentile(maxes, 10), np.percentile(maxes, 50), np.percentile(maxes, 90)))
