"""Python transcription of firmware compute_features / logistic score.

The numbers that go into the report come from the compiled firmware
(host/fw_replay_host). This module exists so a consistency test can compare
the same window against that binary, and so the previous PC script's
approximations can be quantified.

Float32 order follows vf_detect.cpp: sequential sums, zero-state SOS forward
then zero-state backward, median of the two central |x| samples, zero-cross
defined by `v > 0` (zero counts as non-positive).
"""
from __future__ import annotations

import numpy as np

WIN_N = 1250
FS = 250
WIN_S = 5.0
THETA = np.float32(0.15)
CONFIRM = 2
ALARM_VF_GATE_MS = 2500
ALARM_VF_RMS_MIN = np.float32(0.05)
ALARM_ASYSTOLE_MS = 4000
DC_OFFSET_REMOVE = np.float32(1.65)

# Truncated coefficients stored in vf_detect.cpp (not a fresh butter() call).
SOS = np.array([
    [2.67349040e-05, 5.34698080e-05, 2.67349040e-05, 1.0, -1.81135233, 0.846093824],
    [1.0, 2.0, 1.0, 1.0, -1.87777543, 0.893812835],
    [1.0, -2.0, 1.0, 1.0, -1.86519459, 0.922431828],
    [1.0, -2.0, 1.0, 1.0, -1.95573823, 0.966220895],
], dtype=np.float32)

MEAN = np.array([
    0.42513171332031363, 0.24363859527220647, 0.25254772741265197,
    0.04137449856733724, 43.309992836676216, 4.105229226361016,
], dtype=np.float32)
STD = np.array([
    0.4191231586420349, 0.3631392072566181, 0.12891100945922151,
    0.004458203265709307, 9.81194160995198, 1.7036455584085064,
], dtype=np.float32)
COEF = np.array([
    0.6866138077335683, 4.134092762658282, -0.8109653996849735,
    1.2396861763592981, -1.2619362566532957, 1.3262778596404012,
], dtype=np.float32)
INTERCEPT = np.float32(0.24291412543887136)

FEAT_NAMES = ["rms", "med_abs", "vf_ratio", "vf_zcr", "pv_rate", "dom_freq"]


def _f32(v) -> np.float32:
    return np.float32(v)


def bandpass_zero_state(x: np.ndarray) -> np.ndarray:
    """4-section transposed DF2, forward then backward, states cleared both times."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    sos = SOS
    fwd = np.empty(n, dtype=np.float32)
    s = np.zeros((4, 2), dtype=np.float32)
    for i in range(n):
        v = x[i]
        for k in range(4):
            b0, b1, b2, _a0, a1, a2 = sos[k]
            y = _f32(b0 * v + s[k, 0])
            s[k, 0] = _f32(b1 * v - a1 * y + s[k, 1])
            s[k, 1] = _f32(b2 * v - a2 * y)
            v = y
        fwd[i] = v
    out = np.empty(n, dtype=np.float32)
    s[:] = 0
    for i in range(n - 1, -1, -1):
        v = fwd[i]
        for k in range(4):
            b0, b1, b2, _a0, a1, a2 = sos[k]
            y = _f32(b0 * v + s[k, 0])
            s[k, 0] = _f32(b1 * v - a1 * y + s[k, 1])
            s[k, 1] = _f32(b2 * v - a2 * y)
            v = y
        out[i] = v
    return out


def features_firmware(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] != WIN_N:
        raise ValueError(f"expected {WIN_N} samples, got {x.shape[0]}")
    mean = _f32(0)
    for v in x:
        mean = _f32(mean + v)
    mean = _f32(mean / _f32(WIN_N))
    work = np.empty(WIN_N, dtype=np.float32)
    acc = _f32(0)
    for i, v in enumerate(x):
        d = _f32(v - mean)
        work[i] = d
        acc = _f32(acc + d * d)
    rms = _f32(np.sqrt(_f32(acc / _f32(WIN_N))))

    ab = np.abs(work).astype(np.float32, copy=False)
    ab = np.sort(ab, kind="quicksort")
    med = _f32(_f32(ab[WIN_N // 2 - 1] + ab[WIN_N // 2]) * _f32(0.5))

    xf = bandpass_zero_state(work)
    vf_e = _f32(0)
    zc = 0
    prev_pos = bool(xf[0] > 0)
    vf_e = _f32(vf_e + xf[0] * xf[0])
    for i in range(1, WIN_N):
        vf_e = _f32(vf_e + xf[i] * xf[i])
        pos = bool(xf[i] > 0)
        if pos != prev_pos:
            zc += 1
        prev_pos = pos
    vf_ratio = _f32(vf_e / _f32(rms * rms * _f32(WIN_N) + _f32(1e-12)))
    vf_zcr = _f32(_f32(zc) / _f32(WIN_N))

    n_pk = 0
    for i in range(1, WIN_N - 1):
        if work[i] > work[i - 1] and work[i] >= work[i + 1]:
            n_pk += 1
    pv_rate = _f32(_f32(n_pk) / _f32(5.0))

    zc_raw = 0
    prev_pos = bool(work[0] > 0)
    for i in range(1, WIN_N):
        pos = bool(work[i] > 0)
        if pos != prev_pos:
            zc_raw += 1
        prev_pos = pos
    dom = _f32(_f32(zc_raw) / _f32(WIN_N) * _f32(125.0))
    return np.array([rms, med, vf_ratio, vf_zcr, pv_rate, dom], dtype=np.float32)


def score_features(feat: np.ndarray) -> float:
    z = INTERCEPT
    feat = np.asarray(feat, dtype=np.float32)
    for i in range(6):
        z = _f32(z + COEF[i] * _f32((feat[i] - MEAN[i]) / STD[i]))
    zf = float(z)
    # expf is libm; float32 sigmoid matches the firmware expression.
    return float(_f32(1.0) / _f32(1.0 + np.exp(np.float32(-zf))))


def legacy_features(x: np.ndarray) -> np.ndarray:
    """The previous attachment script (32_vf_rules_eval.py), kept for the gap table.

    That script used scipy sosfiltfilt (default padding was even replaced by
    padlen=0 in a later in-repo port), an amplitude ratio, np.sign zero-cross
    counts, and extrema rather than peaks. Here padlen=0 scipy filtfilt is used
    when scipy is importable; otherwise the bandpass is the firmware one and
    only the other formula differences show up.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    med_abs = float(np.median(np.abs(x)))
    try:
        from scipy.signal import sosfiltfilt
        xf = sosfiltfilt(SOS.astype(np.float64), x, padlen=0)
    except Exception:
        xf = bandpass_zero_state(x.astype(np.float32)).astype(np.float64)
    vf_ratio = float(np.sqrt(np.mean(xf * xf)) / (rms + 1e-12))
    zc = float(np.sum(np.abs(np.diff(np.sign(xf)))) / (2 * len(xf)))
    d = np.diff(x)
    s = np.sign(d)
    s[s == 0] = 1
    ext = int(np.sum(s[:-1] * s[1:] < 0))
    pvr = float(ext / (len(x) / FS))
    dom = float(zc * FS / 2.0)
    return np.array([rms, med_abs, vf_ratio, zc, pvr, dom], dtype=np.float64)
