from pathlib import Path
import numpy as np

CACHE = Path('/tmp/pvcexp/cache')
OFF = np.arange(-125, 125)
LABELS = {0: 'N', 1: 'S', 2: 'V', 3: 'Q', 4: 'unmatched', 5: 'other'}


def load(split):
    S = np.load(CACHE / f'{split}_stream.npy', mmap_mode='r')
    with np.load(CACHE / f'{split}_meta.npz') as f:
        M = {k: f[k] for k in f.files}
    M['t_g'] = M['seg_offset'][M['t_seg']] + M['t_anchor']
    M['r_g'] = M['seg_offset'][M['r_seg']] + M['r_sample']
    M['t_patient'] = M['seg_patient'][M['t_seg']]
    M['r_patient'] = M['seg_patient'][M['r_seg']]
    return S, M


def zwin(S, g):
    """Per-window z-score exactly as ecg_ai.cpp zscore() (population std, floor 1e-6 -> 1)."""
    x = np.asarray(S[np.asarray(g)[:, None] + OFF], np.float32)
    x = x - x.mean(1, keepdims=True)
    sd = np.sqrt((x * x).mean(1, keepdims=True))
    sd[sd < 1e-6] = 1.0
    return (x / sd)[..., None]


def rwin(S, g, scale):
    """Relative-amplitude window: centred, divided by a patient-adaptive scale (not the window's own std),
    log-compressed so that 3x-larger beats stay inside the INT8 input range [-5.77, 6.39]."""
    x = np.asarray(S[np.asarray(g)[:, None] + OFF], np.float32)
    x = x - x.mean(1, keepdims=True)
    u = x / np.asarray(scale, np.float32)[:, None]
    v = 3.0 * np.sign(u) * np.log1p(np.abs(u) / 3.0)
    return np.clip(v, -5.77, 6.39)[..., None]


def pools(M, arm):
    """Training pool per arm. 'ref': literal N/V at reference anchors (current recipe).
    'trig': every firmware trigger; positive iff matched to a reference V."""
    if arm == 'ref':
        keep = M['r_ok'] & np.isin(M['r_label'], [0, 2])
        return M['r_g'][keep], (M['r_label'][keep] == 2).astype(np.int32), M['r_patient'][keep]
    if arm == 'trig':
        return M['t_g'], (M['t_label'] == 2).astype(np.int32), M['t_patient']
    if arm == 'trig_noQ':
        keep = M['t_label'] != 3
        return M['t_g'][keep], (M['t_label'][keep] == 2).astype(np.int32), M['t_patient'][keep]
    if arm == 'trigNV':  # literal N/V like 'ref', but windows at the actual trigger anchor
        keep = np.isin(M['t_label'], [0, 2])
        return M['t_g'][keep], (M['t_label'][keep] == 2).astype(np.int32), M['t_patient'][keep]
    raise ValueError(arm)


def ctx_features(M, S):
    """Causal-plus-one-beat RR and amplitude context per trigger (cheap on device)."""
    a, seg = M['t_anchor'].astype(np.float64), M['t_seg']
    n = len(a)
    pre = np.full(n, np.nan); post = np.full(n, np.nan)
    same_prev = np.r_[False, seg[1:] == seg[:-1]]
    same_next = np.r_[seg[1:] == seg[:-1], False]
    d = np.diff(a)
    pre[1:][same_prev[1:]] = d[same_prev[1:]]
    post[:-1][same_next[:-1]] = d[same_next[:-1]]
    ref = np.full(n, np.nan)
    sd = np.empty(n, np.float32)
    g = M['t_g']
    for s in range(0, n, 65536):
        x = np.asarray(S[g[s:s + 65536, None] + OFF], np.float32)
        sd[s:s + 65536] = x.std(1)
    sref = np.full(n, np.nan)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], n]
    for b, e in zip(starts, ends):
        p = pre[b:e]
        for i in range(e - b):
            w = p[max(0, i - 8):i]
            w = w[np.isfinite(w)]
            if len(w) >= 3:
                ref[b + i] = np.median(w)
            v = sd[b + max(0, i - 16):b + i]
            if len(v) >= 4:
                sref[b + i] = np.median(v)
    f = np.zeros((n, 4), np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        f[:, 0] = np.log(pre / ref)
        f[:, 1] = np.log(post / ref)
        f[:, 2] = np.log(post / pre)
        f[:, 3] = np.log(sd / sref)
    f[~np.isfinite(f)] = 0.0
    return np.clip(f, -3, 3)
