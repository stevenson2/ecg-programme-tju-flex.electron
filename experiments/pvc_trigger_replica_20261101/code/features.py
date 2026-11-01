"""Per-trigger context features, cached per split.

F    (n,4) float32, clipped log-ratios: [preRR/ref8, postRR/ref8, post/pre, window_std/ref16]
sd   window std (raw stream units); sref median of the previous 16 window stds (NaN when <4 available)
All quantities are available on device: RR from the trigger train, stds from the ring buffer.
"""
import numpy as np
from common import CACHE, OFF
from dissect import prev_median


def trigger_features(split, M, S):
    p = CACHE / f'{split}_feat.npz'
    if p.exists():
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    a, seg = M['t_anchor'].astype(np.float64), M['t_seg']
    n = len(a)
    same = seg[1:] == seg[:-1]
    d = np.diff(a)
    pre, post = np.full(n, np.nan), np.full(n, np.nan)
    pre[1:][same] = d[same]
    post[:-1][same] = d[same]
    ref = prev_median(pre, 8, seg)
    sd = np.empty(n, np.float64)
    g = M['t_g']
    for s in range(0, n, 65536):
        sd[s:s + 65536] = np.asarray(S[g[s:s + 65536, None] + OFF], np.float32).std(1)
    sref = prev_median(sd, 16, seg)
    with np.errstate(all='ignore'):
        F = np.stack([np.log(pre / ref), np.log(post / ref), np.log(post / pre), np.log(sd / sref)], 1)
    F[~np.isfinite(F)] = 0
    out = dict(F=np.clip(F, -3, 3).astype(np.float32), sd=sd.astype(np.float32), sref=sref.astype(np.float32))
    np.savez(p, **out)
    return out


def rel_scale(feat):
    """Patient-adaptive amplitude scale for rwin: previous-16 median std, own std for the first triggers."""
    s = np.where(np.isfinite(feat['sref']), feat['sref'], feat['sd']).astype(np.float32)
    s[s < 1e-6] = 1.0
    return s


CTX_COLS = {'trigctx': [0, 1, 2, 3], 'trigctx_causal': [0, 3]}
