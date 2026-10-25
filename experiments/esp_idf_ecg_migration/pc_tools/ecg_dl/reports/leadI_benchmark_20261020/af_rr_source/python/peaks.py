"""Peak matching and RR construction (all AF paths share this RR recipe)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from constants import PEAK_MATCH_TOL_S


def rr_from_peaks(t_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rr_s, t_end_s) from strictly increasing peak times in seconds."""
    t = np.asarray(t_s, dtype=np.float64)
    t = np.unique(t)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    rr = np.diff(t)
    return rr, t[1:]


def match_peaks(
    ref_s: np.ndarray,
    det_s: np.ndarray,
    tol_s: float = PEAK_MATCH_TOL_S,
) -> dict:
    """Greedy unique matching within ±tol. Returns TP/FP/FN plus index maps."""
    ref = np.asarray(ref_s, dtype=np.float64)
    det = np.asarray(det_s, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    det = det[np.isfinite(det)]
    n_ref, n_det = ref.size, det.size
    if n_ref == 0:
        return {
            "tp": 0,
            "fn": 0,
            "fp": int(n_det),
            "se": float("nan"),
            "ppv": 0.0 if n_det else float("nan"),
            "ref_matched": np.zeros(0, dtype=bool),
            "det_matched": np.zeros(n_det, dtype=bool),
            "err_s": np.zeros(0, dtype=np.float64),
        }
    matched_ref = np.zeros(n_ref, dtype=bool)
    matched_det = np.zeros(n_det, dtype=bool)
    err = []
    j0 = 0
    for i, t in enumerate(ref):
        while j0 < n_det and det[j0] < t - tol_s:
            j0 += 1
        best = None
        best_abs = tol_s + 1
        j = j0
        while j < n_det and det[j] <= t + tol_s:
            if not matched_det[j]:
                d = abs(det[j] - t)
                if d < best_abs:
                    best_abs = d
                    best = j
            j += 1
        if best is not None:
            matched_ref[i] = True
            matched_det[best] = True
            err.append(det[best] - t)
    tp = int(matched_ref.sum())
    fn = n_ref - tp
    fp = n_det - int(matched_det.sum())
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "se": (tp / n_ref) if n_ref else float("nan"),
        "ppv": (tp / n_det) if n_det else float("nan"),
        "ref_matched": matched_ref,
        "det_matched": matched_det,
        "err_s": np.asarray(err, dtype=np.float64),
    }


def segment_mask(times_s: np.ndarray, mask_sec: np.ndarray) -> np.ndarray:
    """Boolean mask over peak times using a 1 Hz boolean series."""
    t = np.asarray(times_s, dtype=np.float64)
    m = np.asarray(mask_sec, dtype=bool)
    n = len(m)
    idx = np.floor(t).astype(int)
    out = np.zeros(t.size, dtype=bool)
    ok = (idx >= 0) & (idx < n)
    out[ok] = m[idx[ok]]
    return out
