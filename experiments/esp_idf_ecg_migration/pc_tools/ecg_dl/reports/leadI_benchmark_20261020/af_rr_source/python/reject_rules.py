"""Firmware v6 candidate-beat reject rules, applied to a reference RR stream.

These are the three gates named in the hypothesis (heartrate.cpp isQRSValid):
  1. absolute short RR: rr < MIN_RR_SEC (0.480 s)
  2. relative median RR: rr < RR_RATIO_MIN * median of last accepted RRs (0.65)
  3. relative amplitude: rr < 0.9 s AND peak < 0.55 * previous accepted peak

The old ±30% RR_CONSISTENCY block is gated by MIN_CONF_FEAT=1000 and is off.

`filter_rr_stream` keeps the firmware's 'time since last ACCEPTED beat' semantics:
dropping a true beat merges two short intervals into one longer one, which is
exactly how hrProcess would feed AF.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from constants import (
    AMP_FRAC_PREV,
    AMP_FRAC_PREV_RR_MAX,
    MIN_RR_SEC,
    RR_RATIO_MIN,
)


def rolling_median(values: Sequence[float], k: int = 8) -> float:
    if not values:
        return float("nan")
    sl = list(values[-k:])
    sl.sort()
    n = len(sl)
    if n % 2:
        return float(sl[n // 2])
    return 0.5 * (sl[n // 2 - 1] + sl[n // 2])


def classify_ref_rr(
    rr: float,
    accepted_rr: Sequence[float],
    peak: Optional[float] = None,
    prev_peak: Optional[float] = None,
    min_rr: float = MIN_RR_SEC,
    ratio_min: float = RR_RATIO_MIN,
    amp_frac: float = AMP_FRAC_PREV,
    amp_rr_max: float = AMP_FRAC_PREV_RR_MAX,
    min_for_rel: int = 3,
) -> str:
    """Return 'ok' or the first matching reject reason."""
    if rr < min_rr:
        return "short_abs"
    if len(accepted_rr) >= min_for_rel:
        med = rolling_median(accepted_rr)
        if np.isfinite(med) and med >= 0.001 and rr < ratio_min * med:
            return "short_rel_median"
    if (
        peak is not None
        and prev_peak is not None
        and rr < amp_rr_max
        and prev_peak >= 0.0001
        and peak < amp_frac * prev_peak
    ):
        return "low_amp_prev"
    return "ok"


def filter_rr_stream(
    peak_t: np.ndarray,
    peak_amp: Optional[np.ndarray] = None,
    min_rr: float = MIN_RR_SEC,
    ratio_min: float = RR_RATIO_MIN,
) -> dict:
    """Apply abs+rel RR gates with 'since last accepted' timing.

    Amplitude is optional (needs ECG). Returns kept peak times, reject counts,
    and the RR series that an AF backend would see.
    """
    t = np.asarray(peak_t, dtype=np.float64)
    amp = None if peak_amp is None else np.asarray(peak_amp, dtype=np.float64)
    if t.size == 0:
        return {
            "kept_t": t,
            "kept_rr": np.zeros(0),
            "reasons": [],
            "n_in": 0,
            "n_kept": 0,
        }
    kept_t: List[float] = [float(t[0])]
    kept_rr: List[float] = []
    reasons: List[str] = ["first"]
    accepted_rr: List[float] = []
    prev_peak = None if amp is None else float(amp[0])
    for i in range(1, t.size):
        rr = float(t[i] - kept_t[-1])
        pk = None if amp is None else float(amp[i])
        reason = classify_ref_rr(
            rr,
            accepted_rr,
            peak=pk,
            prev_peak=prev_peak,
            min_rr=min_rr,
            ratio_min=ratio_min,
        )
        reasons.append(reason)
        if reason == "ok":
            kept_t.append(float(t[i]))
            kept_rr.append(rr)
            accepted_rr.append(rr)
            if pk is not None:
                prev_peak = pk
    return {
        "kept_t": np.asarray(kept_t, dtype=np.float64),
        "kept_rr": np.asarray(kept_rr, dtype=np.float64),
        "reasons": reasons,
        "n_in": int(t.size),
        "n_kept": len(kept_t),
    }


def rr_cv(rr: np.ndarray) -> float:
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 2:
        return float("nan")
    mu = float(np.mean(rr))
    if mu <= 1e-9:
        return float("nan")
    return float(np.std(rr, ddof=0) / mu)


def short_rr_fraction(rr: np.ndarray, thr: float = MIN_RR_SEC) -> float:
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size == 0:
        return float("nan")
    return float(np.mean(rr < thr))
