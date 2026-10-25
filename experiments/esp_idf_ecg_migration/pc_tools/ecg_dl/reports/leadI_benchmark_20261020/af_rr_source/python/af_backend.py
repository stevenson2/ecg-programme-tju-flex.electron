"""Firmware-faithful AF RR irregularity detector (af_detect.cpp).

Streaming 10 s window, CV + Shannon entropy, 50% time drop after a
0/1 decision.  min-RR / SQI failures do not slide the window.
SQI defaults to 1.0 so RR-source comparisons are not confounded by
firmware SQI; callers may pass real SQI if they have it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from constants import (
    AF_CV_THR,
    AF_ENT_BINS,
    AF_ENT_HI,
    AF_ENT_LO,
    AF_ENT_THR,
    AF_MAX_RR_S,
    AF_MIN_RR,
    AF_RR_BUF,
    AF_SQI_GATE,
    AF_WIN_S,
)


def shannon_entropy_fw(
    rr: Sequence[float],
    lo: float = AF_ENT_LO,
    hi: float = AF_ENT_HI,
    bins: int = AF_ENT_BINS,
) -> float:
    """Match afShannonEntropy: clip to [lo, hi], C-style bin, natural log."""
    hist = np.zeros(bins, dtype=np.int64)
    span = hi - lo
    n = 0
    for v in rr:
        x = float(v)
        if x < lo:
            x = lo
        if x > hi:
            x = hi
        b = int((x - lo) / span * bins)
        if b >= bins:
            b = bins - 1
        if b < 0:
            b = 0
        hist[b] += 1
        n += 1
    if n <= 0:
        return 0.0
    ent = 0.0
    for c in hist:
        if c == 0:
            continue
        p = c / n
        ent -= p * np.log(p)
    return float(ent)


@dataclass
class AFWindow:
    t_end: float
    t_start: float
    label: int  # 0 normal, 1 AF, 2 undetermined
    cv: float
    entropy: float
    score: float
    n_rr: int
    window_ready: bool
    reason: str


# Compact on-disk / in-RAM form. A 24 h record can emit ~50k windows;
# dataclass lists of those across LTAFDB do not fit in 16 GB.
REASON_CODE = {
    "normal": 0,
    "af": 1,
    "min_rr": 2,
    "sqi": 3,
    "warmup": 4,
    "no_beat": 5,
}
REASON_NAME = {v: k for k, v in REASON_CODE.items()}
SKIP_RELABEL_CODES = (2, 3, 4, 5)  # min_rr / sqi / warmup / no_beat
SKIP_RELABEL_REASONS = frozenset({"min_rr", "sqi", "warmup", "no_beat"})

WIN_DTYPE = np.dtype(
    [
        ("t_end", "f8"),
        ("t_start", "f8"),
        ("label", "i1"),
        ("cv", "f8"),
        ("entropy", "f8"),
        ("score", "f4"),
        ("n_rr", "i2"),
        ("window_ready", "i1"),
        ("reason", "i1"),
    ]
)

WindowsLike = Union[Sequence[AFWindow], np.ndarray]


def windows_to_array(windows: WindowsLike) -> np.ndarray:
    if isinstance(windows, np.ndarray):
        return np.asarray(windows, dtype=WIN_DTYPE)
    n = len(windows)
    arr = np.empty(n, dtype=WIN_DTYPE)
    for i, w in enumerate(windows):
        arr[i]["t_end"] = w.t_end
        arr[i]["t_start"] = w.t_start
        arr[i]["label"] = w.label
        arr[i]["cv"] = w.cv
        arr[i]["entropy"] = w.entropy
        arr[i]["score"] = w.score
        arr[i]["n_rr"] = w.n_rr
        arr[i]["window_ready"] = 1 if w.window_ready else 0
        arr[i]["reason"] = REASON_CODE.get(w.reason, 0)
    return arr


def array_to_windows(arr: np.ndarray) -> List[AFWindow]:
    out: List[AFWindow] = []
    for w in np.asarray(arr):
        out.append(
            AFWindow(
                t_end=float(w["t_end"]),
                t_start=float(w["t_start"]),
                label=int(w["label"]),
                cv=float(w["cv"]),
                entropy=float(w["entropy"]),
                score=float(w["score"]),
                n_rr=int(w["n_rr"]),
                window_ready=bool(w["window_ready"]),
                reason=REASON_NAME.get(int(w["reason"]), "normal"),
            )
        )
    return out


class FirmwareAF:
    """Stateful replica of afProcess() called once per accepted beat."""

    def __init__(
        self,
        win_s: float = AF_WIN_S,
        min_rr: int = AF_MIN_RR,
        cv_thr: float = AF_CV_THR,
        ent_thr: float = AF_ENT_THR,
        sqi_gate: float = AF_SQI_GATE,
        rr_buf: int = AF_RR_BUF,
        max_rr: float = AF_MAX_RR_S,
    ) -> None:
        self.win_s = float(win_s)
        self.min_rr = int(min_rr)
        self.cv_thr = float(cv_thr)
        self.ent_thr = float(ent_thr)
        self.sqi_gate = float(sqi_gate)
        self.buf_n = int(rr_buf)
        self.max_rr = float(max_rr)
        self.reset()

    def reset(self) -> None:
        self.rr_buf = np.zeros(self.buf_n, dtype=np.float64)
        self.sqi_buf = np.zeros(self.buf_n, dtype=np.float64)
        self.head = 0
        self.count = 0
        self.elapsed = 0.0
        self.t_abs = 0.0

    def process_beat(self, rr: float, sqi: float = 1.0) -> AFWindow:
        rr = float(rr)
        sqi = float(sqi)
        out = AFWindow(
            t_end=self.t_abs,
            t_start=self.t_abs,
            label=2,
            cv=0.0,
            entropy=0.0,
            score=0.0,
            n_rr=0,
            window_ready=False,
            reason="no_beat",
        )
        if rr <= 0.0 or rr > self.max_rr:
            return out
        self.t_abs += rr
        out.t_end = self.t_abs
        self.rr_buf[self.head] = rr
        self.sqi_buf[self.head] = sqi
        self.head = (self.head + 1) % self.buf_n
        if self.count < self.buf_n:
            self.count += 1
        self.elapsed += rr
        if self.elapsed < self.win_s:
            out.reason = "warmup"
            return out

        acc = 0.0
        s = 0.0
        s2 = 0.0
        sq = 0.0
        n = 0
        i0 = (self.head + self.buf_n - self.count) % self.buf_n
        rr_win: List[float] = []
        for k in range(self.count):
            idx = (i0 + k) % self.buf_n
            v = float(self.rr_buf[idx])
            acc += v
            s += v
            s2 += v * v
            sq += float(self.sqi_buf[idx])
            n += 1
            rr_win.append(v)
            if acc >= self.win_s:
                break
        out.n_rr = n
        out.window_ready = acc >= self.win_s
        out.t_start = self.t_abs - acc
        if n < self.min_rr:
            out.reason = "min_rr"
            return out
        if (sq / n) < self.sqi_gate:
            out.reason = "sqi"
            return out

        mean = s / n
        var = s2 / n - mean * mean
        if var < 0.0:
            var = 0.0
        sd = float(np.sqrt(var))
        cv = (sd / mean) if mean > 1e-6 else 0.0
        ent = shannon_entropy_fw(rr_win)
        score = 0.5 * (cv / 0.2) + 0.5 * (ent / 4.5)
        if cv > self.cv_thr and ent > self.ent_thr:
            label = 1
            reason = "af"
        else:
            label = 0
            reason = "normal"
        out.label = label
        out.cv = float(cv)
        out.entropy = float(ent)
        out.score = float(score)
        out.reason = reason

        drop = 0.0
        n_keep = 0
        for k in range(self.count):
            idx = (i0 + k) % self.buf_n
            if drop >= self.win_s * 0.5:
                n_keep += 1
            drop += float(self.rr_buf[idx])
        drop_n = self.count - n_keep
        tmp_rr = np.zeros(n_keep, dtype=np.float64)
        tmp_sqi = np.zeros(n_keep, dtype=np.float64)
        start = (i0 + drop_n) % self.buf_n
        for k in range(n_keep):
            idx = (start + k) % self.buf_n
            tmp_rr[k] = self.rr_buf[idx]
            tmp_sqi[k] = self.sqi_buf[idx]
        self.rr_buf[:] = 0.0
        self.sqi_buf[:] = 0.0
        if n_keep:
            self.rr_buf[:n_keep] = tmp_rr
            self.sqi_buf[:n_keep] = tmp_sqi
        self.head = n_keep
        self.count = n_keep
        self.elapsed -= self.win_s * 0.5
        return out

    def evaluate_rr(
        self, rr: Iterable[float], sqi: Optional[Iterable[float]] = None
    ) -> List[AFWindow]:
        self.reset()
        rr = list(rr)
        if sqi is None:
            sqi_l = [1.0] * len(rr)
        else:
            sqi_l = list(sqi)
            if len(sqi_l) != len(rr):
                raise ValueError("sqi length must match rr")
        out: List[AFWindow] = []
        for r, q in zip(rr, sqi_l):
            w = self.process_beat(r, q)
            # Firmware still classifies when s_elapsed >= WIN even if the
            # summed RR in the ring is slightly short (elapsed drift after
            # subtracting WIN/2). Keep 0/1 decisions and min_rr/sqi refusals.
            if w.label in (0, 1) or w.reason in ("min_rr", "sqi"):
                out.append(w)
        return out


def relabel_windows(
    windows: WindowsLike,
    cv_thr: float,
    ent_thr: float,
) -> WindowsLike:
    """Thresholds only flip 0/1. Window geometry is independent of CV/entropy gates."""
    if isinstance(windows, np.ndarray):
        out = np.array(windows, dtype=WIN_DTYPE, copy=True)
        reason = out["reason"]
        skip = np.isin(reason, SKIP_RELABEL_CODES)
        af = (~skip) & (out["cv"] > cv_thr) & (out["entropy"] > ent_thr)
        out["label"] = np.where(skip, out["label"], np.where(af, 1, 0)).astype(np.int8)
        out["reason"] = np.where(skip, reason, np.where(af, 1, 0)).astype(np.int8)
        return out
    out_l: List[AFWindow] = []
    for w in windows:
        if w.reason in SKIP_RELABEL_REASONS:
            out_l.append(w)
            continue
        if w.cv > cv_thr and w.entropy > ent_thr:
            out_l.append(replace(w, label=1, reason="af"))
        else:
            out_l.append(replace(w, label=0, reason="normal"))
    return out_l


def windows_to_second_labels(
    windows: WindowsLike,
    duration_s: float,
    win_s: float = AF_WIN_S,
) -> np.ndarray:
    """Causal 1 Hz series: later windows overwrite earlier ones.

    Unpainted seconds stay 2 (undetermined), including warmup.
    """
    n = max(1, int(np.ceil(float(duration_s))))
    lab = np.full(n, 2, dtype=np.int8)
    if isinstance(windows, np.ndarray):
        if windows.size == 0:
            return lab
        t_end = windows["t_end"]
        labels = windows["label"]
        for te, lb in zip(t_end, labels):
            t1 = int(np.floor(te))
            t0 = int(np.floor(te - win_s))
            if t1 <= 0:
                continue
            t0 = max(0, t0)
            t1 = min(n, max(t1, t0 + 1))
            lab[t0:t1] = int(lb)
        return lab
    for w in windows:
        t1 = int(np.floor(w.t_end))
        t0 = int(np.floor(w.t_end - win_s))
        if t1 <= 0:
            continue
        t0 = max(0, t0)
        t1 = min(n, max(t1, t0 + 1))
        lab[t0:t1] = int(w.label)
    return lab
