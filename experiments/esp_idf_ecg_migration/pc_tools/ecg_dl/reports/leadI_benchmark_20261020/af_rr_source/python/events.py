"""Event construction and matching. Undetermined time stays in the denominator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from constants import ALARM_MERGE_GAP_S, MIN_EPISODE_S
from labels import Interval


@dataclass
class Episode:
    start_s: float
    end_s: float
    detected: bool = False
    delay_s: Optional[float] = None
    miss_reason: str = ""  # hit / called_normal / undetermined / none

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def runs_from_mask(mask: np.ndarray, merge_gap_s: float = 0.0) -> List[Tuple[int, int]]:
    """Inclusive-exclusive integer runs [start, end) on a 1 Hz mask, optional gap merge."""
    m = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < n and m[j]:
            j += 1
        runs.append((i, j))
        i = j
    if merge_gap_s <= 0 or len(runs) < 2:
        return runs
    merged = [runs[0]]
    for a, b in runs[1:]:
        pa, pb = merged[-1]
        if a - pb <= merge_gap_s:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    return merged


def reference_episodes(
    y_sec: np.ndarray,
    min_duration_s: float = MIN_EPISODE_S,
) -> List[Episode]:
    out = []
    for a, b in runs_from_mask(np.asarray(y_sec) == 1, merge_gap_s=0.0):
        if (b - a) >= min_duration_s:
            out.append(Episode(float(a), float(b)))
    return out


def predicted_alarm_episodes(
    pred_sec: np.ndarray,
    merge_gap_s: float = ALARM_MERGE_GAP_S,
    min_duration_s: float = 1.0,
) -> List[Episode]:
    out = []
    for a, b in runs_from_mask(np.asarray(pred_sec) == 1, merge_gap_s=merge_gap_s):
        if (b - a) >= min_duration_s:
            out.append(Episode(float(a), float(b)))
    return out


def _overlap(a: Episode, b: Episode) -> float:
    return max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))


def match_events(
    ref: Sequence[Episode],
    pred: Sequence[Episode],
    pred_sec: np.ndarray,
) -> Tuple[List[Episode], List[Episode]]:
    """Mark TP/FN on a copy of ref; return (ref_marked, independent FP episodes).

    A reference episode is detected if any predicted episode overlaps it.
    Delay = first predicted-AF second inside the episode minus onset (>=0).
    If the overlapping alarm seconds inside the episode are all missing (shouldn't
    happen if overlap>0), fall back to predicted episode start.
    Miss reason:
      hit | called_normal (AF seconds exist as label 0, none as 1)
      | undetermined (AF seconds only label 2 / unpainted)
    """
    pred_sec = np.asarray(pred_sec)
    n = len(pred_sec)
    ref_out = [
        Episode(r.start_s, r.end_s, False, None, "none") for r in ref
    ]
    pred_hit = [False] * len(pred)
    for i, r in enumerate(ref_out):
        hit_j = None
        for j, p in enumerate(pred):
            if _overlap(r, p) > 0:
                hit_j = j
                break
        a = max(0, int(np.floor(r.start_s)))
        b = min(n, int(np.ceil(r.end_s)))
        sl = pred_sec[a:b] if b > a else np.array([], dtype=pred_sec.dtype)
        if hit_j is not None:
            r.detected = True
            r.miss_reason = "hit"
            pred_hit[hit_j] = True
            # also mark every overlapping pred as used so it is not an FP
            for j, p in enumerate(pred):
                if _overlap(r, p) > 0:
                    pred_hit[j] = True
            if sl.size and np.any(sl == 1):
                first = int(np.argmax(sl == 1))
                r.delay_s = float(first)
            else:
                r.delay_s = max(0.0, pred[hit_j].start_s - r.start_s)
            continue
        if sl.size == 0:
            r.miss_reason = "undetermined"
            continue
        n1 = int(np.sum(sl == 1))
        n0 = int(np.sum(sl == 0))
        n2 = int(np.sum(sl == 2))
        if n1 > 0:
            r.detected = True
            r.miss_reason = "hit"
            r.delay_s = float(int(np.argmax(sl == 1)))
        elif n0 > 0:
            r.miss_reason = "called_normal"
        else:
            r.miss_reason = "undetermined"
            _ = n2
    fp = [p for p, h in zip(pred, pred_hit) if not h]
    # An FP that lands entirely in excluded seconds (label -1 on y) is filtered
    # by the caller via pred constructed from (pred==1 & y!=-1) if desired.
    return ref_out, fp


def analyzable_stats(pred_sec: np.ndarray) -> dict:
    """Report undetermined time separately; do not drop it from the denominator."""
    x = np.asarray(pred_sec)
    n = len(x)
    if n == 0:
        return {
            "duration_s": 0.0,
            "analyzable_s": 0.0,
            "undetermined_s": 0.0,
            "analyzable_frac": 0.0,
            "undetermined_frac": 0.0,
            "longest_undetermined_s": 0.0,
            "n_pred_af_s": 0.0,
            "n_pred_normal_s": 0.0,
        }
    und = x == 2
    if n == 0:
        longest = 0
    else:
        padded = np.concatenate(([0], und.astype(np.int8, copy=False), [0]))
        d = np.diff(padded)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        longest = int((ends - starts).max()) if starts.size else 0
    return {
        "duration_s": float(n),
        "analyzable_s": float(np.sum(x != 2)),
        "undetermined_s": float(np.sum(und)),
        "analyzable_frac": float(np.mean(x != 2)),
        "undetermined_frac": float(np.mean(und)),
        "longest_undetermined_s": float(longest),
        "n_pred_af_s": float(np.sum(x == 1)),
        "n_pred_normal_s": float(np.sum(x == 0)),
    }
