"""Rhythm interval parsing. AF and AFL are first-class and kept separate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from constants import AFL_CODES, AFIB_CODES, MIN_EPISODE_S, QRS_BEAT_SYMBOLS, WINDOW_OVERLAP_POS


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float
    code: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def parse_rhythm_code(aux: Optional[str]) -> Optional[str]:
    if not aux:
        return None
    s = str(aux).replace("\x00", " ").strip()
    if not s:
        return None
    if s[0] == "(":
        s = s[1:]
    s = s.strip().upper()
    # strip trailing punctuation / extra tokens
    token = s.replace("\t", " ").split(" ")[0].strip(" \t\r\n.,;")
    if not token:
        return None
    if token in AFIB_CODES:
        return "AFIB"
    if token in AFL_CODES:
        return "AFL"
    return token


def rhythm_intervals_from_ann(
    samples: Sequence[int],
    aux_notes: Sequence,
    fs: float,
    duration_s: float,
    symbols: Optional[Sequence[str]] = None,
) -> List[Interval]:
    """Build contiguous rhythm segments from PhysioNet aux notes.

    If the first rhythm marker is after t=0, the leading gap is 'N'.
    If there are no rhythm markers at all, the whole record is 'N'.
    """
    fs = float(fs)
    duration_s = float(duration_s)
    marks: List[Tuple[float, str]] = []
    for i, samp in enumerate(samples):
        aux = aux_notes[i] if i < len(aux_notes) else None
        code = parse_rhythm_code(aux)
        if code is None:
            continue
        t = float(samp) / fs
        if t < 0:
            t = 0.0
        if t > duration_s:
            t = duration_s
        marks.append((t, code))
    if not marks:
        return [Interval(0.0, duration_s, "N")]
    marks.sort(key=lambda x: x[0])
    if marks[0][0] > 0:
        marks = [(0.0, "N")] + marks
    else:
        marks[0] = (0.0, marks[0][1])
    out: List[Interval] = []
    for i, (t, code) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else duration_s
        if end <= t:
            continue
        if out and out[-1].code == code:
            out[-1] = Interval(out[-1].start_s, end, code)
        else:
            out.append(Interval(t, end, code))
    if out and out[-1].end_s < duration_s:
        out[-1] = Interval(out[-1].start_s, duration_s, out[-1].code)
    return out


def beat_times_from_ann(
    samples: Sequence[int],
    symbols: Sequence[str],
    fs: float,
) -> np.ndarray:
    fs = float(fs)
    times = []
    for samp, sym in zip(samples, symbols):
        if str(sym) in QRS_BEAT_SYMBOLS:
            times.append(float(samp) / fs)
    return np.asarray(times, dtype=np.float64)


def episodes_of(
    intervals: Sequence[Interval],
    codes: Iterable[str],
    min_duration_s: float = MIN_EPISODE_S,
) -> List[Interval]:
    want = set(codes)
    return [iv for iv in intervals if iv.code in want and iv.duration_s >= min_duration_s]


def mask_from_intervals(
    intervals: Sequence[Interval],
    duration_s: float,
    codes: Iterable[str],
) -> np.ndarray:
    n = max(1, int(np.ceil(float(duration_s))))
    m = np.zeros(n, dtype=bool)
    want = set(codes)
    for iv in intervals:
        if iv.code not in want:
            continue
        a = max(0, int(np.floor(iv.start_s)))
        b = min(n, int(np.ceil(iv.end_s)))
        if b > a:
            m[a:b] = True
    return m


def second_true_label(
    intervals: Sequence[Interval],
    duration_s: float,
    scheme: str = "af_only",
) -> np.ndarray:
    """Per-second reference label: 1=positive, 0=negative, -1=excluded (AFL in af_only)."""
    n = max(1, int(np.ceil(float(duration_s))))
    y = np.zeros(n, dtype=np.int8)
    af = mask_from_intervals(intervals, duration_s, AFIB_CODES)
    afl = mask_from_intervals(intervals, duration_s, AFL_CODES)
    if scheme == "af_only":
        y[af] = 1
        y[afl & ~af] = -1
    elif scheme == "af_or_afl":
        y[af | afl] = 1
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return y


def window_true_label(
    t_start: float,
    t_end: float,
    y_sec: np.ndarray,
    pos_frac: float = WINDOW_OVERLAP_POS,
) -> int:
    """1 if enough positive seconds, -1 if excluded (AFL) dominates remainder, else 0.

    Undetermined is a detector output, not a reference label.
    """
    n = len(y_sec)
    a = max(0, int(np.floor(t_start)))
    b = min(n, int(np.ceil(t_end)))
    if b <= a:
        return 0
    sl = y_sec[a:b]
    pos = float(np.mean(sl == 1))
    exc = float(np.mean(sl == -1))
    if pos >= pos_frac:
        return 1
    if exc >= pos_frac:
        return -1
    return 0
