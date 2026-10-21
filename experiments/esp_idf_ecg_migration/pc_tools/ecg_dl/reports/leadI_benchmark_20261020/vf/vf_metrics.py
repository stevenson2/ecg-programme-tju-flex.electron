"""Window labels, beat gate, episode scores, and confidence intervals.

Event definition (also stated in the report):

* A reference episode is one contiguous rhythm label whose token is shockable
  (VFIB, VF, VT, VFL). Adjacent different tokens stay separate episodes.
* The firmware confirms on two non-overlapping 5 s windows, so an episode
  shorter than 10 s cannot contain two full windows. Primary event sensitivity
  uses episodes of duration >= 10 s. Episodes >= 15 s are also reported: that
  length always covers two complete 5 s windows no matter where it starts.
* A confirmed alarm window is the half-open interval [5i, 5(i+1)) seconds on
  the host clock. The episode is detected when any gated alarm window overlaps
  it. Overlap is any shared time, not a 50% rule.
* Window sensitivity uses a different cut: a window is positive when shockable
  time is >= 50% and noise is < 50%. Noise-majority, unknown-majority, and
  asystole-majority windows are excluded from the VF confusion counts.
* A false-alarm episode is a run of consecutive gated alarm windows that does
  not overlap any shockable reference episode.
* The beat gate matches main.cc: the record must already have produced a QRS
  (s_beatEverSeen), and the completing sample of the alarm window must be
  strictly more than 2500 ms after the latest QRS at or before that sample.
  Frame math on the 500 Hz front-end: time_ms = frame_index * 2, and the
  comparison is `> 2500`, so a gap of exactly 2500 ms does not open the gate.
  RMS uses the completing window only (`vf.lastRms >= 0.05`).
"""
from __future__ import annotations

import numpy as np

SHOCK_TOKENS = {"VFIB", "VF", "VT", "VFL"}
ASYS_TOKENS = {"ASYS", "ASYSTOLE"}
NOISE_PREFIXES = ("NOISE",)
BEAT_SYMS = set("NLRBAaJSVrejnE/fQ?")

# MIT-BIH 48, VFDB 22, CUDB 35. Names checked after download.
MIT_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
VFDB_RECORDS = [
    "418", "419", "420", "421", "422", "423", "424", "425", "426", "427",
    "428", "429", "430", "602", "605", "607", "609", "610", "611", "612",
    "614", "615",
]
CUDB_RECORDS = [f"cu{i:02d}" for i in range(1, 36)]


def rhythm_token(aux: str | None) -> str | None:
    if aux is None:
        return None
    s = str(aux).replace("\x00", "").strip().upper()
    if not s:
        return None
    if s[0] == "(":
        s = s[1:].strip()
    if not s:
        return None
    tok = s.split()[0].split(",")[0].strip()
    return tok or None


def is_shock(tok: str | None) -> bool:
    return tok in SHOCK_TOKENS


def is_noise(tok: str | None) -> bool:
    return bool(tok) and tok.startswith(NOISE_PREFIXES)


def is_asys(tok: str | None) -> bool:
    return tok in ASYS_TOKENS


def rhythm_segments(sample, symbol, aux, duration_s: float, fs: float):
    """Rhythm track from `.atr`.

    Parenthetical rhythm labels are taken only from symbol `+` (AHA/PhysioNet
    rhythm annotations). A `~` or beat symbol that happens to carry `(N` does
    not change the rhythm. Time before the first label is UNK.

    CUDB marks VF onset with `[` and offset with `]`. The `]` annotation has
    an empty aux note, so it is not a `(...)` label. `[` starts VF when the
    current rhythm is not already shockable (`+ (VF` a few samples earlier is
    the same onset and is not split). `]` ends the shockable rhythm and
    restores the previous non-shockable label, or `N` when the file never
    named one. CUDB does not write a rhythm after `]`; leaving the tail as VF
    would score post-VF sinus as shockable.
    """
    events = []
    order = {"rhythm": 0, "vf_on": 1, "vf_off": 2}
    for s, sym, note in zip(sample, symbol, aux):
        t = float(s) / float(fs)
        if sym == "+":
            tok = rhythm_token(note)
            if tok:
                events.append((t, "rhythm", tok))
        elif sym == "[":
            tok = rhythm_token(note) or "VF"
            if not is_shock(tok):
                tok = "VF"
            events.append((t, "vf_on", tok))
        elif sym == "]":
            events.append((t, "vf_off", ""))
    events.sort(key=lambda e: (e[0], order[e[1]]))
    if not events:
        return [(0.0, float(duration_s), "UNK")]

    segs = []
    cur = "UNK"
    # `]` with no following rhythm: outside the bracket is not VF.
    # CUDB's own note: at least 61 non-VF beats precede VF, and `[` is only
    # an approximate onset. Time before the first rhythm mark is that
    # non-VF prefix, not an unlabeled unknown. A record with no rhythm
    # marks at all (cu14) stays UNK so a missing VF label is not scored
    # as a clean negative window.
    restore = "N"
    if events and events[0][0] > 0:
        cur = "N"
    t_cur = 0.0

    def close(t1: float) -> None:
        nonlocal t_cur
        if t1 > t_cur:
            segs.append((t_cur, t1, cur))
        t_cur = max(t_cur, t1)

    for t, kind, tok in events:
        if kind == "rhythm":
            if tok == cur:
                continue
            close(t)
            cur = tok
            if not is_shock(tok):
                restore = tok
        elif kind == "vf_on":
            if is_shock(cur):
                continue
            close(t)
            cur = tok
        elif kind == "vf_off":
            if not is_shock(cur):
                continue
            close(t)
            cur = restore
    close(float(duration_s))
    return segs


def add_flutter_wave_episodes(segs, sample, symbol, fs: float, duration_s: float):
    """MIT marks ventricular flutter waves as beat symbol '!' without always
    opening a (VFL rhythm. A run of '!' closer than 1 s becomes a VFL episode
    when that span is not already inside a shockable rhythm."""
    idx = [int(s) for s, y in zip(sample, symbol) if y == "!"]
    if len(idx) < 3:
        return segs
    runs = []
    run = [idx[0]]
    for s in idx[1:]:
        if (s - run[-1]) / fs <= 1.0:
            run.append(s)
        else:
            runs.append(run)
            run = [s]
    runs.append(run)
    extra = []
    for run in runs:
        t0 = run[0] / fs
        t1 = run[-1] / fs
        if t1 - t0 < 1.0:
            continue
        covered = 0.0
        for a, b, tok in segs:
            if is_shock(tok):
                covered += max(0.0, min(b, t1) - max(a, t0))
        if covered >= 0.5 * (t1 - t0):
            continue
        extra.append((t0, t1, "VFL"))
    if not extra:
        return segs
    return sorted(segs + extra, key=lambda z: z[0])


def beat_times_from_ann(sample, symbol, fs: float) -> np.ndarray:
    t = [float(s) / float(fs) for s, y in zip(sample, symbol) if y in BEAT_SYMS]
    return np.asarray(t, dtype=np.float64)


def overlap_seconds(segs, t0: float, t1: float, pred) -> float:
    acc = 0.0
    for a, b, tok in segs:
        if pred(tok):
            acc += max(0.0, min(b, t1) - max(a, t0))
    return acc


def window_class(segs, t0: float, t1: float) -> int:
    """1 positive, 0 organized negative, -1 excluded (noise/unk), -2 asystole."""
    dur = t1 - t0
    if dur <= 0:
        return -1
    noise = overlap_seconds(segs, t0, t1, is_noise)
    unk = overlap_seconds(segs, t0, t1, lambda tok: tok == "UNK")
    shock = overlap_seconds(segs, t0, t1, is_shock)
    asys = overlap_seconds(segs, t0, t1, is_asys)
    half = 0.5 * dur
    if noise >= half:
        return -1
    if unk >= half:
        return -1
    if shock >= half:
        return 1
    if asys >= half:
        return -2
    return 0


def shock_episodes(segs, min_dur: float = 0.0):
    eps = []
    for a, b, tok in segs:
        if is_shock(tok) and (b - a) >= min_dur:
            eps.append((a, b, tok))
    return eps


def window_end_frame(i: int) -> int:
    """500 Hz frame index of the sample that completes VF window i.

    Even frames are fed to vfProcess. Window i's last VF sample is frame
    2500*(i+1) - 2.
    """
    return 2500 * (i + 1) - 2


def last_leq(sorted_idx: np.ndarray, frame: int) -> int | None:
    if sorted_idx.size == 0:
        return None
    k = int(np.searchsorted(sorted_idx, frame, side="right")) - 1
    if k < 0:
        return None
    return int(sorted_idx[k])


def beat_gate_open(frame_end: int, beats: np.ndarray, gate_ms: int = 2500, require_arm: bool = True) -> bool:
    last = last_leq(beats, frame_end)
    if last is None:
        return not require_arm
    return (frame_end - last) * 2 > gate_ms


def flat_intervals(beats: np.ndarray, n_frames: int):
    """Frames where main.cc would set FLAT: armed and >= 4000 ms since last QRS.

    Returns half-open frame intervals [start, end).
    """
    if beats.size == 0 or n_frames <= 0:
        return []
    b = np.asarray(beats, dtype=np.int64)
    b = b[(b >= 0) & (b < n_frames)]
    if b.size == 0:
        return []
    edges = list(b) + [int(n_frames)]
    out = []
    for a, nxt in zip(edges[:-1], edges[1:]):
        start = int(a) + 2000  # 4000 ms at 2 ms/frame, `>=`
        if start < int(nxt):
            out.append((start, int(nxt)))
    return out


def alarm_episodes(flags: np.ndarray, win_s: float = 5.0):
    flags = np.asarray(flags, dtype=bool)
    eps = []
    i = 0
    n = flags.size
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        eps.append((i * win_s, (j + 1) * win_s, i, j))
        i = j + 1
    return eps


def overlaps_any(t0: float, t1: float, spans) -> bool:
    for a, b, *_rest in spans:
        if min(t1, b) > max(t0, a):
            return True
    return False


def match_beats(ref_t: np.ndarray, det_t: np.ndarray, tol_s: float = 0.150):
    """Greedy one-to-one match. Returns (n_matched, n_ref, n_det)."""
    ref = np.asarray(ref_t, dtype=np.float64)
    det = np.asarray(det_t, dtype=np.float64)
    if ref.size == 0 or det.size == 0:
        return 0, int(ref.size), int(det.size)
    used = np.zeros(det.size, dtype=bool)
    matched = 0
    j = 0
    for t in ref:
        while j < det.size and det[j] < t - tol_s:
            j += 1
        k = j
        best = None
        best_dt = tol_s
        while k < det.size and det[k] <= t + tol_s:
            if not used[k]:
                dt = abs(det[k] - t)
                if dt <= best_dt:
                    best_dt = dt
                    best = k
            k += 1
        if best is not None:
            used[best] = True
            matched += 1
    return matched, int(ref.size), int(det.size)


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return None
    ph = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (ph + z2 / (2.0 * n)) / den
    half = z * np.sqrt(ph * (1.0 - ph) / n + z2 / (4.0 * n * n)) / den
    return {
        "k": int(k),
        "n": int(n),
        "est": float(ph),
        "lo": float(max(0.0, center - half)),
        "hi": float(min(1.0, center + half)),
        "method": "wilson",
    }


def poisson_rate(k: int, hours: float, alpha: float = 0.05):
    if hours <= 0:
        return None
    from scipy.stats import chi2
    k = int(k)
    lo = 0.0 if k == 0 else 0.5 * float(chi2.ppf(alpha / 2.0, 2 * k)) / hours
    hi = 0.5 * float(chi2.ppf(1.0 - alpha / 2.0, 2 * (k + 1))) / hours
    return {
        "k": k,
        "hours": float(hours),
        "est": float(k / hours),
        "lo": float(lo),
        "hi": float(hi),
        "method": "poisson_chi2",
    }


def bootstrap_ratio(kn, n_boot: int = 4000, seed: int = 20261021):
    """Record-cluster bootstrap of pooled k/n. kn is a list of (k, n)."""
    arr = np.asarray(list(kn), dtype=np.float64)
    if arr.size == 0 or arr[:, 1].sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    m = arr.shape[0]
    draw = rng.integers(0, m, size=(n_boot, m))
    ks = arr[draw, 0].sum(axis=1)
    ns = arr[draw, 1].sum(axis=1)
    ok = ns > 0
    p = ks[ok] / ns[ok]
    lo, hi = np.quantile(p, [0.025, 0.975])
    return {
        "est": float(arr[:, 0].sum() / arr[:, 1].sum()),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "method": "record_cluster_bootstrap",
    }


def bootstrap_rate(counts_hours, n_boot: int = 4000, seed: int = 20261021):
    arr = np.asarray(list(counts_hours), dtype=np.float64)
    if arr.size == 0 or arr[:, 1].sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    m = arr.shape[0]
    draw = rng.integers(0, m, size=(n_boot, m))
    ks = arr[draw, 0].sum(axis=1)
    hs = arr[draw, 1].sum(axis=1)
    ok = hs > 0
    p = ks[ok] / hs[ok]
    lo, hi = np.quantile(p, [0.025, 0.975])
    return {
        "est": float(arr[:, 0].sum() / arr[:, 1].sum()),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "method": "record_cluster_bootstrap",
    }


def confirm_from_scores(scores: np.ndarray, theta: float) -> np.ndarray:
    suspect = np.asarray(scores) >= theta
    alarm = np.zeros(suspect.shape[0], dtype=bool)
    if suspect.size == 0:
        return alarm
    prev = False
    for i, s in enumerate(suspect):
        alarm[i] = bool(s and prev)
        prev = bool(s)
    return alarm


def apply_profile(alarm: np.ndarray, rms: np.ndarray, gate: np.ndarray | None,
                  use_rms: bool, use_gate: bool, rms_min: float = 0.05) -> np.ndarray:
    out = np.asarray(alarm, dtype=bool).copy()
    if use_rms:
        out &= np.asarray(rms) >= rms_min
    if use_gate:
        if gate is None:
            raise ValueError("gate required")
        out &= np.asarray(gate, dtype=bool)
    return out
