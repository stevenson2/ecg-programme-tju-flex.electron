"""WFDB helpers: locate records, pick a channel, load rhythm + beat annotations."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import wfdb

from labels import beat_times_from_ann, rhythm_intervals_from_ann, Interval


PREFERRED_CHANNELS = ("MLII", "ECG1", "I", "LEAD I", "LEADI", "V5", "ECG")


def list_records(db_dir: Path) -> List[str]:
    heas = sorted(db_dir.rglob("*.hea"))
    recs = []
    seen = set()
    for h in heas:
        if h.name.startswith("."):
            continue
        rec = h.stem
        # skip noise-only headers without matching dat of the same stem in some layouts
        dat = h.with_suffix(".dat")
        if not dat.exists():
            continue
        if rec in seen:
            continue
        seen.add(rec)
        recs.append(rec)
    return recs


def record_path(db_dir: Path, rec: str) -> str:
    direct = db_dir / rec
    if direct.with_suffix(".hea").exists():
        return str(direct)
    matches = list(db_dir.rglob(f"{rec}.hea"))
    if not matches:
        raise FileNotFoundError(f"no header for {rec} under {db_dir}")
    return str(matches[0].with_suffix(""))


def pick_channel(sig_name) -> Tuple[int, str]:
    names = [str(n).upper().strip() for n in list(sig_name)]
    for pref in PREFERRED_CHANNELS:
        if pref in names:
            return names.index(pref), names[names.index(pref)]
    return 0, (names[0] if names else "?")


def load_signal(db_dir: Path, rec: str) -> dict:
    path = record_path(db_dir, rec)
    obj = wfdb.rdrecord(path)
    ch, chname = pick_channel(obj.sig_name)
    sig = np.asarray(obj.p_signal[:, ch], dtype=np.float64)
    # replace NaN by linear interpolation then zeros
    if np.isnan(sig).any():
        n = np.arange(sig.size)
        ok = np.isfinite(sig)
        if ok.any():
            sig = np.interp(n, n[ok], sig[ok])
        sig = np.nan_to_num(sig, nan=0.0)
    fs = float(obj.fs)
    duration_s = sig.size / fs
    return {
        "path": path,
        "sig": sig,
        "fs": fs,
        "duration_s": duration_s,
        "channel": ch,
        "channel_name": chname,
        "sig_name": list(obj.sig_name),
        "rec": rec,
    }


def _try_ann(path: str, ext: str):
    try:
        return wfdb.rdann(path, ext)
    except Exception:
        return None


def load_rhythm(db_dir: Path, rec: str, duration_s: float, fs: float) -> List[Interval]:
    path = record_path(db_dir, rec)
    for ext in ("atr", "qrs", "ann"):
        ann = _try_ann(path, ext)
        if ann is None:
            continue
        aux = list(ann.aux_note) if getattr(ann, "aux_note", None) is not None else []
        if any(parse_has_rhythm(a) for a in aux):
            return rhythm_intervals_from_ann(ann.sample, aux, float(ann.fs or fs), duration_s)
    return rhythm_intervals_from_ann([], [], fs, duration_s)


def parse_has_rhythm(aux) -> bool:
    if not aux:
        return False
    return "(" in str(aux)


def load_beats(db_dir: Path, rec: str, prefer: str = "auto") -> Tuple[np.ndarray, str]:
    """Return (peak_times_s, source_ext).

    prefer='manual' uses .atr (MIT-BIH / LTAFDB reviewed beats).
    prefer='auto' uses .qrs if present else .atr.
    """
    path = record_path(db_dir, rec)
    order = ["atr"] if prefer == "manual" else ["qrs", "atr"]
    last_err = None
    for ext in order:
        ann = _try_ann(path, ext)
        if ann is None:
            continue
        fs = float(ann.fs) if ann.fs else None
        if not fs:
            hea = wfdb.rdheader(path)
            fs = float(hea.fs)
        t = beat_times_from_ann(ann.sample, ann.symbol, fs)
        if t.size == 0 and ext == "qrs":
            t = np.asarray(ann.sample, dtype=np.float64) / fs
        if t.size > 0:
            return t, ext
        last_err = f"{ext} empty"
    _ = last_err
    return np.zeros(0, dtype=np.float64), "none"
