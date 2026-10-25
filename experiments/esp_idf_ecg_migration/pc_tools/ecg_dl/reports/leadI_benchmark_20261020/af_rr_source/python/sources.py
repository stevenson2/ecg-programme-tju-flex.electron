"""RR sources. AF backend is not here — only peak times in seconds."""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from constants import FW_FS
from io_wfdb import load_beats

HERE = Path(__file__).resolve().parent
HOST_BIN = HERE.parent / "host" / "hr_host"

SOURCE_NAMES = ("ann", "ann_manual", "fw_strict", "fw_aiqrs", "xqrs")


def resample_to_fw(sig: np.ndarray, fs: float, out_fs: int = FW_FS) -> np.ndarray:
    fs = float(fs)
    if abs(fs - out_fs) < 1e-9:
        return np.ascontiguousarray(sig, dtype=np.float32)
    up, down = out_fs, int(round(fs))
    g = math.gcd(up, down)
    y = resample_poly(np.asarray(sig, dtype=np.float64), up // g, down // g)
    return np.ascontiguousarray(y, dtype=np.float32)


def _run_hr_host(sig500: np.ndarray, mode: int) -> np.ndarray:
    if not HOST_BIN.exists():
        raise FileNotFoundError(f"hr_host missing: {HOST_BIN} (run host/build.sh)")
    x = np.ascontiguousarray(sig500, dtype=np.float32)
    proc = subprocess.run(
        [str(HOST_BIN), str(mode)],
        input=x.tobytes(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")[:500]
        raise RuntimeError(f"hr_host mode={mode} failed rc={proc.returncode}: {err}")
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    idx = []
    for ln in lines:
        if not ln.strip():
            continue
        idx.append(int(ln.split()[0]))
    if not idx:
        return np.zeros(0, dtype=np.float64)
    pk = np.asarray(idx, dtype=np.int64)
    pk = np.unique(pk)
    pk = pk[(pk >= 0) & (pk < x.size)]
    return pk.astype(np.float64) / float(FW_FS)


def fw_strict_peaks(sig: np.ndarray, fs: float) -> np.ndarray:
    return _run_hr_host(resample_to_fw(sig, fs), mode=1)


def fw_aiqrs_peaks(sig: np.ndarray, fs: float) -> np.ndarray:
    return _run_hr_host(resample_to_fw(sig, fs), mode=2)


def _xqrs_once(x: np.ndarray, fs: float) -> np.ndarray:
    from wfdb.processing import xqrs_detect

    try:
        ind = xqrs_detect(x, fs=fs, verbose=False)
    except TypeError:
        ind = xqrs_detect(x, fs=fs)
    return np.asarray(ind, dtype=np.int64)


def xqrs_peaks(sig: np.ndarray, fs: float) -> np.ndarray:
    """wfdb XQRS at native fs.

    24 h Holter can stall in XQRS `_backsearch` (hours on a noisy record).
    Records longer than 40 min are split into 10 min chunks with 8 s overlap.
    """
    x = np.asarray(sig, dtype=np.float64)
    fs = float(fs)
    if x.size < int(fs):
        return np.zeros(0, dtype=np.float64)

    def _with_polarity(arr: np.ndarray) -> np.ndarray:
        ind = _xqrs_once(arr, fs)
        if ind.size >= 5:
            return ind
        ind2 = _xqrs_once(-arr, fs)
        return ind2 if ind2.size > ind.size else ind

    max_oneshot = int(round(fs * 40 * 60))
    if x.size <= max_oneshot:
        ind = _with_polarity(x)
        return ind.astype(np.float64) / fs

    chunk = int(round(fs * 600.0))
    overlap = int(round(fs * 8.0))
    parts = []
    start = 0
    n = x.size
    while start < n:
        end = min(n, start + chunk)
        loc = _with_polarity(x[start:end])
        if loc.size:
            parts.append(loc.astype(np.int64) + start)
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    if not parts:
        return np.zeros(0, dtype=np.float64)
    ind = np.unique(np.concatenate(parts))
    # collapse overlap duplicates (150 ms)
    if ind.size >= 2:
        keep = np.ones(ind.size, dtype=bool)
        last = ind[0]
        for i in range(1, ind.size):
            if (ind[i] - last) / fs < 0.150:
                keep[i] = False
            else:
                last = ind[i]
        ind = ind[keep]
    return ind.astype(np.float64) / fs


def annotation_peaks(db_dir: Path, rec: str, manual: bool) -> np.ndarray:
    t, ext = load_beats(db_dir, rec, prefer="manual" if manual else "auto")
    if manual and ext != "atr":
        return np.zeros(0, dtype=np.float64)
    return t


def mix_nstdb_noise(
    sig: np.ndarray,
    fs: float,
    noise: np.ndarray,
    noise_fs: float,
    snr_db: float,
    seed: int = 0,
) -> np.ndarray:
    """Add NSTDB-style noise at a target SNR (dB, signal std / noise std)."""
    s = np.asarray(sig, dtype=np.float64)
    n = resample_to_fw(noise, noise_fs, out_fs=int(round(fs)))
    n = np.asarray(n, dtype=np.float64)
    if n.size < s.size:
        reps = int(np.ceil(s.size / n.size))
        n = np.tile(n, reps)
    n = n[: s.size]
    rng = np.random.default_rng(seed)
    # random start
    if n.size > 10:
        off = int(rng.integers(0, max(1, min(int(fs * 10), n.size // 4))))
        n = np.roll(n, off)
    ss = float(np.std(s))
    ns = float(np.std(n))
    if ns < 1e-12 or ss < 1e-12:
        return s
    scale = ss / (ns * (10 ** (snr_db / 20.0)))
    return s + scale * n
