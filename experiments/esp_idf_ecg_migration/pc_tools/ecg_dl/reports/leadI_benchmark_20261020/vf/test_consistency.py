#!/usr/bin/env python3
"""Compare the Python feature port to compiled vf_detect.cpp, and check gates.

Exit 0 when the firmware binary and the Python port agree within tolerance
and the vfProcess confirm bit matches the two-window rule. Writes
results/consistency.json with the measured gaps, including the previous
attachment script and scipy sosfiltfilt(padlen=0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from vf_host import ReplayHost
from vf_metrics import (
    beat_gate_open,
    confirm_from_scores,
    flat_intervals,
    rhythm_segments,
    shock_episodes,
    window_class,
    window_end_frame,
)
from vf_port import (
    FEAT_NAMES,
    WIN_N,
    features_firmware,
    legacy_features,
    score_features,
)

HERE = Path(__file__).resolve().parent
BIN = HERE / "host" / "fw_replay_host"
OUT = HERE / "results" / "consistency.json"

# Tolerances vs the firmware binary. Score uses expf; a few ulps are enough.
SCORE_ATOL = 5e-4
FEAT_ATOL = np.array([1e-4, 1e-4, 1e-4, 1e-6, 1e-4, 1e-3], dtype=np.float64)


def _windows() -> list[tuple[str, np.ndarray]]:
    n = WIN_N
    t = np.arange(n, dtype=np.float32) / np.float32(250.0)
    rng = np.random.default_rng(20261021)
    out = [
        ("zeros", np.zeros(n, np.float32)),
        ("const", np.ones(n, np.float32)),
        ("sine5", (0.8 * np.sin(2 * np.pi * 5.0 * t)).astype(np.float32)),
        ("sine1", (1.0 * np.sin(2 * np.pi * 1.0 * t)).astype(np.float32)),
        ("sine7_small", (0.05 * np.sin(2 * np.pi * 7.0 * t)).astype(np.float32)),
        ("impulses", np.zeros(n, np.float32)),
        ("gauss", rng.normal(0, 0.4, size=n).astype(np.float32)),
        ("with_zeros", rng.normal(0, 0.3, size=n).astype(np.float32)),
    ]
    out[5][1][::200] = np.float32(1.5)
    # Exact zeros in the waveform: np.sign and `> 0` disagree here.
    z = out[7][1]
    z[::17] = np.float32(0)
    z[3] = np.float32(0)
    z[4] = np.float32(0)
    return out


def _gate_unit_tests() -> None:
    # Completing frame of window 0 is 2498. A beat there is 0 ms ago: gate closed.
    assert window_end_frame(0) == 2498
    assert window_end_frame(1) == 4998
    beats = np.array([2498], dtype=np.int32)
    assert beat_gate_open(2498, beats, 2500, True) is False
    # 1250 frames * 2 ms = 2500 ms exactly: firmware uses `>`, so still closed.
    beats = np.array([2498 - 1250], dtype=np.int32)
    assert (2498 - int(beats[0])) * 2 == 2500
    assert beat_gate_open(2498, beats, 2500, True) is False
    beats = np.array([2498 - 1251], dtype=np.int32)
    assert beat_gate_open(2498, beats, 2500, True) is True
    # No beat yet: armed gate stays shut; no-arm variant stays open.
    assert beat_gate_open(2498, np.zeros(0, np.int32), 2500, True) is False
    assert beat_gate_open(2498, np.zeros(0, np.int32), 2500, False) is True
    # FLAT at exactly 4000 ms (2000 frames).
    flats = flat_intervals(np.array([0], dtype=np.int32), 5000)
    assert flats == [(2000, 5000)]
    # Rhythm parser: prefix before the first label is UNK; VF/VT/VFL split.
    segs = rhythm_segments(
        [10, 100, 200],
        ["+", "+", "+"],
        ["(N", "(VT", "(VFIB"],
        duration_s=1.0,
        fs=250.0,
    )
    # First mark is non-shock and not at t=0: the prefix is non-VF, not UNK.
    assert [s[2] for s in segs] == ["N", "VT", "VFIB"]
    eps = shock_episodes(segs, min_dur=0.0)
    assert [e[2] for e in eps] == ["VT", "VFIB"]
    # 50% rule. Window [0.04, 5.04) on this short fixture is mostly past the end;
    # check a window that sits inside VT: VT is samples 100..200 -> 0.4..0.8 s.
    assert window_class(segs, 0.4, 0.8) == 1
    assert window_class(segs, 0.04, 0.4) == 0
    # CUDB: `+ (VF` then `[`, and `]` with an empty aux. Tail must not stay VF.
    # A `~` carrying `(N` in the middle of VT must not end the episode.
    fs = 250.0
    segs = rhythm_segments(
        [1000, 1005, 2000, 8000],
        ["+", "[", "~", "]"],
        ["(VF", "", "(N", ""],
        duration_s=40.0,
        fs=fs,
    )
    assert [(round(a, 3), round(b, 3), tok) for a, b, tok in segs] == [
        (0.0, 4.0, "N"),
        (4.0, 32.0, "VF"),
        (32.0, 40.0, "N"),
    ]
    segs = rhythm_segments(
        [0, 1250, 5000],
        ["+", "~", "+"],
        ["(VT", "(N", "(N"],
        duration_s=30.0,
        fs=fs,
    )
    assert [s[2] for s in segs] == ["VT", "N"]
    assert shock_episodes(segs, 0.0)[0][1] - shock_episodes(segs, 0.0)[0][0] == 20.0
    # Brackets alone, no parenthetical rhythm.
    segs = rhythm_segments(
        [2500, 7500],
        ["[", "]"],
        ["", ""],
        duration_s=40.0,
        fs=fs,
    )
    assert [(round(a, 3), round(b, 3), tok) for a, b, tok in segs] == [
        (0.0, 10.0, "N"),
        (10.0, 30.0, "VF"),
        (30.0, 40.0, "N"),
    ]
    # No rhythm marks at all: do not invent a negative class.
    segs = rhythm_segments([100, 200], ["N", "N"], ["", ""], duration_s=10.0, fs=fs)
    assert segs == [(0.0, 10.0, "UNK")]


def _synth_qrs(seconds: float = 20.0, fs: int = 500, hr: float = 60.0) -> np.ndarray:
    n = int(seconds * fs)
    x = np.zeros(n, dtype=np.float32)
    period = int(round(fs * 60.0 / hr))
    half = 8  # 16 ms wide spike, amplitude 1.5 mV
    for c in range(period, n - half, period):
        for k in range(-half, half + 1):
            x[c + k] += np.float32(1.5 * (1.0 - abs(k) / half))
    return x


def main() -> int:
    _gate_unit_tests()
    if not BIN.exists():
        print(f"missing {BIN}; run host/build.sh", file=sys.stderr)
        return 2
    host = ReplayHost(BIN, HERE / "results" / "host_stderr.log")
    rows = []
    worst_score = 0.0
    worst_feat = np.zeros(6, dtype=np.float64)
    legacy_gap = np.zeros(6, dtype=np.float64)
    suspect_bad = 0
    scipy_band_gap = None
    try:
        from scipy.signal import sosfilt, sosfiltfilt
        from vf_port import SOS
    except Exception:
        sosfilt = None
        sosfiltfilt = None
        SOS = None

    for name, x in _windows():
        py = features_firmware(x)
        py_score = score_features(py)
        c = host.score_window(x)
        df = np.abs(py.astype(np.float64) - c["feat"].astype(np.float64))
        ds = abs(py_score - c["score"])
        worst_feat = np.maximum(worst_feat, df)
        worst_score = max(worst_score, ds)
        if int(py_score >= 0.15) != c["suspect"] and abs(c["score"] - 0.15) > 1e-3:
            suspect_bad += 1
        leg = legacy_features(x)
        legacy_gap = np.maximum(legacy_gap, np.abs(leg - c["feat"].astype(np.float64)))
        band_gap = None
        if sosfiltfilt is not None:
            work = x.astype(np.float64)
            work = work - work.mean()
            ref = sosfiltfilt(SOS.astype(np.float64), work, padlen=0)
            # Firmware bandpass output is not exported; compare energy ratio only
            # through a local zero-state double sosfilt, which is what we claim
            # scipy padlen=0 is not.
            zi = np.zeros((SOS.shape[0], 2), dtype=np.float64)
            fwd, _ = sosfilt(SOS.astype(np.float64), work, zi=zi)
            back, _ = sosfilt(SOS.astype(np.float64), fwd[::-1], zi=zi)
            zero = back[::-1]
            band_gap = float(np.max(np.abs(ref - zero)))
            scipy_band_gap = band_gap if scipy_band_gap is None else max(scipy_band_gap, band_gap)
        rows.append({
            "name": name,
            "c_score": c["score"],
            "py_score": py_score,
            "c_suspect": c["suspect"],
            "abs_score": ds,
            "abs_feat": [float(v) for v in df],
            "c_feat": [float(v) for v in c["feat"]],
            "legacy_abs": [float(v) for v in np.abs(leg - c["feat"].astype(np.float64))],
            "scipy_filtfilt_pad0_max_abs": band_gap,
            "rms_matches_feat0": abs(c["rms"] - float(c["feat"][0])) == 0.0,
        })
        print(f"{name:12s} score C {c['score']:.6f} py {py_score:.6f} "
              f"dscore {ds:.3e} dfeat {np.array2string(df, precision=2)}")

    # Confirm bit: three copies of the 5 Hz window through vfProcess (cmd 3).
    sine = (0.8 * np.sin(2 * np.pi * 5.0 * np.arange(WIN_N) / 250.0)).astype(np.float32)
    quiet = (0.2 * np.sin(2 * np.pi * 1.2 * np.arange(WIN_N) / 250.0)).astype(np.float32)
    stream = np.concatenate([sine, sine, quiet, sine, sine])
    got = host.vf_stream(stream)
    expect_scores = []
    for w in (sine, sine, quiet, sine, sine):
        expect_scores.append(host.score_window(w)["score"])
    expect_alarm = confirm_from_scores(np.array(expect_scores), 0.15)
    alarm_ok = bool(np.all(got[:, 3] >= 0.5) == np.all(expect_alarm)) and bool(
        np.array_equal(got[:, 3] >= 0.5, expect_alarm)
    )
    score_stream_gap = float(np.max(np.abs(got[:, 0] - np.array(expect_scores, dtype=np.float32))))
    print("stream alarm", (got[:, 3] >= 0.5).tolist(), "expect", expect_alarm.tolist(),
          "score gap", score_stream_gap)

    # Firmware QRS on a 60 bpm spike train. Not a database score: just proof
    # the linked heartrate.cpp emits beats on a mV-scale signal.
    qrs = _synth_qrs()
    _wins, beats = host.frontend(qrs)
    print(f"synth QRS beats in 20 s @ 60 bpm: {beats.size}")
    host.close()

    feat_ok = bool(np.all(worst_feat <= FEAT_ATOL))
    score_ok = worst_score <= SCORE_ATOL
    qrs_ok = 15 <= int(beats.size) <= 22
    passed = feat_ok and score_ok and suspect_bad == 0 and alarm_ok and score_stream_gap < 1e-5 and qrs_ok
    doc = {
        "passed": passed,
        "score_atol": SCORE_ATOL,
        "feat_atol": [float(v) for v in FEAT_ATOL],
        "max_abs_score": worst_score,
        "max_abs_feat": {n: float(v) for n, v in zip(FEAT_NAMES, worst_feat)},
        "suspect_mismatches_away_from_theta": suspect_bad,
        "stream_alarm_matches_confirm": alarm_ok,
        "stream_score_gap_vs_score_window": score_stream_gap,
        "legacy_attachment_max_abs_feat": {n: float(v) for n, v in zip(FEAT_NAMES, legacy_gap)},
        "scipy_sosfiltfilt_padlen0_vs_zero_state_max_abs": scipy_band_gap,
        "synth_qrs_beats_20s": int(beats.size),
        "windows": rows,
        "notes": [
            "Python port is float32 and sequential, matching vf_detect.cpp.",
            "scipy.signal.sosfiltfilt(..., padlen=0) still applies sosfilt_zi initial conditions; the firmware clears state to zero before both passes.",
            "legacy_attachment_max_abs_feat is the previous 32_vf_rules_eval.py formulas against the firmware features on the same windows.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print("wrote", OUT, "passed" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
