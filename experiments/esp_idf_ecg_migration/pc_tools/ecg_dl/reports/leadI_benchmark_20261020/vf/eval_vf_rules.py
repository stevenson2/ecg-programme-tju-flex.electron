#!/usr/bin/env python3
"""Replay the firmware VF rule on VFDB, CUDB, MIT-BIH, and optional NSTDB.

The logistic features are computed by the compiled firmware
(host/fw_replay_host), not by a second copy of the math. Two inputs are
scored:

* frontend — shipped replay chain at 500 Hz: subtract 1.65, double 10-tap
  comb, HP 0.5 + LP 40, hrProcess on every sample, vfProcess on even samples
  with no ×0.763 (replay is already mV). This is the headline.
* raw — database signal resampled to 250 Hz and fed straight to vfProcess.
  That is the domain the logistic coefficients were fit in.

Beat gate and RMS gate are applied here the way main.cc applies them.
Nothing in this script changes firmware thresholds.

    python3 eval_vf_rules.py
    python3 eval_vf_rules.py --data-root /path/to/ecg
    ECG_DATA_ROOT=/path/to/ecg python3 eval_vf_rules.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BIN = HERE / "host" / "fw_replay_host"

from vf_host import ReplayHost
from vf_metrics import (
    BEAT_SYMS,
    CUDB_RECORDS,
    MIT_RECORDS,
    VFDB_RECORDS,
    alarm_episodes,
    beat_gate_open,
    beat_times_from_ann,
    bootstrap_rate,
    bootstrap_ratio,
    confirm_from_scores,
    flat_intervals,
    is_asys,
    is_noise,
    is_shock,
    match_beats,
    poisson_rate,
    rhythm_segments,
    shock_episodes,
    wilson,
    window_class,
    window_end_frame,
    add_flutter_wave_episodes,
)
from vf_port import ALARM_VF_RMS_MIN, THETA


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def resample_to(x: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    fs_in_i = int(round(float(fs_in)))
    if fs_in_i == fs_out:
        return np.ascontiguousarray(x, dtype=np.float64)
    from scipy.signal import resample_poly
    g = math.gcd(fs_in_i, fs_out)
    return resample_poly(np.asarray(x, dtype=np.float64), fs_out // g, fs_in_i // g)


def data_root_from(cli: str | None) -> Path:
    if cli:
        return Path(cli).expanduser().resolve()
    for key in ("ECG_DATA_ROOT", "ECG_REFS_ROOT"):
        if os.environ.get(key):
            return Path(os.environ[key]).expanduser().resolve()
    return HERE / "data"


def available_records(folder: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}

    def take(hea: Path) -> None:
        stem = hea.stem
        if stem in {"RECORDS", "SHA256SUMS", "MD5SUMS"}:
            return
        if hea.with_suffix(".dat").exists() or hea.with_suffix(".mat").exists():
            found[stem] = hea.with_suffix("")

    if not folder.exists():
        return found
    for hea in sorted(folder.glob("*.hea")):
        take(hea)
    if not found:
        for hea in sorted(folder.glob("*/*.hea")):
            take(hea)
    return found


def ensure_db(root: Path, name: str, needed: list[str], download: bool) -> tuple[dict[str, Path], list[str]]:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    have = available_records(folder)
    missing = [n for n in needed if n not in have]
    if missing and download:
        log(f"downloading {name} ({len(missing)} missing) via wfdb")
        import wfdb
        wfdb.dl_database(name, str(folder))
        have = available_records(folder)
        missing = [n for n in needed if n not in have]
    return have, missing


def load_signal(path: Path):
    import wfdb
    rec = wfdb.rdrecord(str(path), channels=[0])
    if rec.p_signal is not None:
        sig = np.asarray(rec.p_signal[:, 0], dtype=np.float64)
    else:
        gain = float(rec.adc_gain[0] or 1.0)
        base = float(rec.baseline[0] or 0.0)
        sig = (np.asarray(rec.d_signal[:, 0], dtype=np.float64) - base) / gain
    units = (rec.units[0] if rec.units else "") or ""
    ul = units.lower()
    if ul in {"v", "volt", "volts"}:
        sig = sig * 1000.0
        units = "mV"
    elif ul in {"uv", "µv", "μv"}:
        sig = sig / 1000.0
        units = "mV"
    nans = int(np.isnan(sig).sum())
    if nans:
        idx = np.arange(sig.size)
        good = np.isfinite(sig)
        if int(good.sum()) < 10:
            raise RuntimeError(f"{path.name}: signal is almost all NaN")
        sig = np.interp(idx, idx[good], sig[good])
    fs = float(rec.fs)
    sig_name = rec.sig_name[0] if rec.sig_name else "ch0"
    return sig, fs, units, sig_name, nans


def load_annotations(path: Path, fs: float, duration_s: float):
    import wfdb
    try:
        ann = wfdb.rdann(str(path), "atr")
    except Exception:
        return [(0.0, duration_s, "UNK")], np.zeros(0, np.float64), Counter(), "missing"
    segs = rhythm_segments(ann.sample, ann.symbol, ann.aux_note, duration_s, fs)
    segs = add_flutter_wave_episodes(segs, ann.sample, ann.symbol, fs, duration_s)
    beats = beat_times_from_ann(ann.sample, ann.symbol, fs)
    vocab = Counter()
    for note in ann.aux_note or []:
        from vf_metrics import rhythm_token
        tok = rhythm_token(note)
        if tok:
            vocab[tok] += 1
    sym = Counter(ann.symbol)
    return segs, beats, vocab, sym


def scale_segs(segs, scale: float):
    return [(a * scale, b * scale, tok) for a, b, tok in segs]


def window_labels(segs, nwin: int) -> np.ndarray:
    lab = np.empty(nwin, dtype=np.int8)
    for i in range(nwin):
        lab[i] = window_class(segs, 5.0 * i, 5.0 * (i + 1))
    return lab


def gate_array(nwin: int, beats: np.ndarray, gate_ms: int, require_arm: bool) -> np.ndarray:
    g = np.empty(nwin, dtype=bool)
    for i in range(nwin):
        g[i] = beat_gate_open(window_end_frame(i), beats, gate_ms, require_arm)
    return g


def flags_of(alarm: np.ndarray, rms: np.ndarray, gate: np.ndarray | None, use_rms: bool, use_gate: bool) -> np.ndarray:
    out = np.asarray(alarm, dtype=bool).copy()
    if use_rms:
        out &= np.asarray(rms, dtype=np.float32) >= np.float32(ALARM_VF_RMS_MIN)
    if use_gate:
        out &= np.asarray(gate, dtype=bool)
    return out


def confusion(flags: np.ndarray, labels: np.ndarray):
    pos = labels == 1
    neg = labels == 0
    tp = int(np.count_nonzero(flags & pos))
    fn = int(np.count_nonzero(~flags & pos))
    fp = int(np.count_nonzero(flags & neg))
    tn = int(np.count_nonzero(~flags & neg))
    return tp, fn, fp, tn


def first_hit_delay(flags: np.ndarray, a: float, b: float):
    for i, f in enumerate(flags):
        if not f:
            continue
        w0, w1 = 5.0 * i, 5.0 * (i + 1)
        if min(w1, b) > max(w0, a):
            return (w1 - 0.004) - a
    return None


def event_counts(flags: np.ndarray, segs, min_dur: float):
    eps = shock_episodes(segs, min_dur)
    by = {}
    delays = []
    hits = 0
    for a, b, tok in eps:
        slot = by.setdefault(tok, [0, 0])
        slot[1] += 1
        dly = first_hit_delay(flags, a, b)
        if dly is not None:
            hits += 1
            slot[0] += 1
            delays.append(float(dly))
    shock_all = shock_episodes(segs, 0.0)
    fp = 0
    fp_after10 = 0
    for t0, t1, _i, _j in alarm_episodes(flags):
        if any(min(t1, b) > max(t0, a) for a, b, _tok in shock_all):
            continue
        fp += 1
        if t0 >= 10.0:
            fp_after10 += 1
    return {
        "hits": hits,
        "n": len(eps),
        "by": {k: {"hit": v[0], "n": v[1]} for k, v in by.items()},
        "delays": delays,
        "fp_episodes": fp,
        "fp_episodes_after_10s": fp_after10,
    }


def beats_in_negative_windows(beat_frames: np.ndarray, labels: np.ndarray) -> float | None:
    if labels.size == 0 or np.count_nonzero(labels == 0) == 0:
        return None
    org_s = float(np.count_nonzero(labels == 0) * 5.0)
    if beat_frames.size == 0:
        return 0.0
    bt = beat_frames.astype(np.float64) / 500.0
    c = 0
    for t in bt:
        i = int(t // 5.0)
        if 0 <= i < labels.size and labels[i] == 0:
            c += 1
    return c / org_s


def beats_in_shock(beat_frames: np.ndarray, segs) -> tuple[int, float]:
    shock = shock_episodes(segs, 0.0)
    shock_s = float(sum(b - a for a, b, _ in shock))
    if beat_frames.size == 0 or shock_s <= 0:
        return 0, shock_s
    bt = beat_frames.astype(np.float64) / 500.0
    c = 0
    for t in bt:
        for a, b, _ in shock:
            if a <= t < b:
                c += 1
                break
    return c, shock_s


def flat_report(beat_frames: np.ndarray, n_frames: int, segs, ann_beats_s: np.ndarray, dur_s: float):
    spans = [(a / 500.0, b / 500.0) for a, b in flat_intervals(beat_frames, n_frames)]
    asys = [(a, b) for a, b, tok in segs if is_asys(tok) and b > a]
    pauses = []
    if ann_beats_s.size:
        edges = list(ann_beats_s.astype(float)) + [dur_s]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a >= 4.0:
                pauses.append((a + 4.0, b))
    shock = [(a, b) for a, b, tok in segs if is_shock(tok)]

    def overlaps(span, refs) -> bool:
        a0, b0 = span
        return any(min(b0, b) > max(a0, a) for a, b in refs)

    fp_org = 0
    during_shock = 0
    for sp in spans:
        if overlaps(sp, shock):
            during_shock += 1
            continue
        if overlaps(sp, asys) or overlaps(sp, pauses):
            continue
        fp_org += 1
    asys_hit = sum(1 for ep in asys if overlaps(ep, spans))
    return {
        "n_flat": len(spans),
        "n_flat_during_shock": during_shock,
        "n_flat_fp_organized": fp_org,
        "n_asys": len(asys),
        "n_asys_hit": asys_hit,
    }


def qrs_report(beat_frames: np.ndarray, ann_beats_s: np.ndarray):
    det = beat_frames.astype(np.float64) / 500.0
    matched, nref, ndet = match_beats(ann_beats_s, det, 0.150)
    return {
        "n_det": ndet,
        "n_ref": nref,
        "matched": matched,
        "se": (matched / nref) if nref else None,
        "ppv": (matched / ndet) if (ndet and nref) else None,
    }


def process_record(host: ReplayHost, path: Path, db: str, skip_raw: bool):
    sig, fs, units, sig_name, nans = load_signal(path)
    dur_orig = len(sig) / float(fs) if fs else 0.0
    segs, ann_beats, vocab, symbols = load_annotations(path, fs, dur_orig)
    x500 = np.ascontiguousarray(resample_to(sig, fs, 500), dtype=np.float32)
    dur_host = len(x500) / 500.0
    scale = (dur_host / dur_orig) if dur_orig > 0 else 1.0
    segs_h = scale_segs(segs, scale)
    ann_h = ann_beats * scale
    wins, beats = host.frontend(x500)
    n_even = (len(x500) + 1) // 2
    expect = n_even // 1250
    if len(wins) != expect:
        raise RuntimeError(f"{path.name}: host windows {len(wins)} != {expect}")
    nwin = len(wins)
    labels = window_labels(segs_h, nwin)
    alarm = wins[:, 3] >= 0.5
    suspect = wins[:, 2] >= 0.5
    expect_alarm = np.zeros(nwin, dtype=bool)
    prev = False
    for i, s in enumerate(suspect):
        expect_alarm[i] = bool(s and prev)
        prev = bool(s)
    if nwin and not np.array_equal(alarm, expect_alarm):
        raise RuntimeError(f"{path.name}: vfProcess alarm bit != two-window confirm")
    ann_frames = np.round(ann_h * 500.0).astype(np.int32)
    ann_frames = ann_frames[(ann_frames >= 0) & (ann_frames < len(x500))]
    ann_frames.sort()
    raw = None
    if not skip_raw:
        x250 = np.ascontiguousarray(resample_to(sig, fs, 250), dtype=np.float32)
        rw = host.vf_stream(x250)
        dur_raw = len(x250) / 250.0
        scale_r = (dur_raw / dur_orig) if dur_orig > 0 else 1.0
        segs_r = scale_segs(segs, scale_r)
        nwin_r = len(rw)
        # Beat times expressed on the 250 Hz clock.
        beat_raw_s = (beats.astype(np.float64) / 500.0) * (dur_raw / dur_host if dur_host else 1.0)
        raw = {
            "score": rw[:, 0].copy(),
            "rms": rw[:, 1].copy(),
            "alarm": rw[:, 3] >= 0.5,
            "labels": window_labels(segs_r, nwin_r),
            "segs": segs_r,
            "beat_s": beat_raw_s,
            "hours": nwin_r * 5.0 / 3600.0,
        }
    b_shock, shock_s = beats_in_shock(beats, segs_h)
    return {
        "db": db,
        "record": path.name,
        "sig_name": sig_name,
        "units": units,
        "fs": fs,
        "nans": nans,
        "hours": nwin * 5.0 / 3600.0,
        "n_win": nwin,
        "n_frames": int(len(x500)),
        "labels": labels,
        "score": wins[:, 0].copy(),
        "rms": wins[:, 1].copy(),
        "alarm": alarm,
        "beats": beats.astype(np.int32, copy=False),
        "segs": segs_h,
        "ann_frames": ann_frames,
        "ann_beats_s": ann_h,
        "vocab": vocab,
        "symbols": symbols,
        "qrs": qrs_report(beats, ann_h),
        "flat": flat_report(beats, len(x500), segs_h, ann_h, dur_host),
        "beats_in_shock": b_shock,
        "shock_seconds": shock_s,
        "beats_per_s_shock": (b_shock / shock_s) if shock_s > 0 else None,
        "beats_per_s_organized": beats_in_negative_windows(beats, labels),
        "raw": raw,
        "median_rms": float(np.median(wins[:, 1])) if nwin else None,
    }


def profile_flags(rec, kind: str, theta: float = 0.15, gate_ms: int = 2500) -> np.ndarray:
    if kind == "confirm":
        alarm = confirm_from_scores(rec["score"], theta) if theta != 0.15 else rec["alarm"]
        # At the shipped theta, trust the firmware bit (float32 compare).
        if abs(theta - 0.15) < 1e-9:
            alarm = rec["alarm"]
        return flags_of(alarm, rec["rms"], None, False, False)
    alarm = rec["alarm"] if abs(theta - 0.15) < 1e-9 else confirm_from_scores(rec["score"], theta)
    if kind == "rms":
        return flags_of(alarm, rec["rms"], None, True, False)
    if kind == "firmware":
        gate = gate_array(rec["n_win"], rec["beats"], gate_ms, True)
        return flags_of(alarm, rec["rms"], gate, True, True)
    if kind == "noarm":
        gate = gate_array(rec["n_win"], rec["beats"], gate_ms, False)
        return flags_of(alarm, rec["rms"], gate, True, True)
    if kind == "ann":
        if rec["ann_frames"].size == 0:
            return None
        gate = gate_array(rec["n_win"], rec["ann_frames"], gate_ms, True)
        return flags_of(alarm, rec["rms"], gate, True, True)
    raise KeyError(kind)


def raw_firmware_flags(rec, theta: float = 0.15) -> np.ndarray | None:
    raw = rec["raw"]
    if raw is None or raw["score"].size == 0:
        return None
    alarm = raw["alarm"] if abs(theta - 0.15) < 1e-9 else confirm_from_scores(raw["score"], theta)
    nwin = alarm.size
    gate = np.zeros(nwin, dtype=bool)
    beats = raw["beat_s"]
    armed = False
    # Walk windows; a beat at or before the completing sample opens the arm
    # and is subject to `> 2500 ms`.
    for i in range(nwin):
        t_end = ((i + 1) * 1250 - 1) / 250.0
        if beats.size == 0:
            gate[i] = False
            continue
        k = int(np.searchsorted(beats, t_end, side="right")) - 1
        if k < 0:
            gate[i] = False
            continue
        armed = True
        gate[i] = (t_end - float(beats[k])) * 1000.0 > 2500.0
    _ = armed
    return flags_of(alarm, raw["rms"], gate, True, True)


def summarize_flags(recs, flags_by_record: dict[str, np.ndarray], min_dur: float):
    """Pooled counts. flags_by_record maps record name -> bool array."""
    tp = fn = fp = tn = 0
    evt_hits = evt_n = 0
    evt15_hits = evt15_n = 0
    evt_all_hits = evt_all_n = 0
    by = {}
    delays = []
    fp_eps = 0
    fp_eps_after = 0
    hours = 0.0
    rec_n = rec_hit = 0
    rec_any_n = rec_any_hit = 0
    kn_evt = []
    kn_win = []
    ch_fp = []
    per = []
    for rec in recs:
        flags = flags_by_record.get(rec["record"])
        if flags is None:
            continue
        hours += rec["hours"]
        a, b, c, d = confusion(flags, rec["labels"])
        tp += a
        fn += b
        fp += c
        tn += d
        kn_win.append((a, a + b))
        ev = event_counts(flags, rec["segs"], min_dur)
        ev15 = event_counts(flags, rec["segs"], 15.0)
        ev0 = event_counts(flags, rec["segs"], 0.0)
        evt_hits += ev["hits"]
        evt_n += ev["n"]
        evt15_hits += ev15["hits"]
        evt15_n += ev15["n"]
        evt_all_hits += ev0["hits"]
        evt_all_n += ev0["n"]
        kn_evt.append((ev["hits"], ev["n"]))
        delays.extend(ev["delays"])
        fp_eps += ev["fp_episodes"]
        fp_eps_after += ev["fp_episodes_after_10s"]
        ch_fp.append((ev["fp_episodes"], rec["hours"]))
        for tok, slot in ev["by"].items():
            acc = by.setdefault(tok, {"hit": 0, "n": 0})
            acc["hit"] += slot["hit"]
            acc["n"] += slot["n"]
        has = ev["n"] > 0
        hit = ev["hits"] > 0
        if has:
            rec_n += 1
            rec_hit += int(hit)
        if ev0["n"] > 0:
            rec_any_n += 1
            rec_any_hit += int(ev0["hits"] > 0)
        per.append({
            "record": rec["record"],
            "tp": a, "fn": b, "fp": c, "tn": d,
            "evt_hits": ev["hits"], "evt_n": ev["n"],
            "evt15_hits": ev15["hits"], "evt15_n": ev15["n"],
            "evt_all_hits": ev0["hits"], "evt_all_n": ev0["n"],
            "fp_episodes": ev["fp_episodes"],
            "fp_episodes_after_10s": ev["fp_episodes_after_10s"],
            "hours": rec["hours"],
            "record_hit": int(hit) if has else None,
        })
    se_rec = wilson(rec_hit, rec_n)
    se_rec_any = wilson(rec_any_hit, rec_any_n)
    se_evt = wilson(evt_hits, evt_n)
    se_evt_boot = bootstrap_ratio([(k, n) for k, n in kn_evt if n > 0])
    se_win = wilson(tp, tp + fn)
    se_win_boot = bootstrap_ratio([(k, n) for k, n in kn_win if n > 0])
    fp_boot = bootstrap_rate(ch_fp)
    fp_pois = poisson_rate(fp_eps, hours)
    delay = None
    if delays:
        delay = {
            "n": len(delays),
            "median_s": float(np.median(delays)),
            "p90_s": float(np.quantile(delays, 0.9)),
            "mean_s": float(np.mean(delays)),
        }
    by_ci = {}
    for tok, slot in sorted(by.items()):
        by_ci[tok] = wilson(slot["hit"], slot["n"])
    return {
        "hours": hours,
        "n_records_scored": len(per),
        "window": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "se": se_win, "se_bootstrap": se_win_boot},
        "event_ge10": {"hits": evt_hits, "n": evt_n, "se": se_evt, "se_bootstrap": se_evt_boot, "by_rhythm": by_ci, "delay_s": delay},
        # Duration >= 15 s, any overlap. Not Se15, and not the latched user alarm.
        "event_ge15": {"hits": evt15_hits, "n": evt15_n, "se": wilson(evt15_hits, evt15_n)},
        "event_all": {"hits": evt_all_hits, "n": evt_all_n, "se": wilson(evt_all_hits, evt_all_n)},
        "record_ge10": se_rec,
        "record_any": se_rec_any,
        "fp_episodes": fp_eps,
        "fp_episodes_after_10s": fp_eps_after,
        "fp_per_h": fp_pois,
        "fp_per_h_bootstrap": fp_boot,
        "per_record": per,
    }


def firmware_pin() -> dict:
    d = HERE
    root = None
    for _ in range(12):
        d = d.parent
        cand = d / "experiments/esp_idf_ecg_migration/components/ecg_core/src/vf_detect.cpp"
        if cand.exists():
            root = d
            break
    if root is None:
        return {}
    vf = root / "experiments/esp_idf_ecg_migration/components/ecg_core/src/vf_detect.cpp"
    hdr = root / "experiments/esp_idf_ecg_migration/components/ecg_core/include/vf_detect/vf_detect.h"
    main = root / "experiments/esp_idf_ecg_migration/main/main.cc"
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    rev = ""
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception:
        rev = ""
    return {
        "git": rev,
        "vf_detect_cpp_sha256": sha(vf),
        "vf_detect_h_sha256": sha(hdr),
        "main_cc_sha256": sha(main),
    }


def py(o):
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, np.ndarray):
        return py(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def fmt_pct(block, bootstrap=None) -> str:
    if not block:
        return "—"
    s = f"{100 * block['est']:.1f}%（{block['k']}/{block['n']}，Wilson 95% CI {100 * block['lo']:.1f}–{100 * block['hi']:.1f}%）"
    if bootstrap:
        s += f"；记录聚类 bootstrap 95% CI {100 * bootstrap['lo']:.1f}–{100 * bootstrap['hi']:.1f}%"
    return s


def fmt_rate(pois, boot) -> str:
    if not pois:
        return "—"
    s = f"{pois['est']:.3f}/h（{pois['k']} 次 / {pois['hours']:.2f} h，Poisson 95% CI {pois['lo']:.3f}–{pois['hi']:.3f}）"
    if boot:
        s += f"；记录聚类 bootstrap 95% CI {boot['lo']:.3f}–{boot['hi']:.3f}"
    return s


def write_report(summary: dict, path: Path) -> None:
    c = summary.get("consistency") or {}
    fe = summary["frontend"]
    raw = summary.get("raw") or {}
    lines = []
    a = lines.append
    a("# VF 规则 PC 评测 v2")
    a("")
    a("评测对象是当前固件里的室颤/室速规则，不是上一版附件脚本的近似。")
    a("特征、分数和两窗确认来自编译后的 `vf_detect.cpp`（host 同步路径）；")
    a("2.5 s 拍间门控和 0.05 mV RMS 门控按 `main.cc` 在窗外施加。")
    a("默认阈值没有改。")
    a("")
    a("## 怎么复跑")
    a("")
    a("```bash")
    a("cd experiments/esp_idf_ecg_migration/pc_tools/ecg_dl/reports/leadI_benchmark_20261020/vf")
    a("./run_eval.sh")
    a("# 或者把库放在别处：")
    a("ECG_DATA_ROOT=/path/to/cache ./run_eval.sh")
    a("./run_eval.sh --data-root /path/to/cache")
    a("```")
    a("")
    a("依赖：`numpy`、`scipy`、`matplotlib`、`wfdb`，以及 `g++`。")
    a("缺 VFDB / CUDB / MIT-BIH 时脚本会用 wfdb 从 PhysioNet 下载到数据根目录。")
    a("波形在 `data/`（或 `ECG_DATA_ROOT`）下，已被 gitignore，不要提交。")
    a("一致性测试不通过就停，不会接着出主表。")
    a("")
    a(f"本次数据根目录：`{summary['data_root']}`。")
    a(f"固件 git `{summary['firmware'].get('git', '')}`，`vf_detect.cpp` SHA256 `{summary['firmware'].get('vf_detect_cpp_sha256', '')}`。")
    a("")
    a("## 结论")
    a("")
    a("主结果是**固件重放链**：500 Hz 重采样后走 `main.cc` 的梳状滤波 + `applyFilter`（0.5–40 Hz），")
    a("心率检测器每样本调用，VF 在偶数样本上调用，输入保持 mV，不乘 0.763（回放路径）。")
    a("拍位置用的就是这次 `hrProcess` 的 `beatDetected`。门控要先见过至少一拍（`s_beatEverSeen`），")
    a("并且报警窗结束时刻距最近一拍 **严格大于** 2500 ms，同时该窗 RMS ≥ 0.05 mV。")
    a("")
    a("事件的定义写在下一节。下面的事件灵敏度都是 **时长 ≥ 10 s** 的可电击节律（VF/VFIB、VT、VFL）。")
    a("")
    a("| 库 | 记录 | 记录级 Se（≥10 s 事件） | 事件级 Se ≥10 s | 窗级 Se | 非可电击段 FP/h |")
    a("|---|---:|---|---|---|---|")
    for db in ("vfdb", "cudb", "mitdb"):
        block = fe.get(db)
        if not block:
            a(f"| {db} | — | — | — | — | — |")
            continue
        ev = block["event_ge10"]
        a("| {db} | {n} | {rec} | {evt} | {win} | {fp} |".format(
            db=db,
            n=block["n_records_scored"],
            rec=fmt_pct(block["record_ge10"]),
            evt=fmt_pct(ev["se"], ev.get("se_bootstrap")),
            win=fmt_pct(block["window"]["se"], block["window"].get("se_bootstrap")),
            fp=fmt_rate(block["fp_per_h"], block.get("fp_per_h_bootstrap")),
        ))
    a("")
    a("MIT-BIH 的「FP/h」是全部 48 条、通道 0 上的假报警**段**（连续报警窗并成一段）每小时，")
    a("与可电击参考段重叠的不算假阳。记录级灵敏度的分母是至少有一段 ≥10 s 可电击节律的记录。")
    a("MIT-BIH 里这样的记录很少（这次是 2 条），假阳率才是它要回答的问题。")
    a("")
    prof = summary.get("profiles") or {}
    conf_v = ((prof.get("confirm") or {}).get("vfdb") or {}).get("event_ge10", {}).get("se")
    conf_c = ((prof.get("confirm") or {}).get("cudb") or {}).get("event_ge10", {}).get("se")
    ann_c = ((prof.get("ann") or {}).get("cudb") or {}).get("event_ge10", {}).get("se")
    if conf_v and conf_c:
        a("差在 2.5 s 拍间门，不在逻辑回归。不把门控加上、只留两窗确认时，")
        a(f"VFDB ≥10 s 事件是 {conf_v['k']}/{conf_v['n']}，CUDB 是 {conf_c['k']}/{conf_c['n']}，窗级灵敏度也接近 1。")
        a("同一套分数在非可电击窗上几乎全部报警（窗特异度在后面的门控小节）。")
        a("这时按「段」算的假阳会偏少：报警从窦律一直连进室颤，整段都和参考重叠，定义上不算假阳。")
        a("加上固件门控之后窗特异度回到 0.99 附近，但可电击段里 `heartrate.cpp` 仍以大约每秒 1 次以上的速度报拍，")
        a("间隔短于 2.5 s，门是关的。报出来的事件中位延迟远长于两窗确认的 10–15 s，是后段 QRS 失锁才放行。")
    if ann_c:
        a(f"CUDB 改用数据库心拍做同一套 2.5 s 门（`ann` 列）时，事件灵敏度是 {ann_c['k']}/{ann_c['n']}：")
        a("库里的拍符号不把室颤波当成 QRS，固件检测器会。这是两套门控的差别，不是阈值调出来的。")
    a("")
    if summary.get("headline_notes"):
        for note in summary["headline_notes"]:
            a(f"- {note}")
        a("")
    holes = []
    for db in ("vfdb", "cudb"):
        for row in (fe.get(db) or {}).get("per_record") or []:
            if int(row.get("evt_all_n") or 0) == 0:
                holes.append(f"{row['record']}（{db}，假阳段 {row['fp_episodes']}）")
    if holes:
        a("下面这些记录的 `.atr` 里没有可电击节律，不进灵敏度分母。报警仍然算假阳段：")
        a("、".join(holes) + "。")
        a("PhysioNet 对 CUDB 的说明写过：参考标注不是定论，cu01、cu03、cu12 的 VF 起止是 2006 年补上的。")
        a("cu14 现在的 `.atr` 仍然没有 `[` 或 `(VF` / `(VT`。主表没有把落在它上面的假阳段拿掉。")
        a("")
    a("## 方法")
    a("")
    a("### 固件在算什么")
    a("")
    a("- 窗：1250 点，250 Hz，5 s，**不重叠**。窗满才算一次。`vf_detect.cpp` 写明不是滑动窗。")
    a("- 特征 6 维，与 `compute_features` 一致：去均值 RMS、`|x|` 中位数、4–10 Hz 能量比、带通后 `v>0` 过零率、局部峰个数/5、原始去均值信号过零率 ×125。")
    a("- 带通是固件里写死的 4 节 SOS，先正向再反向，**两遍都把状态清零**。")
    a("- 逻辑回归用头文件里的 mean/std/coef/intercept，θ = 0.15。连续两窗 `score ≥ θ` 才把 `vfAlarm` 置上。第一窗不会报。")
    a("- RMS 门：只看**这一窗**的 `lastRms ≥ 0.05`。")
    a("- 拍间门：`main.cc` 里 `s_beatEverSeen && (nowMs - s_lastBeatSeenMs) > 2500`。评测用 500 Hz 帧号，1 帧 = 2 ms，完成窗的帧是 `2500*(i+1)-2`。正好 2500 ms 不放行。")
    a("- 停搏/FLAT：见过拍之后，距最近一拍 ≥ 4000 ms 就成一段。这是附带结果，不拿来改 VF 阈值。")
    a("")
    a("### 事件、窗、记录")
    a("")
    a("参考节律只认 `.atr` 里符号为 `+` 的 `(...)` 标注，一段节律持续到下一次这类标注。")
    a("拍符号或 `~` 上碰巧带的 `(N` 不改节律。")
    a("CUDB 用 `[` 标 VF 起点、`]` 标终点，`]` 的 aux 是空的。`]` 结束可电击节律，并恢复进入 VF 之前的非可电击标注；文件里从没写过的话，尾段记为 `N`（括号外不是 VF）。`+ (VF` 和紧随的 `[` 算同一次起点，不拆成两段。")
    a("第一条节律标注如果不是从 0 秒开始，它之前的时间也记为 `N`：CUDB 说明里 VF 之前至少有 61 个非 VF 拍，`[` 只是大约的起点。整条记录一个节律标注都没有时保持 UNK，避免把缺标注的记录当成干净阴性窗。")
    a("可电击 token 只认整词：`VF`、`VFIB`、`VT`、`VFL`。`SVTA` 不是 VT。其余 token（`N`、`NOISE`、`ASYS`、`BI`、`PM`、`HGEA` 等）都当非可电击。")
    a("MIT-BIH 里成串的 `!`（室扑波）若没有被节律标注盖住，也算一段 VFL。")
    a("")
    a("- **事件级（主结果）**：一段可电击节律时长 ≥ 10 s。两窗确认的最短几何长度就是 10 s；短于 10 s 的段另报，不进主分母。≥ 15 s 的段保证无论相位如何都能罩住两个完整 5 s 窗，也另报。")
    a("- **event_ge15 不是 Se15**。上一条里「≥ 15 s 另报」只按时长进分母，命中仍是任意重叠，时延不设上限。Se15 用同一分母，但要求首次命中延迟 ≤ 15 s（`vf_layers.score_episodes`）。这次没有把旧 `event_ge15` 改记成 Se15，也没有为这个口径重跑整库。")
    a("- **分层**。L0 是分数和当窗疑似（`score ≥ 0.15`）。L1 是 `vfAlarm`（连续两窗）。L2 是 L1 再经过 RMS 和拍间门，主表的 `firmware` 列就是 L2。L3 是 `alarmSecondTick` 擎住后的 BLE 第 8 列；VF 单独命中只打遥测，不擎住。主表不是 L3，也不是最终用户报警。")
    a("- 检出：任一被门控放行的报警窗与该段有时间重叠（不要求 50%）。")
    a("- **窗级**：窗内可电击时间 ≥ 50%，且噪声 < 50%，算阳性。噪声或未标注占多数的窗排除。停搏占多数的窗不进 VF 的灵敏度和假阳，单独交给 FLAT。")
    a("- **记录级**：记录里至少有一段 ≥ 10 s 可电击节律，且其中至少一段被检出。")
    a("- **假阳段**：连续报警窗并成一段，且整段都不与任何可电击参考重叠。FP/h = 假阳段数 / 分析小时。分析小时 = 完整窗数 × 5 / 3600，尾部不足一窗丢掉。")
    a("- 置信区间：比例用 Wilson；同时给**按记录重抽样**的 bootstrap 95% 区间（4000 次，种子 20261021），因为同一条记录里的窗和事件不独立。计数率另给 Poisson/χ² 区间。记录少、事件挤在少数记录里时，以 bootstrap 为准。")
    a("")
    a("### 和标定域的对照")
    a("")
    a("系数是在原始 mV、250 Hz 窗上拟合的（`pc_tools/ecg_dl/eval_vf_detect_v2.py --input raw`）。")
    a("重放链多了梳状滤波和 0.5–40 Hz。下表是同一套门控打在「直接 250 Hz、不过前端」输入上的对照，用来看前端把数字拉开了多少。")
    a("")
    if raw:
        a("| 库 | 事件级 Se ≥10 s（原始 250 Hz） | 窗级 Se | FP/h |")
        a("|---|---|---|---|")
        for db in ("vfdb", "cudb", "mitdb"):
            block = raw.get(db)
            if not block:
                continue
            ev = block["event_ge10"]
            a(f"| {db} | {fmt_pct(ev['se'], ev.get('se_bootstrap'))} | {fmt_pct(block['window']['se'], block['window'].get('se_bootstrap'))} | {fmt_rate(block['fp_per_h'], block.get('fp_per_h_bootstrap'))} |")
        a("")
    a("### 拍从哪来")
    a("")
    a("主结果的拍是固件 `heartrate.cpp`（内部 8–25 Hz，500 Hz）。")
    a("`main.cc` 只在 `filterInit()` 时清零，不调用 `filterWarmup()`，这里同样不清零，开头几秒有高通瞬态。")
    a("数据库 QRS 标注（MIT-BIH 的心拍符号，不含 `!`）用来做第二套门控，并报告检测器相对标注的 Se/PPV（容差 150 ms）。")
    a("VFDB 的 `.atr` 通常只有节律、没有逐拍 QRS，所以 VFDB 上没有「标注拍门控」的灵敏度，只有固件拍。")
    a("")
    q = summary.get("qrs_vs_ann") or {}
    if q:
        a("| 库 | 有标注拍的记录 | 检出/参考 | Se | PPV |")
        a("|---|---:|---|---|---|")
        for db, row in q.items():
            if not row or not row.get("n_ref"):
                a(f"| {db} | {row.get('n_records', 0) if row else 0} | 无逐拍标注 | — | — |")
            else:
                a(f"| {db} | {row['n_records']} | {row['matched']}/{row['n_ref']} 参考，检出 {row['n_det']} | {row['se']:.3f} | {row['ppv']:.3f} |")
        a("")
    a("可电击段上固件拍的密度决定 2.5 s 门控会不会被室颤波形自己顶住。")
    a("密度高于 0.4 次/秒（平均间隔 < 2.5 s）时，门控在真 VF 上会一直关着。")
    a("")
    dens = summary.get("beat_density") or {}
    if dens:
        a("| 库 | 可电击时段内拍/秒 | 非可电击窗内拍/秒 |")
        a("|---|---:|---:|")
        for db, row in dens.items():
            def f3(v):
                return "—" if v is None else f"{v:.3f}"
            a(f"| {db} | {f3(row.get('per_s_shock'))} | {f3(row.get('per_s_organized'))} |")
        a("")
    vocab = summary.get("vocab") or {}
    if vocab:
        a("本次 `.atr` aux 里出现过的节律 token，以及评测怎么归类（可电击 / 噪声 / 停搏 / 其他非可电击）：")
        a("")
        a("| token | 出现次数 | 归类 |")
        a("|---|---:|---|")
        for tok, n in sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0])):
            if is_shock(tok):
                kind = "可电击"
            elif is_noise(tok):
                kind = "噪声（窗级排除）"
            elif is_asys(tok):
                kind = "停搏（窗级不进 VF 分母）"
            else:
                kind = "非可电击"
            a(f"| {tok} | {n} | {kind} |")
        a("")
        a("次数是 aux 出现次数，不是事件段数。没进可电击词表的 token 不会抬灵敏度。")
        a("")
    a("## 与固件逐项一致，以及和旧脚本的差")
    a("")
    if c:
        feat = c.get("max_abs_feat") or {}
        feat_s = ", ".join(f"{k} {v:.3g}" for k, v in feat.items())
        a(f"一致性测试通过：`{c.get('passed')}`。8 段合成窗上，Python float32 移植对编译后的 `compute_features` / 分数：")
        score_gap = c.get("max_abs_score")
        score_s = "—" if score_gap is None else f"{float(score_gap):.3g}"
        a(f"分数最大绝对差 {score_s}，六维特征最大绝对差 {feat_s}。")
        a(f"`vfProcess` 的报警位与「当前窗和上一窗都疑似」一致：{c.get('stream_alarm_matches_confirm')}。")
        a(f"20 s、60 bpm 的窄脉冲上，链接进来的 `heartrate.cpp` 检出 {c.get('synth_qrs_beats_20s')} 拍。")
        a("")
        leg = c.get("legacy_attachment_max_abs_feat") or {}
        if leg:
            a("上一版附件 `32_vf_rules_eval.py` 在同一批窗上和固件特征的最大绝对差：")
            a("")
            a("| 特征 | 最大绝对差 |")
            a("|---|---:|")
            for k, v in leg.items():
                a(f"| {k} | {v:.4g} |")
            a("")
            a("差集中在后四维：旧脚本把能量比做成了幅度比，过零用 `np.sign`（0 单独成一类），峰谷率数的是极值而不是峰，主频用的是带通后的过零而不是去均值原始信号。")
            gap = c.get("scipy_sosfiltfilt_padlen0_vs_zero_state_max_abs")
            gap_s = "—" if gap is None else f"{float(gap):.3g}"
            a(f"另外，`scipy.signal.sosfiltfilt(..., padlen=0)` 相对固件零状态正反向滤波，波形最大绝对差 {gap_s}。")
            a("scipy 即使 `padlen=0` 仍用 `sosfilt_zi` 做稳态初值；固件两遍都把状态清零。头文件里「等价 sosfiltfilt padlen=0」这句并不成立。")
            a("")
    a("## 结果")
    a("")
    a("### 按节律拆开（固件重放链，≥10 s）")
    a("")
    a("| 库 | 节律 | 事件 Se |")
    a("|---|---|---|")
    for db in ("vfdb", "cudb", "mitdb"):
        block = fe.get(db) or {}
        by = (block.get("event_ge10") or {}).get("by_rhythm") or {}
        if not by:
            a(f"| {db} | — | 无 ≥10 s 事件 |")
        for tok, w in by.items():
            a(f"| {db} | {tok} | {fmt_pct(w)} |")
    a("")
    a("### ≥15 s、全部事件、去掉前 10 s 的假阳")
    a("")
    a("| 库 | event_ge15（时长≥15 s，任意时延，不是 Se15） | 事件 Se（不限时长） | 记录级 Se（任意时长） | 前 10 s 之后的 FP 段 |")
    a("|---|---|---|---|---:|")
    for db in ("vfdb", "cudb", "mitdb"):
        block = fe.get(db) or {}
        if not block:
            continue
        a(f"| {db} | {fmt_pct(block['event_ge15']['se'])} | {fmt_pct(block['event_all']['se'])} | {fmt_pct(block['record_any'])} | {block['fp_episodes_after_10s']} |")
    a("")
    a("检出事件从起点到报警窗结束的延迟（只统计 ≥10 s 且被检出的段）：")
    a("")
    for db in ("vfdb", "cudb"):
        block = fe.get(db) or {}
        dly = (block.get("event_ge10") or {}).get("delay_s")
        if not dly:
            a(f"- {db}：没有被检出的 ≥10 s 事件，或库缺失。")
        else:
            a(f"- {db}：n={dly['n']}，中位 {dly['median_s']:.1f} s，均值 {dly['mean_s']:.1f} s，P90 {dly['p90_s']:.1f} s。")
    a("")
    a("两窗不重叠确认的理论下限大约是 10 s（第二窗结束），相位不巧时接近 15 s。")
    a("头文件写的「时延 ≤ 7.5 s」对应的是 50% 重叠、2.5 s 一步；**固件实现是 5 s 一步**，7.5 s 达不到。")
    a("")
    a("### 门控各自贡献了什么")
    a("")
    a("同一条重放链、θ=0.15，只换门。`confirm` 是两窗确认本身；`rms` 再加 RMS；`firmware` 是 RMS + 2.5 s 拍间门 + 必须见过拍；`noarm` 去掉「必须见过拍」；`ann` 用数据库心拍代替固件心拍（没有逐拍标注的记录不进这一列）。")
    a("")
    a("| 库 | 门 | 事件 Se ≥10 s | 窗 Se | FP/h |")
    a("|---|---|---|---|---|")
    for db in ("vfdb", "cudb", "mitdb"):
        for kind in ("confirm", "rms", "firmware", "noarm", "ann"):
            block = (summary.get("profiles") or {}).get(kind, {}).get(db)
            if not block:
                continue
            ev = block["event_ge10"]
            a(f"| {db} | {kind} | {fmt_pct(ev['se'])} | {fmt_pct(block['window']['se'])} | {fmt_rate(block['fp_per_h'], None)} |")
    a("")
    a("同一张表换成窗级特异度（非可电击窗里没有报警的比例）。")
    a("两窗确认不加门时特异度接近 0，是因为 θ=0.15 把窦律也点亮了；")
    a("假阳段数在那一列看起来不大，是段定义把「连进真室颤的成片误报」算成了一次真阳段。")
    a("")
    a("| 库 | 门 | 非可电击窗 | 其中报警 | 特异度 |")
    a("|---|---|---:|---:|---:|")
    for db in ("vfdb", "cudb", "mitdb"):
        for kind in ("confirm", "firmware", "ann"):
            block = (summary.get("profiles") or {}).get(kind, {}).get(db)
            if not block:
                continue
            w = block["window"]
            neg = w["fp"] + w["tn"]
            spec = (w["tn"] / neg) if neg else float("nan")
            a(f"| {db} | {kind} | {neg} | {w['fp']} | {spec:.3f} |")
    a("")
    a("### 停搏 / FLAT（≥4 s 无拍）")
    a("")
    flat = summary.get("flat") or {}
    if flat:
        a("| 库 | 固件 FLAT 段 | 其中落在可电击段 | 有序节律上的假 FLAT 段 | 参考停搏段 | 检出 |")
        a("|---|---:|---:|---:|---:|---:|")
        for db, row in flat.items():
            a(f"| {db} | {row['n_flat']} | {row['n_flat_during_shock']} | {row['n_flat_fp_organized']} | {row['n_asys']} | {row['n_asys_hit']} |")
        a("")
        a("有序节律上的假 FLAT，是检测器把真拍漏掉、空档拉到 4 s 以上，而且参考标注里也没有 ≥4 s 的停搏。")
        a("落在可电击段上的 FLAT 不是停搏真阳，是室颤期间检测器失锁。")
        a("")
    ns = summary.get("nstdb")
    a("### NSTDB 噪声压力（可选）")
    a("")
    if not ns:
        a("这次没有可用的 NSTDB 记录，压力测试空缺。")
    else:
        a("MIT-BIH 噪声压力库的 ECG 记录（去掉纯噪声 `bw`/`em`/`ma`），通道 0，固件重放链 + 全部门控。")
        a("这些记录不应含室颤，数字是假阳率。")
        a("")
        a("| 记录 | SNR (dB) | 小时 | FP 段 | FP/h |")
        a("|---|---:|---:|---:|---:|")
        for row in ns.get("records", []):
            snr = "—" if row["snr_db"] is None else str(row["snr_db"])
            fph = 0.0 if row["hours"] <= 0 else row["fp_episodes"] / row["hours"]
            a(f"| {row['record']} | {snr} | {row['hours']:.3f} | {row['fp_episodes']} | {fph:.3f} |")
        a("")
        tot = ns.get("total") or {}
        a(f"合计 FP/h：{fmt_rate(tot.get('fp_per_h'), tot.get('fp_per_h_bootstrap'))}。")
    a("")
    a("### 阈值扫描（建议，不改固件）")
    a("")
    a("θ 从 0.05 扫到 0.90，步长 0.05。每次都重新做两窗确认，RMS 和 2.5 s 拍间门保持固件默认。")
    a("纵轴是 VFDB+CUDB 合在一起的 ≥10 s 事件灵敏度（按事件数合并，不是两条库先平均）。")
    a("横轴是 MIT-BIH 48 条的 FP/h。默认 θ=0.15 在图上单独标出。")
    a("")
    a("| θ | 合并事件 Se ≥10 s | VFDB 事件 Se | CUDB 事件 Se | MIT FP/h |")
    a("|---:|---:|---:|---:|---:|")
    def fnum(v):
        return "—" if v is None or not math.isfinite(float(v)) else f"{float(v):.3f}"

    for row in summary.get("tradeoff_theta") or []:
        a(f"| {row['theta']:.2f} | {fnum(row['se_pooled'])} | {fnum(row['se_vfdb'])} | {fnum(row['se_cudb'])} | {fnum(row['fp_per_h_mit'])} |")
    a("")
    a("拍间门限也扫了（θ 和 RMS 不动）：")
    a("")
    a("| 门限 ms | 必须先见过拍 | 合并事件 Se ≥10 s | 其中 VT | 其中 VF/VFIB | 其中 VFL | MIT FP/h |")
    a("|---:|---|---:|---:|---:|---:|---:|")
    for row in summary.get("tradeoff_gate") or []:
        a("| {ms} | {arm} | {se} | {vt} | {vf} | {vfl} | {fp} |".format(
            ms=row["gate_ms"],
            arm=row["require_arm"],
            se=fnum(row["se_pooled"]),
            vt=fnum(row["se_vt"]),
            vf=fnum(row["se_vf"]),
            vfl=fnum(row["se_vfl"]),
            fp=fnum(row["fp_per_h_mit"]),
        ))
    a("")
    a("门限 1000 ms 时 MIT 的假阳段数会冲高，是短空档把成片报警切成很多段，不是检测器突然更吵。窗级假阳没有这个峰。")
    a("")
    sug = summary.get("suggestion") or []
    if sug:
        a("建议（只写在这里，固件默认值未改）：")
        a("")
        for s in sug:
            a(f"- {s}")
        a("")
    a("## 固件里对不上或值得单独看的点")
    a("")
    a("这些是读源码和这次对照得到的，**没有改固件**。")
    a("")
    for item in summary.get("firmware_findings") or []:
        a(f"- {item}")
    a("")
    a("## 不足")
    a("")
    for item in summary.get("limitations") or []:
        a(f"- {item}")
    a("")
    a("逐条记录的数字在 `results/per_record.csv`，机器可读汇总在 `results/summary.json`，图是 `results/vf_rules_eval.png`。")
    a("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_figure(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if Path(font).exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # Panel 1: Se
    fe = summary["frontend"]
    labels = []
    recs, evs, wins = [], [], []
    rec_err, ev_err, win_err = [], [], []
    for db, title in (("vfdb", "VFDB"), ("cudb", "CUDB")):
        block = fe.get(db)
        if not block:
            continue
        labels.append(title)
        def grab(block_se, boot=None):
            if not block_se:
                return 0.0, 0.0, 0.0
            est = block_se["est"]
            lo = boot["lo"] if boot else block_se["lo"]
            hi = boot["hi"] if boot else block_se["hi"]
            return est, max(0.0, est - lo), max(0.0, hi - est)
        r, rlo, rhi = grab(block.get("record_ge10"))
        e, elo, ehi = grab(block["event_ge10"]["se"], block["event_ge10"].get("se_bootstrap"))
        w, wlo, whi = grab(block["window"]["se"], block["window"].get("se_bootstrap"))
        recs.append(r); rec_err.append((rlo, rhi))
        evs.append(e); ev_err.append((elo, ehi))
        wins.append(w); win_err.append((wlo, whi))
    if labels:
        x = np.arange(len(labels))
        width = 0.25
        def yerr(pairs):
            return np.array([[p[0] for p in pairs], [p[1] for p in pairs]])
        ax[0].bar(x - width, recs, width, yerr=yerr(rec_err), label="记录 ≥10 s", color="#c62828", capsize=3)
        ax[0].bar(x, evs, width, yerr=yerr(ev_err), label="事件 ≥10 s", color="#ef6c00", capsize=3)
        ax[0].bar(x + width, wins, width, yerr=yerr(win_err), label="窗 ≥50%", color="#1565c0", capsize=3)
        ax[0].set_xticks(x, labels)
        ax[0].set_ylim(0, 1.05)
        ax[0].set_ylabel("灵敏度")
        ax[0].legend(fontsize=8, frameon=False)
        ax[0].set_title("固件重放链 Se")
    ax[0].axhline(1.0, color="#eee", lw=0.5)

    # Panel 2: MIT FP by profile
    names, vals, los, his = [], [], [], []
    for kind, title in (("confirm", "仅两窗"), ("rms", "+RMS"), ("firmware", "+2.5s拍门")):
        block = (summary.get("profiles") or {}).get(kind, {}).get("mitdb")
        if not block or not block.get("fp_per_h"):
            continue
        est = block["fp_per_h"]["est"]
        boot = block.get("fp_per_h_bootstrap")
        lo = boot["lo"] if boot else block["fp_per_h"]["lo"]
        hi = boot["hi"] if boot else block["fp_per_h"]["hi"]
        names.append(title)
        vals.append(est)
        los.append(max(0.0, est - lo))
        his.append(max(0.0, hi - est))
    if names:
        ax[1].bar(names, vals, color="#455a64", yerr=np.array([los, his]), capsize=3)
        ax[1].set_ylabel("假阳段 / 小时")
        ax[1].set_title("MIT-BIH 48 条 FP/h")

    # Panel 3: tradeoff
    rows = summary.get("tradeoff_theta") or []
    if rows:
        xs = [r["fp_per_h_mit"] for r in rows]
        ys = [r["se_pooled"] for r in rows]
        ax[2].plot(xs, ys, color="#37474f", marker="o", ms=3.5)
        for r in rows:
            if abs(r["theta"] - 0.15) < 1e-6:
                ax[2].scatter([r["fp_per_h_mit"]], [r["se_pooled"]], s=60, color="#c62828", zorder=3, label="θ=0.15")
                ax[2].annotate("θ=0.15", (r["fp_per_h_mit"], r["se_pooled"]), textcoords="offset points", xytext=(6, 6), color="#c62828")
        ax[2].set_xlabel("MIT FP/h")
        ax[2].set_ylabel("VFDB+CUDB 事件 Se ≥10 s")
        ax[2].set_ylim(0, 1.05)
        ax[2].set_title("θ 扫描（门控保持默认）")
        ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pooled_se(recs, flags_map, min_dur=10.0):
    hit = n = 0
    by = {}
    for rec in recs:
        flags = flags_map.get(rec["record"])
        if flags is None:
            continue
        ev = event_counts(flags, rec["segs"], min_dur)
        hit += ev["hits"]
        n += ev["n"]
        for tok, slot in ev["by"].items():
            acc = by.setdefault(tok, [0, 0])
            acc[0] += slot["hit"]
            acc[1] += slot["n"]
    return hit, n, by


def se_of(hit, n):
    return (hit / n) if n else float("nan")


def build_tradeoff(groups):
    thetas = [round(float(x), 2) for x in np.arange(0.05, 0.91, 0.05)]
    rows = []
    vfdb = groups.get("vfdb") or []
    cudb = groups.get("cudb") or []
    mit = groups.get("mitdb") or []
    for th in thetas:
        maps = {}
        for db, recs in (("vfdb", vfdb), ("cudb", cudb), ("mitdb", mit)):
            maps[db] = {}
            for rec in recs:
                maps[db][rec["record"]] = profile_flags(rec, "firmware", theta=th, gate_ms=2500)
        hv, nv, _ = pooled_se(vfdb, maps["vfdb"])
        hc, nc, _ = pooled_se(cudb, maps["cudb"])
        hp, np_, _ = pooled_se(vfdb + cudb, {**maps["vfdb"], **maps["cudb"]})
        fp = 0
        hours = 0.0
        for rec in mit:
            ev = event_counts(maps["mitdb"][rec["record"]], rec["segs"], 10.0)
            fp += ev["fp_episodes"]
            hours += rec["hours"]
        rows.append({
            "theta": th,
            "se_pooled": se_of(hp, np_),
            "se_vfdb": se_of(hv, nv),
            "se_cudb": se_of(hc, nc),
            "fp_per_h_mit": (fp / hours) if hours else float("nan"),
            "fp_episodes_mit": fp,
            "hours_mit": hours,
        })
    gate_rows = []
    for gate_ms in (0, 1000, 1500, 2000, 2500, 4000, 8000):
        for require_arm, kind in ((True, "firmware"), (False, "noarm")):
            if gate_ms == 0 and require_arm:
                # 0 ms with arm still requires a beat to have existed; practically open after the first beat.
                pass
            maps = {}
            for db, recs in (("vfdb", vfdb), ("cudb", cudb), ("mitdb", mit)):
                maps[db] = {rec["record"]: profile_flags(rec, kind, theta=0.15, gate_ms=gate_ms) for rec in recs}
            both = {**maps["vfdb"], **maps["cudb"]}
            hp, np_, by = pooled_se(vfdb + cudb, both)
            vt_h = vt_n = vf_h = vf_n = 0
            vfl_h = vfl_n = 0
            for tok, (h, n) in by.items():
                if tok == "VT":
                    vt_h += h
                    vt_n += n
                if tok in {"VF", "VFIB"}:
                    vf_h += h
                    vf_n += n
                if tok == "VFL":
                    vfl_h += h
                    vfl_n += n
            fp = 0
            hours = 0.0
            for rec in mit:
                ev = event_counts(maps["mitdb"][rec["record"]], rec["segs"], 10.0)
                fp += ev["fp_episodes"]
                hours += rec["hours"]
            gate_rows.append({
                "gate_ms": gate_ms,
                "require_arm": require_arm,
                "se_pooled": se_of(hp, np_),
                "se_vt": se_of(vt_h, vt_n),
                "se_vf": se_of(vf_h, vf_n),
                "se_vfl": se_of(vfl_h, vfl_n),
                "fp_per_h_mit": (fp / hours) if hours else float("nan"),
            })
    return rows, gate_rows


def suggestion_text(theta_rows, gate_rows) -> list[str]:
    out = []
    if not theta_rows:
        return out
    base = next((r for r in theta_rows if abs(r["theta"] - 0.15) < 1e-6), None)
    if base is None or not math.isfinite(base["se_pooled"]):
        return out
    flat = [r for r in theta_rows if r["theta"] <= 0.80 + 1e-9 and math.isfinite(r["se_pooled"])]
    if flat and max(r["se_pooled"] for r in flat) - min(r["se_pooled"] for r in flat) < 0.02:
        hi = max(flat, key=lambda r: r["theta"])
        lo = min(flat, key=lambda r: r["theta"])
        out.append(
            f"θ 从 {lo['theta']:.2f} 到 {hi['theta']:.2f}，合并事件 Se 停在 {base['se_pooled']:.3f}，"
            f"MIT FP/h 从 {lo['fp_per_h_mit']:.3f} 降到 {hi['fp_per_h_mit']:.3f}。"
            "灵敏度几乎不随 θ 动，因为 2.5 s 拍间门已经把大部分真室颤挡在外面。"
            "把默认 θ 抬到 0.80 甚至 0.85 只能再削掉一点假阳，补不回被门控吃掉的灵敏度。这次不改 θ。"
        )
    else:
        cand = [r for r in theta_rows if math.isfinite(r["se_pooled"]) and r["se_pooled"] >= base["se_pooled"] - 0.05]
        if cand:
            best = min(cand, key=lambda r: (r["fp_per_h_mit"], -r["se_pooled"]))
            if abs(best["theta"] - 0.15) > 1e-6 and best["fp_per_h_mit"] < base["fp_per_h_mit"] * 0.8:
                out.append(
                    f"在事件 Se 下降不超过 0.05 的前提下，θ={best['theta']:.2f} 的 MIT FP/h "
                    f"从 {base['fp_per_h_mit']:.3f} 降到 {best['fp_per_h_mit']:.3f}，"
                    f"合并事件 Se 从 {base['se_pooled']:.3f} 变为 {best['se_pooled']:.3f}。"
                    "这只说明公开库上的工作点，不能直接改板上的默认 θ。"
                )
    def find(ms, arm):
        return next((r for r in gate_rows if r["gate_ms"] == ms and bool(r["require_arm"]) is arm), None)
    g = find(2500, True)
    opened = find(0, False)
    if g and opened and math.isfinite(g["se_pooled"]) and math.isfinite(opened["se_pooled"]):
        out.append(
            f"去掉拍间门（0 ms，且不要求先见过拍）时，合并事件 Se 从 {g['se_pooled']:.3f} 变为 {opened['se_pooled']:.3f}，"
            f"MIT FP/h 从 {g['fp_per_h_mit']:.3f} 变为 {opened['fp_per_h_mit']:.3f}。"
            f"VT {g['se_vt']:.3f}→{opened['se_vt']:.3f}，"
            f"VF/VFIB {g['se_vf']:.3f}→{opened['se_vf']:.3f}，"
            f"VFL {g['se_vfl']:.3f}→{opened['se_vfl']:.3f}。"
            "三类一起掉，不是只有室速被吃掉。可电击段里固件拍大约每秒 1 次以上，间隔短于 2.5 s，门在真室颤上是关着的。"
            "假阳段数在去掉门之后不一定更高：报警若从窦律连到室颤，整段都和参考重叠，按段计假阳会把成片误报算成 0。看窗级特异度。"
            "更合理的改法是不要用「距最近 QRS > 2.5 s」当作室颤的放行条件。本次未改固件。"
        )
    return out


def static_findings() -> list[str]:
    return [
        "部署路径是 5 s 不重叠窗。头文件里的「连续 2 窗、时延 ≤ 7.5 s」按 50% 重叠才成立；不重叠时第二窗结束最早在 10 s，相位不利时接近 15 s。",
        "`vfAlarm` 还要 `s_beatEverSeen`。一次会话如果从室颤开始、检测器又始终不报拍，VF 永远不会放行。`noarm` 列就是把这一条拿掉之后的数字。",
        "2.5 s 门控看的是检测器的拍，不是病人是否真有脉搏。室颤波形若被当成 QRS，门控会在真 VF 上关闭。可电击段上的拍密度写在结果里。",
        "注释写「等价 scipy sosfiltfilt padlen=0」。scipy 在 padlen=0 时仍用稳态初值 `sosfilt_zi`；固件正向、反向之前都把二阶节状态清零。一致性测试里量了这个差。",
        "`eval_vf_detect_v2.py` 训练时窗移是 2.5 s（50% 重叠），而且 MIT 对照没有从 360 Hz 重采样到 250 Hz 就按 1250 点切窗。头文件里的 VFDB Se 0.985 / CUDB Se 0.921 / MIT Sp 0.888 不是本报告这套协议，不能和这里的数字比。",
        "真机 `vfProcess` 把重计算丢给 `vf_worker`。算不赢时会丢掉一整窗并把 `s_lastSuspect` 清零，两窗确认被打断。PC 走的是 worker 起不来时的同步分支，不模拟丢窗。",
        "同步分支在窗的最后一个样本上把报警交出来；异步分支要等 worker 把结果塞回队列，门控时间戳会再晚几毫秒到若干采样。相对 2.5 s 可以忽略，但不是同一个采样点。",
        "AFE/模拟器路径把滤波后的样本乘 0.763 再送 VF；回放路径不乘。本评测按回放（输入已是 mV）。0.763 是为了把 AFE 的伏特量级拉回训练用的 mV，不是本实验要估计的增益。",
        "旧 PC 脚本没有 2.5 s 门控，特征公式也和 `compute_features` 不一致。那次 VFDB 14/22、窗 Se 0.40、MIT 约 0.5 h 的数字作废。",
    ]


def static_limitations() -> list[str]:
    return [
        "公开库通道 0 不是产品上的柔性单导联近似 Lead I。MIT-BIH 通道 0 多半是 MLII。导联不同，QRS 幅度和室颤形态都会变。",
        "重放链的数字滤波器增益约 1，吃的是 mV。板上心率检测器是按 AFE 滤波后大约零点几伏标定的。mV 数值碰巧落在同一量级，但不是同一条模拟前端。QRS 相对标注的 Se/PPV 用来暴露这件事，不把它当成心率模块的正式成绩。",
        "没有做 50 Hz 以外的工频模型。梳状滤波按固件固定在 500/10。美式记录里的 60 Hz 不会被这个梳状滤掉。",
        "NSTDB 只说明加噪后的假阳，不是室颤灵敏度。噪声段没有室颤参考。",
        "事件重叠规则（只要沾到报警窗就算检出）对「晚几十秒才响」仍然给真阳。延迟分布单独给了，临床能不能接受要看中位数，不能只看 Se。",
        "假阳按段合并。不加拍间门时，报警经常从非可电击段一直连进可电击段，整段不算假阳，FP/h 会显得很低。窗级特异度才反映 θ=0.15 在窦律上的误报。",
        "Wilson 区间把事件当独立样本，偏窄。报告里的 bootstrap 按记录重抽，记录数只有二十几条时区间仍然宽。",
        "短于 10 s 的室速/室扑不在主分母里。它们在「不限时长」那一行，很多是窗长和确认规则从结构上盖不住的。",
    ]


def csv_rows(recs, flag_maps) -> list[dict]:
    rows = []
    for rec in recs:
        fw = flag_maps["firmware"].get(rec["record"])
        if fw is None:
            continue
        ev = event_counts(fw, rec["segs"], 10.0)
        ev15 = event_counts(fw, rec["segs"], 15.0)
        ev0 = event_counts(fw, rec["segs"], 0.0)
        tp, fn, fp, tn = confusion(fw, rec["labels"])
        q = rec["qrs"]
        flat = rec["flat"]
        row = {
            "db": rec["db"],
            "record": rec["record"],
            "sig_name": rec["sig_name"],
            "units": rec["units"],
            "fs": rec["fs"],
            "hours": rec["hours"],
            "n_win": rec["n_win"],
            "n_nans": rec["nans"],
            "median_rms": rec["median_rms"],
            "n_beats_fw": int(rec["beats"].size),
            "n_beats_ann": int(rec["ann_beats_s"].size),
            "qrs_matched": q["matched"],
            "qrs_se": q["se"],
            "qrs_ppv": q["ppv"],
            "shock_seconds": rec["shock_seconds"],
            "beats_per_s_shock": rec["beats_per_s_shock"],
            "beats_per_s_organized": rec["beats_per_s_organized"],
            "tp_win": tp, "fn_win": fn, "fp_win": fp, "tn_win": tn,
            "se_win": (tp / (tp + fn)) if (tp + fn) else None,
            "n_evt10": ev["n"], "n_evt10_hit": ev["hits"],
            "se_evt10": (ev["hits"] / ev["n"]) if ev["n"] else None,
            "n_evt15": ev15["n"], "n_evt15_hit": ev15["hits"],
            "n_evt_all": ev0["n"], "n_evt_all_hit": ev0["hits"],
            "record_has_evt10": int(ev["n"] > 0),
            "record_hit_evt10": int(ev["hits"] > 0) if ev["n"] else None,
            "fp_episodes": ev["fp_episodes"],
            "fp_episodes_after_10s": ev["fp_episodes_after_10s"],
            "fp_per_h": (ev["fp_episodes"] / rec["hours"]) if rec["hours"] else None,
            "n_flat": flat["n_flat"],
            "n_flat_during_shock": flat["n_flat_during_shock"],
            "n_flat_fp_organized": flat["n_flat_fp_organized"],
            "n_asys": flat["n_asys"],
            "n_asys_hit": flat["n_asys_hit"],
        }
        rows.append(row)
    return rows


def nstdb_snr(name: str):
    if "e_6" in name:
        return -6
    import re
    m = re.search(r"e(\d{2})", name)
    if m:
        return int(m.group(1))
    return None


def aggregate_qrs(recs):
    matched = nref = ndet = 0
    nrec = 0
    for rec in recs:
        if rec["qrs"]["n_ref"] <= 0:
            continue
        nrec += 1
        matched += rec["qrs"]["matched"]
        nref += rec["qrs"]["n_ref"]
        ndet += rec["qrs"]["n_det"]
    return {
        "n_records": nrec,
        "matched": matched,
        "n_ref": nref,
        "n_det": ndet,
        "se": (matched / nref) if nref else None,
        "ppv": (matched / ndet) if ndet else None,
    }


def aggregate_density(recs):
    b_s = s_s = b_o = 0.0
    o_s = 0.0
    any_shock = False
    any_org = False
    for rec in recs:
        if rec["shock_seconds"] > 0 and rec["beats_per_s_shock"] is not None:
            any_shock = True
            b_s += rec["beats_in_shock"]
            s_s += rec["shock_seconds"]
        if rec["beats_per_s_organized"] is not None:
            any_org = True
            o_s += float(np.count_nonzero(rec["labels"] == 0) * 5.0)
            # recompute count from rate * seconds to avoid storing it twice
            b_o += rec["beats_per_s_organized"] * float(np.count_nonzero(rec["labels"] == 0) * 5.0)
    return {
        "per_s_shock": (b_s / s_s) if any_shock and s_s > 0 else None,
        "per_s_organized": (b_o / o_s) if any_org and o_s > 0 else None,
    }


def aggregate_flat(recs):
    acc = {k: 0 for k in ("n_flat", "n_flat_during_shock", "n_flat_fp_organized", "n_asys", "n_asys_hit")}
    for rec in recs:
        for k in acc:
            acc[k] += int(rec["flat"][k])
    return acc


def headline_notes(summary_frontend, profiles) -> list[str]:
    notes = []
    fw = profiles.get("firmware") or {}
    noarm = profiles.get("noarm") or {}
    for db in ("vfdb", "cudb"):
        a = (fw.get(db) or {}).get("event_ge10", {}).get("se")
        b = (noarm.get(db) or {}).get("event_ge10", {}).get("se")
        if a and b and a["n"] and abs(a["est"] - b["est"]) >= 0.02:
            notes.append(
                f"{db}：要求「先见过一拍」使 ≥10 s 事件 Se 从 {b['est']:.3f}（不要求）变为 {a['est']:.3f}（固件）。"
            )
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Firmware VF-rule evaluation on VFDB, CUDB, MIT-BIH.")
    ap.add_argument("--data-root", default=None, help="database root (or set ECG_DATA_ROOT / ECG_REFS_ROOT)")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="score whatever records are already on disk")
    ap.add_argument("--limit", type=int, default=0, help="max records per database (debug)")
    ap.add_argument("--skip-nstdb", action="store_true")
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args()
    if not BIN.exists():
        log("building host")
        subprocess.check_call(["bash", str(HERE / "host" / "build.sh")])
    root = data_root_from(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    log("data root", root)
    download = not args.no_download
    needed = {
        "vfdb": VFDB_RECORDS,
        "cudb": CUDB_RECORDS,
        "mitdb": MIT_RECORDS,
    }
    have = {}
    missing = {}
    for name, names in needed.items():
        got, miss = ensure_db(root, name, names, download and not args.allow_partial)
        have[name] = got
        missing[name] = miss
        log(name, "readable", len(got), "missing", len(miss))
    if any(missing.values()) and not args.allow_partial:
        for name, miss in missing.items():
            if miss:
                log("MISSING", name, ", ".join(miss))
        log("refusing to publish a partial score; pass --allow-partial to override")
        return 2
    order = []
    for db, names in needed.items():
        use = [n for n in names if n in have[db]]
        if args.limit:
            use = use[: args.limit]
        # Also include unexpected extra records only when not requiring the canonical set? No: stick to the canonical names.
        for n in use:
            order.append((db, n, have[db][n]))
    nstdb_order = []
    if not args.skip_nstdb:
        nstdb_have = available_records(root / "nstdb")
        if not nstdb_have and download and not args.allow_partial:
            try:
                import wfdb
                log("downloading nstdb")
                (root / "nstdb").mkdir(parents=True, exist_ok=True)
                wfdb.dl_database("nstdb", str(root / "nstdb"))
                nstdb_have = available_records(root / "nstdb")
            except Exception as exc:
                log("NSTDB download failed", exc)
        skip_noise = {"bw", "em", "ma"}
        names = sorted(n for n in nstdb_have if n not in skip_noise)
        if args.limit:
            names = names[: args.limit]
        nstdb_order = [(n, nstdb_have[n]) for n in names]
        log("nstdb ecg records", len(nstdb_order))

    host = ReplayHost(BIN, HERE / "results" / "host_stderr.log")
    groups = {"vfdb": [], "cudb": [], "mitdb": []}
    failures = []
    t0 = time.time()
    try:
        for db, name, path in order:
            try:
                rec = process_record(host, path, db, args.skip_raw)
                groups[db].append(rec)
                log(db, name, "win", rec["n_win"], "rms", None if rec["median_rms"] is None else round(rec["median_rms"], 4),
                    "beats", int(rec["beats"].size), "shock_s", round(rec["shock_seconds"], 1))
            except Exception as exc:
                failures.append({"db": db, "record": name, "error": str(exc)})
                log("FAIL", db, name, exc)
                traceback.print_exc()
        nstdb_recs = []
        for name, path in nstdb_order:
            try:
                rec = process_record(host, path, "nstdb", skip_raw=True)
                rec["snr_db"] = nstdb_snr(name)
                nstdb_recs.append(rec)
                log("nstdb", name, "snr", rec["snr_db"], "beats", int(rec["beats"].size))
            except Exception as exc:
                failures.append({"db": "nstdb", "record": name, "error": str(exc)})
                log("FAIL nstdb", name, exc)
    finally:
        host.close()
    log(f"scored in {time.time() - t0:.0f}s, failures {len(failures)}")
    if failures and not args.allow_partial:
        log("failures on a complete run, not writing a headline report")
        return 3

    kinds = ("confirm", "rms", "firmware", "noarm", "ann")
    profiles = {}
    flag_maps = {}
    for kind in kinds:
        profiles[kind] = {}
        flag_maps[kind] = {}
        for db, recs in groups.items():
            local = {}
            for rec in recs:
                flags = profile_flags(rec, kind)
                if flags is None:
                    continue
                local[rec["record"]] = flags
                flag_maps[kind][rec["record"]] = flags
            if local:
                # summarize only records that produced flags
                use = [r for r in recs if r["record"] in local]
                profiles[kind][db] = summarize_flags(use, local, 10.0)

    raw_summary = {}
    if not args.skip_raw:
        for db, recs in groups.items():
            local = {}
            for rec in recs:
                flags = raw_firmware_flags(rec)
                if flags is None:
                    continue
                # event_counts uses rec["segs"] which is the frontend clock.
                # Raw flags must be scored against raw segments. Swap in a view.
                shadow = dict(rec)
                shadow["labels"] = rec["raw"]["labels"]
                shadow["segs"] = rec["raw"]["segs"]
                shadow["hours"] = rec["raw"]["hours"]
                shadow["n_win"] = int(flags.size)
                local[rec["record"]] = (shadow, flags)
            if not local:
                continue
            shadows = [local[r["record"]][0] for r in recs if r["record"] in local]
            fmap = {name: pair[1] for name, pair in local.items()}
            raw_summary[db] = summarize_flags(shadows, fmap, 10.0)

    log("tradeoff")
    theta_rows, gate_rows = build_tradeoff(groups)
    notes = headline_notes(profiles.get("firmware"), profiles)
    # RMS sanity
    for db, recs in groups.items():
        rms = [r["median_rms"] for r in recs if r["median_rms"] is not None]
        if not rms:
            continue
        med = float(np.median(rms))
        if med < 0.01 or med > 5.0:
            notes.append(f"{db} 窗 RMS 中位数 {med:.4f} mV，离训练均值 0.425 mV 很远，单位或前端可能不对。")

    vocab = Counter()
    for recs in groups.values():
        for rec in recs:
            vocab.update(rec["vocab"])

    consistency = {}
    cpath = HERE / "results" / "consistency.json"
    if cpath.exists():
        consistency = json.loads(cpath.read_text(encoding="utf-8"))

    nstdb_block = None
    if nstdb_recs:
        fmap = {r["record"]: profile_flags(r, "firmware") for r in nstdb_recs}
        total = summarize_flags(nstdb_recs, fmap, 10.0)
        nstdb_block = {
            "records": [
                {
                    "record": r["record"],
                    "snr_db": r.get("snr_db"),
                    "hours": r["hours"],
                    "fp_episodes": event_counts(fmap[r["record"]], r["segs"], 10.0)["fp_episodes"],
                    "median_rms": r["median_rms"],
                }
                for r in nstdb_recs
            ],
            "total": total,
        }

    summary = {
        "data_root": str(root),
        "firmware": firmware_pin(),
        "theta": 0.15,
        "confirm_windows": 2,
        "rms_min_mV": 0.05,
        "gate_ms": 2500,
        "asystole_ms": 4000,
        "input_headline": "frontend_replay_mV",
        "missing": missing,
        "failures": failures,
        "vocab": dict(vocab),
        "frontend": profiles["firmware"],
        "profiles": profiles,
        "raw": raw_summary,
        "qrs_vs_ann": {db: aggregate_qrs(recs) for db, recs in groups.items()},
        "beat_density": {db: aggregate_density(recs) for db, recs in groups.items()},
        "flat": {db: aggregate_flat(recs) for db, recs in groups.items()},
        "nstdb": nstdb_block,
        "tradeoff_theta": theta_rows,
        "tradeoff_gate": gate_rows,
        "suggestion": suggestion_text(theta_rows, gate_rows),
        "headline_notes": notes,
        "firmware_findings": static_findings(),
        "limitations": static_limitations(),
        "consistency": {
            "passed": consistency.get("passed"),
            "max_abs_score": consistency.get("max_abs_score"),
            "max_abs_feat": consistency.get("max_abs_feat"),
            "stream_alarm_matches_confirm": consistency.get("stream_alarm_matches_confirm"),
            "legacy_attachment_max_abs_feat": consistency.get("legacy_attachment_max_abs_feat"),
            "scipy_sosfiltfilt_padlen0_vs_zero_state_max_abs": consistency.get("scipy_sosfiltfilt_padlen0_vs_zero_state_max_abs"),
            "synth_qrs_beats_20s": consistency.get("synth_qrs_beats_20s"),
        },
        "versions": {
            "numpy": np.__version__,
        },
    }
    try:
        import scipy
        import wfdb
        import matplotlib
        summary["versions"]["scipy"] = scipy.__version__
        summary["versions"]["wfdb"] = wfdb.__version__
        summary["versions"]["matplotlib"] = matplotlib.__version__
    except Exception:
        pass

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop bulky per-record event lists from the json profiles? They're useful and small.
    (out_dir / "summary.json").write_text(json.dumps(py(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    rows = csv_rows([r for recs in groups.values() for r in recs], flag_maps)
    if rows:
        with (out_dir / "per_record.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    with (out_dir / "tradeoff_theta.csv").open("w", newline="", encoding="utf-8") as f:
        if theta_rows:
            w = csv.DictWriter(f, fieldnames=list(theta_rows[0].keys()))
            w.writeheader()
            w.writerows(theta_rows)
    with (out_dir / "tradeoff_gate.csv").open("w", newline="", encoding="utf-8") as f:
        if gate_rows:
            w = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
            w.writeheader()
            w.writerows(gate_rows)
    plot_figure(summary, out_dir / "vf_rules_eval.png")
    write_report(summary, HERE / "REPORT_VF规则评测_v2.md")
    log("wrote", out_dir / "summary.json")
    # Print the headline so the log stands alone.
    for db in ("vfdb", "cudb", "mitdb"):
        block = profiles["firmware"].get(db)
        if not block:
            continue
        ev = block["event_ge10"]["se"]
        fp = block["fp_per_h"]
        log("HEAD", db,
            "Se_evt10", None if not ev else round(ev["est"], 3),
            "Se_rec", None if not block["record_ge10"] else round(block["record_ge10"]["est"], 3),
            "FP/h", None if not fp else round(fp["est"], 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
