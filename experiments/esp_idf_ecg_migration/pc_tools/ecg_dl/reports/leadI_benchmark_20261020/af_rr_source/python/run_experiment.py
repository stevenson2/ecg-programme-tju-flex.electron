#!/usr/bin/env python3
"""AF RR-source mechanism experiment. One-command entry: ../run.sh

Fixed AF backend (firmware CV+entropy 10 s window). Only the RR generator changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from af_backend import (
    FirmwareAF,
    WIN_DTYPE,
    relabel_windows,
    windows_to_array,
    windows_to_second_labels,
)
from constants import (
    AF_CV_THR,
    AF_ENT_THR,
    AF_WIN_S,
    BOOTSTRAP_N,
    CAL_CV_GRID,
    CAL_ENT_GRID,
    CAL_MIN_ANALYZABLE,
    MIN_EPISODE_S,
    MIN_RR_SEC,
    SEED,
)
from io_wfdb import list_records, load_rhythm, load_signal
from labels import AFIB_CODES, Interval, mask_from_intervals, second_true_label
from metrics import bootstrap_ci, pool_records, record_eval, youden_like
from peaks import match_peaks, rr_from_peaks, segment_mask
from reject_rules import filter_rr_stream, rr_cv, short_rr_fraction
from sources import (
    SOURCE_NAMES,
    annotation_peaks,
    fw_aiqrs_peaks,
    fw_strict_peaks,
    mix_nstdb_noise,
    xqrs_peaks,
)
from splits import split_table
from plots import make_all_plots
from write_report import write_report


SCHEMES = ("af_only", "af_or_afl")
SETTINGS = ("default", "calibrated")


def _json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_to_py), encoding="utf-8")


def _to_py(x):
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x if x is None else str(x)


def discover(data_dir: Path, smoke: bool, max_records: Optional[int]) -> Dict[str, List[str]]:
    out = {}
    smoke_want = {
        "afdb": ["04015", "04043", "04936", "07162", "08219"],
        "mitdb": ["100", "101", "103", "201", "202", "210", "219"],
        "ltafdb": None,  # first few
        "nstdb": ["em", "ma", "bw"],
    }
    for db in ("afdb", "mitdb", "ltafdb", "nstdb"):
        d = data_dir / db
        if not d.exists():
            out[db] = []
            continue
        recs = list_records(d)
        if db == "nstdb":
            recs = [r for r in recs if r in ("em", "ma", "bw") or r[:2].isdigit()]
        if smoke:
            want = smoke_want.get(db)
            if want:
                recs = [r for r in recs if r in want] or recs[:3]
            else:
                recs = recs[:2]
        if max_records is not None:
            recs = recs[: max_records]
        out[db] = recs
    return out


def cache_path(cache: Path, db: str, rec: str, name: str) -> Path:
    return cache / db / rec / name


def load_or_build_meta(data_dir: Path, cache: Path, db: str, rec: str) -> dict:
    p = cache_path(cache, db, rec, "meta.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    db_dir = data_dir / db
    sig = load_signal(db_dir, rec)
    ivs = load_rhythm(db_dir, rec, sig["duration_s"], sig["fs"])
    meta = {
        "db": db,
        "rec": rec,
        "fs": sig["fs"],
        "duration_s": sig["duration_s"],
        "channel": sig["channel"],
        "channel_name": sig["channel_name"],
        "n_samples": int(sig["sig"].size),
        "intervals": [
            {"start_s": iv.start_s, "end_s": iv.end_s, "code": iv.code} for iv in ivs
        ],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def intervals_from_meta(meta: dict) -> List[Interval]:
    return [
        Interval(float(x["start_s"]), float(x["end_s"]), str(x["code"]))
        for x in meta.get("intervals", [])
    ]


DET_SOURCES = ("fw_strict", "fw_aiqrs", "xqrs")


def _cache_npz(path: Path, t: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=np.asarray(t, dtype=np.float64))


def compute_detectors_job(args) -> str:
    """Worker: load ECG once, run three detectors, cache peaks. Returns rec id."""
    data_dir, cache_dir, db, rec = args
    data_dir = Path(data_dir)
    cache = Path(cache_dir)
    sig = load_signal(data_dir / db, rec)
    mapping = {
        "fw_strict": fw_strict_peaks,
        "fw_aiqrs": fw_aiqrs_peaks,
        "xqrs": xqrs_peaks,
    }
    for src, fn in mapping.items():
        p = cache_path(cache, db, rec, f"peaks_{src}.npz")
        if p.exists():
            continue
        t = fn(sig["sig"], sig["fs"])
        _cache_npz(p, t)
    return f"{db}/{rec}"


def get_peaks(
    data_dir: Path,
    cache: Path,
    db: str,
    rec: str,
    source: str,
    sig: Optional[dict],
) -> np.ndarray:
    p = cache_path(cache, db, rec, f"peaks_{source}.npz")
    if p.exists():
        return np.load(p)["t"]
    db_dir = data_dir / db
    if source == "ann":
        t = annotation_peaks(db_dir, rec, manual=False)
    elif source == "ann_manual":
        t = annotation_peaks(db_dir, rec, manual=True)
    else:
        if sig is None:
            sig = load_signal(db_dir, rec)
        if source == "fw_strict":
            t = fw_strict_peaks(sig["sig"], sig["fs"])
        elif source == "fw_aiqrs":
            t = fw_aiqrs_peaks(sig["sig"], sig["fs"])
        elif source == "xqrs":
            t = xqrs_peaks(sig["sig"], sig["fs"])
        else:
            raise ValueError(source)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, t=np.asarray(t, dtype=np.float64))
    return np.asarray(t, dtype=np.float64)


def score_windows(peak_t: np.ndarray):
    """Window geometry is threshold-independent (0 and 1 both slide)."""
    rr, _ = rr_from_peaks(peak_t)
    af = FirmwareAF(cv_thr=-1.0, ent_thr=-1.0)
    return rr, windows_to_array(af.evaluate_rr(rr))


def save_window_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(arr, dtype=WIN_DTYPE) if getattr(arr, "size", 0) else np.zeros(0, dtype=WIN_DTYPE)
    np.savez_compressed(path, w=w)


def load_window_array(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros(0, dtype=WIN_DTYPE)
    return np.load(path)["w"]


def slim_ev(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if k not in ("ref_events", "fp_events")}


def eval_windows(
    windows,
    duration_s: float,
    intervals: List[Interval],
    scheme: str,
    cv_thr: float,
    ent_thr: float,
    n_peaks: int,
    n_rr: int,
    compute_window_metrics: bool = True,
) -> dict:
    wins = relabel_windows(windows, cv_thr, ent_thr)
    pred = windows_to_second_labels(wins, duration_s)
    y = second_true_label(intervals, duration_s, scheme=scheme)
    ev = record_eval(
        y, pred, wins, min_episode_s=MIN_EPISODE_S,
        compute_window_metrics=compute_window_metrics,
    )
    ev["n_peaks"] = int(n_peaks)
    ev["n_rr"] = int(n_rr)
    ev["cv_thr"] = float(cv_thr)
    ev["ent_thr"] = float(ent_thr)
    return ev, wins, pred, y


def eval_rr_path(
    peak_t: np.ndarray,
    duration_s: float,
    intervals: List[Interval],
    scheme: str,
    cv_thr: float,
    ent_thr: float,
) -> dict:
    rr, windows = score_windows(peak_t)
    ev, wins, pred, y = eval_windows(
        windows, duration_s, intervals, scheme, cv_thr, ent_thr,
        n_peaks=int(np.asarray(peak_t).size), n_rr=int(rr.size),
    )
    return ev, wins, pred, y, rr


def mechanism_block(
    ref_t: np.ndarray,
    det_t: np.ndarray,
    y_af: np.ndarray,
    rr_ref: np.ndarray,
    rr_det: np.ndarray,
) -> dict:
    m = match_peaks(ref_t, det_t)
    af_ref = segment_mask(ref_t, y_af)
    na_ref = segment_mask(ref_t, ~y_af)
    # FN in AF vs non-AF
    fn_af = int(np.sum((~m["ref_matched"]) & af_ref)) if ref_t.size else 0
    fn_na = int(np.sum((~m["ref_matched"]) & na_ref)) if ref_t.size else 0
    tp_af = int(np.sum(m["ref_matched"] & af_ref)) if ref_t.size else 0
    tp_na = int(np.sum(m["ref_matched"] & na_ref)) if ref_t.size else 0
    n_af = int(np.sum(af_ref))
    n_na = int(np.sum(na_ref))
    # RR of reference in AF: times of the later peak of each RR
    ref_end = ref_t[1:] if ref_t.size else ref_t
    af_rr_mask = segment_mask(ref_end, y_af) if ref_end.size else np.zeros(0, dtype=bool)
    na_rr_mask = segment_mask(ref_end, ~y_af) if ref_end.size else np.zeros(0, dtype=bool)
    rr_ref_af = rr_ref[af_rr_mask] if rr_ref.size == af_rr_mask.size else np.zeros(0)
    rr_ref_na = rr_ref[na_rr_mask] if rr_ref.size == na_rr_mask.size else np.zeros(0)
    det_end = det_t[1:] if det_t.size else det_t
    det_af_mask = segment_mask(det_end, y_af) if det_end.size else np.zeros(0, dtype=bool)
    rr_det_af = rr_det[det_af_mask] if rr_det.size == det_af_mask.size else np.zeros(0)

    # Would firmware RR gates drop the *reference* AF beats?
    filt = filter_rr_stream(ref_t)
    # reasons aligned to peaks; skip first
    reasons = filt["reasons"]
    short_abs_af = rel_af = 0
    if ref_t.size and reasons:
        for i in range(1, min(len(reasons), ref_t.size)):
            if not af_ref[i]:
                continue
            if reasons[i] == "short_abs":
                short_abs_af += 1
            elif reasons[i] == "short_rel_median":
                rel_af += 1
    return {
        "qrs_tp": m["tp"],
        "qrs_fn": m["fn"],
        "qrs_fp": m["fp"],
        "qrs_se": m["se"],
        "qrs_ppv": m["ppv"],
        "qrs_se_af": (tp_af / n_af) if n_af else float("nan"),
        "qrs_se_non_af": (tp_na / n_na) if n_na else float("nan"),
        "fn_af": fn_af,
        "fn_non_af": fn_na,
        "tp_af": tp_af,
        "n_ref_af": n_af,
        "n_ref_non_af": n_na,
        "rr_cv_ref_af": rr_cv(rr_ref_af),
        "rr_cv_ref_non_af": rr_cv(rr_ref_na),
        "rr_cv_det_af": rr_cv(rr_det_af),
        "short_rr_frac_ref_af": short_rr_fraction(rr_ref_af, MIN_RR_SEC),
        "short_rr_frac_det_af": short_rr_fraction(rr_det_af, MIN_RR_SEC),
        "rule_short_abs_af": short_abs_af,
        "rule_short_rel_af": rel_af,
        "rule_drop_frac_af": (
            (short_abs_af + rel_af) / n_af if n_af else float("nan")
        ),
    }


def _cal_record_grid(args) -> dict:
    """All (cv, ent) cells for one development record / one RR source."""
    windows, duration_s, iv_dicts, n_peaks, n_rr = args
    ivs = [Interval(float(d["start_s"]), float(d["end_s"]), str(d["code"])) for d in iv_dicts]
    grid = {}
    for cv in CAL_CV_GRID:
        for ent in CAL_ENT_GRID:
            ev, *_ = eval_windows(
                windows, duration_s, ivs, "af_only", cv, ent, n_peaks, n_rr,
                compute_window_metrics=False,
            )
            grid[(float(cv), float(ent))] = slim_ev(ev)
    return grid


def calibrate_on_dev(dev_payloads: List[dict], cache: Path, jobs: int = 1) -> dict:
    """Grid-search CV/entropy per RR source. Windows are loaded from cache."""
    best = {}
    nj = max(1, int(jobs))
    for source in SOURCE_NAMES:
        job_args = []
        for p in dev_payloads:
            arr = load_window_array(cache_path(cache, p["db"], p["rec"], f"windows_{source}.npz"))
            peaks = p["peaks"].get(source, np.zeros(0))
            job_args.append(
                (
                    arr,
                    p["duration_s"],
                    p["intervals"],
                    int(np.asarray(peaks).size),
                    int(p.get("n_rr", {}).get(source, 0)),
                )
            )
        if not job_args:
            continue
        if nj == 1 or len(job_args) == 1:
            grids = [_cal_record_grid(a) for a in job_args]
        else:
            with ProcessPoolExecutor(max_workers=min(nj, len(job_args))) as ex:
                grids = list(ex.map(_cal_record_grid, job_args))
        winner = None
        for cv in CAL_CV_GRID:
            for ent in CAL_ENT_GRID:
                key = (float(cv), float(ent))
                evs = [g[key] for g in grids]
                pooled = pool_records(evs)
                if pooled.get("analyzable_frac", 0) < CAL_MIN_ANALYZABLE:
                    continue
                score = youden_like(pooled)
                cand = {
                    "cv_thr": float(cv),
                    "ent_thr": float(ent),
                    "score": float(score),
                    "dev_event_se": pooled.get("event_se"),
                    "dev_fph": pooled.get("fp_per_hour_total"),
                    "dev_analyzable_frac": pooled.get("analyzable_frac"),
                }
                if winner is None or cand["score"] > winner["score"]:
                    winner = cand
        if winner is None:
            winner = {
                "cv_thr": AF_CV_THR,
                "ent_thr": AF_ENT_THR,
                "score": None,
                "fallback": "default_unanalyzable_grid",
            }
        best[source] = winner
    return best


def score_one_record(args) -> dict:
    """Worker: annotation peaks + AF windows for every RR source. Windows go to npz."""
    data_dir, cache_dir, db, rec, split = args
    data_dir = Path(data_dir)
    cache = Path(cache_dir)
    meta = load_or_build_meta(data_dir, cache, db, rec)
    duration_s = float(meta["duration_s"])
    ivs = intervals_from_meta(meta)
    y_af = mask_from_intervals(ivs, duration_s, AFIB_CODES)
    peaks = {}
    n_rr: Dict[str, int] = {}
    sig = None
    for src in SOURCE_NAMES:
        try:
            need_sig = (
                src in DET_SOURCES
                and not cache_path(cache, db, rec, f"peaks_{src}.npz").exists()
            )
            if need_sig and sig is None:
                sig = load_signal(data_dir / db, rec)
            t = get_peaks(data_dir, cache, db, rec, src, sig)
            peaks[src] = np.asarray(t, dtype=np.float64)
            print(f"  {db}/{rec} {src}: {peaks[src].size} peaks", flush=True)
        except Exception as e:
            print(f"  {db}/{rec} {src} FAIL {e}", flush=True)
            traceback.print_exc()
            peaks[src] = np.zeros(0, dtype=np.float64)
    sig = None
    for src, t in peaks.items():
        wpath = cache_path(cache, db, rec, f"windows_{src}.npz")
        rpath = cache_path(cache, db, rec, f"rr_{src}.npz")
        if wpath.exists() and rpath.exists():
            rr = np.load(rpath)["rr"]
            n_rr[src] = int(np.asarray(rr).size)
            continue
        rr, wins = score_windows(t)
        n_rr[src] = int(rr.size)
        save_window_array(wpath, wins)
        np.savez_compressed(rpath, rr=np.asarray(rr, dtype=np.float64))
    return {
        "db": db,
        "rec": rec,
        "split": split,
        "duration_s": duration_s,
        "intervals": [
            {"start_s": iv.start_s, "end_s": iv.end_s, "code": iv.code} for iv in ivs
        ],
        "peaks": peaks,
        "y_af": y_af,
        "n_rr": n_rr,
        "channel_name": meta.get("channel_name"),
        "fs": meta.get("fs"),
    }


def eval_one_record(args) -> tuple:
    payload, cal, cache_dir = args
    cache = Path(cache_dir)
    db, rec, split = payload["db"], payload["rec"], payload["split"]
    duration_s = payload["duration_s"]
    ivs = intervals_from_meta({"intervals": payload["intervals"]})
    peaks = payload["peaks"]
    y_af = payload["y_af"]
    n_rr_map = payload.get("n_rr") or {}
    per_record = []
    mech_store = []
    example_candidates = []
    ref_t = peaks.get("ann_manual")
    if ref_t is None or np.asarray(ref_t).size == 0:
        ref_t = peaks.get("ann", np.zeros(0))
    ref_t = np.asarray(ref_t)
    rr_ref, _ = rr_from_peaks(ref_t)
    for src, t in peaks.items():
        t = np.asarray(t)
        rr_det, _ = rr_from_peaks(t)
        mech = None
        if src not in ("ann",) and ref_t.size:
            mech = mechanism_block(ref_t, t, y_af, rr_ref, rr_det)
            mech_store.append({"db": db, "rec": rec, "split": split, "source": src, **mech})
        windows = load_window_array(cache_path(cache, db, rec, f"windows_{src}.npz"))
        n_rr = int(n_rr_map.get(src, rr_det.size))
        for setting in SETTINGS:
            if setting == "default":
                cv, ent = AF_CV_THR, AF_ENT_THR
            else:
                c = cal.get(src, {})
                cv = float(c.get("cv_thr", AF_CV_THR))
                ent = float(c.get("ent_thr", AF_ENT_THR))
            for scheme in SCHEMES:
                ev, _wins, _pred, _y = eval_windows(
                    windows, duration_s, ivs, scheme, cv, ent,
                    n_peaks=int(t.size), n_rr=n_rr,
                    compute_window_metrics=True,
                )
                extra = {
                    "n_peaks": int(t.size),
                    "channel_name": payload.get("channel_name"),
                }
                per_record.append(
                    flatten_eval(
                        db, rec, src, setting, scheme, split, ev,
                        mech if setting == "default" and scheme == "af_only" else None,
                        extra,
                    )
                )
                if setting == "default" and scheme == "af_only" and src == "fw_strict":
                    for fe in ev.get("ref_events", []):
                        if not fe["detected"]:
                            example_candidates.append(
                                {
                                    "kind": "miss",
                                    "db": db,
                                    "rec": rec,
                                    "start_s": fe["start_s"],
                                    "end_s": fe["end_s"],
                                    "miss_reason": fe["miss_reason"],
                                    "source": src,
                                }
                            )
                    for fe in ev.get("fp_events", [])[:3]:
                        example_candidates.append(
                            {
                                "kind": "fp",
                                "db": db,
                                "rec": rec,
                                "start_s": fe["start_s"],
                                "end_s": fe["end_s"],
                                "source": src,
                            }
                        )
    return per_record, mech_store, example_candidates


def flatten_eval(db, rec, source, setting, scheme, split, ev, mech, extra) -> dict:
    row = {
        "db": db,
        "rec": rec,
        "source": source,
        "setting": setting,
        "scheme": scheme,
        "split": split,
    }
    skip = {"ref_events", "fp_events"}
    for k, v in ev.items():
        if k in skip:
            continue
        if isinstance(v, (float, int, np.floating, np.integer)) or v is None:
            row[k] = _to_py(v)
    if mech:
        for k, v in mech.items():
            row[f"mech_{k}"] = _to_py(v)
    row.update(extra or {})
    return row


def nstdb_stress(data_dir: Path, cache: Path, mit_recs: List[str], out_dir: Path) -> dict:
    noise_dir = data_dir / "nstdb"
    if not noise_dir.exists() or not mit_recs:
        return {"skipped": True, "reason": "missing nstdb or mit records"}
    noises = {}
    for name in ("em", "ma"):
        try:
            noises[name] = load_signal(noise_dir, name)
        except Exception as e:
            print(f"[nstdb] cannot load {name}: {e}", flush=True)
    if not noises:
        return {"skipped": True, "reason": "no em/ma"}
    mit_dir = data_dir / "mitdb"
    rows = []
    for rec in mit_recs[:4]:
        try:
            sig = load_signal(mit_dir, rec)
            meta = load_or_build_meta(data_dir, cache, "mitdb", rec)
            ivs = intervals_from_meta(meta)
        except Exception as e:
            print(f"[nstdb] skip {rec}: {e}", flush=True)
            continue
        clean_t = get_peaks(data_dir, cache, "mitdb", rec, "fw_strict", sig)
        for nname, nobj in noises.items():
            for snr in (12.0, 6.0):
                mixed = mix_nstdb_noise(sig["sig"], sig["fs"], nobj["sig"], nobj["fs"], snr, seed=SEED)
                t = fw_strict_peaks(mixed, sig["fs"])
                ev, *_ = eval_rr_path(t, sig["duration_s"], ivs, "af_only", AF_CV_THR, AF_ENT_THR)
                rows.append(
                    {
                        "rec": rec,
                        "noise": nname,
                        "snr_db": snr,
                        "event_se": ev["event_se"],
                        "fp_per_hour_total": ev["fp_per_hour_total"],
                        "analyzable_frac": ev["analyzable_frac"],
                        "n_peaks": ev["n_peaks"],
                        "clean_n_peaks": int(clean_t.size),
                    }
                )
    _json_dump(out_dir / "nstdb_stress.json", {"rows": rows})
    return {"skipped": False, "n": len(rows), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE.parent)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--skip-nstdb", action="store_true")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("AF_RR_JOBS", "2")))
    args = ap.parse_args()
    root: Path = args.root
    data_dir = root / "data"
    cache = root / "cache"
    out_dir = root / "results"
    fig_dir = root / "figures"
    split_dir = root / "splits"
    for d in (data_dir, cache, out_dir, fig_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    n_boot = 50 if args.smoke else (args.bootstrap or BOOTSTRAP_N)
    t0 = time.time()

    if not args.skip_download:
        from download import download_dbs
        rec_map = None
        if args.smoke:
            rec_map = {
                "mitdb": ["100", "101", "103", "201", "202", "210", "219"],
                "afdb": ["04015", "04043", "04936", "07162", "08219"],
                "ltafdb": ["00", "01"],
                "nstdb": ["em", "ma"],
            }
            dbs = "mitdb,afdb,nstdb"
        else:
            dbs = "mitdb,afdb,ltafdb,nstdb"
        rc = download_dbs(data_dir, dbs, record_map=rec_map)
        if rc != 0:
            print(f"[download] rc={rc}, continuing with whatever is present", flush=True)

    found = discover(data_dir, args.smoke, args.max_records)
    print("[discover]", {k: len(v) for k, v in found.items()}, flush=True)
    split_rows = split_table({k: v for k, v in found.items() if k != "nstdb"})
    _json_dump(split_dir / "patient_split.json", {"records": split_rows, "seed": SEED})

    todo = []
    for row in split_rows:
        db, rec = row["db"], row["rec"]
        if any(
            not cache_path(cache, db, rec, f"peaks_{s}.npz").exists()
            for s in DET_SOURCES
        ):
            todo.append((str(data_dir), str(cache), db, rec))
    if todo:
        nj = max(1, int(args.jobs))
        print(f"[detectors] {len(todo)} records, jobs={nj}", flush=True)
        with ProcessPoolExecutor(max_workers=nj) as ex:
            futs = [ex.submit(compute_detectors_job, j) for j in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    name = fut.result()
                    print(f"  detectors {i}/{len(todo)} {name}", flush=True)
                except Exception as e:
                    print(f"  detectors FAIL {e}", flush=True)
                    traceback.print_exc()

    per_record: List[dict] = []
    mech_store: List[dict] = []
    example_candidates = []
    rec_payloads = []
    nj = max(1, int(args.jobs))

    score_jobs = [
        (str(data_dir), str(cache), row["db"], row["rec"], row["split"])
        for row in split_rows
    ]
    print(f"[score] {len(score_jobs)} records, jobs={nj}", flush=True)
    if nj == 1 or len(score_jobs) <= 1:
        scored = []
        for j in score_jobs:
            try:
                scored.append(score_one_record(j))
            except Exception as e:
                print(f"  score FAIL {j[2]}/{j[3]} {e}", flush=True)
                traceback.print_exc()
    else:
        scored = []
        with ProcessPoolExecutor(max_workers=nj) as ex:
            futs = {ex.submit(score_one_record, j): j for j in score_jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    payload = fut.result()
                    scored.append(payload)
                    print(f"  score {i}/{len(score_jobs)} {j[2]}/{j[3]}", flush=True)
                except Exception as e:
                    print(f"  score FAIL {j[2]}/{j[3]} {e}", flush=True)
                    traceback.print_exc()
    rec_payloads = scored
    dev_payloads = [p for p in rec_payloads if p.get("split") == "dev"]

    print("\n[calibrate] AFDB grid search per source...", flush=True)
    cal = calibrate_on_dev(dev_payloads, cache, jobs=nj)
    _json_dump(out_dir / "calibration.json", cal)
    for src, c in cal.items():
        print(
            f"  cal {src}: cv={c.get('cv_thr')} ent={c.get('ent_thr')} "
            f"dev_se={c.get('dev_event_se')} fph={c.get('dev_fph')}",
            flush=True,
        )

    print(f"[eval] {len(rec_payloads)} records, jobs={nj}", flush=True)
    eval_jobs = [(p, cal, str(cache)) for p in rec_payloads]
    if nj == 1 or len(eval_jobs) <= 1:
        eval_out = []
        for job in eval_jobs:
            eval_out.append(eval_one_record(job))
    else:
        eval_out = []
        with ProcessPoolExecutor(max_workers=nj) as ex:
            futs = {ex.submit(eval_one_record, job): job[0]["rec"] for job in eval_jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = futs[fut]
                try:
                    eval_out.append(fut.result())
                    print(f"  eval {i}/{len(eval_jobs)} {rec}", flush=True)
                except Exception as e:
                    print(f"  eval FAIL {rec} {e}", flush=True)
                    traceback.print_exc()
    for rows, mechs, exs in eval_out:
        per_record.extend(rows)
        mech_store.extend(mechs)
        example_candidates.extend(exs)

    # write per_record.csv
    csv_path = out_dir / "per_record.csv"
    if per_record:
        keys = sorted({k for r in per_record for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_record:
                w.writerow({k: r.get(k, "") for k in keys})
    print(f"[csv] {csv_path} n={len(per_record)}", flush=True)

    def subset(split=None, source=None, setting=None, scheme=None, db=None):
        rows = per_record
        if split:
            rows = [r for r in rows if r["split"] == split]
        if source:
            rows = [r for r in rows if r["source"] == source]
        if setting:
            rows = [r for r in rows if r["setting"] == setting]
        if scheme:
            rows = [r for r in rows if r["scheme"] == scheme]
        if db:
            rows = [r for r in rows if r["db"] == db]
        return rows

    summary = {
        "hypothesis": (
            "在连续单导联监测中，为稳定心率而设计的候选心搏筛选"
            "（短 RR / 相对中位 RR / 相对幅度）会丢掉真实节律信息，从而削弱 AF 事件检出。"
        ),
        "firmware_af": {
            "file": "experiments/esp_idf_ecg_migration/components/ecg_core/src/af_detect.cpp",
            "win_s": AF_WIN_S,
            "cv_thr": AF_CV_THR,
            "ent_thr": AF_ENT_THR,
            "min_rr": 6,
            "host_main_cc_discards_af": True,
            "comb50_included": False,
            "sqi_forced_1": True,
        },
        "reject_gates": {
            "MIN_RR_SEC": MIN_RR_SEC,
            "RR_RATIO_MIN": 0.65,
            "AMP_FRAC_PREV": 0.55,
        },
        "smoke": bool(args.smoke),
        "n_bootstrap": n_boot,
        "elapsed_s": None,
        "records_found": {k: len(v) for k, v in found.items()},
        "calibration": cal,
        "by_split": {},
        "mechanism_pooled": {},
    }

    for split in ("dev", "test_ltaf", "test_mit"):
        summary["by_split"][split] = {}
        for setting in SETTINGS:
            summary["by_split"][split][setting] = {}
            for scheme in SCHEMES:
                summary["by_split"][split][setting][scheme] = {}
                for src in SOURCE_NAMES:
                    rows = subset(split=split, source=src, setting=setting, scheme=scheme)
                    if not rows:
                        continue
                    pooled = pool_records(rows)
                    ci = bootstrap_ci(rows, n_boot=n_boot, seed=SEED)
                    summary["by_split"][split][setting][scheme][src] = {
                        "pooled": pooled,
                        "ci95": ci,
                        "n_records": len(rows),
                    }

    def _pool_mech(ms):
        def _nanmean(key):
            xs = []
            for m in ms:
                v = m.get(key, np.nan)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    xs.append(fv)
            return float(np.mean(xs)) if xs else None
        if not ms:
            return {}
        return {
            "n_records": len(ms),
            "qrs_se_af": _nanmean("qrs_se_af"),
            "qrs_se_non_af": _nanmean("qrs_se_non_af"),
            "short_rr_frac_ref_af": _nanmean("short_rr_frac_ref_af"),
            "short_rr_frac_det_af": _nanmean("short_rr_frac_det_af"),
            "rr_cv_ref_af": _nanmean("rr_cv_ref_af"),
            "rr_cv_det_af": _nanmean("rr_cv_det_af"),
            "rule_drop_frac_af": _nanmean("rule_drop_frac_af"),
            "fn_af_sum": int(sum(m.get("fn_af", 0) for m in ms)),
            "fn_non_af_sum": int(sum(m.get("fn_non_af", 0) for m in ms)),
        }

    summary["mechanism_pooled"] = {}
    for split in ("dev", "test_ltaf", "test_mit"):
        summary["mechanism_pooled"][split] = {}
        for src in SOURCE_NAMES:
            ms = [m for m in mech_store if m["source"] == src and m.get("split") == split]
            blob = _pool_mech(ms)
            if blob:
                summary["mechanism_pooled"][split][src] = blob
    # gold-peak subset (MIT + LTAF) used by the hypothesis checks
    gold = [m for m in mech_store if m.get("split") in ("test_ltaf", "test_mit")]
    summary["mechanism_pooled"]["gold_peaks"] = {
        src: _pool_mech([m for m in gold if m["source"] == src])
        for src in SOURCE_NAMES
        if any(m["source"] == src for m in gold)
    }

    if not args.skip_nstdb:
        print("\n[nstdb] optional stress...", flush=True)
        try:
            summary["nstdb"] = nstdb_stress(data_dir, cache, found.get("mitdb", []), out_dir)
        except Exception as e:
            summary["nstdb"] = {"skipped": True, "reason": str(e)}
            traceback.print_exc()

    # hypothesis verdict from TEST LTAF default af_only, fallback MIT
    verdict = decide_hypothesis(summary)
    summary["hypothesis_verdict"] = verdict
    summary["elapsed_s"] = time.time() - t0
    _json_dump(out_dir / "summary.json", summary)
    _json_dump(out_dir / "example_candidates.json", example_candidates[:80])
    _json_dump(out_dir / "mechanism_per_record.json", mech_store)

    print("\n[plots]", flush=True)
    make_all_plots(
        root=root,
        per_record=per_record,
        summary=summary,
        rec_payloads=rec_payloads,
        example_candidates=example_candidates,
        data_dir=data_dir,
    )
    report_path = root / "REPORT_AF_RR来源机制实验.md"
    write_report(report_path, summary, per_record)
    print(f"[done] {time.time()-t0:.1f}s  report={report_path}", flush=True)
    return 0


def decide_hypothesis(summary: dict) -> dict:
    """Data-driven, can conclude the hypothesis is FALSE."""
    def grab(split):
        node = (
            summary.get("by_split", {})
            .get(split, {})
            .get("default", {})
            .get("af_only", {})
        )
        return node

    split = "test_ltaf" if grab("test_ltaf") else "test_mit"
    node = grab(split)
    cal_node = (
        summary.get("by_split", {})
        .get(split, {})
        .get("calibrated", {})
        .get("af_only", {})
    )
    se = {}
    fph = {}
    for src, blob in node.items():
        se[src] = blob["pooled"].get("event_se")
        fph[src] = blob["pooled"].get("fp_per_hour_total")
    mech_all = summary.get("mechanism_pooled", {})
    mech = mech_all.get("gold_peaks") or mech_all.get(split) or {}
    fw = mech.get("fw_strict", {})
    se_ann = se.get("ann_manual") or se.get("ann")
    se_fw = se.get("fw_strict")
    se_ai = se.get("fw_aiqrs")
    se_xq = se.get("xqrs")
    gap = None
    if se_ann is not None and se_fw is not None and np.isfinite(se_ann) and np.isfinite(se_fw):
        gap = float(se_ann) - float(se_fw)
    short_ref = fw.get("short_rr_frac_ref_af")
    short_det = fw.get("short_rr_frac_det_af")
    cv_ref = fw.get("rr_cv_ref_af")
    cv_det = fw.get("rr_cv_det_af")
    qrs_af = fw.get("qrs_se_af")
    qrs_na = fw.get("qrs_se_non_af")
    # calibration gap
    se_fw_cal = None
    if cal_node.get("fw_strict"):
        se_fw_cal = cal_node["fw_strict"]["pooled"].get("event_se")
    cal_closes = False
    if gap is not None and se_ann is not None and se_fw_cal is not None:
        gap_cal = float(se_ann) - float(se_fw_cal)
        cal_closes = gap_cal < 0.5 * gap if gap > 0 else True

    holds_gap = gap is not None and gap > 0.05
    holds_short = (
        short_ref is not None
        and short_det is not None
        and np.isfinite(short_ref)
        and np.isfinite(short_det)
        and short_ref > short_det + 0.05
    )
    holds_cv = (
        cv_ref is not None
        and cv_det is not None
        and np.isfinite(cv_ref)
        and np.isfinite(cv_det)
        and cv_ref > cv_det + 0.02
    )
    holds_qrs = (
        qrs_af is not None
        and qrs_na is not None
        and np.isfinite(qrs_af)
        and np.isfinite(qrs_na)
        and qrs_af < qrs_na - 0.05
    )
    evidence_for = sum([holds_gap, holds_short, holds_cv, holds_qrs])
    if se_fw is None and se_ann is None:
        holds = None
        text = "数据不足，无法判定。"
    elif evidence_for >= 3 and holds_gap and not cal_closes:
        holds = True
        text = (
            f"在 {split} 上固件严格检测器事件灵敏度低于标注峰"
            f"（ΔSe={gap:.3f}），且 AF 段短 RR 被截断、RR CV 下降；开发集校准未能消除差距。"
        )
    elif holds_gap and cal_closes:
        holds = False
        text = (
            "默认阈值下严格检测器较差，但各路径在开发集校准后差距明显缩小"
            "（阈值不匹配解释）。"
        )
    elif not holds_gap:
        holds = False
        text = (
            "固定 AF 后端后，严格检测器的事件灵敏度并不系统性地低于"
            "人工修正峰 / 宽松触发 / 外部检测器。心率筛选会丢掉短 RR，"
            "但不是本实验中 AF 事件漏检的主因。"
        )
    else:
        holds = False
        text = (
            "事件灵敏度有差距，但机制侧未能同时看到 AF 段短 RR 截断与 CV 下降。"
        )
    return {
        "holds": holds,
        "split_used": split,
        "text": text,
        "event_se": se,
        "fp_per_hour_total": fph,
        "gap_ann_minus_fw": gap,
        "cal_closes_gap": cal_closes,
        "se_fw_calibrated": se_fw_cal,
        "checks": {
            "event_gap": holds_gap,
            "short_rr_truncated": holds_short,
            "cv_reduced": holds_cv,
            "qrs_se_worse_in_af": holds_qrs,
        },
        "se_aiqrs": se_ai,
        "se_xqrs": se_xq,
    }


if __name__ == "__main__":
    raise SystemExit(main())
