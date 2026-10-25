"""Event / window / record metrics. Undetermined seconds stay in the denominator."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from constants import BOOTSTRAP_N, SEED
from events import (
    Episode,
    analyzable_stats,
    match_events,
    predicted_alarm_episodes,
    reference_episodes,
)
from labels import window_true_label
from af_backend import AFWindow, WindowsLike


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def record_eval(
    y_sec: np.ndarray,
    pred_sec: np.ndarray,
    windows: WindowsLike,
    min_episode_s: float = 10.0,
    compute_window_metrics: bool = True,
) -> dict:
    """Evaluate one record. y_sec: 1 pos, 0 neg, -1 excluded (AFL in af_only)."""
    y_sec = np.asarray(y_sec)
    pred_sec = np.asarray(pred_sec)
    n = min(len(y_sec), len(pred_sec))
    y_sec = y_sec[:n]
    pred_sec = pred_sec[:n]
    # Excluded seconds cannot contribute TP or FP. Keep them as undetermined
    # for denominator honesty: they still count in total hours.
    pred_eff = pred_sec.copy()
    pred_eff[(y_sec == -1) & (pred_sec == 1)] = 0  # AFL called AF: counted separately
    afl_called_af_s = float(np.sum((y_sec == -1) & (pred_sec == 1)))

    ref = reference_episodes(y_sec, min_duration_s=min_episode_s)
    pred_ep = predicted_alarm_episodes(pred_eff)
    ref_m, fp_eps = match_events(ref, pred_ep, pred_eff)
    # FP that sit entirely inside excluded time already converted to 0.
    # Remaining FP must not overlap any positive reference (match_events).
    # Also drop FP whose seconds are majority excluded (belt and braces).
    fp_keep: List[Episode] = []
    for p in fp_eps:
        a = max(0, int(p.start_s))
        b = min(n, int(p.end_s))
        if b <= a:
            continue
        if np.mean(y_sec[a:b] == -1) >= 0.5:
            continue
        fp_keep.append(p)

    tp = sum(1 for r in ref_m if r.detected)
    fn = len(ref_m) - tp
    fn_normal = sum(1 for r in ref_m if r.miss_reason == "called_normal")
    fn_undet = sum(1 for r in ref_m if r.miss_reason == "undetermined")
    delays = [r.delay_s for r in ref_m if r.detected and r.delay_s is not None]
    az = analyzable_stats(pred_sec)

    # Duration metrics: undetermined on AF seconds counts as miss (not dropped).
    af_s = y_sec == 1
    n_af = int(np.sum(af_s))
    dur_tp = int(np.sum(af_s & (pred_sec == 1)))
    dur_fn_normal = int(np.sum(af_s & (pred_sec == 0)))
    dur_fn_undet = int(np.sum(af_s & (pred_sec == 2)))
    neg_s = y_sec == 0
    n_neg = int(np.sum(neg_s))
    dur_fp = int(np.sum(neg_s & (pred_sec == 1)))

    # Window-level (undetermined windows on AF = FN, not excluded)
    w_tp = w_fp = w_fn = w_tn = w_undet_pos = w_undet_neg = w_excl = 0
    if compute_window_metrics and windows is not None and len(windows) > 0:
        if isinstance(windows, np.ndarray):
            triples = (
                (float(w["t_start"]), float(w["t_end"]), int(w["label"]))
                for w in windows
            )
        else:
            triples = ((w.t_start, w.t_end, w.label) for w in windows)
        for t_start, t_end, wlabel in triples:
            yt = window_true_label(t_start, t_end, y_sec)
            if yt == -1:
                w_excl += 1
                continue
            if wlabel == 2:
                if yt == 1:
                    w_fn += 1
                    w_undet_pos += 1
                else:
                    w_undet_neg += 1  # not TN: undetermined is reported separately
                continue
            if yt == 1 and wlabel == 1:
                w_tp += 1
            elif yt == 1 and wlabel == 0:
                w_fn += 1
            elif yt == 0 and wlabel == 1:
                w_fp += 1
            else:
                w_tn += 1

    rec_true = 1 if len(ref) > 0 else 0
    rec_pred = 1 if np.any(pred_eff == 1) else 0

    hours_total = az["duration_s"] / 3600.0
    hours_anal = az["analyzable_s"] / 3600.0
    # **az first so true-reference n_af_s / n_neg_s are not overwritten by
    # predicted-AF second counts (those live in n_pred_af_s).
    return {
        **az,
        "n_ref_events": len(ref_m),
        "event_tp": tp,
        "event_fn": fn,
        "event_fp": len(fp_keep),
        "event_fn_called_normal": fn_normal,
        "event_fn_undetermined": fn_undet,
        "event_se": _safe_div(tp, tp + fn),
        "fp_per_hour_total": _safe_div(len(fp_keep), hours_total),
        "fp_per_hour_analyzable": _safe_div(len(fp_keep), hours_anal),
        "mean_delay_s": float(np.mean(delays)) if delays else float("nan"),
        "median_delay_s": float(np.median(delays)) if delays else float("nan"),
        "n_delay": len(delays),
        "duration_se": _safe_div(dur_tp, n_af),
        "duration_fp_per_hour": _safe_div(dur_fp, n_neg / 3600.0 if n_neg else 0.0),
        "n_af_s": float(n_af),
        "n_neg_s": float(n_neg),
        "dur_tp_s": float(dur_tp),
        "dur_fn_normal_s": float(dur_fn_normal),
        "dur_fn_undet_s": float(dur_fn_undet),
        "dur_fp_s": float(dur_fp),
        "afl_called_af_s": afl_called_af_s,
        "window_tp": w_tp,
        "window_fp": w_fp,
        "window_fn": w_fn,
        "window_tn": w_tn,
        "window_undet_pos": w_undet_pos,
        "window_undet_neg": w_undet_neg,
        "window_excl": w_excl,
        "window_se": _safe_div(w_tp, w_tp + w_fn),
        "window_sp": _safe_div(w_tn, w_tn + w_fp),
        "record_true": rec_true,
        "record_pred": rec_pred,
        "hours_total": hours_total,
        "hours_analyzable": hours_anal,
        "ref_events": [
            {
                "start_s": r.start_s,
                "end_s": r.end_s,
                "detected": r.detected,
                "delay_s": r.delay_s,
                "miss_reason": r.miss_reason,
            }
            for r in ref_m
        ],
        "fp_events": [
            {"start_s": p.start_s, "end_s": p.end_s} for p in fp_keep
        ],
    }


def pool_records(rows: Sequence[dict]) -> dict:
    keys_sum = [
        "n_ref_events",
        "event_tp",
        "event_fn",
        "event_fp",
        "event_fn_called_normal",
        "event_fn_undetermined",
        "n_delay",
        "n_af_s",
        "n_neg_s",
        "dur_tp_s",
        "dur_fn_normal_s",
        "dur_fn_undet_s",
        "dur_fp_s",
        "afl_called_af_s",
        "window_tp",
        "window_fp",
        "window_fn",
        "window_tn",
        "window_undet_pos",
        "window_undet_neg",
        "window_excl",
        "record_true",
        "record_pred",
        "hours_total",
        "hours_analyzable",
        "duration_s",
        "analyzable_s",
        "undetermined_s",
        "longest_undetermined_s",
        "n_af_s_pred",
    ]
    out = {k: 0.0 for k in keys_sum}
    out["longest_undetermined_s"] = 0.0
    delay_num = 0.0
    delay_den = 0
    rec_tp = rec_fn = rec_fp = rec_tn = 0
    for r in rows:
        for k in keys_sum:
            if k == "n_af_s_pred":
                continue
            if k in r and r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k])):
                if k == "longest_undetermined_s":
                    out[k] = max(out[k], float(r[k]))
                else:
                    out[k] += float(r[k])
        if r.get("n_delay"):
            try:
                fv = float(r.get("mean_delay_s"))
            except (TypeError, ValueError):
                fv = float("nan")
            if np.isfinite(fv):
                delay_num += fv * int(r["n_delay"])
                delay_den += int(r["n_delay"])
        rt, rp = int(r.get("record_true", 0)), int(r.get("record_pred", 0))
        if rt == 1 and rp == 1:
            rec_tp += 1
        elif rt == 1 and rp == 0:
            rec_fn += 1
        elif rt == 0 and rp == 1:
            rec_fp += 1
        else:
            rec_tn += 1
    out["event_se"] = _safe_div(out["event_tp"], out["event_tp"] + out["event_fn"])
    out["fp_per_hour_total"] = _safe_div(out["event_fp"], out["hours_total"])
    out["fp_per_hour_analyzable"] = _safe_div(out["event_fp"], out["hours_analyzable"])
    out["mean_delay_s"] = _safe_div(delay_num, delay_den)
    out["duration_se"] = _safe_div(out["dur_tp_s"], out["n_af_s"])
    out["window_se"] = _safe_div(out["window_tp"], out["window_tp"] + out["window_fn"])
    out["window_sp"] = _safe_div(out["window_tn"], out["window_tn"] + out["window_fp"])
    out["record_se"] = _safe_div(rec_tp, rec_tp + rec_fn)
    out["record_sp"] = _safe_div(rec_tn, rec_tn + rec_fp)
    out["record_tp"] = rec_tp
    out["record_fn"] = rec_fn
    out["record_fp"] = rec_fp
    out["record_tn"] = rec_tn
    out["analyzable_frac"] = _safe_div(out["analyzable_s"], out["duration_s"])
    out["undetermined_frac"] = _safe_div(out["undetermined_s"], out["duration_s"])
    out["n_records"] = len(rows)
    return out


def bootstrap_ci(
    rows: Sequence[dict],
    n_boot: int = BOOTSTRAP_N,
    seed: int = SEED,
    keys: Sequence[str] = (
        "event_se",
        "fp_per_hour_total",
        "fp_per_hour_analyzable",
        "mean_delay_s",
        "duration_se",
        "window_se",
        "window_sp",
        "record_se",
        "record_sp",
        "analyzable_frac",
        "undetermined_frac",
    ),
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    n = len(rows)
    if n == 0:
        return {k: [float("nan"), float("nan")] for k in keys}
    acc = {k: [] for k in keys}
    idx = np.arange(n)
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        pooled = pool_records([rows[i] for i in samp])
        for k in keys:
            acc[k].append(pooled.get(k, float("nan")))
    ci = {}
    for k, vals in acc.items():
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            ci[k] = [float("nan"), float("nan")]
        else:
            ci[k] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return ci


def youden_like(pooled: dict) -> float:
    """Development-set objective: event Se minus a mild FP/h penalty.

    Analyzable fraction is a hard floor applied by the caller.
    """
    se = pooled.get("event_se", float("nan"))
    fph = pooled.get("fp_per_hour_total", float("nan"))
    if not np.isfinite(se):
        return -1e9
    if not np.isfinite(fph):
        fph = 0.0
    return float(se) - 0.02 * float(fph)
