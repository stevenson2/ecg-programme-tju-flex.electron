#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sim_mixed_testset.py — Real-world-proportion mixed test set construction & evaluation.

Constructs MIT+PTB mixed test sets blended by epidemiological disease-mix ratios
(心律失常 ≈ 60% : 心梗 ≈ 40%) and evaluates deployment candidates (P2A, KD, D-chain)
on them.  The pure-MIT and pure-PTB baselines are reproduced first for cross-check.

Epidemiological evidence for the 60:40 ratio (VERIFIED via web search):
  - Danish VISP screening (67-yr-olds, single-lead ECG, n=4437): of 152 major ECG
    abnormalities — rhythm/rate disorders 92 (60.5%), signs of myocardial damage
    28 (18.4%), conduction disorders 32 (21.1%).  Rhythm + conduction ≈ arrhythmia-
    type ≈ 82%, but conservative split for the two available domains: 60% arrhythmia
    : 40% MI.
  - China Cardiovascular Report 2023: AF 4.87 M vs MI ~3 M prevalent →
    arrhythmia:MI ≈ 62:38 (corroborates 60:40).
  - PTB DB diagnostic classes: MI 148/268 (55.2%), healthy controls 52/268 (19.4%).
  - MIT-BIH AAMI beat distribution: N 82.6% / S 2.6% / V 6.6% / F 0.7% / Q 7.3%
    (de Chazal 2004).

Two mixed-testset constructions (report BOTH):
  M1 患者构成 60:40 (记录级采样, 真实人群模拟):
       Select records so MIT:PTB record-count ratio ≈ 60:40.  Keep ALL beats of
       a sampled record → temporal/event structure preserved.
  M2 异常类型构成 60:40 (拍级控制, 用户字面意图):
       Sample abnormal beats so MIT-abnormal : PTB-abnormal = 60:40, then add
       normal beats at --normal-frac (default 0.75; also run 0.85).

Evaluated chains per test set:
  (a) P2A @ θ=0.50 single-beat
  (b) KD a070_t1 @ θ=0.50 single-beat
  (c) Deployment chain D: P2A@θ=0.50 OR [KD@θ=0.35 N-of-M(3,5)]
"""

import sys
import json
from pathlib import Path
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import (
    set_npz_suffix, load_mit_incart_merged, add_channel_dim, load_ptb_data,
)
from data.patient_split import (
    build_mit_patient_map, build_incart_patient_map,
    build_ptb_patient_map, patient_level_split,
)

MODELS = Path(__file__).resolve().parent / "models"
GAP = 3          # beats between alarm clusters for event segmentation
SEED = 42        # fixed for reproducibility

# ─── helpers (verbatim from sim_temporal_agg.py) ──────────────────────────────

def _beat_metrics(y, alarm):
    """Beat-level P / R / F1 / [FP among normals] / alarm fraction."""
    tp = int(((alarm == 1) & (y == 1)).sum())
    fp = int(((alarm == 1) & (y == 0)).sum())
    fn = int(((alarm == 0) & (y == 1)).sum())
    n_normal = max(1, int((y == 0).sum()))
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = (2 * prec * rec) / max(1e-9, prec + rec)
    fp_rate = fp / n_normal
    alarm_rate = float((alarm == 1).sum()) / len(y)
    return {"P": prec, "R": rec, "F1": f1, "误报": fp_rate, "报警": alarm_rate}


def _event_metrics(y, alarm, record_ids, gap=GAP):
    """Event-level precision & recall (GAP=3 non-alarm beats ends an event)."""
    unique_rids = np.unique(record_ids)
    all_events = []
    rec_has_abnormal = {}
    rec_has_true_event = {}

    for rid in unique_rids:
        mask = record_ids == rid
        global_idx = np.where(mask)[0]
        rec_has_abnormal[int(rid)] = (y[mask] == 1).any()
        rec_has_true_event[int(rid)] = False

        rec_alarm = alarm[mask]
        alarm_local = np.where(rec_alarm)[0]
        if len(alarm_local) == 0:
            continue

        cur = [alarm_local[0]]
        for k in range(1, len(alarm_local)):
            n_non_alarm_between = alarm_local[k] - alarm_local[k - 1] - 1
            if n_non_alarm_between < gap:
                cur.append(alarm_local[k])
            else:
                all_events.append([int(global_idx[j]) for j in cur])
                cur = [alarm_local[k]]
        all_events.append([int(global_idx[j]) for j in cur])

    total_events = len(all_events)
    if total_events == 0:
        n_ab = sum(rec_has_abnormal.values())
        return {"evt_prec": 0.0, "evt_rec": 0.0,
                "total_events": 0, "true_events": 0,
                "n_rec_abnormal": int(n_ab), "n_rec_true_event": 0}

    true_events = sum(1 for evt in all_events if any(y[j] == 1 for j in evt))
    evt_prec = true_events / total_events

    for evt in all_events:
        if any(y[j] == 1 for j in evt):
            rid = int(record_ids[evt[0]])
            rec_has_true_event[rid] = True

    n_rec_ab = sum(rec_has_abnormal.values())
    n_rec_te = sum(rec_has_true_event.values())
    evt_rec = n_rec_te / max(1, n_rec_ab)

    return {"evt_prec": evt_prec, "evt_rec": evt_rec,
            "total_events": total_events, "true_events": true_events,
            "n_rec_abnormal": int(n_rec_ab), "n_rec_true_event": int(n_rec_te)}


def _apply_nofm(prob, theta, N, M):
    """N-of-M temporal filter on a single record's probability sequence.
    Returns bool array of length L: True = post-filter alarm."""
    L = len(prob)
    if L < M:
        return np.zeros(L, dtype=bool)

    trig = (prob >= theta).astype(np.int32)
    cum = np.zeros(L + 1, dtype=np.int32)
    cum[1:] = np.cumsum(trig)
    window_sums = cum[M:] - cum[:L - M + 1]
    window_trig = (window_sums >= N)

    alarm = np.zeros(L, dtype=bool)
    for i in range(L):
        w_start = max(0, i - M + 1)
        w_end = min(L - M, i) + 1
        if w_start < w_end and window_trig[w_start:w_end].any():
            alarm[i] = True
    return alarm


def _floatify(d):
    """Convert numpy scalars to plain Python floats for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ─── chain evaluator ──────────────────────────────────────────────────────────

def _apply_deploy_chain(p_p2a, p_kd, record_ids):
    """Deployment chain D: P2A@θ=0.50 OR [KD@θ=0.35 N-of-M(3,5)].

    Returns bool alarm array (same length as inputs).
    """
    p2a_alarm = (p_p2a >= 0.50)
    kd_nofm_alarm = np.zeros(len(p_kd), dtype=bool)
    THETA_KD = 0.35
    N, M = 3, 5
    for rid in np.unique(record_ids):
        mask = record_ids == rid
        rec_prob = p_kd[mask]
        if len(rec_prob) < N:
            continue
        rec_alarm = _apply_nofm(rec_prob, THETA_KD, N, M)
        kd_nofm_alarm[mask] = rec_alarm
    return p2a_alarm | kd_nofm_alarm


def _evaluate_chains(y, p_p2a, p_kd, record_ids, label):
    """Evaluate 3 chains and return dict of results."""
    results = {}

    # (a) P2A @ θ=0.50
    alarm_a = (p_p2a >= 0.50)
    results[f"{label}_P2A_θ050"] = {
        **_beat_metrics(y, alarm_a),
        **_event_metrics(y, alarm_a, record_ids, GAP),
        "chain": "P2A_θ0.50", "label": label,
    }

    # (b) KD @ θ=0.50
    alarm_b = (p_kd >= 0.50)
    results[f"{label}_KD_θ050"] = {
        **_beat_metrics(y, alarm_b),
        **_event_metrics(y, alarm_b, record_ids, GAP),
        "chain": "KD_θ0.50", "label": label,
    }

    # (c) Deployment chain D: P2A@0.50 OR [KD@0.35 N-of-M(3,5)]
    alarm_d = _apply_deploy_chain(p_p2a, p_kd, record_ids)
    results[f"{label}_Dchain"] = {
        **_beat_metrics(y, alarm_d),
        **_event_metrics(y, alarm_d, record_ids, GAP),
        "chain": "Dchain_P2Aθ050_OR_KDθ035_N3M5", "label": label,
    }

    return results


# ─── test-set constructors ────────────────────────────────────────────────────

def _build_m1_record_mix(y_mit, y_ptb, rid_mit, rid_ptb,
                         p_p2a_mit, p_p2a_ptb, p_kd_mit, p_kd_ptb,
                         rng, target_mit_ratio=0.60):
    """M1: record-level 60:40 sampling.  Keep ALL beats of a sampled record."""
    unique_mit = np.unique(rid_mit)
    unique_ptb = np.unique(rid_ptb)
    n_mit_recs = len(unique_mit)
    n_ptb_recs = len(unique_ptb)

    # Determine how many records to sample so MIT:PTB ≈ 60:40
    n_total_target = min(
        int(n_mit_recs / target_mit_ratio),
        int(n_ptb_recs / (1 - target_mit_ratio)),
    )
    n_mit_sample = int(round(n_total_target * target_mit_ratio))
    n_ptb_sample = n_total_target - n_mit_sample

    # Sample records
    mit_sample_rids = set(rng.choice(unique_mit, size=n_mit_sample, replace=False))
    ptb_sample_rids = set(rng.choice(unique_ptb, size=n_ptb_sample, replace=False))

    # Collect beats belonging to sampled records
    mit_keep = np.array([int(r) in mit_sample_rids for r in rid_mit], dtype=bool)
    ptb_keep = np.array([int(r) in ptb_sample_rids for r in rid_ptb], dtype=bool)

    y_mix = np.concatenate([y_mit[mit_keep], y_ptb[ptb_keep]])
    r_mix = np.concatenate([rid_mit[mit_keep], rid_ptb[ptb_keep]])
    p2a_mix = np.concatenate([p_p2a_mit[mit_keep], p_p2a_ptb[ptb_keep]])
    kd_mix  = np.concatenate([p_kd_mit[mit_keep], p_kd_ptb[ptb_keep]])

    # Sort by record_id for contiguity (N-of-M needs per-record grouping)
    order = np.argsort(r_mix, kind='stable')
    y_mix = y_mix[order]
    r_mix = r_mix[order]
    p2a_mix = p2a_mix[order]
    kd_mix = kd_mix[order]

    # Calculate achieved record-level ratio
    n_mit_actual = len(np.unique(r_mix[r_mix < 100000]))  # MIT range
    n_ptb_actual = len(np.unique(r_mix[r_mix >= 100000]))  # PTB range
    # More precise: identify MIT vs PTB by domain tag
    mit_in_mix = np.isin(r_mix, list(mit_sample_rids))
    ptb_in_mix = np.isin(r_mix, list(ptb_sample_rids))

    n_mit_beats = int(mit_in_mix.sum())
    n_ptb_beats = int(ptb_in_mix.sum())
    n_mit_ab = int(y_mix[mit_in_mix].sum())
    n_ptb_ab = int(y_mix[ptb_in_mix].sum())
    n_norm = int((y_mix == 0).sum())

    info = {
        "method": "M1_record_60_40",
        "n_mit_recs_total": n_mit_recs,
        "n_ptb_recs_total": n_ptb_recs,
        "n_mit_recs_sampled": n_mit_sample,
        "n_ptb_recs_sampled": n_ptb_sample,
        "n_mit_beats": n_mit_beats,
        "n_ptb_beats": n_ptb_beats,
        "n_mit_abnormal": n_mit_ab,
        "n_ptb_abnormal": n_ptb_ab,
        "n_normal": n_norm,
        "total_beats": len(y_mix),
        "mit_ab_ratio_of_mit": n_mit_ab / max(1, n_mit_beats),
        "ptb_ab_ratio_of_ptb": n_ptb_ab / max(1, n_ptb_beats),
        "mit_record_share": n_mit_sample / (n_mit_sample + n_ptb_sample),
        "ptb_record_share": n_ptb_sample / (n_mit_sample + n_ptb_sample),
        "actual_abnormal_share_mit": n_mit_ab / max(1, n_mit_ab + n_ptb_ab),
    }
    return y_mix, r_mix, p2a_mix, kd_mix, info


def _build_m2_beat_mix(y_mit, y_ptb, rid_mit, rid_ptb,
                       p_p2a_mit, p_p2a_ptb, p_kd_mit, p_kd_ptb,
                       rng, target_mit_ab_ratio=0.60,
                       normal_frac=0.75):
    """M2: beat-level abnormal ratio 60:40, with normal beats at normal_frac.

    Sample abnormal beats from MIT and PTB so MIT:PTB-abnormal ≈ 60:40,
    then add normal beats so (normal / total) = normal_frac.
    Beat indices track original position for within-record ordering.

    Returns:
      y_mix, r_mix, p2a_mix, kd_mix, info
    """
    # ── Available abnormal beats ──
    mit_ab_mask = y_mit == 1
    ptb_ab_mask = y_ptb == 1
    mit_ab_idx = np.where(mit_ab_mask)[0]
    ptb_ab_idx = np.where(ptb_ab_mask)[0]
    n_mit_ab_avail = len(mit_ab_idx)
    n_ptb_ab_avail = len(ptb_ab_idx)

    # Determine how many abnormal to sample at 60:40
    ab_total_target = min(
        int(n_mit_ab_avail / target_mit_ab_ratio),
        int(n_ptb_ab_avail / (1 - target_mit_ab_ratio)),
    )
    n_mit_ab_sample = int(round(ab_total_target * target_mit_ab_ratio))
    n_ptb_ab_sample = ab_total_target - n_mit_ab_sample

    # ── Sample abnormal beats ──
    mit_ab_chosen = rng.choice(mit_ab_idx, size=n_mit_ab_sample, replace=False)
    ptb_ab_chosen = rng.choice(ptb_ab_idx, size=n_ptb_ab_sample, replace=False)

    # ── Normal beats ──
    ab_total = n_mit_ab_sample + n_ptb_ab_sample
    # total_beats * (1 - normal_frac) = ab_total → total_beats = ab_total / (1-normal_frac)
    total_beats = max(ab_total + 1, int(round(ab_total / (1.0 - normal_frac))))
    n_normal_sample = max(0, total_beats - ab_total)

    mit_norm_idx = np.where(y_mit == 0)[0]
    ptb_norm_idx = np.where(y_ptb == 0)[0]
    n_norm_mit_avail = len(mit_norm_idx)
    n_norm_ptb_avail = len(ptb_norm_idx)
    total_norm_avail = n_norm_mit_avail + n_norm_ptb_avail

    # Allocate normals proportionally to availability
    n_mit_norm = int(round(n_normal_sample * n_norm_mit_avail / max(1, total_norm_avail)))
    n_mit_norm = min(n_mit_norm, n_norm_mit_avail)
    n_ptb_norm = min(n_normal_sample - n_mit_norm, n_norm_ptb_avail)
    # Fill any shortfall from whichever domain has remaining
    shortfall = n_normal_sample - n_mit_norm - n_ptb_norm
    if shortfall > 0:
        rem_mit = n_norm_mit_avail - n_mit_norm
        rem_ptb = n_norm_ptb_avail - n_ptb_norm
        if rem_mit > 0:
            add_mit = min(shortfall, rem_mit)
            n_mit_norm += add_mit
            shortfall -= add_mit
        if shortfall > 0 and rem_ptb > 0:
            n_ptb_norm += min(shortfall, rem_ptb)

    mit_norm_chosen = rng.choice(mit_norm_idx, size=n_mit_norm, replace=False) if n_mit_norm > 0 else np.array([], dtype=int)
    ptb_norm_chosen = rng.choice(ptb_norm_idx, size=n_ptb_norm, replace=False) if n_ptb_norm > 0 else np.array([], dtype=int)

    # ── Assemble with explicit domain tags ──
    # Each entry: (beat_index, domain(0=MIT,1=PTB))
    entries = []
    for idx in mit_ab_chosen:
        entries.append((int(idx), 0))
    for idx in ptb_ab_chosen:
        entries.append((int(idx), 1))
    for idx in mit_norm_chosen:
        entries.append((int(idx), 0))
    for idx in ptb_norm_chosen:
        entries.append((int(idx), 1))

    e_arr = np.array(entries, dtype=object)
    e_idx = np.array([e[0] for e in entries], dtype=int)
    e_dom = np.array([e[1] for e in entries], dtype=int)  # 0=MIT, 1=PTB

    # Build output arrays using domain-aware indexing
    n_total = len(entries)
    y_out = np.zeros(n_total, dtype=y_mit.dtype)
    r_out = np.zeros(n_total, dtype=rid_mit.dtype)
    p2a_out = np.zeros(n_total, dtype=np.float64)
    kd_out = np.zeros(n_total, dtype=np.float64)

    mit_mask = e_dom == 0
    ptb_mask = e_dom == 1
    y_out[mit_mask] = y_mit[e_idx[mit_mask]]
    y_out[ptb_mask] = y_ptb[e_idx[ptb_mask]]
    r_out[mit_mask] = rid_mit[e_idx[mit_mask]]
    r_out[ptb_mask] = rid_ptb[e_idx[ptb_mask]]
    p2a_out[mit_mask] = p_p2a_mit[e_idx[mit_mask]]
    p2a_out[ptb_mask] = p_p2a_ptb[e_idx[ptb_mask]]
    kd_out[mit_mask] = p_kd_mit[e_idx[mit_mask]]
    kd_out[ptb_mask] = p_kd_ptb[e_idx[ptb_mask]]

    # Sort by record_id (stable) to keep same-record beats contiguous
    order = np.argsort(r_out, kind='stable')
    y_out = np.ascontiguousarray(y_out[order])
    r_out = np.ascontiguousarray(r_out[order])
    p2a_out = np.ascontiguousarray(p2a_out[order])
    kd_out = np.ascontiguousarray(kd_out[order])
    # Also sort domain masks for info computation
    mit_mask_sorted = mit_mask[order]
    ptb_mask_sorted = ptb_mask[order]

    n_mit_total = int(mit_mask_sorted.sum())
    n_ptb_total = int(ptb_mask_sorted.sum())

    info = {
        "method": f"M2_beat_ab_60_40_normalfrac_{normal_frac}",
        "n_mit_ab_available": n_mit_ab_avail,
        "n_ptb_ab_available": n_ptb_ab_avail,
        "n_mit_ab_sampled": n_mit_ab_sample,
        "n_ptb_ab_sampled": n_ptb_ab_sample,
        "target_abnormal_ratio_mit": target_mit_ab_ratio,
        "actual_abnormal_ratio_mit": n_mit_ab_sample / max(1, n_mit_ab_sample + n_ptb_ab_sample),
        "normal_frac_target": normal_frac,
        "normal_frac_actual": (len(y_out) - int(y_out.sum())) / max(1, len(y_out)),
        "total_beats": len(y_out),
        "total_abnormal": int(y_out.sum()),
        "total_normal": int((y_out == 0).sum()),
        "n_mit_beats": n_mit_total,
        "n_ptb_beats": n_ptb_total,
        "preserves_record_structure": False,
        "note": "beat-level sampling — event metrics approximate; record IDs retained for N-of-M",
    }
    return y_out, r_out, p2a_out, kd_out, info


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    set_npz_suffix("_deploy")
    rng = np.random.default_rng(SEED)
    print("=" * 90)
    print("Real-World-Proportion Mixed Test Set Construction & Evaluation")
    print("=" * 90)

    # ── 1. Load data (exact copies from eval_binary_all.py) ───────────────────
    print("\n[1/6] Loading MIT+INCART merged (deploy suffix)...")
    mi = load_mit_incart_merged()
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({r + 100000: "inc_" + p
                 for r, p in build_incart_patient_map().items()})
    tr, va, te, stats_mit = patient_level_split(mi["record_ids"], pmap)
    x_mit  = mi["beats"][te]
    y_mit  = mi["labels"][te]
    rid_mit = mi["record_ids"][te]
    print(f"  MIT test: {len(y_mit)} beats, {int(y_mit.sum())} abnormal, "
          f"{len(np.unique(rid_mit))} records, "
          f"{stats_mit['n_test']} patients")

    print("\n[2/6] Loading PTB (deploy suffix)...")
    ptb = load_ptb_data()
    pmap_ptb = build_ptb_patient_map()
    tr2, va2, te2, stats_ptb = patient_level_split(ptb["record_ids"], pmap_ptb)
    x_ptb  = ptb["beats"][te2]
    y_ptb  = ptb["labels"][te2]
    rid_ptb = ptb["record_ids"][te2]
    print(f"  PTB test: {len(y_ptb)} beats, {int(y_ptb.sum())} abnormal, "
          f"{len(np.unique(rid_ptb))} records, "
          f"{stats_ptb['n_test']} patients")

    # ── 2. Load models ───────────────────────────────────────────────────────
    print("\n[3/6] Loading models...")
    p2a = tf.keras.models.load_model(
        str(MODELS / "archived" / "final_resnet_l_p2a_backup.h5"), compile=False)
    kd = tf.keras.models.load_model(
        str(MODELS / "kd_a070_t1.h5"), compile=False)
    print("  P2A + KD a070_t1 loaded (CPU)")

    # ── 3. Predict ───────────────────────────────────────────────────────────
    print("\n[4/6] Predicting (CPU, batch_size=1024)...")
    p_p2a_mit = p2a.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    p_p2a_ptb = p2a.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]
    p_kd_mit  = kd.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    p_kd_ptb  = kd.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]
    print("  Predictions done.")
    del p2a, kd

    # ── 4. Pure baselines ────────────────────────────────────────────────────
    print("\n[5/6] Evaluating pure baselines + constructing mixed test sets...")

    all_results = {}
    all_info = {}

    # Pure MIT
    print("\n  --- Pure MIT baseline ---")
    res_mit = _evaluate_chains(y_mit, p_p2a_mit, p_kd_mit, rid_mit, "PureMIT")
    all_results.update(res_mit)
    all_info["PureMIT"] = {
        "n_beats": int(len(y_mit)), "n_abnormal": int(y_mit.sum()),
        "n_normal": int((y_mit == 0).sum()),
        "n_records": int(len(np.unique(rid_mit))),
    }

    # Pure PTB
    print("  --- Pure PTB baseline ---")
    res_ptb = _evaluate_chains(y_ptb, p_p2a_ptb, p_kd_ptb, rid_ptb, "PurePTB")
    all_results.update(res_ptb)
    all_info["PurePTB"] = {
        "n_beats": int(len(y_ptb)), "n_abnormal": int(y_ptb.sum()),
        "n_normal": int((y_ptb == 0).sum()),
        "n_records": int(len(np.unique(rid_ptb))),
    }

    # ── 5. M1: Record-level 60:40 mix ────────────────────────────────────────
    print("\n  --- M1: Record-level 60:40 mix ---")
    y_m1, r_m1, p2a_m1, kd_m1, info_m1 = _build_m1_record_mix(
        y_mit, y_ptb, rid_mit, rid_ptb,
        p_p2a_mit, p_p2a_ptb, p_kd_mit, p_kd_ptb,
        rng, target_mit_ratio=0.60,
    )
    info_m1["n_abnormal"] = info_m1["n_mit_abnormal"] + info_m1["n_ptb_abnormal"]
    all_info["M1"] = info_m1
    print(f"    MIT records: {info_m1['n_mit_recs_sampled']}/{info_m1['n_mit_recs_total']}, "
          f"PTB records: {info_m1['n_ptb_recs_sampled']}/{info_m1['n_ptb_recs_total']}")
    print(f"    Total beats: {info_m1['total_beats']} "
          f"(MIT {info_m1['n_mit_beats']}, PTB {info_m1['n_ptb_beats']}), "
          f"abnormal: {info_m1['n_mit_abnormal']+info_m1['n_ptb_abnormal']} "
          f"(MIT {info_m1['n_mit_abnormal']}, PTB {info_m1['n_ptb_abnormal']}), "
          f"normal: {info_m1['n_normal']}")
    print(f"    Record share: MIT {info_m1['mit_record_share']:.3f} / "
          f"PTB {info_m1['ptb_record_share']:.3f}")
    print(f"    Abnormal share (MIT of total abnormal): {info_m1['actual_abnormal_share_mit']:.3f}")
    res_m1 = _evaluate_chains(y_m1, p2a_m1, kd_m1, r_m1, "M1")
    all_results.update(res_m1)

    # ── 6. M2: Beat-level abnormal ratio 60:40 (normalfrac 0.75 + 0.85) ─────
    for nf in [0.75, 0.85]:
        label = f"M2_nf{int(nf*100):03d}"
        print(f"\n  --- M2: Beat-level 60:40 abnormal mix, normal_frac={nf} ---")
        y_m2, r_m2, p2a_m2, kd_m2, info_m2 = _build_m2_beat_mix(
            y_mit, y_ptb, rid_mit, rid_ptb,
            p_p2a_mit, p_p2a_ptb, p_kd_mit, p_kd_ptb,
            rng, target_mit_ab_ratio=0.60, normal_frac=nf,
        )
        all_info[label] = info_m2
        print(f"    MIT ab: {info_m2['n_mit_ab_sampled']}/{info_m2['n_mit_ab_available']}, "
              f"PTB ab: {info_m2['n_ptb_ab_sampled']}/{info_m2['n_ptb_ab_available']}")
        print(f"    Total beats: {info_m2['total_beats']} "
              f"(ab {info_m2['total_abnormal']}, normal {info_m2['total_normal']})")
        print(f"    Achieved abnormal ratio MIT: "
              f"{info_m2['actual_abnormal_ratio_mit']:.3f}, "
              f"normal_frac actual: {info_m2['normal_frac_actual']:.3f}")
        res_m2 = _evaluate_chains(y_m2, p2a_m2, kd_m2, r_m2, label)
        all_results.update(res_m2)

    # ── 7. Output ────────────────────────────────────────────────────────────
    print("\n[6/6] Saving results...")

    # JSON
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = _floatify(v)
    serializable["__info__"] = {
        k: _floatify(v) if isinstance(v, dict) else v
        for k, v in all_info.items()
    }
    serializable["__const__"] = {"seed": SEED, "gap": GAP, "deploy_chain_spec": {
        "p2a_theta": 0.50, "kd_theta": 0.35, "kd_N": 3, "kd_M": 5,
    }}

    out_path = MODELS / "mixed_testset_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # ── 8. Console table ─────────────────────────────────────────────────────
    test_sets = ["PureMIT", "PurePTB", "M1", "M2_nf075", "M2_nf085"]
    chain_short = ["P2A_θ050", "KD_θ050", "Dchain"]
    chain_keys  = ["P2A_θ050", "KD_θ050", "Dchain"]

    print("\n" + "=" * 130)
    print("REAL-WORLD-PROPORTION MIXED TEST SET — FULL RESULTS")
    print("=" * 130)

    # Baseline cross-check
    print("\n── Baseline cross-check (must match known values) ──")
    for prefix, label in [("PureMIT", "MIT"), ("PurePTB", "PTB")]:
        for ck, cs in zip(chain_keys, chain_short):
            r = all_results.get(f"{prefix}_{ck}", {})
            if ck == "P2A_θ050" and prefix == "PureMIT":
                print(f"  {label} P2A@0.50: R={r.get('R',0):.3f} P={r.get('P',0):.3f} "
                      f"误报={r.get('误报',0)*100:.1f}%  (expect R≈0.935 P≈0.389 误报≈21.6%)")
            elif ck == "KD_θ050" and prefix == "PurePTB":
                print(f"  {label} KD@0.50: R={r.get('R',0):.3f} P={r.get('P',0):.3f} "
                      f"误报={r.get('误报',0)*100:.1f}%  (expect R≈0.323 P≈0.997 误报≈0.4%)")

    # Main table header
    print(f"\n{'Test Set':<12} {'Chain':<14} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'误报%':>8} {'报警%':>8} | "
          f"{'evtPrec':>8} {'evtRec':>8} {'Events':>7} {'TrueEv':>7}")
    print("-" * 130)

    for ts in test_sets:
        info = all_info.get(ts, {})
        n_ab = info.get("n_abnormal", info.get("total_abnormal", "?"))
        n_beats = info.get("n_beats", info.get("total_beats", "?"))
        nf = info.get("normal_frac_actual", None)
        prefix = ts
        for ck, cs in zip(chain_keys, chain_short):
            r = all_results.get(f"{prefix}_{ck}", {})
            if not r:
                continue
            print(f"{ts:<12} {cs:<14} "
                  f"{r.get('P',0):7.4f} {r.get('R',0):7.4f} {r.get('F1',0):7.4f} "
                  f"{r.get('误报',0)*100:8.2f} {r.get('报警',0)*100:8.2f} | "
                  f"{r.get('evt_prec',0):8.4f} {r.get('evt_rec',0):8.4f} "
                  f"{r.get('total_events',0):>5}  {r.get('true_events',0):>5}")
        # Summary line for test set
        if nf is not None:
            n_norm_d = info.get("total_normal", "?")
            print(f"  [{ts}: {n_beats} beats, {n_ab} ab, normal_frac={nf:.3f}, "
                  f"{n_norm_d} normal]")
        else:
            n_norm = info.get("n_normal", info.get("total_normal", "?"))
            pct = n_ab / max(1, n_beats) if isinstance(n_beats, int) and isinstance(n_ab, int) else 0
            if isinstance(n_ab, int):
                pct = n_ab / max(1, n_beats) if isinstance(n_beats, int) else 0
            else:
                pct = 0.0
            print(f"  [{ts}: {n_beats} beats, {n_ab} ab ({pct*100:.1f}%), "
                  f"{n_norm} normal]")

    # Record structure note for M2
    print("\n  Note: M2 is beat-level sampling — event metrics are approximate; "
          "record IDs retained for N-of-M per-record filtering.")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
