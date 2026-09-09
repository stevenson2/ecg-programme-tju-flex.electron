#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sim_temporal_agg.py — N-of-M temporal aggregation simulation (zero-training).

Quantifies how much false-alarm rate drops when applying multi-beat confirmation
filtering on top of existing single-beat model probabilities.  Uses the exact
patient-level test splits from eval_binary_all.py for direct comparability.

Models: P2A = archived/final_resnet_l_p2a_backup.h5 (MIT expert)
        KD a070_t1 = kd_a070_t1.h5 (PTB expert)

Algorithm:
  - Group test beats by record_id (already in chronological order).
  - Slide a window of M beats (stride 1); window "triggered" if >=N beats >=θ.
  - A beat is "post-filter alarm" iff it belongs to >=1 triggered window.
  - Event-level: cluster alarm beats within each record (GAP=3 non-alarm beats
    between events).  True event = contains >=1 abnormal beat.
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
GAP = 3  # beats between alarm clusters for event segmentation

# ─── helpers ─────────────────────────────────────────────────────────────────

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
    """Event-level precision & recall (GAP=3 non-alarm beats ends an event).

    Event precision = true_events / total_events  (across all records).
    Event recall    = # records_with_abn_beats_that_got_true_event
                      / # records_with_any_abnormal_beat.
    """
    unique_rids = np.unique(record_ids)
    all_events = []          # list of lists of *global* beat indices
    rec_has_abnormal = {}    # rid -> bool
    rec_has_true_event = {}  # rid -> bool

    for rid in unique_rids:
        mask = record_ids == rid
        global_idx = np.where(mask)[0]
        n_beats = len(global_idx)
        rec_has_abnormal[int(rid)] = (y[mask] == 1).any()
        rec_has_true_event[int(rid)] = False

        rec_alarm = alarm[mask]
        alarm_local = np.where(rec_alarm)[0]
        if len(alarm_local) == 0:
            continue

        # Cluster alarm beats within this record (GAP = gap non-alarm beats)
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

    Returns bool array of length L: True = post-filter alarm.
    Beats not covered by any full M-window remain non-alarm.
    """
    L = len(prob)
    if L < M:                     # no full window exists
        return np.zeros(L, dtype=bool)

    trig = (prob >= theta).astype(np.int32)
    cum = np.zeros(L + 1, dtype=np.int32)
    cum[1:] = np.cumsum(trig)
    window_sums = cum[M:] - cum[:L - M + 1]     # length L-M+1
    window_trig = (window_sums >= N)             # bool, length L-M+1

    # Beat i is alarm if any covering window triggered.
    # Covering windows for beat i: [max(0,i-M+1) … min(L-M,i)]
    alarm = np.zeros(L, dtype=bool)
    for i in range(L):
        w_start = max(0, i - M + 1)
        w_end = min(L - M, i) + 1            # exclusive
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

# ─── main ────────────────────────────────────────────────────────────────────

def main():
    set_npz_suffix("_deploy")
    print("=" * 90)
    print("N-of-M Temporal Aggregation Simulation (Zero-Training)")
    print("=" * 90)

    # ---- 1. Load data (exact copies from eval_binary_all.py) ------------------
    print("\n[1/5] Loading MIT+INCART merged (deploy suffix)...")
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

    print("\n[2/5] Loading PTB (deploy suffix)...")
    ptb = load_ptb_data()
    pmap_ptb = build_ptb_patient_map()
    tr2, va2, te2, stats_ptb = patient_level_split(ptb["record_ids"], pmap_ptb)
    x_ptb  = ptb["beats"][te2]
    y_ptb  = ptb["labels"][te2]
    rid_ptb = ptb["record_ids"][te2]
    print(f"  PTB test: {len(y_ptb)} beats, {int(y_ptb.sum())} abnormal, "
          f"{len(np.unique(rid_ptb))} records, "
          f"{stats_ptb['n_test']} patients")

    # ---- 2. Load models -------------------------------------------------------
    print("\n[3/5] Loading models...")
    p2a = tf.keras.models.load_model(
        str(MODELS / "archived" / "final_resnet_l_p2a_backup.h5"), compile=False)
    kd  = tf.keras.models.load_model(
        str(MODELS / "kd_a070_t1.h5"), compile=False)
    print("  P2A + KD a070_t1 loaded (CPU)")

    # ---- 3. Predict -----------------------------------------------------------
    print("\n[4/5] Predicting (CPU, batch_size=1024)...")
    p_p2a_mit = p2a.predict(add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    p_p2a_ptb = p2a.predict(add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]
    p_kd_mit  = kd.predict( add_channel_dim(x_mit), verbose=0, batch_size=1024)[:, 1]
    p_kd_ptb  = kd.predict( add_channel_dim(x_ptb), verbose=0, batch_size=1024)[:, 1]
    print("  Predictions done.")
    del p2a, kd

    # ---- 4. Record contiguity check ------------------------------------------
    print("\n[5/5] Record contiguity check + grid sweep...")
    for label, rids in [("MIT+INCART", rid_mit), ("PTB", rid_ptb)]:
        n_switches = int((np.diff(rids) != 0).sum())
        n_recs = len(np.unique(rids))
        ok = "OK" if n_switches == n_recs - 1 else "NON-CONTIGUOUS"
        print(f"  {label}: {n_recs} records, {n_switches} switches → {ok}")

    # ---- 5. N-of-M grid -------------------------------------------------------
    thetas = [0.30, 0.35, 0.50]
    nofm_grid = [(2, 3), (3, 5), (3, 7), (4, 7), (5, 10)]

    # (model_name, domain_name, probs, labels, record_ids)
    configs = [
        ("P2A",        "MIT", p_p2a_mit, y_mit, rid_mit),
        ("P2A",        "PTB", p_p2a_ptb, y_ptb, rid_ptb),
        ("KD_a070_t1", "MIT", p_kd_mit,  y_mit, rid_mit),
        ("KD_a070_t1", "PTB", p_kd_ptb,  y_ptb, rid_ptb),
    ]

    results = {}

    for model_name, domain, probs, y, rids in configs:
        unique_rids = np.unique(rids)
        n_recs = len(unique_rids)
        n_ab_beats = int(y.sum())

        print(f"\n{'=' * 80}")
        print(f"  {model_name}  on  {domain}  "
              f"({n_recs} records, {len(y)} beats, {n_ab_beats} abnormal)")
        print(f"  {'Config':<14} {'P':>6} {'R':>6} {'F1':>6} "
              f"{'误报%':>7} {'报警%':>7} | "
              f"{'evtPrec':>7} {'evtRec':>7} {'Events':>7}")
        print(f"  {'-' * 89}")

        prefix = f"{model_name}_{domain}"

        for theta in thetas:
            # --- baseline (no filter) ---
            bl_alarm = (probs >= theta)
            bl_bm = _beat_metrics(y, bl_alarm)
            bl_em = _event_metrics(y, bl_alarm, rids, gap=GAP)
            bl_key = f"{prefix}_θ{theta}_baseline"
            results[bl_key] = {**bl_bm, **bl_em,
                               "theta": theta, "N": None, "M": None,
                               "type": "baseline"}

            print(f"  θ={theta:.2f} "
                  f"{'baseline':<10} "
                  f"{bl_bm['P']:6.3f} {bl_bm['R']:6.3f} {bl_bm['F1']:6.3f} "
                  f"{bl_bm['误报']*100:7.2f} {bl_bm['报警']*100:7.2f} | "
                  f"{bl_em['evt_prec']:7.4f} {bl_em['evt_rec']:7.4f} "
                  f"{bl_em['total_events']:>5}")

            for N, M in nofm_grid:
                # Apply N-of-M per record
                alarm = np.zeros(len(probs), dtype=bool)
                for rid in unique_rids:
                    mask = rids == rid
                    rec_probs = probs[mask]
                    if len(rec_probs) < N:
                        # Not enough beats to confirm anything
                        continue
                    rec_alarm = _apply_nofm(rec_probs, theta, N, M)
                    alarm[mask] = rec_alarm

                bm = _beat_metrics(y, alarm)
                em = _event_metrics(y, alarm, rids, gap=GAP)
                key = f"{prefix}_θ{theta}_N{N}_M{M}"
                results[key] = {**bm, **em,
                                "theta": theta, "N": N, "M": M,
                                "type": "nofm"}

                delta_fp = (bl_bm['误报'] - bm['误报']) * 100
                delta_r  = (bl_bm['R'] - bm['R'])
                print(f"         "
                      f"({N},{M}){'':<6}"
                      f"{bm['P']:6.3f} {bm['R']:6.3f} {bm['F1']:6.3f} "
                      f"{bm['误报']*100:7.2f} {bm['报警']*100:7.2f} | "
                      f"{em['evt_prec']:7.4f} {em['evt_rec']:7.4f} "
                      f"{em['total_events']:>5}"
                      f"  (ΔFP{delta_fp:+.1f}pp ΔR{delta_r:+.3f})")

    # ---- 6. Save JSON ---------------------------------------------------------
    serializable = {}
    for k, v in results.items():
        serializable[k] = _floatify(v)

    out_path = MODELS / "temporal_agg_sim.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n\n[Save] {out_path}")

    # ---- 7. Analysis summary --------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY ANALYSIS")
    print("=" * 90)

    # -- 7a. P2A on MIT: false-alarm reduction --
    print("\n┌─ [1] P2A on MIT — False-alarm reduction from 21.6% baseline (θ=0.50)")
    for theta in thetas:
        bl = results[f"P2A_MIT_θ{theta}_baseline"]
        print(f"│ θ={theta:.2f} baseline: 误报={bl['误报']*100:.2f}%  "
              f"R={bl['R']:.4f}  evtPrec={bl['evt_prec']:.4f}")
        for N, M in nofm_grid:
            r = results[f"P2A_MIT_θ{theta}_N{N}_M{M}"]
            dfp = (bl['误报'] - r['误报']) * 100
            dr  = (bl['R'] - r['R'])
            dep = r['evt_prec'] - bl['evt_prec']
            print(f"│   ({N},{M}): 误报={r['误报']*100:.2f}% (Δ{dfp:+.1f}pp)  "
                  f"R={r['R']:.4f} (Δ{dr:+.3f})  "
                  f"evtPrec={r['evt_prec']:.4f} (Δ{dep:+.4f})")

    # -- 7b. KD a070_t1 on PTB: low-θ temporal confirmation --
    print("\n┌─ [2] KD a070_t1 on PTB — Low-θ temporal confirmation (R baseline @0.50 = 0.323)")
    for theta in [0.30, 0.35, 0.50]:
        bl = results[f"KD_a070_t1_PTB_θ{theta}_baseline"]
        print(f"│ θ={theta:.2f} baseline: R={bl['R']:.4f}  "
              f"误报={bl['误报']*100:.2f}%  evtPrec={bl['evt_prec']:.4f}")
        for N, M in nofm_grid:
            r = results[f"KD_a070_t1_PTB_θ{theta}_N{N}_M{M}"]
            dfp = (bl['误报'] - r['误报']) * 100
            dr  = (bl['R'] - r['R'])
            dep = r['evt_prec'] - bl['evt_prec']
            print(f"│   ({N},{M}): R={r['R']:.4f} (Δ{dr:+.3f})  "
                  f"误报={r['误报']*100:.2f}% (Δ{dfp:+.1f}pp)  "
                  f"evtPrec={r['evt_prec']:.4f} (Δ{dep:+.4f})")

    # -- 7c. Cross-domain reference --
    print("\n┌─ [3] Cross-domain reference — P2A on PTB, KD on MIT")
    for model, dom, th_val in [("P2A", "PTB", 0.5), ("KD_a070_t1", "MIT", 0.5)]:
        bl = results[f"{model}_{dom}_θ{th_val}_baseline"]
        print(f"│ {model} on {dom} @θ={th_val} baseline: R={bl['R']:.4f}  "
              f"误报={bl['误报']*100:.2f}%  evtPrec={bl['evt_prec']:.4f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
