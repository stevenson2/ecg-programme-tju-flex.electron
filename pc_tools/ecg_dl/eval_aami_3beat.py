#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_aami_3beat.py — AAMI-class recall breakdown for the 3-beat (750pt) model.

Mirrors eval_aami_breakdown.py but:
  - data source: mit_incart_3beat_deploy.npz (3-beat sequences, 750pt)
  - symbol alignment: center-beat symbol of each triple (label = center beat);
    symbols recovered from .atr per record, tiled 6x for MIT augmentation,
    then stitch window [i-1, i, i+1] -> symbol of index i.
  - both beat-level (all sequences) and patient-level modes.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROCESSED_DIR, MIT_BIH_LOCAL_DIR, MIT_BIH_RECORDS,
                    AAMI_CLASSES)
from data.dataset import set_npz_suffix, add_channel_dim
from data.patient_split import (build_mit_patient_map, build_incart_patient_map,
                                patient_level_split)
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

SYM2AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V',
    'F': 'F', 'f': 'F',
    '!': 'V',
    '/': 'Q', '?': 'Q',
}
VALID_SYMS = set(AAMI_CLASSES.keys())


def recover_mit_symbols_per_record() -> dict:
    out = {}
    for rec in MIT_BIH_RECORDS:
        rec_str = str(rec)
        ann = wfdb.rdann(str(MIT_BIH_LOCAL_DIR / rec_str), "atr")
        syms = [SYM2AAMI.get(s, 'Q') for s in ann.symbol if s in VALID_SYMS]
        out[rec] = np.array(syms, dtype=object)
    return out


def recover_incart_symbols_per_record(incart_dir) -> dict:
    out = {}
    for rid in range(1, 76):
        rec_name = f"I{rid:02d}"
        try:
            ann = wfdb.rdann(str(incart_dir / rec_name), "atr")
        except Exception:
            continue
        syms = [SYM2AAMI.get(s, 'Q') for s in ann.symbol if s in VALID_SYMS]
        out[100000 + rid] = np.array(syms, dtype=object)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_cnn_m_large.h5")
    parser.add_argument("--tag", default="exp6_3beat")
    parser.add_argument("--beat-level", action="store_true")
    args = parser.parse_args()

    model_path = Path(__file__).resolve().parent / "models" / args.model
    if not model_path.exists():
        print(f"MODEL MISSING: {model_path}")
        return 1

    # load 3-beat deploy data
    set_npz_suffix("_deploy")
    from data.dataset import load_3beat_merged
    data = load_3beat_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    print(f"3-beat data: {len(beats)} sequences ({beats.shape[1]}pt)")

    # recover symbols: single-beat per-record -> tile 6x (MIT aug) -> stitch
    print("Recovering AAMI symbols ...")
    per_rec = recover_mit_symbols_per_record()
    incart_dir = Path(__file__).resolve().parent / "data" / "raw" / "incart"
    per_rec.update(recover_incart_symbols_per_record(incart_dir))

    # Build single-beat symbol array aligned to single-beat npz order
    single_rids = np.concatenate([
        np.load(PROCESSED_DIR / "mit_bih_processed_deploy.npz")["record_ids"],
        np.load(PROCESSED_DIR / "incart_processed_deploy.npz")["record_ids"] + 100000,
    ]).astype(int)
    single_syms = np.empty(len(single_rids), dtype=object)
    idx = 0
    for rid in np.unique(single_rids):
        mask = single_rids == rid
        n = int(mask.sum())
        syms = per_rec.get(int(rid))
        if rid < 100000:
            n_raw = len(syms)
            if n_raw * 6 != n:
                print(f"  !! MIT {rid}: {n_raw}x6 != {n}")
                return 1
            single_syms[idx:idx + n] = np.tile(syms, 6)
        else:
            if syms is not None and len(syms) == n:
                single_syms[idx:idx + n] = syms
            else:
                single_syms[idx:idx + n] = 'U'
        idx += n
    print(f"  single-beat symbols aligned: {len(single_syms)} == {len(single_rids)}")

    # stitch symbols to 3-beat center (mirror stitch_3beat semantics)
    # stitch_3beat: per record with k single beats, produces k-2 sequences;
    # sequence j (0-based) uses single beats [j, j+1, j+2], center = beat j+1.
    center_syms = np.empty(len(rids), dtype=object)
    ci = 0
    for rec in np.unique(rids):
        m = np.where(rids == rec)[0]
        sm = np.where(single_rids == rec)[0]
        k = len(sm)
        # sequences for this record occupy m[0..len(m)-1]; center of seq j = sm[j+1]
        for j, seq_pos in enumerate(m):
            if j + 1 >= k:
                print(f"  !! record {rec}: sequence index {j} beyond beats {k}")
                return 1
            center_syms[ci] = single_syms[sm[j + 1]]
            ci += 1
    print(f"  center symbols: {len(center_syms)} == {len(rids)}  "
          f"(U={int((center_syms=='U').sum())})")

    # split mode
    if args.beat_level:
        te = np.ones(len(beats), dtype=bool)
        split_desc = "beat-level (all sequences)"
    else:
        pmap = {}
        pmap.update(build_mit_patient_map())
        pmap.update({rid + 100000: "inc_" + pat
                     for rid, pat in build_incart_patient_map().items()})
        tr, va, te, stats = patient_level_split(rids, pmap)
        split_desc = "patient-level 60/20/20 seed42"

    x_te, y_te = beats[te], labels[te]
    sym_te = center_syms[te]
    print(f"  test: {len(x_te)} sequences")

    m = tf.keras.models.load_model(str(model_path), compile=False)
    xi = add_channel_dim(x_te)
    prob_raw = m.predict(xi, verbose=0, batch_size=256)
    if isinstance(prob_raw, (list, tuple)):
        prob_raw = prob_raw[0]
    prob = prob_raw[:, 1]

    classes = ['N', 'S', 'V', 'F', 'Q']
    thresholds = [0.35, 0.5, 0.65]
    print("\n" + "=" * 78)
    print(f"AAMI-class breakdown ({split_desc}) — 3-beat 750pt")
    print("=" * 78)
    print(f"{'AAMI':<6}{'n':>8}{'abn%':>7}" + "".join(
        f"  R@t{t:.2f}  P@t{t:.2f}" for t in thresholds))
    per_class = {}
    known = sym_te != 'U'
    for c in classes:
        mask_c = (sym_te == c) & known
        n_c = int(mask_c.sum())
        if n_c == 0:
            continue
        y_c = y_te[mask_c]
        p_c = prob[mask_c]
        row = f"{c:<6}{n_c:>8}{y_c.sum()/n_c*100:>7.1f}"
        vals = {}
        for t in thresholds:
            pred = (p_c >= t).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_c, pred, average="binary", zero_division=0)
            row += f"  {rec:>7.3f}  {prec:>7.3f}"
            vals[f"thr_{t}"] = {"recall": float(rec), "precision": float(prec),
                                "f1": float(f1)}
        print(row)
        per_class[c] = {"n": n_c, "n_abn": int(y_c.sum()), "thr": vals}
        if len(np.unique(y_c)) > 1:
            per_class[c]["auc"] = float(roc_auc_score(y_c, p_c))
    print("-" * 78)
    row = f"{'ALL':<6}{len(y_te):>8}{y_te.sum()/len(y_te)*100:>7.1f}"
    agg = {}
    for t in thresholds:
        pred = (prob >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="binary", zero_division=0)
        row += f"  {rec:>7.3f}  {prec:>7.3f}"
        agg[f"thr_{t}"] = {"recall": float(rec), "precision": float(prec),
                           "f1": float(f1)}
    print(row)
    print(f"\nOverall AUC: {roc_auc_score(y_te, prob):.4f}")

    out = {"meta": {"model": model_path.name, "tag": args.tag, "split": split_desc,
                    "n_test": int(len(y_te))},
           "per_class": per_class, "aggregate": agg,
           "auc": float(roc_auc_score(y_te, prob))}
    out_path = Path(__file__).resolve().parent / "models" / f"aami_breakdown_{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
