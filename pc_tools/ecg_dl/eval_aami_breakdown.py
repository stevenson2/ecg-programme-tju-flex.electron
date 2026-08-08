#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AAMI-category recall breakdown for the patient-split evaluation.

Goal: show that aggregate beat-level recall (0.81-0.85 plateau) hides a big
disparity across AAMI superclasses:
  N(normal) / S(SVEB) / V(VEB) / F(fusion) / Q(other: paced, unclassifiable)

Method:
  - Recover per-beat AAMI symbol by re-reading raw .atr annotations in the
    SAME order preprocess.py extracts beats (per record, annotation order,
    with the same window-coverage keep rule), then re-apply the 6x augmentation
    block-expansion so symbols align 1:1 with the *_deploy.npz arrays.
  - Reuse the patient-level test masks from data.patient_split (same seed 42)
    to select test beats; report per-class recall/precision for a few thresholds.

Aligning augmented beats: augment_data() concatenates in order
  [raw] + [noise_std ...] + [scale ...] + [drift ...] with 1:1 label copies,
  so augmented symbol array = np.tile(symbols, n_aug_blocks) when each block is
  full-length (raw no-aug first block + 5 augmented blocks = 6x).
  BUT: noise uses randn (random), scale/drift are deterministic. All blocks keep
  the same ordering, so tile(symbols, 6) is exact for the 6x case.

Usage: python3 eval_aami_breakdown.py --model best_resnet_large_exp6_deploy.h5 \
       --tag exp6_deploy [--deploy-suffix]
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
from sklearn.metrics import precision_recall_fscore_support

# ---- AAMI symbol mapping — use config.AAMI_CLASSES as the single source of truth ----
# AAMI superclass per symbol (config only gives binary 0/1; we add the 5-way split)
SYM2AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',      # normal (label 0)
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',                 # SVEB
    'V': 'V',                                               # VEB
    'F': 'F', 'f': 'F',                                     # fusion
    '!': 'V',                                               # ventricular flutter -> VEB group
    '/': 'Q', '?': 'Q',                                     # paced / unclassifiable
}
# symbols that participate in training (== AAMI_CLASSES keys)
VALID_SYMS = set(AAMI_CLASSES.keys())


def recover_mit_symbols_per_record() -> dict:
    """Per-record raw AAMI symbol arrays from .atr (annotation order).

    Returns {record_id(int): np.ndarray of AAMI superclass symbols}.
    Only symbols in VALID_SYMS are kept (mirrors _aami_r_idx filter step).
    """
    out = {}
    for rec in MIT_BIH_RECORDS:
        rec_str = str(rec)
        ann = wfdb.rdann(str(MIT_BIH_LOCAL_DIR / rec_str), "atr")
        syms = [SYM2AAMI.get(s, 'Q') for s in ann.symbol if s in VALID_SYMS]
        out[rec] = np.array(syms, dtype=object)
    return out


def recover_incart_symbols_per_record(incart_dir) -> dict:
    """Per-record raw AAMI symbol arrays for INCART (I01..I75)."""
    out = {}
    for rid in range(1, 76):
        rec_name = f"I{rid:02d}"
        try:
            ann = wfdb.rdann(str(incart_dir / rec_name), "atr")
        except Exception as e:
            print(f"  INCART {rec_name}: ERR {e}")
            continue
        syms = [SYM2AAMI.get(s, 'Q') for s in ann.symbol if s in VALID_SYMS]
        out[100000 + rid] = np.array(syms, dtype=object)
    return out


def align_symbols_to_npz(per_rec_syms: dict, record_ids: np.ndarray,
                         n_aug_mit: int) -> np.ndarray:
    """Build per-beat symbol array matching npz order, per record.

    MIT beats are 6x augmented (block-tiled) -> each raw symbol repeats
    n_aug_mit times. INCART beats are raw-only but .atr coverage is incomplete
    (7/75 records available); INCART beats get symbol 'U' (unknown) so they
    are excluded from the per-class table but still counted in aggregate.
    """
    out = np.empty(len(record_ids), dtype=object)
    idx = 0
    n_incart_unknown = 0
    for rid in np.unique(record_ids):
        mask = record_ids == rid
        n = int(mask.sum())
        syms = per_rec_syms.get(int(rid))
        if rid < 100000:  # MIT — must align exactly
            if syms is None:
                print(f"  !! MIT record {rid} symbols missing")
                return None, -1
            n_raw = len(syms)
            if n_raw * n_aug_mit != n:
                print(f"  !! MIT record {rid}: {n_raw} syms x{n_aug_mit} = "
                      f"{n_raw*n_aug_mit} != npz {n} beats")
                return None, -1
            out[idx:idx + n] = np.tile(syms, n_aug_mit)
        else:  # INCART
            if syms is not None and len(syms) == n:
                out[idx:idx + n] = syms
            else:
                out[idx:idx + n] = 'U'  # unknown -> excluded from per-class
                n_incart_unknown += n
        idx += n
    return out, n_incart_unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_resnet_large_exp6_deploy.h5")
    parser.add_argument("--tag", default="exp6_deploy")
    parser.add_argument("--deploy-suffix", default="_deploy",
                        help="npz suffix: '' for filtfilt-era, '_deploy' for deploy chain")
    parser.add_argument("--no-expand", action="store_true",
                        help="if MIT npz is raw-only (no augmentation)")
    parser.add_argument("--beat-level", action="store_true",
                        help="evaluate on ALL beats (beat-level, no patient split)")
    args = parser.parse_args()

    model_path = Path(__file__).resolve().parent / "models" / args.model
    if not model_path.exists():
        print(f"MODEL MISSING: {model_path}")
        return 1

    if args.deploy_suffix:
        set_npz_suffix(args.deploy_suffix)

    # ---- Load merged data (MIT deploy has 6x augmentation) ----
    from data.dataset import load_mit_incart_merged
    mit_inc = load_mit_incart_merged()
    beats, labels, rids = mit_inc["beats"], mit_inc["labels"], mit_inc["record_ids"]

    # ---- Recover MIT symbols per record and align ----
    print("Recovering MIT-BIH AAMI symbols from .atr (per record) ...")
    per_rec_syms = recover_mit_symbols_per_record()
    print("Recovering INCART AAMI symbols from .atr (per record) ...")
    incart_dir = Path(__file__).resolve().parent / "data" / "raw" / "incart"
    per_rec_syms.update(recover_incart_symbols_per_record(incart_dir))
    n_aug = 1 if args.no_expand else 6
    sym_full, n_incart_unknown = align_symbols_to_npz(per_rec_syms, rids, n_aug)
    if sym_full is None:
        print("  SYMBOL ALIGNMENT FAILED — aborting (do not trust per-class numbers)")
        return 1
    print(f"  aligned symbols: {len(sym_full)} == beats {len(beats)}  OK "
          f"(INCART unknown: {n_incart_unknown})")

    # ---- Patient-level test mask (seed 42, same as patient_split_eval.json) ----
    if args.beat_level:
        te = np.ones(len(beats), dtype=bool)
        print("  beat-level mode: using ALL beats")
    else:
        pmap = {}
        pmap.update(build_mit_patient_map())
        pmap.update({rid + 100000: "inc_" + pat
                     for rid, pat in build_incart_patient_map().items()})
        tr, va, te, stats = patient_level_split(rids, pmap)
        print(f"  patient test mask: {te.sum()} beats")

    x_te, y_te = beats[te], labels[te]
    sym_te = sym_full[te]

    # ---- Load model & predict ----
    print(f"Loading model: {model_path.name}")
    m = tf.keras.models.load_model(str(model_path), compile=False)
    xi = add_channel_dim(x_te)
    prob_raw = m.predict(xi, verbose=0, batch_size=512)
    if isinstance(prob_raw, (list, tuple)):
        prob_raw = prob_raw[0]
    prob = prob_raw[:, 1]

    # ---- Per-AAMI-class breakdown ----
    classes = ['N', 'S', 'V', 'F', 'Q']
    thresholds = [0.35, 0.5, 0.65]
    out = {}
    print("\n" + "=" * 78)
    print("AAMI-class breakdown (patient-level test, MIT+INCART)")
    print("=" * 78)
    hdr = f"{'AAMI':<6}{'n':>8}{'abn%':>7}" + "".join(
        f"  R@t{t:.2f}  P@t{t:.2f}" for t in thresholds)
    print(hdr)
    total_abn = 0
    per_class = {}
    # exclude 'U' (INCART unknown-symbol beats) from per-class table
    known = sym_te != 'U'
    for c in classes:
        mask_c = (sym_te == c) & known
        n_c = int(mask_c.sum())
        if n_c == 0:
            continue
        y_c = y_te[mask_c]
        p_c = prob[mask_c]
        n_abn_c = int(y_c.sum())
        total_abn += n_abn_c
        row = f"{c:<6}{n_c:>8}{n_abn_c/max(n_c,1)*100:>7.1f}"
        vals = {}
        for t in thresholds:
            pred = (p_c >= t).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_c, pred, average="binary", zero_division=0)
            # ⚠️ 2026-08-06 (TH §三十): 类内精确率在异常类内无负样本时为恒等式 1.000,
            # 无统计意义 (正常类误报不计入) —— 仅保留 recall 供逐类报告, precision 不作数。
            row += f"  {rec:>7.3f}  {prec:>7.3f}"
            vals[f"thr_{t}"] = {"recall": float(rec), "precision": float(prec),
                                "f1": float(f1)}
        print(row)
        per_class[c] = {"n": n_c, "n_abn": n_abn_c,
                        "auc": None, "thr": vals}
        if len(np.unique(y_c)) > 1:
            from sklearn.metrics import roc_auc_score
            per_class[c]["auc"] = float(roc_auc_score(y_c, p_c))

    # Aggregate (all abnormal beats regardless of class) — 全局精确率 (有效口径)
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
    print(row + "   <- 全局 P 在此列 (有效口径)")

    split_desc = ("beat-level (all beats, no split)" if args.beat_level
                  else "patient-level 60/20/20 seed42 (MIT+INCART)")
    out = {
        "meta": {"model": model_path.name, "tag": args.tag,
                 "split": split_desc,
                 "n_test": int(len(y_te)),
                 "n_incart_unknown": int(n_incart_unknown),
                 "note": ("per-class table covers beats with recovered AAMI "
                          "symbols (MIT full + INCART I01,I10-I15); INCART beats "
                          "without .atr marked 'U' and excluded from per-class "
                          "but included in aggregate")},
        "per_class": per_class,
        "aggregate_recall": {
            f"thr_{t}": {
                "recall": float(precision_recall_fscore_support(
                    y_te, (prob >= t).astype(int), average="binary",
                    zero_division=0)[1])
            } for t in thresholds},
        "aggregate_precision": {
            f"thr_{t}": {
                "precision": float(precision_recall_fscore_support(
                    y_te, (prob >= t).astype(int), average="binary",
                    zero_division=0)[0])
            } for t in thresholds},   # 2026-08-06 (TH §三十): 全局精确率 (有效口径)
    }
    out_path = Path(__file__).resolve().parent / "models" / f"aami_breakdown_{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
