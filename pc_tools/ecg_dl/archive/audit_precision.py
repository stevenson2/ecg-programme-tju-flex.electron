#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean Precision audit for exp6_deploy model.
Answers:
  A) MIT-only (known AAMI symbols) patient-test Precision/R recall
  B) Aggregate Precision excluding INCART 'U' beats
  C) N-class false-alarm rate (how many normal beats predicted abnormal)
  D) Whether INCART 'U' beats pollute the aggregate precision
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import tensorflow as tf

_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpus[0], True)
    except Exception:
        pass

from data.dataset import set_npz_suffix, load_mit_incart_merged, add_channel_dim
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_aami_breakdown import (recover_mit_symbols_per_record,
                                 recover_incart_symbols_per_record,
                                 align_symbols_to_npz)
from sklearn.metrics import precision_recall_fscore_support

def main():
    set_npz_suffix("_deploy")
    mit_inc = load_mit_incart_merged()
    beats, labels, rids = mit_inc["beats"], mit_inc["labels"], mit_inc["record_ids"]

    per_rec_syms = recover_mit_symbols_per_record()
    incart_dir = Path(__file__).resolve().parent / "data" / "raw" / "incart"
    per_rec_syms.update(recover_incart_symbols_per_record(incart_dir))
    sym_full, n_incart_unknown = align_symbols_to_npz(per_rec_syms, rids, 6)

    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat for rid, pat in build_incart_patient_map().items()})
    tr, va, te, stats = patient_level_split(rids, pmap)
    print(f"patient test: {te.sum()} beats, INCART-unknown in test: "
          f"{(sym_full[te] == 'U').sum()}")

    x_te, y_te = beats[te], labels[te]
    sym_te = sym_full[te]

    m = tf.keras.models.load_model("models/best_resnet_large_exp6_deploy.h5", compile=False)
    xi = add_channel_dim(x_te)
    prob_raw = m.predict(xi, verbose=0, batch_size=512)
    prob = prob_raw[:, 1]

    def pr(y, p, t):
        pred = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)
        return prec, rec, f1, int(pred.sum())

    for t in [0.35, 0.5, 0.65]:
        print(f"\n{'='*70}\nθ={t}\n{'='*70}")
        # A) MIT-only (known symbols: N/S/V/F/Q all from .atr)
        known = sym_te != 'U'
        # B) Aggregate known-only
        prec_k, rec_k, f1_k, npos_k = pr(y_te[known], prob[known], t)
        print(f"[B] 已知标签(排除U): P={prec_k:.4f} R={rec_k:.4f} F1={f1_k:.4f} "
              f"预测异常数={npos_k}/{known.sum()}")
        # C) N-class false alarm rate
        n_mask = sym_te == 'N'
        fn = (prob[n_mask] >= t).sum()
        print(f"[C] N类(正常)误报: {fn}/{n_mask.sum()} = {fn/n_mask.sum()*100:.2f}%")
        # D) full aggregate incl U (as script does)
        prec_all, rec_all, f1_all, npos_all = pr(y_te, prob, t)
        print(f"[D] 全量(含U, 原脚本口径): P={prec_all:.4f} R={rec_all:.4f} "
              f"预测异常数={npos_all}/{len(y_te)}")
        # U beats: what labels & predictions
        u_mask = sym_te == 'U'
        if u_mask.sum() > 0:
            u_pos = (prob[u_mask] >= t).sum()
            u_abn_label = y_te[u_mask].sum()
            print(f"[U] INCART未知拍: n={u_mask.sum()}, 预测异常={u_pos}, "
                  f"标签异常={u_abn_label}, 预测异常占比={u_pos/u_mask.sum()*100:.1f}%")

if __name__ == "__main__":
    main()
