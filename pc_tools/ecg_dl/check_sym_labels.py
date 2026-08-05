#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 sym_te (AAMI符号) 与 labels (二分类) 的一致性"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import set_npz_suffix, load_mit_incart_merged
from data.patient_split import build_mit_patient_map, build_incart_patient_map, patient_level_split
from eval_aami_breakdown import recover_mit_symbols_per_record, align_symbols_to_npz

set_npz_suffix("_deploy")
mi = load_mit_incart_merged()
per = recover_mit_symbols_per_record()
sym, nuk = align_symbols_to_npz(per, mi["record_ids"], 6)
pmap = {}
pmap.update(build_mit_patient_map())
pmap.update({r + 100000: "inc_" + p for r, p in build_incart_patient_map().items()})
tr, va, te, _ = patient_level_split(mi["record_ids"], pmap)
sym_te = sym[te]
y_te = mi["labels"][te]

for cls in ["N", "S", "V", "F", "Q"]:
    m = sym_te == cls
    n = int(m.sum())
    n1 = int((m & (y_te == 1)).sum())
    print(f"{cls}: n={n}, label=1(异常)={n1}, label=0(正常)={n - n1}")
print(f"U(未知): n={(sym_te=='U').sum()}")
