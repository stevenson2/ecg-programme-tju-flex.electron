#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p3b_student_readout.py — P3b: 学生 fc1 线性读出对照 (与 P3 教师同协议)
================================================================================
P3 判 INCONCLUSIVE 后的对照实验 (2026-09-01 用户授权):
  教师 1024 维特征留一记录线性读出 AUC=0.7345 (< GO 线 0.80)。
  缺一个苹果对苹果的比较: 学生 fc1 (Dense 64) 特征在同一批 16000 拍、
  同一协议 (L2 归一化 + 留一记录 LogReg C=1.0) 下的线性读出是多少?
    * 学生 ≈ 0.73 → 教师无增量线性信息 → 蒸馏有对照依据判 KILL
    * 学生 ≪ 0.73 → 教师确有增量, 但仍低于 GO 线, 权衡工程成本再定

复现保证:
  - 分组/抽样与 P3 完全一致 (同种子 0、同 rng 消耗顺序), 拍群 = P3 同批
  - selfcheck 与 P1/P2/P3 同一 npz AUC 自证
  - 线性读出代码逐行镜像 P3

输出: models/deploy_match/p3_student_readout.json
运行: WSL (export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
OUT = CACHE / "p3_student_readout.json"
TEACHER_JSON = CACHE / "p3_teacher_separability.json"
MODEL = MODELS / "best_resnet_large_clean_baseline_v3.h5"
DIAG_V3 = CACHE / "diag_v3.json"

FC_FP_THETA = 0.95
TN_THETA = 0.50
SEED = 0
CAP = 4000
GNAME = ("FP", "TN", "TP", "gray_norm", "miss_abn")
EMBED_LAYER = "fc1"


def subsample_groups(labels, probs, cap, seed):
    """与 probe_teacher_separability.subsample_groups 逐行一致。"""
    group_ids = np.where(
        (labels == 0) & (probs > FC_FP_THETA), 0,
        np.where((labels == 0) & (probs <= TN_THETA), 1,
                 np.where((labels == 1) & (probs > TN_THETA), 2,
                          np.where(labels == 0, 3, 4))))
    keep = np.zeros(len(labels), dtype=bool)
    rng = np.random.default_rng(seed)
    for gid, gn in enumerate(GNAME):
        if gn == "gray_norm":
            continue
        idx = np.where(group_ids == gid)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep[idx] = True
    return keep, group_ids


def main():
    t0 = time.time()
    import tensorflow as tf
    sys.path.insert(0, str(BASE))
    from eval_clean_test import beat_metrics
    from data.split_guard import get_guard, load_arrays, INCART_RID_OFFSET

    model = tf.keras.models.load_model(str(MODEL), compile=False)
    assert EMBED_LAYER in [l.name for l in model.layers], \
        f"嵌入层 {EMBED_LAYER!r} 不在模型层中"

    # ---- selfcheck (与 P1/P2/P3 同一自证) ----
    diag = json.loads(DIAG_V3.read_text(encoding="utf-8"))
    dmi = np.load(CACHE / "mit_deploy_causal_match.npz")
    p_cmi = model.predict(np.asarray(dmi["beats"], dtype=np.float32)[..., None],
                          batch_size=512, verbose=0)[:, 1]
    c_mi_auc = float(beat_metrics(np.asarray(dmi["labels"]).astype(np.int32),
                                  p_cmi)["auc"])
    diag_mi_auc = diag["per_domain"]["mit_incart_merged"]["beat"]["auc"]
    auc_ok = abs(c_mi_auc - diag_mi_auc) < 1e-3
    print(f"[P3b] selfcheck: npz MIT+INCART AUC={c_mi_auc:.4f} "
          f"diag={diag_mi_auc:.4f} match={auc_ok}", flush=True)
    if not auc_ok:
        OUT.write_text(json.dumps(
            {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
             "aborted": True, "reason": "auc selfcheck failed"},
            indent=2, ensure_ascii=False), encoding="utf-8")
        return

    # ---- val 拍 (与 P3 同源) ----
    def split_arrays(tag):
        b, l, r = load_arrays(tag)
        g = get_guard(tag)
        return (np.asarray(b, dtype=np.float32), np.asarray(l).astype(np.int32),
                np.asarray(r), g)

    mi_b, mi_l, mi_r, g_mit = split_arrays("mit_bih")
    in_b, in_l, in_r, g_inc = split_arrays("incart")
    vb = np.concatenate([mi_b[g_mit.val_mask], in_b[g_inc.val_mask]])
    vl = np.concatenate([mi_l[g_mit.val_mask], in_l[g_inc.val_mask]])
    vr = np.concatenate([mi_r[g_mit.val_mask],
                         np.asarray(in_r[g_inc.val_mask]) + INCART_RID_OFFSET])
    probs = model.predict(vb[..., None], batch_size=512, verbose=0)[:, 1]
    print(f"[P3b] val MIT+INCART: {len(vb)} beats, {len(np.unique(vr))} records",
          flush=True)

    keep, group_ids_full = subsample_groups(vl, probs, CAP, SEED)
    census = {}
    for gid, gn in enumerate(GNAME):
        census[gn] = {"full": int((group_ids_full == gid).sum()),
                      "keep": int((keep & (group_ids_full == gid)).sum())}
        print(f"  {gn:9s} full={census[gn]['full']:6d} "
              f"keep={census[gn]['keep']:6d}", flush=True)

    beats = vb[keep]
    labels, rids = vl[keep], vr[keep]
    group_ids = group_ids_full[keep]
    n = len(beats)

    # ---- 学生 fc1 特征 ----
    embed_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer(EMBED_LAYER).output)
    feats = embed_model.predict(beats[..., None], batch_size=512, verbose=0)
    print(f"[P3b] student fc1 feats {feats.shape} ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- 留一记录线性读出 (逐行镜像 P3) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import normalize
    fnorm = normalize(feats, axis=1)
    y_bin = labels.astype(np.int64)
    recs = np.unique(rids)
    pred = np.full(n, np.nan)
    for rec in recs:
        tgt = rids == rec
        if len(np.unique(y_bin[~tgt])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(fnorm[~tgt], y_bin[~tgt])
        pred[tgt] = clf.predict_proba(fnorm[tgt])[:, 1]
    m = ~np.isnan(pred)
    lr_auc_all = float(roc_auc_score(y_bin[m], pred[m]))
    fp_vs_tp_mask = m & ((group_ids == 0) | (group_ids == 2))
    lr_auc_fp_vs_tp = (float(roc_auc_score(y_bin[fp_vs_tp_mask],
                                           pred[fp_vs_tp_mask]))
                       if fp_vs_tp_mask.sum() and len(set(y_bin[fp_vs_tp_mask])) == 2
                       else None)

    # ---- 与教师对照 ----
    teacher = json.loads(TEACHER_JSON.read_text(encoding="utf-8"))
    t_all = teacher["values"]["linear_readout_auc_all"]
    t_ft = teacher["values"]["linear_readout_auc_fp_vs_tp"]
    delta_all = t_all - lr_auc_all
    delta_ft = (t_ft - lr_auc_fp_vs_tp
                if (t_ft is not None and lr_auc_fp_vs_tp is not None) else None)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "P3b: 学生 fc1 特征同协议线性读出, 对照教师 0.7345; "
                   "定蒸馏路线的增量信息归属",
        "student_model": str(MODEL.relative_to(BASE)),
        "embed_layer": EMBED_LAYER,
        "selfcheck": {"npz_mit_incart_auc": c_mi_auc,
                      "diag_v3_auc": diag_mi_auc, "match": auc_ok},
        "sampling": {"per_group_cap": CAP, "seed": SEED, "census": census,
                     "same_as_p3": True},
        "n_beats": int(n), "n_records": int(len(recs)),
        "method": "留一记录线性读出 (L2 归一化 + LogReg C=1.0, max_iter=2000), "
                  "与 P3 逐行同协议",
        "values": {
            "student_linear_readout_auc_all": lr_auc_all,
            "student_linear_readout_auc_fp_vs_tp": lr_auc_fp_vs_tp,
            "teacher_linear_readout_auc_all": t_all,
            "teacher_linear_readout_auc_fp_vs_tp": t_ft,
            "delta_teacher_minus_student_all": delta_all,
            "delta_teacher_minus_student_fp_vs_tp": delta_ft,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[P3b] student readout AUC={lr_auc_all:.4f} "
          f"(FP vs TP: {lr_auc_fp_vs_tp:.4f})", flush=True)
    print(f"[P3b] teacher readout AUC={t_all:.4f} "
          f"(FP vs TP: {t_ft:.4f})", flush=True)
    print(f"[P3b] delta (teacher-student) all={delta_all:+.4f} "
          f"fp_vs_tp={delta_ft:+.4f}", flush=True)
    print(f"[P3b] saved {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
