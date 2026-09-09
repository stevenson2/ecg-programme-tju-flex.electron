#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_embed_separation.py — P2: 被误报的正常拍在嵌入空间里挨着谁? (TH §105 之后)
================================================================================
判定目标 (决定"原型头/距离校准"值不值得做):
  - 被误报的正常拍 (label=0 & prob>0.95) 在 v3 A fc1 嵌入空间里
    更靠近"异常原型"还是"正常原型"?
    * 更近异常 → 嵌入信息层就没把这两类分开 → 换任何头都修不了过度报警 (表征瓶颈)
    * 更近正常 → 嵌入其实可分, 是 softmax 头把边缘点推过了界 → 距离校准/reject 可修

组定义 (prob = 模型输出的异常概率):
  FP    : label=0 & prob>0.95     过度报警源头 (P1 的 frac_gt_0.95 同口径)
  TN    : label=0 & prob<=0.5     明确判对
  TP    : label=1 & prob>0.5      真阳性
  (label=0 & 0.5<prob<=0.95 为灰区, label=1 & prob<=0.5 为漏检, 仅计数)

方法: g=SplitGuard val_mask 全量拍 → v3 A fc1 嵌入 (Dense64 relu, softmax 前)。
      leave-one-record-out 原型: 对每条 val 记录, 用"去掉该记录"的其余拍算
      正常原型/异常原型, 再量该记录内 FP/TN/TP 拍到两原型的距离与最近归属。
      这样规避"自己拍的嵌入混进自己原型 → 同源偏置"。

度量:
  embedding_AUC        = roc_auc(label, dist_ratio), dist_ratio=dn/(dn+da),
                         正常拍小/异常拍大 → 越高表示嵌入空间越可分
  frac_nearest_abn     = 某组拍到异常原型更近的占比
  mean_dist_ratio      = 某组平均 dist_ratio

输出: models/deploy_match/p2_embed_separation.json
运行: WSL (export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clean_test import make_predictor, beat_metrics
from data.split_guard import get_guard, load_arrays, INCART_RID_OFFSET

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
OUT = CACHE / "p2_embed_separation.json"
MODEL = MODELS / "best_resnet_large_clean_baseline_v3.h5"
DIAG_V3 = CACHE / "diag_v3.json"
EMBED_LAYER = "fc1"          # Dense(64, relu), softmax 前唯一嵌入层
FC_FP_THETA = 0.95
TN_THETA = 0.50


def main():
    t0 = time.time()

    # ---- 自证: 模型 + fc1 层 + npz 概率 AUC 对齐 diag_v3 ----
    model = tf.keras.models.load_model(str(MODEL), compile=False)
    assert EMBED_LAYER in [l.name for l in model.layers], \
        f"嵌入层 {EMBED_LAYER!r} 不在模型层中"
    _emb_in = model.get_layer(EMBED_LAYER).input.shape
    print(f"[P2] loaded {MODEL.name}; layers={len(model.layers)}; "
          f"embed={EMBED_LAYER} in={_emb_in}", flush=True)

    diag = json.loads(DIAG_V3.read_text(encoding="utf-8"))
    dmi = np.load(CACHE / "mit_deploy_causal_match.npz")
    p_cmi = model.predict(np.asarray(dmi["beats"], dtype=np.float32)[..., None],
                          batch_size=512, verbose=0)[:, 1]
    c_mi_auc = float(beat_metrics(np.asarray(dmi["labels"]).astype(np.int32), p_cmi)["auc"])
    diag_mi_auc = diag["per_domain"]["mit_incart_merged"]["beat"]["auc"]
    auc_ok = abs(c_mi_auc - diag_mi_auc) < 1e-3
    print(f"[P2] selfcheck: npz MIT+INCART AUC={c_mi_auc:.4f} diag={diag_mi_auc:.4f} "
          f"match={auc_ok}", flush=True)
    if not auc_ok:
        OUT.write_text(json.dumps(
            {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
             "aborted": True, "reason": "auc selfcheck failed"},
            indent=2, ensure_ascii=False), encoding="utf-8")
        return

    embed_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer(EMBED_LAYER).output)

    def predict_emb(beats):
        return embed_model.predict(np.asarray(beats, dtype=np.float32)[..., None],
                                   batch_size=512, verbose=0)

    def split_arrays(tag):
        b, l, r = load_arrays(tag)
        g = get_guard(tag)
        return (np.asarray(b, dtype=np.float32), np.asarray(l).astype(np.int32),
                np.asarray(r), g)

    mi_b, mi_l, mi_r, g_mit = split_arrays("mit_bih")
    in_b, in_l, in_r, g_inc = split_arrays("incart")

    # val 全量拍 (合并空间) — 与 P1 同源
    vmi = (mi_b[g_mit.val_mask], mi_l[g_mit.val_mask], mi_r[g_mit.val_mask])
    vin = (in_b[g_inc.val_mask], in_l[g_inc.val_mask],
           np.asarray(in_r[g_inc.val_mask]) + INCART_RID_OFFSET)
    vmi_b = np.concatenate([vmi[0], vin[0]])
    vmi_l = np.concatenate([vmi[1], vin[1]])
    vmi_r = np.concatenate([vmi[2], vin[2]])
    print(f"[P2] val MIT+INCART: {len(vmi_b)} beats, {len(np.unique(vmi_r))} records",
          flush=True)

    probs = model.predict(vmi_b[..., None], batch_size=512, verbose=0)[:, 1]
    emb = predict_emb(vmi_b)
    print(f"[P2] emb shape={emb.shape}", flush=True)

    # ---- leave-one-record-out 原型 + 最近归属 ----
    dom_res = {}
    for domain, rids, labels, p, e in (
            ("mit_incart", vmi_r, vmi_l, probs, emb),):
        recs = np.unique(rids)
        rec_group = {rec: [] for rec in recs}     # rec -> (group_idx, label, prob)
        # group: 0=FP,1=TN,2=TP,3=gray_norm,4=miss_abn
        group_ids = np.where(
            (labels == 0) & (p > FC_FP_THETA), 0,
            np.where((labels == 0) & (p <= TN_THETA), 1,
                     np.where((labels == 1) & (p > TN_THETA), 2,
                              np.where(labels == 0, 3, 4))))
        per_rec_group_idx = np.full((len(recs), 5), 0, dtype=np.int64)

        # 全局聚合 (每组): dist_ratio, nearest_abn 占比, prob
        agg = {g: {"dn": [], "da": [], "ratio": [], "nearest_abn": 0,
                   "prob": []} for g in ("FP", "TN", "TP", "gray_norm", "miss_abn")}
        gname = ("FP", "TN", "TP", "gray_norm", "miss_abn")

        for rec in recs:
            tgt = rids == rec
            oth = ~tgt
            emb_o, lab_o = e[oth], labels[oth]
            norm_proto = emb_o[lab_o == 0].mean(axis=0)
            abn_proto = emb_o[lab_o == 1].mean(axis=0)
            if not np.all(np.isfinite(norm_proto)) or not np.all(np.isfinite(abn_proto)):
                continue
            tgt_idx = np.where(tgt)[0]
            for i in tgt_idx:
                gid = group_ids[i]
                gn = gname[gid]
                dn = float(np.linalg.norm(e[i] - norm_proto))
                da = float(np.linalg.norm(e[i] - abn_proto))
                ratio = dn / (dn + da) if (dn + da) > 0 else 0.5
                agg[gn]["dn"].append(dn)
                agg[gn]["da"].append(da)
                agg[gn]["ratio"].append(ratio)
                agg[gn]["nearest_abn"] += int(da < dn)
                agg[gn]["prob"].append(float(p[i]))

        # 嵌入空间可分度: 正常(0/1) vs 异常(2/4) 用 dist_ratio
        ratio_all = np.array(agg["TN"]["ratio"] + agg["TP"]["ratio"]
                             + agg["FP"]["ratio"] + agg["gray_norm"]["ratio"]
                             + agg["miss_abn"]["ratio"])
        lab_all = np.array([0] * (len(agg["TN"]["ratio"]) + len(agg["FP"]["ratio"])
                                  + len(agg["gray_norm"]["ratio"]))
                           + [1] * (len(agg["TP"]["ratio"]) + len(agg["miss_abn"]["ratio"])))
        emb_auc = float(roc_auc_score(lab_all, ratio_all)) if len(set(lab_all)) == 2 else None

        summary = {}
        for gn in gname:
            a = agg[gn]
            n = len(a["ratio"])
            summary[gn] = {
                "n": n,
                "recs_present": int(sum(1 for r in recs if (group_ids[rids == r] == gname.index(gn)).any())),
                "mean_dist_norm": float(np.mean(a["dn"])) if n else None,
                "mean_dist_abn": float(np.mean(a["da"])) if n else None,
                "mean_dist_ratio": float(np.mean(a["ratio"])) if n else None,
                "frac_nearest_abn": float(a["nearest_abn"] / n) if n else None,
                "mean_prob": float(np.mean(a["prob"])) if n else None,
                "p95_prob": float(np.quantile(a["prob"], 0.95)) if n else None,
            }
        dom_res[domain] = {"records": int(len(recs)),
                           "values": {"embedding_auc_label_vs_ratio": emb_auc},
                           "groups": summary}
        print(f"\n[P2] ===== domain {domain}: records={len(recs)} "
              f"embedding_AUC={emb_auc:.4f} =====", flush=True)
        for gn in gname:
            s = summary[gn]
            if s["n"] == 0:
                continue
            print(f"  {gn:9s} n={s['n']:6d} dist_ratio={s['mean_dist_ratio']:.3f} "
                  f"nearest_abn={s['frac_nearest_abn']:.3f} prob={s['mean_prob']:.3f}",
                  flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "P2: 被误报正常拍在 fc1 嵌入空间里靠近正常还是异常原型; "
                   "决定原型/距离校准头是否值得做",
        "model": str(MODEL.relative_to(BASE)),
        "embed_layer": EMBED_LAYER,
        "selfcheck": {"npz_mit_incart_auc": c_mi_auc,
                      "diag_v3_mit_incart_auc": diag_mi_auc, "match": auc_ok},
        "group_defs": {"FP": "label=0 & prob>0.95 (过度报警源头)",
                       "TN": "label=0 & prob<=0.5 (明确判对)",
                       "TP": "label=1 & prob>0.5 (真阳性)",
                       "gray_norm": "label=0 & 0.5<prob<=0.95 (灰区)",
                       "miss_abn": "label=1 & prob<=0.5 (漏检)"},
        "method": "leave-one-record-out 原型; dist_ratio=dn/(dn+da), 正常小/异常大",
        "domains": dom_res,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[P2] saved {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
