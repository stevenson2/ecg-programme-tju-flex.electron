#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_teacher_separability.py — P3: 教师特征能否分开"被误报正常拍"与"异常拍"? (val 侧)
================================================================================
判定目标 (蒸馏路线的最便宜先行验证):
  P2 已证明学生 fc1 嵌入分不开 FP 正常拍与异常拍 (embedding AUC=0.543)。
  本探针问: ECGFounder 1-lead 教师的 1024 维特征, 在同一 val 拍群上,
  能否把 FP 正常拍 (label=0 & v3A prob>0.95) 与异常拍分开?
    * 分不开 → 蒸馏路线当场否决 (教师也没有这个信息)
    * 分得开 → 教师有学生没有的信息, 蒸馏 (B1/B2/B3) 进入规格阶段

数据纪律:
  - 只用 SplitGuard val_mask 拍 (mit_bih + incart, incart rid +100000), 零测试集接触
  - v3 A 概率自证: npz 缓存 AUC 对齐 diag_v3 (与 P1/P2 同一自证)

拍 → 教师输入转换 (预注册的设计决策, 局限写入报告):
  - 250@250Hz 拍 → resample_poly 上采到 500Hz (500 样本)
  - 平铺 10 份填充 5000 样本窗 (教师原生 10s@500Hz 输入流形;
    局限: 周期化输入会强化节律周期特征, 弱化一次性形态事件)
  - 预处理严格复刻教师: filter_bandpass(500Hz) + 逐段 z-score

抽样 (控制 CPU 推理时长, 组定义与 P2 完全一致):
  FP / TN / TP / miss_abn 各上限 --per-group-cap (默认 4000), 种子 0 固定;
  gray_norm 不参与判定, 不抽。原型用其余记录的抽样拍算 (leave-one-record-out)。

输出: models/deploy_match/p3_teacher_separability.json
      models/deploy_match/p3_teacher_features.npy (备查/复算)
运行: WSL (export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data)。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
OUT = CACHE / "p3_teacher_separability.json"
FEAT_NPY = CACHE / "p3_teacher_features.npy"
MODEL = MODELS / "best_resnet_large_clean_baseline_v3.h5"
DIAG_V3 = CACHE / "diag_v3.json"

PROJECT_ROOT = BASE.parents[1]
ECGFOUNDER_DIR = PROJECT_ROOT.parent / "ECGFounder"
CKPT_1LEAD = ECGFOUNDER_DIR / "checkpoint" / "1_lead_ECGFounder.pth"

FS_STUDENT = 250
FS_TEACHER = 500
SEG_LEN = 5000
FC_FP_THETA = 0.95
TN_THETA = 0.50
SEED = 0
GNAME = ("FP", "TN", "TP", "gray_norm", "miss_abn")


# ---------------------------------------------------------------- TF 侧
def tf_part():
    """自证 + val 拍概率。返回后调用方释放 TF, 再进 torch。"""
    import tensorflow as tf
    sys.path.insert(0, str(BASE))
    from eval_clean_test import beat_metrics
    from data.split_guard import get_guard, load_arrays, INCART_RID_OFFSET

    model = tf.keras.models.load_model(str(MODEL), compile=False)
    diag = json.loads(DIAG_V3.read_text(encoding="utf-8"))
    dmi = np.load(CACHE / "mit_deploy_causal_match.npz")
    p_cmi = model.predict(np.asarray(dmi["beats"], dtype=np.float32)[..., None],
                          batch_size=512, verbose=0)[:, 1]
    c_mi_auc = float(beat_metrics(np.asarray(dmi["labels"]).astype(np.int32),
                                  p_cmi)["auc"])
    diag_mi_auc = diag["per_domain"]["mit_incart_merged"]["beat"]["auc"]
    auc_ok = abs(c_mi_auc - diag_mi_auc) < 1e-3
    print(f"[P3] selfcheck: npz MIT+INCART AUC={c_mi_auc:.4f} "
          f"diag={diag_mi_auc:.4f} match={auc_ok}", flush=True)
    if not auc_ok:
        return None

    def split_arrays(tag):
        b, l, r = load_arrays(tag)
        g = get_guard(tag)
        return (np.asarray(b, dtype=np.float32), np.asarray(l).astype(np.int32),
                np.asarray(r), g)

    mi_b, mi_l, mi_r, g_mit = split_arrays("mit_bih")
    in_b, in_l, in_r, g_inc = split_arrays("incart")
    vmi = (mi_b[g_mit.val_mask], mi_l[g_mit.val_mask], mi_r[g_mit.val_mask])
    vin = (in_b[g_inc.val_mask], in_l[g_inc.val_mask],
           np.asarray(in_r[g_inc.val_mask]) + INCART_RID_OFFSET)
    vb = np.concatenate([vmi[0], vin[0]])
    vl = np.concatenate([vmi[1], vin[1]])
    vr = np.concatenate([vmi[2], vin[2]])
    probs = model.predict(vb[..., None], batch_size=512, verbose=0)[:, 1]
    print(f"[P3] val MIT+INCART: {len(vb)} beats, {len(np.unique(vr))} records",
          flush=True)

    del model, dmi
    tf.keras.backend.clear_session()
    return {"auc_ok": auc_ok, "selfcheck_auc": c_mi_auc,
            "diag_auc": diag_mi_auc,
            "beats": vb, "labels": vl, "rids": vr, "probs": probs}


# ---------------------------------------------------------------- 拍→教师段
def chunk_to_teacher_input(beats_chunk):
    """(m,250)@250Hz → (m,1,5000)@500Hz float32, 教师原生预处理。

    步骤: resample_poly ×2 上采 → 平铺 10 份 → filter_bandpass(500Hz, 逐行)
    → 逐段 z-score。filter_bandpass 契约是 2D (channels,time), 故按 (m,5000)
    行阵传入 (每行一"导联"), 3D 传入会让 medfilt 跨 batch 轴, 禁用。
    """
    from scipy.signal import resample_poly
    sys.path.insert(0, str(ECGFOUNDER_DIR))
    from util import filter_bandpass
    up = resample_poly(beats_chunk.astype(np.float64), 2, 1, axis=1)  # (m,500)
    seg = np.tile(up, (1, 10))                                        # (m,5000)
    filt = filter_bandpass(seg, FS_TEACHER)                           # 逐行
    mu = filt.mean(axis=-1, keepdims=True)
    sd = filt.std(axis=-1, keepdims=True) + 1e-8
    z = ((filt - mu) / sd).astype(np.float32)
    return z[:, None, :]


def build_teacher():
    import torch
    sys.path.insert(0, str(ECGFOUNDER_DIR))
    from net1d import Net1D
    if not CKPT_1LEAD.exists():
        raise FileNotFoundError(f"缺少 checkpoint: {CKPT_1LEAD}")
    model = Net1D(
        in_channels=1, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=False, use_do=False,
        n_classes=150, return_features=True,
    )
    ck = torch.load(str(CKPT_1LEAD), map_location="cpu")
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    print(f"[P3] teacher loaded; missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    model.eval()
    return model


# ---------------------------------------------------------------- 分析
def subsample_groups(labels, probs, rids, cap, seed):
    """按 P2 组定义分组并抽样; 返回 (keep_mask, group_ids)。"""
    group_ids = np.where(
        (labels == 0) & (probs > FC_FP_THETA), 0,
        np.where((labels == 0) & (probs <= TN_THETA), 1,
                 np.where((labels == 1) & (probs > TN_THETA), 2,
                          np.where(labels == 0, 3, 4))))
    keep = np.zeros(len(labels), dtype=bool)
    rng = np.random.default_rng(seed)
    for gid, gn in enumerate(GNAME):
        if gn == "gray_norm":
            continue                      # 不参与判定
        idx = np.where(group_ids == gid)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep[idx] = True
    return keep, group_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group-cap", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()

    info = tf_part()
    if info is None:
        OUT.write_text(json.dumps(
            {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
             "aborted": True, "reason": "auc selfcheck failed"},
            indent=2, ensure_ascii=False), encoding="utf-8")
        return

    labels, probs, rids = info["labels"], info["probs"], info["rids"]
    keep, group_ids_full = subsample_groups(labels, probs, rids,
                                            args.per_group_cap, SEED)
    print("[P3] group census (val full -> sampled):", flush=True)
    for gid, gn in enumerate(GNAME):
        n_full = int((group_ids_full == gid).sum())
        n_keep = int((keep & (group_ids_full == gid)).sum())
        print(f"  {gn:9s} full={n_full:6d} keep={n_keep:6d}", flush=True)

    beats = info["beats"][keep]
    labels, probs, rids = labels[keep], probs[keep], rids[keep]
    group_ids = group_ids_full[keep]
    n = len(beats)

    # ---- 拍 → 教师特征 (分块构建输入, 控制内存) ----
    import torch
    teacher = build_teacher()
    feats = np.zeros((n, 1024), dtype=np.float32)
    t1 = time.time()
    bs = args.batch_size
    for i0 in range(0, n, bs):
        chunk = chunk_to_teacher_input(beats[i0:i0 + bs])
        with torch.no_grad():
            _logits, f = teacher(torch.from_numpy(chunk))
        feats[i0:i0 + bs] = f.numpy()
        done = min(i0 + bs, n)
        if done % (bs * 10) < bs or done >= n:
            rate = done / max(time.time() - t1, 1e-6)
            print(f"  [teacher] {done}/{n} ({rate:.1f} seg/s, "
                  f"eta {(n - done) / max(rate, 1e-6):.0f}s)", flush=True)
    np.save(FEAT_NPY, feats)
    print(f"[P3] teacher feats {feats.shape} saved ({time.time()-t1:.0f}s)",
          flush=True)

    # ---- leave-one-record-out 原型 (镜像 P2) ----
    from sklearn.metrics import roc_auc_score
    recs = np.unique(rids)
    agg = {gn: {"ratio": [], "nearest_abn": 0, "prob": []} for gn in GNAME}
    for rec in recs:
        tgt = rids == rec
        e_o, l_o = feats[~tgt], labels[~tgt]
        if (l_o == 0).sum() < 5 or (l_o == 1).sum() < 5:
            continue
        norm_proto = e_o[l_o == 0].mean(axis=0)
        abn_proto = e_o[l_o == 1].mean(axis=0)
        dn = np.linalg.norm(feats[tgt] - norm_proto, axis=1)
        da = np.linalg.norm(feats[tgt] - abn_proto, axis=1)
        ratio = dn / (dn + da + 1e-12)
        for j, gi in enumerate(np.where(tgt)[0]):
            gn = GNAME[group_ids[gi]]
            agg[gn]["ratio"].append(float(ratio[j]))
            agg[gn]["nearest_abn"] += int(da[j] < dn[j])
            agg[gn]["prob"].append(float(probs[gi]))

    ratio_all = np.array(sum((agg[gn]["ratio"] for gn in GNAME), []))
    lab_all = np.array([0] * sum(len(agg[gn]["ratio"])
                                 for gn in ("FP", "TN"))
                       + [1] * sum(len(agg[gn]["ratio"])
                                   for gn in ("TP", "miss_abn")))
    emb_auc = float(roc_auc_score(lab_all, ratio_all)) if len(set(lab_all)) == 2 else None

    # ---- 线性读出 (留一记录) : 教师特征里是否存在任何线性可分方向 ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize
    fnorm = normalize(feats, axis=1)
    y_bin = labels.astype(np.int64)
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
    lr_auc_fp_vs_tp = (float(roc_auc_score(y_bin[fp_vs_tp_mask], pred[fp_vs_tp_mask]))
                       if fp_vs_tp_mask.sum() and len(set(y_bin[fp_vs_tp_mask])) == 2
                       else None)

    summary = {}
    for gn in GNAME:
        a = agg[gn]
        k = len(a["ratio"])
        summary[gn] = {
            "n": k,
            "mean_dist_ratio": float(np.mean(a["ratio"])) if k else None,
            "frac_nearest_abn": float(a["nearest_abn"] / k) if k else None,
            "mean_prob": float(np.mean(a["prob"])) if k else None,
        }

    # ---- 判定规则 (预注册) ----
    fp_s = summary["FP"]
    verdict = None
    if fp_s["n"] and emb_auc is not None:
        if fp_s["frac_nearest_abn"] >= 0.9 and fp_s["mean_dist_ratio"] >= 0.6 \
                and emb_auc < 0.65:
            verdict = ("KILL_DISTILLATION: 教师特征同样把 FP 正常拍混入异常簇 "
                       f"(nearest_abn={fp_s['frac_nearest_abn']:.3f}, "
                       f"ratio={fp_s['mean_dist_ratio']:.3f}, AUC={emb_auc:.3f}); "
                       "教师没有学生缺失的分离信息")
        elif fp_s["frac_nearest_abn"] <= 0.5 and emb_auc >= 0.80:
            verdict = ("GO_DISTILLATION: 教师特征把 FP 正常拍归回正常簇且整体可分 "
                       f"(nearest_abn={fp_s['frac_nearest_abn']:.3f}, "
                       f"AUC={emb_auc:.3f}); 进入蒸馏规格阶段 (B1/B2/B3)")
        else:
            verdict = ("INCONCLUSIVE: 教师特征部分可分, 需人工复核数字后再定路线")

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "P3: ECGFounder 1-lead 教师 1024 维特征能否在 val 侧分开 "
                   "FP 正常拍与异常拍; 蒸馏路线先行验证",
        "teacher": {"ckpt": str(CKPT_1LEAD), "feature_dim": 1024,
                    "input": "10s@500Hz (1,5000)",
                    "preprocess": "filter_bandpass(500Hz) + per-segment z-score"},
        "beat_to_segment": {
            "resample": "250@250Hz -> resample_poly x2 -> 500@500Hz",
            "padding": "tile x10 -> 5000 samples",
            "limitation": "周期化输入强化节律特征、弱化一次性形态事件; "
                          "教师在真实 10s 段上训练, 域偏移对本探针保守 "
                          "(分不开→否决蒸馏仍成立; 分得开→需蒸馏阶段复核)"},
        "student_model": str(MODEL.relative_to(BASE)),
        "selfcheck": info["auc_ok"] and {"npz_mit_incart_auc": info["selfcheck_auc"],
                                         "diag_v3_auc": info["diag_auc"]},
        "group_defs": {"FP": "label=0 & prob>0.95", "TN": "label=0 & prob<=0.5",
                       "TP": "label=1 & prob>0.5",
                       "gray_norm": "label=0 & 0.5<prob<=0.95 (不抽样/不判定)",
                       "miss_abn": "label=1 & prob<=0.5"},
        "sampling": {"per_group_cap": args.per_group_cap, "seed": SEED},
        "n_beats": n, "n_records": int(len(recs)),
        "method": "leave-one-record-out 原型; dist_ratio=dn/(dn+da); "
                  "另加留一记录线性读出 (L2 归一化 + LogReg)",
        "values": {
            "embedding_auc_label_vs_ratio": emb_auc,
            "linear_readout_auc_all": lr_auc_all,
            "linear_readout_auc_fp_vs_tp": lr_auc_fp_vs_tp,
        },
        "groups": summary,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[P3] ===== teacher separability =====", flush=True)
    print(f"  embedding_AUC={emb_auc}  linear_readout_AUC={lr_auc_all} "
          f"(FP vs TP: {lr_auc_fp_vs_tp})", flush=True)
    for gn in GNAME:
        s = summary[gn]
        if s["n"]:
            print(f"  {gn:9s} n={s['n']:6d} ratio={s['mean_dist_ratio']:.3f} "
                  f"nearest_abn={s['frac_nearest_abn']:.3f}", flush=True)
    print(f"\n[P3] verdict: {verdict}", flush=True)
    print(f"[P3] saved {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
