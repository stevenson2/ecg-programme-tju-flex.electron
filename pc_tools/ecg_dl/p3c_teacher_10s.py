#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p3c_teacher_10s.py — P3c: 教师真实上限探针 (真 10s 上下文窗, 消除 tile 域偏移)
================================================================================
P3 用 拍×10 周期化平铺喂教师 (250样本→500@500Hz→tile 10份→5000), 存在域偏移:
教师训练于真实 10s 段。本探针在同一 val 拍群上改用 **以 R 峰为中心的真实 10s
原始信号窗** (native fs → resample_poly → 5000@500Hz → 教师原生预处理), 消除
该混杂, 回答"教师真实上限是否越过 0.80 GO 线"。

设计决策 (写入报告):
  1. 评估集 = RAW 拍 (MIT 记录块前 1/6 行, INCART 全行)。增强副本与 raw 共享
     同一 10s 上下文 (窗从原始信号切), 无增量信息且引入重复相关性 → 排除。
     故 census 与 P3/P3b 不同, 教师-教师 (10s vs tile) 配对比较 + P3c 内部
     教师-学生对照仍然成立。
  2. 同一拍集同时计算教师 10s 特征与教师 tile 特征 (镜像 P3 变换) →
     域偏移效应的直接归因。
  3. 边界: R 峰 ±5s 越出记录边界的拍 **排除** (不内移, 避免改变窗内对齐),
     记录排除数。
  4. 映射依据 (p3c_map_check.py 已验证, 逐拍 max|Δ|=0): 构建链
     build_deploy_npz --causal = corrected_deployment_chain + extract_beats_deploy;
     MIT 块内顺序 [raw, noise, scale1, scale2, drift1, drift2]; raw 行 k ↔
     AAMI 标注序列中第 k 个保留峰 (跳过规则同 extract_beats_deploy)。

数据纪律: 仅 SplitGuard val_mask (mit_bih + incart), 零测试集接触;
自证同 P1/P2/P3 (npz AUC 对齐 diag_v3)。

输出: models/deploy_match/p3c_teacher_10s.json
      models/deploy_match/p3c_teacher_features_10s.npy (备查)
运行: WSL, export ECG_PROCESSED_DIR=/home/devcontainers/ecg_data
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
OUT = CACHE / "p3c_teacher_10s.json"
FEAT_NPY = CACHE / "p3c_teacher_features_10s.npy"
MODEL = MODELS / "best_resnet_large_clean_baseline_v3.h5"
DIAG_V3 = CACHE / "diag_v3.json"

PROJECT_ROOT = BASE.parents[1]
ECGFOUNDER_DIR = PROJECT_ROOT.parent / "ECGFounder"
CKPT_1LEAD = ECGFOUNDER_DIR / "checkpoint" / "1_lead_ECGFounder.pth"

FS_TEACHER = 500
SEG_LEN = 5000
FC_FP_THETA = 0.95
TN_THETA = 0.50
SEED = 0
GNAME = ("FP", "TN", "TP", "gray_norm", "miss_abn")


# ---------------------------------------------------------------- 构建链复用
from config import TARGET_FS, AAMI_CLASSES  # noqa: E402
from data.preprocess import (  # noqa: E402
    load_mit_bih_record, extract_beats as mit_extract_beats, resample_ecg,
)
from data.preprocess_incart import (  # noqa: E402
    load_incart_record, extract_beats as incart_extract_beats,
)
from eval_deploy_match import (  # noqa: E402
    corrected_deployment_chain, align_stream_lengths,
)


def _aami_ann(ann_idx, ann_sym, fs):
    """AAMI 过滤: 返回 (r 峰 250Hz 索引, 保留掩码)。与 build_deploy_npz 同式。"""
    ann_idx = np.asarray(ann_idx)
    m = np.array([s in AAMI_CLASSES for s in ann_sym])
    return (ann_idx[m] * (TARGET_FS / fs)).astype(int), m


def _kept_mask_deploy(r_idx_250, n_sig, domain):
    """复刻 extract_beats_deploy 的保留/跳过决策 (逐峰), 返回布尔掩码。"""
    half = 125
    keep = np.zeros(len(r_idx_250), dtype=bool)
    for i, ri in enumerate(r_idx_250):
        if domain == "mit":
            start, end = max(0, ri - half), min(n_sig, ri + half)
            keep[i] = (end - start) >= 250 * 0.5
        else:
            lo, hi = ri - half, ri - half + 250
            keep[i] = (lo >= 0) and (hi <= n_sig)
    return keep


def _record_block_layout(rec_ids, rid, n_raw):
    """记录块在合并数组中的起始行; MIT 块长=6*n_raw, INCART=n_raw。"""
    rows = np.where(rec_ids == rid)[0]
    assert len(rows) and rows[0] + len(rows) - 1 == rows[-1], f"rid {rid} 行不连续"
    return int(rows[0]), len(rows)


class RecordCache:
    """按记录缓存: raw 原始信号 (lead0, native fs), fs, 每个保留 raw 拍的
    native R 峰样本索引 (与数组行顺序一致)。"""

    def __init__(self):
        self._cache = {}

    def get(self, domain, rid):
        key = (domain, rid)
        if key in self._cache:
            return self._cache[key]
        if domain == "mit":
            signal, ann_idx, ann_sym, fs = load_mit_bih_record(str(rid))
            lead0 = np.asarray(signal[:, 0], dtype=np.float64)
        else:
            lead0, ann_idx, ann_sym, fs = load_incart_record(f"I{rid:02d}")
            lead0 = np.asarray(lead0, dtype=np.float64)
        r250, _m = _aami_ann(ann_idx, ann_sym, fs)
        # 部署流长度 = resample_ecg(lead0, fs, 250) 的长度 (对齐基准)
        n_sig = int(len(lead0) * TARGET_FS / fs)
        kept = _kept_mask_deploy(r250, n_sig, domain)
        ann_idx = np.asarray(ann_idx)
        # kept 对应 AAMI 过滤后的峰序列, 故索引 ann_idx[_m] 而非全量 ann_idx
        r_native_kept = ann_idx[_m][kept]  # 与数组内 raw 行顺序一致 (map_check 已证)
        out = (lead0, fs, r_native_kept)
        self._cache[key] = out
        return out


def cut_10s_window(lead0, fs, r_native):
    """R 峰为中心的真实 10s 窗 → (5000,)@500Hz; 越界返回 None。"""
    half_native = 5 * fs
    lo, hi = r_native - half_native, r_native + half_native
    if lo < 0 or hi > len(lead0):
        return None
    from scipy.signal import resample_poly
    seg = lead0[lo:hi]                       # 10*fs 样本
    up = resample_poly(seg, FS_TEACHER, fs)  # → 5000 样本 (10*fs*500/fs)
    assert len(up) == SEG_LEN, f"resample len {len(up)} != {SEG_LEN}"
    return up


def teacher_preprocess(segs):
    """(m,5000) → filter_bandpass(500Hz, 逐行) + 逐段 z-score → (m,1,5000)。
    与 P3 chunk_to_teacher_input 同一预处理 (去掉 tile); 统一 float64 进滤波,
    两臂数值路径一致。"""
    sys.path.insert(0, str(ECGFOUNDER_DIR))
    from util import filter_bandpass
    segs = np.asarray(segs, dtype=np.float64)
    filt = filter_bandpass(segs, FS_TEACHER)
    mu = filt.mean(axis=-1, keepdims=True)
    sd = filt.std(axis=-1, keepdims=True) + 1e-8
    return ((filt - mu) / sd).astype(np.float32)[:, None, :]


def chunk_to_teacher_input_tiled(beats_chunk):
    """P3 原变换 (对照臂): (m,250)@250Hz → resample x2 → tile x10 → (m,1,5000)。"""
    from scipy.signal import resample_poly
    up = resample_poly(beats_chunk.astype(np.float64), 2, 1, axis=1)
    seg = np.tile(up, (1, 10))
    return teacher_preprocess(seg)


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
    print(f"[P3c] teacher loaded; missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    model.eval()
    return model


# ---------------------------------------------------------------- TF 侧
def tf_part():
    """自证 + val RAW 拍集 + 学生概率 + 学生 fc1 特征。返回后释放 TF。"""
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
    print(f"[P3c] selfcheck: npz MIT+INCART AUC={c_mi_auc:.4f} "
          f"diag={diag_mi_auc:.4f} match={auc_ok}", flush=True)
    if not auc_ok:
        return None

    # ---- val RAW 拍集: MIT 仅保留每记录块前 1/6 (raw), INCART 全行 ----
    b_m, l_m, r_m = load_arrays("mit_bih")
    b_m, l_m, r_m = np.asarray(b_m), np.asarray(l_m), np.asarray(r_m)
    g_mit = get_guard("mit_bih")
    vm = np.asarray(g_mit.val_mask)
    raw_mask_m = np.zeros(len(r_m), dtype=bool)
    for rid in np.unique(r_m):
        start, blk = _record_block_layout(r_m, int(rid), None)
        assert blk % 6 == 0, f"MIT {rid} 块长 {blk} 非 6 倍数"
        n_raw = blk // 6
        raw_mask_m[start:start + n_raw] = True
    sel_m = vm & raw_mask_m

    b_i, l_i, r_i = load_arrays("incart")
    b_i, l_i, r_i = np.asarray(b_i), np.asarray(l_i), np.asarray(r_i)
    g_inc = get_guard("incart")
    sel_i = np.asarray(g_inc.val_mask)

    vb = np.concatenate([b_m[sel_m].astype(np.float32),
                         b_i[sel_i].astype(np.float32)])
    vl = np.concatenate([l_m[sel_m].astype(np.int32),
                         l_i[sel_i].astype(np.int32)])
    vr = np.concatenate([r_m[sel_m], np.asarray(r_i[sel_i]) + INCART_RID_OFFSET])
    vdom = np.array(["mit"] * int(sel_m.sum()) + ["inc"] * int(sel_i.sum()))
    vrow = np.concatenate([np.where(sel_m)[0], np.where(sel_i)[0]])  # 原数组行号

    # 记录块起始行 (在各标签完整数组中), 键 (域, 原生 rid): 行号→记录内拍序用
    rid_meta = {}
    for rid in np.unique(r_m):
        rid_meta[("mit", int(rid))] = int(np.where(r_m == rid)[0][0])
    for rid in np.unique(r_i):
        rid_meta[("inc", int(rid))] = int(np.where(r_i == rid)[0][0])

    probs = model.predict(vb[..., None], batch_size=512, verbose=0)[:, 1]
    print(f"[P3c] val RAW beats: MIT={int(sel_m.sum())} INCART={int(sel_i.sum())} "
          f"total={len(vb)}, records={len(np.unique(vr))}", flush=True)

    # ---- 学生 fc1 特征 (对照臂) ----
    emb = tf.keras.Model(model.input, model.get_layer("fc1").output)
    s_feats = emb.predict(vb[..., None], batch_size=512, verbose=0)
    print(f"[P3c] student fc1 feats {s_feats.shape}", flush=True)

    del model, emb, dmi
    tf.keras.backend.clear_session()
    return {"selfcheck_auc": c_mi_auc, "diag_auc": diag_mi_auc,
            "beats": vb, "labels": vl, "rids": vr, "doms": vdom,
            "rows": vrow, "probs": probs, "student_feats": s_feats,
            "rid_meta": rid_meta}


def subsample_groups(labels, probs, cap, seed):
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

    labels, probs = info["labels"], info["probs"]
    keep, group_ids_full = subsample_groups(labels, probs,
                                            args.per_group_cap, SEED)
    print("[P3c] group census (val RAW full -> sampled):", flush=True)
    for gid, gn in enumerate(GNAME):
        n_full = int((group_ids_full == gid).sum())
        n_keep = int((keep & (group_ids_full == gid)).sum())
        print(f"  {gn:9s} full={n_full:6d} keep={n_keep:6d}", flush=True)

    idx_keep = np.where(keep)[0]
    beats = info["beats"][idx_keep]
    labels_s = labels[idx_keep]
    probs_s = probs[idx_keep]
    rids_s = info["rids"][idx_keep]
    doms_s = info["doms"][idx_keep]
    rows_s = info["rows"][idx_keep]
    group_ids = group_ids_full[idx_keep]

    # ---- 每拍切真 10s 窗 (按记录缓存原始信号) ----
    rc = RecordCache()
    rid_meta = info["rid_meta"]
    windows = np.zeros((len(beats), SEG_LEN), dtype=np.float32)
    win_ok = np.ones(len(beats), dtype=bool)
    t1 = time.time()
    for j in range(len(beats)):
        dom = doms_s[j]
        rid_key = int(rids_s[j])                      # INCART 带 +100000 偏移
        rid_native = rid_key if dom == "mit" else rid_key - 100000
        lead0, fs, r_native_kept = rc.get(dom, rid_native)
        start = rid_meta[(dom, rid_native)]
        k_raw = int(rows_s[j]) - start                # 记录内 raw 拍序 (INCART 行即 raw)
        if k_raw < 0 or k_raw >= len(r_native_kept):
            raise AssertionError(f"行映射越界: dom={dom} rid={rid_native} k={k_raw} "
                                 f"n_kept={len(r_native_kept)}")
        w = cut_10s_window(lead0, fs, int(r_native_kept[k_raw]))
        if w is None:
            win_ok[j] = False
        else:
            windows[j] = w
        if (j + 1) % 2000 == 0:
            print(f"  [windows] {j+1}/{len(beats)} ({time.time()-t1:.0f}s)",
                  flush=True)
    n_drop = int((~win_ok).sum())
    print(f"[P3c] 10s windows built; boundary-excluded={n_drop} "
          f"({time.time()-t1:.0f}s)", flush=True)

    sel = win_ok
    beats, labels_s, probs_s = beats[sel], labels_s[sel], probs_s[sel]
    rids_s, group_ids, windows = rids_s[sel], group_ids[sel], windows[sel]
    s_feats = info["student_feats"][idx_keep][sel]
    n = len(beats)

    # ---- 教师特征: 10s 臂 + tile 对照臂 ----
    import torch
    teacher = build_teacher()
    f10 = np.zeros((n, 1024), dtype=np.float32)
    f_tile = np.zeros((n, 1024), dtype=np.float32)
    t2 = time.time()
    bs = args.batch_size
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        x10 = teacher_preprocess(windows[i0:i1])
        xtl = chunk_to_teacher_input_tiled(beats[i0:i1])
        with torch.no_grad():
            _, fa = teacher(torch.from_numpy(x10))
            _, fb = teacher(torch.from_numpy(xtl))
        f10[i0:i1] = fa.numpy()
        f_tile[i0:i1] = fb.numpy()
        done = i1
        if done % (bs * 10) < bs or done >= n:
            rate = done / max(time.time() - t2, 1e-6)
            print(f"  [teacher] {done}/{n} ({rate:.1f} seg/s, "
                  f"eta {(n - done) / max(rate, 1e-6):.0f}s)", flush=True)
    np.save(FEAT_NPY, f10)
    print(f"[P3c] teacher feats 10s {f10.shape} + tiled {f_tile.shape} "
          f"({time.time()-t2:.0f}s)", flush=True)

    # ---- 留一记录线性读出 (三臂同协议: L2 归一 + LogReg C=1.0) ----
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    def readout(feats, name):
        fnorm = normalize(feats, axis=1)
        y_bin = labels_s.astype(np.int64)
        pred = np.full(n, np.nan)
        for rec in np.unique(rids_s):
            tgt = rids_s == rec
            if len(np.unique(y_bin[~tgt])) < 2:
                continue
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(fnorm[~tgt], y_bin[~tgt])
            pred[tgt] = clf.predict_proba(fnorm[tgt])[:, 1]
        m = ~np.isnan(pred)
        auc_all = float(roc_auc_score(y_bin[m], pred[m]))
        m2 = m & ((group_ids == 0) | (group_ids == 2))
        auc_ft = (float(roc_auc_score(y_bin[m2], pred[m2]))
                  if m2.sum() and len(set(y_bin[m2])) == 2 else None)
        print(f"[P3c] readout {name}: all={auc_all:.4f} fp_vs_tp={auc_ft}",
              flush=True)
        return auc_all, auc_ft

    lr_10_all, lr_10_ft = readout(f10, "teacher_10s")
    lr_tile_all, lr_tile_ft = readout(f_tile, "teacher_tile")
    lr_stu_all, lr_stu_ft = readout(s_feats, "student_fc1")

    # ---- 教师 10s 特征的原型/距离统计 (镜像 P3) ----
    recs = np.unique(rids_s)
    agg = {gn: {"ratio": [], "nearest_abn": 0, "prob": []} for gn in GNAME}
    for rec in recs:
        tgt = rids_s == rec
        e_o, l_o = f10[~tgt], labels_s[~tgt]
        if (l_o == 0).sum() < 5 or (l_o == 1).sum() < 5:
            continue
        norm_proto = e_o[l_o == 0].mean(axis=0)
        abn_proto = e_o[l_o == 1].mean(axis=0)
        dn = np.linalg.norm(f10[tgt] - norm_proto, axis=1)
        da = np.linalg.norm(f10[tgt] - abn_proto, axis=1)
        ratio = dn / (dn + da + 1e-12)
        for j, gi in enumerate(np.where(tgt)[0]):
            gn = GNAME[group_ids[gi]]
            agg[gn]["ratio"].append(float(ratio[j]))
            agg[gn]["nearest_abn"] += int(da[j] < dn[j])
            agg[gn]["prob"].append(float(probs_s[gi]))
    ratio_all = np.array(sum((agg[gn]["ratio"] for gn in GNAME), []))
    lab_all = np.array([0] * sum(len(agg[gn]["ratio"]) for gn in ("FP", "TN"))
                       + [1] * sum(len(agg[gn]["ratio"]) for gn in ("TP", "miss_abn")))
    emb_auc = float(roc_auc_score(lab_all, ratio_all)) if len(set(lab_all)) == 2 else None

    groups = {}
    for gn in GNAME:
        a = agg[gn]
        k = len(a["ratio"])
        groups[gn] = {
            "n": k,
            "mean_dist_ratio": float(np.mean(a["ratio"])) if k else None,
            "frac_nearest_abn": float(a["nearest_abn"] / k) if k else None,
            "mean_prob": float(np.mean(a["prob"])) if k else None,
        }

    p3_ref = json.loads((CACHE / "p3_teacher_separability.json")
                        .read_text(encoding="utf-8"))["values"]
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "P3c: 教师真实上限 — 真 10s 上下文窗 (非 tile 周期化) 重测, "
                   "消除域偏移混杂后再定蒸馏路线",
        "teacher": {"ckpt": str(CKPT_1LEAD), "feature_dim": 1024,
                    "input": "10s@500Hz (1,5000)",
                    "preprocess": "filter_bandpass(500Hz) + per-segment z-score"},
        "beat_to_segment": {
            "10s_arm": "R 峰为中心 ±5s 原始信号 (lead0, native fs) → resample_poly "
                       "→ 5000@500Hz; 边界拍排除 (不内移)",
            "tile_arm": "同 P3: 250@250Hz → resample x2 → tile x10 → 5000",
            "boundary_excluded": n_drop,
        },
        "evaluation_set": {
            "raw_only": True,
            "reason": "MIT 数组 = raw+5 增强副本; 增强副本与 raw 共享同一 10s "
                      "上下文, 无增量信息 → 仅评 raw (MIT 块前 1/6 行, INCART 全行)",
            "census_note": "census 与 P3/P3b 不同 (P3 含增强副本); 配对比较在 P3c 内部成立",
        },
        "mapping_evidence": "p3c_map_check.py PASS: MIT 201/231 + INCART I01/I02 "
                            "逐拍重建 max|Δ|=0.0, 标签与增强块结构全部一致",
        "student_model": str(MODEL.relative_to(BASE)),
        "selfcheck": {"npz_mit_incart_auc": info["selfcheck_auc"],
                      "diag_v3_auc": info["diag_auc"]},
        "group_defs": {"FP": "label=0 & prob>0.95", "TN": "label=0 & prob<=0.5",
                       "TP": "label=1 & prob>0.5",
                       "gray_norm": "label=0 & 0.5<prob<=0.95 (不抽样)",
                       "miss_abn": "label=1 & prob<=0.5"},
        "sampling": {"per_group_cap": args.per_group_cap, "seed": SEED},
        "n_beats": n, "n_records": int(len(recs)),
        "values": {
            "teacher_10s_linear_readout_auc_all": lr_10_all,
            "teacher_10s_linear_readout_auc_fp_vs_tp": lr_10_ft,
            "teacher_tile_linear_readout_auc_all": lr_tile_all,
            "teacher_tile_linear_readout_auc_fp_vs_tp": lr_tile_ft,
            "student_fc1_linear_readout_auc_all": lr_stu_all,
            "student_fc1_linear_readout_auc_fp_vs_tp": lr_stu_ft,
            "teacher_10s_embedding_auc_label_vs_ratio": emb_auc,
        },
        "p3_reference_tiled": p3_ref,
        "groups_teacher_10s": groups,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[P3c] ===== teacher true ceiling =====", flush=True)
    print(f"  teacher 10s  readout: all={lr_10_all:.4f} fp_vs_tp={lr_10_ft}", flush=True)
    print(f"  teacher tile readout: all={lr_tile_all:.4f} fp_vs_tp={lr_tile_ft}", flush=True)
    print(f"  student fc1  readout: all={lr_stu_all:.4f} fp_vs_tp={lr_stu_ft}", flush=True)
    print(f"  teacher 10s embedding AUC={emb_auc}", flush=True)
    go = lr_10_all is not None and lr_10_all >= 0.80
    print(f"\n[P3c] GO 线 (教师 10s 线性读出 ≥0.80): "
          f"{'CLEAR — 蒸馏路线有证据支持' if go else 'NOT cleared — 蒸馏需再授权'}",
          flush=True)
    print(f"[P3c] saved {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
