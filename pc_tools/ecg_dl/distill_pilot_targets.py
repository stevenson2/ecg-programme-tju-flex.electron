#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""distill_pilot_targets.py — 蒸馏试点 Stage 1: 教师软目标生成 (TH §106)
================================================================================
复现 v3 配置 a 的抽样 (同一 rng 种子与消费顺序, 身份锚点硬断言对照
train_clean_baseline_v3.json), 对全部训练/验证拍计算教师 (ECGFounder 1-lead)
tile 特征, 以训练侧留一记录 (LORO) LogReg 读出产生软伪概率 (绝不接触
val/测试标签), 打包成 Stage 2 一次性数据包 (含置换后训练集)。

铁律:
  - 公共库取数全经 data/split_guard.py; 测试集零接触 (本脚本不读测试缓存)。
  - LORO 分组: 346 个公共库记录组 + 单一伪记录组 real_synth (542 real 重复
    + 600 synth, 全 label 0; 其折只由公共库拍训练)。
  - 预注册中止门: 训练侧 LORO AUC (abn vs norm, 公共库拍) < 0.75 → 写
    aborted JSON, 不落 npz。
  - 预注册替换规则: real_synth 组均值伪概率 > 0.5 → 该组目标替换为 EPS。

输出:
  models/deploy_match/distill_pilot_teacher_targets.npz
  models/deploy_match/distill_pilot_teacher_targets.json
  models/deploy_match/distill_pilot_teacher_features.npy
运行环境: WSL (教师为 CPU 推理, ~54 seg/s)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard, load_arrays
from train_clean_baseline_v3 import (
    SEED, MAIN_RATIO, N_REAL_HOLDOUT, REAL_REPEAT,
    VAL_MI_PER_CLASS, VAL_PTB_PER_CLASS, DATA_REAL,
    sample_val_domain, synth_hard,
)
from p3c_teacher_10s import chunk_to_teacher_input_tiled, build_teacher

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
ANCHOR_JSON = CACHE / "train_clean_baseline_v3.json"
OUT_NPZ = CACHE / "distill_pilot_teacher_targets.npz"
OUT_JSON = CACHE / "distill_pilot_teacher_targets.json"
OUT_FEATS = CACHE / "distill_pilot_teacher_features.npy"

EPS = 1e-7
ABORT_LORO_AUC = 0.75
SWAP_MEAN_P = 0.5
TEACHER_BATCH = 32
DOMAIN_CODES = {"mit_bih": 0, "incart": 1, "ptb": 2, "real": 3, "synth": 4}


def sample_train_domain(tag, n_abn, n_norm, rng):
    """镜像 train_clean_baseline_v3.sample_domain: 先 sample_train_beats
    (唯一 rng 消费点, abn 先 norm 后), 再加载数组做审计。额外返回逐拍
    (索引, 记录号) 以供 LORO 分组 (v3 本体不返回, 故复制 6 行审计粘合)。"""
    g = get_guard(tag)
    sa, sn = g.sample_train_beats(n_abn, n_norm, rng)
    b, _l, r = load_arrays(tag)
    sa, sn = np.asarray(sa), np.asarray(sn)
    r = np.asarray(r)
    # ---- 以下与 sample_domain 函数体逐行对应 ----
    sampled_rids = np.concatenate([r[sa], r[sn]])
    g.assert_train_only(sampled_rids, context=f"distill_pilot_targets({tag})")
    inter_test = np.intersect1d(np.unique(sampled_rids), g.test_record_ids())
    assert len(inter_test) == 0, \
        f"{tag}: 抽样记录与测试记录交集非空: {inter_test.tolist()}"
    # ---- 粘合结束 ----
    return {
        "abn_beats": np.asarray(b[sa], dtype=np.float32),
        "abn_rids": r[sa],
        "norm_beats": np.asarray(b[sn], dtype=np.float32),
        "norm_rids": r[sn],
        "n_unique_records": int(len(np.unique(sampled_rids))),
    }


def teacher_features(beats, teacher, torch, tag):
    t0 = time.time()
    feats = []
    n = len(beats)
    for i0 in range(0, n, TEACHER_BATCH):
        xtl = chunk_to_teacher_input_tiled(beats[i0:i0 + TEACHER_BATCH])
        with torch.no_grad():
            _, f = teacher(torch.from_numpy(xtl))
        feats.append(f.numpy())
        if (i0 // TEACHER_BATCH) % 25 == 0:
            print(f"[DP1] teacher {tag}: {min(i0 + TEACHER_BATCH, n)}/{n}",
                  flush=True)
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    dt = time.time() - t0
    print(f"[DP1] teacher feats {tag}: {feats.shape} "
          f"({n / dt:.1f} seg/s, {dt:.0f}s)", flush=True)
    return feats


def loro_readout(feats, labels, group_ids, real_synth_code):
    """留一记录读出 (镜像探针协议: L2 normalize + LogReg C=1 max_iter=2000)。
    单类折 (该记录组训练侧只含一类) 直接赋该类常数概率。"""
    fn = normalize(feats)
    p = np.zeros(len(labels), dtype=np.float64)
    n_groups = int(group_ids.max()) + 1
    single_class_folds = []
    for g in range(n_groups):
        idx = np.flatnonzero(group_ids == g)
        fit_m = group_ids != g
        y_fit = labels[fit_m]
        classes = np.unique(y_fit)
        if len(classes) == 1:
            p[idx] = float(classes[0])
            single_class_folds.append(int(g))
            continue
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(fn[fit_m], y_fit)
        p[idx] = lr.predict_proba(fn[idx])[:, 1]
        if g % 50 == 0:
            print(f"[DP1] LORO fold {g + 1}/{n_groups}", flush=True)
    return p, single_class_folds


def main():
    import torch  # noqa: 延迟导入, 先让守卫/抽样跑完
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    anchor = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))

    # ---------- 抽样: rng 消费顺序逐字镜像 v3 run_config ----------
    rng = np.random.default_rng(SEED)
    dom = {}
    for tag in ("mit_bih", "incart", "ptb"):
        n_abn, n_norm = MAIN_RATIO[tag]
        dom[tag] = sample_train_domain(tag, n_abn, n_norm, rng)
        print(f"[DP1] sample {tag}: abn={len(dom[tag]['abn_beats'])} "
              f"norm={len(dom[tag]['norm_beats'])} "
              f"records={dom[tag]['n_unique_records']}", flush=True)

    real = np.concatenate([
        np.load(DATA_REAL / "real_normal_beats_exp7c.npy").astype(np.float32),
        np.load(DATA_REAL / "real_normal_beats_rec_latest.npy").astype(np.float32),
    ])
    holdout_idx = rng.choice(len(real), N_REAL_HOLDOUT, replace=False)
    train_real_idx = np.setdiff1d(np.arange(len(real)), holdout_idx)
    real_train = real[train_real_idx]

    hard = synth_hard(real_train, rng)

    # ---- 组装与置换 (与 v3 同序) ----
    x_main = np.concatenate([
        dom["mit_bih"]["abn_beats"], dom["incart"]["abn_beats"],
        dom["ptb"]["abn_beats"],
        dom["mit_bih"]["norm_beats"], dom["incart"]["norm_beats"],
        dom["ptb"]["norm_beats"],
    ])
    y_main = np.concatenate([
        np.ones(len(dom["mit_bih"]["abn_beats"]) + len(dom["incart"]["abn_beats"])
                + len(dom["ptb"]["abn_beats"]), dtype=np.int32),
        np.zeros(len(dom["mit_bih"]["norm_beats"]) + len(dom["incart"]["norm_beats"])
                 + len(dom["ptb"]["norm_beats"]), dtype=np.int32),
    ])
    rid_main = np.concatenate([
        dom["mit_bih"]["abn_rids"], dom["incart"]["abn_rids"],
        dom["ptb"]["abn_rids"],
        dom["mit_bih"]["norm_rids"], dom["incart"]["norm_rids"],
        dom["ptb"]["norm_rids"],
    ])
    dom_main = np.concatenate([
        np.full(len(dom["mit_bih"]["abn_beats"]), DOMAIN_CODES["mit_bih"]),
        np.full(len(dom["incart"]["abn_beats"]), DOMAIN_CODES["incart"]),
        np.full(len(dom["ptb"]["abn_beats"]), DOMAIN_CODES["ptb"]),
        np.full(len(dom["mit_bih"]["norm_beats"]), DOMAIN_CODES["mit_bih"]),
        np.full(len(dom["incart"]["norm_beats"]), DOMAIN_CODES["incart"]),
        np.full(len(dom["ptb"]["norm_beats"]), DOMAIN_CODES["ptb"]),
    ]).astype(np.int32)

    x_real = np.concatenate([real_train] * REAL_REPEAT)
    y_real = np.zeros(len(x_real), dtype=np.int32)
    x_hard = hard.astype(np.float32)
    y_hard = np.zeros(len(x_hard), dtype=np.int32)

    x_train = np.concatenate([x_main, x_real, x_hard])
    y_train = np.concatenate([y_main, y_real, y_hard])

    # ---- LORO 分组键: (域, 原生记录号) → 连续码; real_synth 单一伪记录 ----
    key_to_code, group_ids = {}, []
    for domc, rid in zip(dom_main, rid_main):
        key = (int(domc), int(rid))
        if key not in key_to_code:
            key_to_code[key] = len(key_to_code)
        group_ids.append(key_to_code[key])
    real_synth_code = len(key_to_code)
    group_ids.extend([real_synth_code] * (len(x_real) + len(x_hard)))
    group_ids = np.asarray(group_ids, dtype=np.int32)
    dom_train = np.concatenate([
        dom_main,
        np.full(len(x_real), DOMAIN_CODES["real"], dtype=np.int32),
        np.full(len(x_hard), DOMAIN_CODES["synth"], dtype=np.int32),
    ])

    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    group_ids, dom_train = group_ids[perm], dom_train[perm]
    print(f"[DP1] train total={len(x_train)} abn={int((y_train == 1).sum())} "
          f"norm={int((y_train == 0).sum())} groups={real_synth_code + 1}",
          flush=True)

    # ---------- val (同 v3 顺序: MIT → INCART → PTB) ----------
    va, val, vn, vnl = sample_val_domain("mit_bih", VAL_MI_PER_CLASS,
                                         VAL_MI_PER_CLASS, rng)
    ia, ial, inv, inl = sample_val_domain("incart", VAL_MI_PER_CLASS,
                                          VAL_MI_PER_CLASS, rng)
    pa, pal, pn, pnl = sample_val_domain("ptb", VAL_PTB_PER_CLASS,
                                         VAL_PTB_PER_CLASS, rng)
    x_real_ho = real[holdout_idx]
    x_val = np.concatenate([va, vn, ia, inv, pa, pn, x_real_ho]).astype(np.float32)
    y_val = np.concatenate([val, vnl, ial, inl, pal, pnl,
                            np.zeros(len(holdout_idx), dtype=np.int32)])
    print(f"[DP1] val total={len(x_val)} abn={int((y_val == 1).sum())} "
          f"norm={int((y_val == 0).sum())}", flush=True)

    # ---------- 身份锚点硬断言 (对照 train_clean_baseline_v3.json) ----------
    checks = {}

    def chk(name, got, want):
        ok = got == want
        checks[name] = {"got": got, "want": want, "ok": ok}
        if not ok:
            raise SystemExit(f"[DP1] identity anchor FAIL: {name}: {got} != {want}")

    a_d = anchor["data"]
    for tag, key_prefix in (("mit_bih", "mit_bih"), ("incart", "incart"),
                            ("ptb", "ptb")):
        chk(f"{tag}_abn", int(len(dom[tag]["abn_beats"])),
            a_d["main"][f"{key_prefix}_abn"])
        chk(f"{tag}_norm", int(len(dom[tag]["norm_beats"])),
            a_d["main"][f"{key_prefix}_norm"])
        chk(f"{tag}_unique_records", dom[tag]["n_unique_records"],
            a_d["sampling_audit"][tag]["sampled_unique_records"])
    chk("real_holdout_indices", [int(i) for i in holdout_idx],
        a_d["real_holdout_indices"])
    chk("real_afe_train_unique", int(len(train_real_idx)),
        a_d["real_afe_train_unique"])
    chk("real_afe_train_repeated", int(len(x_real)), a_d["real_afe_train_repeated"])
    chk("synthetic_hard_negative", int(len(x_hard)), a_d["synthetic_hard_negative"])
    chk("train_total", int(len(x_train)), a_d["train_total"])
    chk("train_abn", int((y_train == 1).sum()), a_d["train_abn"])
    chk("train_norm", int((y_train == 0).sum()), a_d["train_norm"])
    chk("val_total", int(len(x_val)), a_d["val_total"])
    chk("val_abn", int((y_val == 1).sum()), a_d["val_abn"])
    chk("val_norm", int((y_val == 0).sum()), a_d["val_norm"])
    print(f"[DP1] identity anchors PASS ({len(checks)} checks "
          f"vs {ANCHOR_JSON.name})", flush=True)

    # ---------- 教师 tile 特征 ----------
    teacher = build_teacher()
    feats_train = teacher_features(x_train, teacher, torch, "train")
    feats_val = teacher_features(x_val, teacher, torch, "val")
    np.save(OUT_FEATS, np.concatenate([feats_train, feats_val], axis=0))

    # ---------- LORO 读出 → 软伪概率 ----------
    p_train, single_folds = loro_readout(feats_train, y_train, group_ids,
                                         real_synth_code)
    pub = group_ids != real_synth_code
    loro_auc = float(roc_auc_score(y_train[pub], p_train[pub]))
    per_dom_auc = {}
    for tag, code in (("mit_bih", 0), ("incart", 1), ("ptb", 2)):
        m = pub & (dom_train == code)
        if len(np.unique(y_train[m])) == 2:
            per_dom_auc[tag] = float(roc_auc_score(y_train[m], p_train[m]))
    print(f"[DP1] LORO readout AUC={loro_auc:.4f} (public beats) "
          f"per-domain={{{', '.join(f'{k}:{v:.4f}' for k, v in per_dom_auc.items())}}} "
          f"single_class_folds={len(single_folds)}", flush=True)

    swap = False
    real_synth_mean = float(p_train[group_ids == real_synth_code].mean())
    if loro_auc < ABORT_LORO_AUC:
        report = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aborted": True,
            "reason": f"LORO readout AUC {loro_auc:.4f} < {ABORT_LORO_AUC} "
                      "(预注册中止门): 教学信号太弱, 不训练",
            "identity_checks": checks,
            "loro_auc_public": loro_auc,
            "loro_auc_per_domain": per_dom_auc,
            "real_synth_mean_p": real_synth_mean,
        }
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"[DP1] ABORT: LORO AUC {loro_auc:.4f} < {ABORT_LORO_AUC}; "
              f"saved {OUT_JSON.name}, no npz", flush=True)
        raise SystemExit(1)
    if real_synth_mean > SWAP_MEAN_P:
        p_train[group_ids == real_synth_code] = EPS
        swap = True
    print(f"[DP1] real_synth mean_p={real_synth_mean:.4f} "
          f"swap_to_EPS={swap}", flush=True)

    # ---------- val 读出: 单个 LogReg 用全部训练拍拟合 ----------
    lr_all = LogisticRegression(C=1.0, max_iter=2000)
    lr_all.fit(normalize(feats_train), y_train)
    p_val = lr_all.predict_proba(normalize(feats_val))[:, 1].astype(np.float32)

    # ---------- 审计统计 ----------
    def pq(m):
        if not m.any():
            return None
        return {"n": int(m.sum()),
                "mean": float(p_train[m].mean()),
                "p05": float(np.quantile(p_train[m], 0.05)),
                "p50": float(np.quantile(p_train[m], 0.5)),
                "p95": float(np.quantile(p_train[m], 0.95)),
                "entropy_bits_mean": float((
                    -p_train[m] * np.log2(np.clip(p_train[m], 1e-12, 1))
                    - (1 - p_train[m]) * np.log2(
                        np.clip(1 - p_train[m], 1e-12, 1))).mean())}

    audit = {
        "p_dist_train": {
            "pub_abn": pq(pub & (y_train == 1)),
            "pub_norm": pq(pub & (y_train == 0)),
            "real_synth": pq(group_ids == real_synth_code),
        },
        "p_val": {"mean": float(p_val.mean()),
                  "frac_gt_0.5": float((p_val > 0.5).mean())},
    }

    np.savez_compressed(
        OUT_NPZ,
        x_train_perm=x_train.astype(np.float32),
        y_train=y_train.astype(np.int32),
        p_train=p_train.astype(np.float32),
        group_ids=group_ids,
        train_domain=dom_train,
        x_val=x_val,
        y_val=y_val.astype(np.int32),
        p_val=p_val,
        x_real_holdout=x_real_ho.astype(np.float32),
    )

    group_keys = {str(code): (f"{ {0:'mit',1:'incart',2:'ptb'}[d] }:{rid}"
                              if d in (0, 1, 2) else "real_synth")
                  for (d, rid), code in key_to_code.items()}
    group_keys[str(real_synth_code)] = "real_synth"
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aborted": False,
        "purpose": "TH §106 蒸馏试点 Stage 1: 教师 tile 特征 → 训练侧 LORO "
                   "读出软伪概率; 测试集零接触",
        "seed": SEED,
        "anchor_file": str(ANCHOR_JSON.relative_to(BASE)),
        "identity_checks": checks,
        "loro": {
            "protocol": "L2 normalize + LogisticRegression(C=1, max_iter=2000), "
                        "leave-one-record-out (train records only)",
            "auc_public": loro_auc,
            "auc_per_domain": per_dom_auc,
            "n_groups": int(real_synth_code + 1),
            "single_class_folds": single_folds,
            "abort_gate": ABORT_LORO_AUC,
        },
        "real_synth": {
            "n_beats": int((group_ids == real_synth_code).sum()),
            "mean_p_before_swap": real_synth_mean,
            "teacher_target_swap": swap,
            "swap_rule": f"mean_p > {SWAP_MEAN_P} → p=EPS (预注册)",
            "grouping_rationale": "rec_latest.ecgr 无盘上元数据; synth 源拍横跨"
                                  "两源文件; 任何拆分均有软泄漏风险 → 单一伪记录组",
        },
        "val_readout": "单个 LogReg 拟合全部训练拍 → p_val 仅用于 val_loss 打包; "
                       "val 患者与训练患者不相交",
        "audit": audit,
        "outputs": {
            "npz": str(OUT_NPZ.relative_to(BASE)),
            "features_npy": str(OUT_FEATS.relative_to(BASE)),
            "features_layout": f"concat(train {len(feats_train)}, "
                               f"val {len(feats_val)}) axis=0",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[DP1] saved {OUT_NPZ.name}, {OUT_JSON.name}, {OUT_FEATS.name} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
