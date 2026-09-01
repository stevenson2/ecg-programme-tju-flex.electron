#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_ptb_diagnosis_strata.py — PTB 诊断分层可检测性审计
================================================================================
回答两个问题:

  Q1 "Lead II + 逐拍 z-score 是否强人所难?"
     把患者级测试记录按诊断分层 (信息来自每条记录 .hea 文件头注释), 分别记分。
     分层依据: 单导联 II 理论上能否承载该诊断的信号。
       lead2_favorable : 下壁 MI (II/III/aVF Q波+ST)、束支阻滞、心律失常、肥厚
       lead2_unreliable: 前壁/前侧/前间壁/侧壁/后壁 MI、心肌病/心衰、心肌炎、
                         瓣膜病、心绞痛、其他——形态学改变主要落在胸前导联
       control         : 健康对照
     若 Q1 成立, 预期: unfavorable 层的 sensitivity 显著低于 favorable 层。

  Q2 "模型认人还是认病?" (患者身份探针)
     取每条记录的平均异常概率, 做 one-vs-rest 记录识别 (随机洗牌对照)。
     身份探针 AUC 高只说明记录间信号风格有个体差异 (泄漏/过拟合会放大它);
     把它和各诊断层的拍级表现对照阅读。

输入:
  测试拍   : models/deploy_match/ptb_deploy_causal_match.npz (原生 rid=400000+i)
  诊断来源 : ECG-Database/patient*/s*.hea (与 data/preprocess_ptb.py 的
             RECORDS 枚举顺序对齐, rid = 400000 + 枚举序号)

已知限制 (写入结果):
  - 除 exp7c_v4 (PTB 泄漏=0) 外, 全部历史模型的训练集均混入大量测试记录
    (见 clean_test_reeval.json), 其层内数字含记忆成分, 只能做相对比较。
  - 逐拍 z-score 已抹掉绝对幅值 (ST 抬/压的幅度信息), 各层结果均在该表示下测得。
  - 部分层样本极少 (心肌炎/瓣膜病等), 仅供方向性参考。

输出: models/deploy_match/ptb_diagnosis_strata_audit.json
运行环境: WSL (ECG_PROCESSED_DIR 或 /home/devcontainers/ecg_data)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clean_test import (make_predictor, beat_metrics, event_metrics,
                             MODELS, PTB_NPZ)

BASE = Path(__file__).resolve().parent
OUT = MODELS / "deploy_match" / "ptb_diagnosis_strata_audit.json"

PTB_DIR = None
for cand in [
    Path(r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\ECG-Database"),
    Path("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/ECG-Database"),
]:
    if cand.exists():
        PTB_DIR = cand
        break
if PTB_DIR is None:
    raise RuntimeError("ECG-Database 目录未找到")

# 与 clean_test_reeval 一致的双谱系: 板上部署 + 其浮点教师 (均含大量泄漏),
# 加 v4 (PTB 泄漏=0) 作诚实参考。
MODELS_TO_AUDIT = [
    ("DEPLOYED_exp7c_int8", "tflite", "ecg_model_exp7c_int8.tflite"),
    ("exp7c", "h5", "best_resnet_large_exp7c.h5"),
    ("exp7c_v4", "h5", "best_resnet_large_exp7c_v4.h5"),
]

REASON_TO_STRATUM = {
    "myocardial infarction": "mi_pending_loc",   # 再看定位
    "bundle branch block": "lead2_favorable",
    "dysrhythmia": "lead2_favorable",
    "hypertrophy": "lead2_favorable",
    "palpitation": "lead2_favorable",            # 主诉为心悸 → 节律问题
    "healthy control": "control",
    "cardiomyopathy": "lead2_unreliable",
    "heart failure (nyha 2)": "lead2_unreliable",
    "heart failure (nyha 3)": "lead2_unreliable",
    "heart failure (nyha 4)": "lead2_unreliable",
    "myocarditis": "lead2_unreliable",
    "valvular heart disease": "lead2_unreliable",
    "stable angina": "lead2_unreliable",
    "unstable angina": "lead2_unreliable",
}


def parse_hea(hea_path):
    """从 .hea 注释里提取 reason / acute loc / former loc。"""
    reason, acute, former = "n/a", "n/a", "n/a"
    for line in hea_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("# reason for admission:"):
            reason = line.split(":", 1)[1].strip()
        elif low.startswith("# acute infarction (localization):"):
            acute = line.split(":", 1)[1].strip()
        elif low.startswith("# former infarction (localization):"):
            former = line.split(":", 1)[1].strip()
    return reason, acute, former


def classify(reason, acute, former):
    """(诊断三字段) -> (stratum, diagnosis 细类, 依据)。"""
    key = reason.strip().lower()
    if key not in REASON_TO_STRATUM:
        return "lead2_unreliable", reason, "unmapped_reason"
    s = REASON_TO_STRATUM[key]
    if s != "mi_pending_loc":
        return s, reason, "reason"
    # MI: 有下壁成分 → II 可见; 纯前壁系/侧壁/后壁 → II 不可靠;
    # 陈旧性 (acute=no + former 有定位) 按 former 同理。
    loc = acute if acute.lower() not in ("no", "n/a", "unknown", "") else former
    l = loc.strip().lower()
    if l in ("no", "n/a", "unknown", ""):
        return "lead2_unreliable", "MI (localization unknown)", "mi_loc_unknown"
    if "infer" in l:
        return "lead2_favorable", f"MI {loc}", "mi_inferior"
    return "lead2_unreliable", f"MI {loc}", "mi_non_inferior"


def build_record_meta():
    """rid -> {record, patient, reason, stratum, diagnosis, rationale}"""
    from data.preprocess_ptb import load_records
    recs = load_records()
    meta = {}
    for i, rec in enumerate(recs):
        rid = 400000 + i
        patient = rec.split("/")[0]
        hea = PTB_DIR / (rec + ".hea")
        if not hea.exists():
            meta[rid] = {"record": rec, "patient": patient, "reason": "MISSING_HEA",
                         "stratum": "lead2_unreliable",
                         "diagnosis": "MISSING_HEA", "rationale": "missing_file"}
            continue
        reason, acute, former = parse_hea(hea)
        stratum, diag, rat = classify(reason, acute, former)
        meta[rid] = {"record": rec, "patient": patient, "reason": reason,
                     "acute_loc": acute, "former_loc": former,
                     "stratum": stratum, "diagnosis": diag, "rationale": rat}
    return meta


def safe_event_metrics(rids, labels, probs, tag):
    """单类标签层 (如 control 全正常) 的 AUC 无定义, 跳过事件级记分。"""
    if len(np.unique(labels)) < 2:
        return {"skipped": "single-class labels (AUC undefined)"}
    return event_metrics(rids, labels, probs, tag)


def identity_probe(rids, probs, seed=42):
    """每条记录一个特征 (平均概率), one-vs-rest 识别; 返回洗牌校正后 AUC。"""
    uniq = sorted(set(int(r) for r in rids))
    if len(uniq) < 2:
        return None
    feats = np.array([probs[rids == r].mean() for r in uniq])
    aucs, shuffled = [], []
    rng = np.random.RandomState(seed)
    for k in range(len(uniq)):
        y = (np.arange(len(uniq)) == k).astype(int)
        aucs.append(float(roc_auc_score(y, feats)))
        fp = feats.copy()
        rng.shuffle(fp)
        shuffled.append(float(roc_auc_score(y, fp)))
    return {"n_records": len(uniq),
            "raw_mean_ovr_auc": float(np.mean(aucs)),
            "shuffled_mean_ovr_auc": float(np.mean(shuffled)),
            "probe_delta": float(np.mean(aucs) - np.mean(shuffled)),
            "max_ovr_auc": float(np.max(aucs)),
            "worst_ovr_auc": float(np.min(aucs))}


def leakage_memory_probe(rids, probs, leaked_ptb):
    """认人探针 (泄漏记忆): 泄漏进训练的测试记录平均概率是否系统性更高。"""
    uniq = sorted(set(int(r) for r in rids))
    leaked = set(int(r) for r in (leaked_ptb or [])) & set(uniq)
    if not leaked or len(leaked) == len(uniq):
        return {"n_leaked": len(leaked), "n_total": len(uniq),
                "probe": "all_or_none_leaked" if uniq else "empty"}
    feats = np.array([probs[rids == r].mean() for r in uniq])
    y = np.array([int(r in leaked) for r in uniq])
    out = {"n_leaked": len(leaked), "n_total": len(uniq),
           "mean_prob_leaked": float(feats[y == 1].mean()),
           "mean_prob_clean": float(feats[y == 0].mean()),
           "leak_vs_clean_auc": float(roc_auc_score(y, feats))}
    out["memory_gap"] = out["mean_prob_leaked"] - out["mean_prob_clean"]
    return out


def evaluate_model(name, kind, fname, beats, labels, rids, meta, leaked_ptb):
    predict = make_predictor(kind, MODELS / fname)
    p = predict(beats)
    leaked = set(int(r) for r in (leaked_ptb or []))
    entry = {"file": fname, "kind": kind,
             "n_leaked_test_records": len(leaked & set(int(r) for r in rids)),
             "overall_beat": beat_metrics(labels, p),
             "overall_event": event_metrics(rids, labels, p, "ptb_test_strata"),
             "strata": {}, "diagnoses": {}}
    rids_i = rids.astype(int)

    for stratum in ("lead2_favorable", "lead2_unreliable", "control"):
        rs = sorted(r for r, m in meta.items() if m["stratum"] == stratum
                    and r in set(rids_i.tolist()))
        if not rs:
            continue
        mask = np.isin(rids_i, rs)
        sl, sp = labels[mask], p[mask]
        st = {"n_records": len(rs),
              "records": rs,
              "n_patients": len({meta[r]["patient"] for r in rs}),
              "beat": beat_metrics(sl, sp),
              "event": safe_event_metrics(rids_i[mask], sl, sp, f"ptb_{stratum}")}
        entry["strata"][stratum] = st

    diag_groups = {}
    for r, m in meta.items():
        if r in set(rids_i.tolist()):
            diag_groups.setdefault(m["diagnosis"], []).append(r)
    for diag, rs in sorted(diag_groups.items()):
        mask = np.isin(rids_i, rs)
        entry["diagnoses"][diag] = {
            "n_records": len(rs), "records": rs,
            "n_leaked_records": len(leaked & set(rs)),
            "stratum": meta[rs[0]]["stratum"],
            "beat": beat_metrics(labels[mask], p[mask]),
        }

    entry["identity_probe"] = identity_probe(rids, p)
    entry["leakage_memory_probe"] = leakage_memory_probe(rids, p, leaked_ptb)
    return entry


def main():
    t0 = time.time()
    d = np.load(PTB_NPZ)
    beats = np.asarray(d["beats"], dtype=np.float32)
    labels = np.asarray(d["labels"]).astype(np.int32)
    rids = np.asarray(d["record_ids"]).astype(np.int64)
    print(f"[STRATA] PTB test beats={len(beats)} "
          f"(abn={int((labels==1).sum())}, records={len(np.unique(rids))})", flush=True)

    meta = build_record_meta()
    test_rids = sorted(set(int(r) for r in rids))
    missing = [r for r in test_rids if r not in meta]
    assert not missing, f"rid 无法映射: {missing[:10]}"

    stratum_counts, diag_counts = {}, {}
    for r in test_rids:
        stratum_counts[meta[r]["stratum"]] = stratum_counts.get(meta[r]["stratum"], 0) + 1
        diag_counts[meta[r]["diagnosis"]] = diag_counts.get(meta[r]["diagnosis"], 0) + 1
    print("[STRATA] strata:", stratum_counts, flush=True)
    for diag, c in sorted(diag_counts.items(), key=lambda kv: -kv[1]):
        print(f"        {diag}: {c}", flush=True)

    reeval = json.loads((MODELS / "deploy_match" / "clean_test_reeval.json")
                        .read_text(encoding="utf-8"))

    results = {}
    for name, kind, fname in MODELS_TO_AUDIT:
        if not (MODELS / fname).exists():
            print(f"[STRATA] SKIP {name}: {fname} 不存在", flush=True)
            continue
        print(f"[STRATA] {name} ...", flush=True)
        leaked_ptb = (reeval["models"].get(name, {}).get("leaked_test_records") or {}).get("ptb")
        entry = evaluate_model(name, kind, fname, beats, labels, rids, meta, leaked_ptb)
        results[name] = entry
        for s, st in entry["strata"].items():
            b = st["beat"]
            sens = f"{b['sensitivity']:.3f}" if b["sensitivity"] is not None else "None"
            print(f"        {s}: records={st['n_records']} sens@0.5={sens} "
                  f"n={b['n_beats']}", flush=True)
        lp = entry["leakage_memory_probe"]
        if "leak_vs_clean_auc" in lp:
            print(f"        leakage probe: leaked={lp['n_leaked']}/{lp['n_total']} "
                  f"meanP leaked={lp['mean_prob_leaked']:.3f} vs "
                  f"clean={lp['mean_prob_clean']:.3f} "
                  f"AUC={lp['leak_vs_clean_auc']:.3f}", flush=True)
        else:
            print(f"        leakage probe: {lp}", flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "PTB 诊断分层可检测性审计: Lead II 信息上限 + 认人/认病探针",
        "reference_policy": {"theta": 0.5, "policy": "1-of-5", "cooldown": 5},
        "test_set": {"beats": int(len(beats)), "records": test_rids,
                     "strata_record_counts": stratum_counts,
                     "diagnosis_record_counts": diag_counts},
        "strata_definition": {
            "lead2_favorable": "下壁 MI / 束支阻滞 / 心律失常 / 肥厚 / 心悸 — "
                               "单导联 II 理论上有承载信号",
            "lead2_unreliable": "前壁系/侧壁/后壁 MI、心肌病/心衰、心肌炎、瓣膜病、"
                                "心绞痛、定位未知 — 形态学改变主要在胸前导联",
            "control": "健康对照",
        },
        "caveats": [
            "除 exp7c_v4 (PTB 泄漏=0) 外, 其余模型训练集均混入大量测试记录, "
            "层内数字含记忆成分, 只能做相对比较。",
            "PTB 为记录级标签: 每个诊断层内标签单类 (层内 AUC 无定义), "
            "判别力读法 = 各层 sensitivity + control 层 FP, 不是层内 AUC。",
            "泄漏模型的按诊断 sensitivity 模式被'哪些记录被背下来'污染, "
            "与临床 Lead II 可检测性无必然对应; 以 v4 为最接近诚实的参照。",
            "逐拍 z-score 抹掉绝对幅值 (ST 抬/压幅度), 所有层结果均在该表示下测得。",
            "部分层样本极少 (心肌炎/瓣膜病/心绞痛), 仅供方向性参考。",
        ],
        "record_meta": {str(r): {k: meta[r][k] for k in
                                 ("record", "patient", "reason", "stratum",
                                  "diagnosis", "rationale")}
                        for r in test_rids},
        "models": results,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[STRATA] saved {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
