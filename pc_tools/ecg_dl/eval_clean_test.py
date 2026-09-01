#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_clean_test.py — 干净测试集重评：exp7c 家族全模型双口径记分表
================================================================================
背景: §97 审计判定 12 个历史训练脚本全部混入测试患者。本脚本把当前谱系的
全部候选模型拉到患者级测试集上重评, 并给出两种口径:

  full   : 完整测试集 (与 §96 同口径; 对泄漏模型偏乐观)
  clean  : 从测试集中剔除曾被抽进该模型训练的记录后的子集
           (泄漏记录由 audit_provenance.replicate 按原始 RNG 流复放重建;
            exp7c_v4 的 INCART 泄漏按 v4 掩码错位审计重建)

测试集:
  MIT+INCART: models/deploy_match/mit_deploy_causal_match.npz (患者级测试拍,
              INCART rid 带 +100000 偏移)
  PTB       : models/deploy_match/ptb_deploy_causal_match.npz (原生 rid)

指标: 拍级 AUC / θ=0.5 混淆矩阵 / Sens / Prec / F1; 事件级 θ=0.5 1-of-5
cooldown=5 (与 §96 参考操作点一致)。

已知限制 (写入结果):
  - clean 子集只剔除"本次训练脚本"见过的测试记录; 祖先模型 (exp7/exp7b)
    未审计, 若祖先亦见过同一记录, clean 数字仍偏乐观。
  - 板上部署 INT8 (ecg_model_exp7c_int8.tflite) 按权重谱系 (finetune_exp7c)
    剔除; 其 PTQ 校准集出处未审计。

输出: models/deploy_match/clean_test_reeval.json
运行环境: WSL (ECG_PROCESSED_DIR 或 /home/devcontainers/ecg_data)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.split_guard import get_guard
import audit_provenance as ap
from eval_exp7c_policy_sweep import evaluate_sequence_set, DEFAULT_GT_GAP
import eval_exp7c_policy_sweep as pol

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
MIT_NPZ = CACHE / "mit_deploy_causal_match.npz"
PTB_NPZ = CACHE / "ptb_deploy_causal_match.npz"
OUT = CACHE / "clean_test_reeval.json"

REF_THETA = 0.50
REF_POLICY = (1, 5)
REF_COOLDOWN = 5

SPEC_A = [("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 100)]
SPEC_B = [("mit_bih", 800, 200), ("incart", 200, 100), ("ptb", 300, 100)]

# 训练脚本 → (抽样规格, 抽样前 RNG 消耗)。同规格只复放一次。
SCRIPT_SPECS = {
    "finetune_exp7c.py": (SPEC_A, None),
    "finetune_exp7c_mild.py": (SPEC_A, None),
    "finetune_exp7c_hardneg.py": (SPEC_A, "synth_hard"),
    "finetune_exp7c_ecgfounder.py": (SPEC_B, None),
    "finetune_exp7c_ecgfounder_v2.py": (SPEC_B, None),
    "finetune_exp7c_ecgfounder_v3.py": (SPEC_B, None),
    "finetune_exp7c_ecgfounder_v4.py": (SPEC_B, None),
    "qat_exp7c.py": ([("mit_bih", 1000, 300), ("incart", 300, 100), ("ptb", 500, 100)], None),
    "qat_exp7c_v3.py": ([("mit_bih", 800, 200), ("incart", 200, 100), ("ptb", 800, 200)], None),
    "qat_exp7c_v3b.py": ([("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 150)], None),
    "qat_exp7c_v4.py": ([("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 150)], None),
    "qat_exp7c_v5.py": ([("mit_bih", 1200, 400), ("incart", 300, 100), ("ptb", 500, 150)], None),
}

# (名称, 类型, 模型文件, 出身脚本或特殊标记)
REGISTRY = [
    ("exp7c", "h5", "best_resnet_large_exp7c.h5", "finetune_exp7c.py"),
    ("exp7c_v2_hardneg", "h5", "best_resnet_large_exp7c_v2.h5", "finetune_exp7c_hardneg.py"),
    ("exp7c_v3_mild", "h5", "best_resnet_large_exp7c_v3.h5", "finetune_exp7c_mild.py"),
    ("exp7c_v4", "h5", "best_resnet_large_exp7c_v4.h5", "__v4_mask__"),
    ("exp7c_ecgfounder", "h5", "best_resnet_large_exp7c_ecgfounder.h5", "finetune_exp7c_ecgfounder.py"),
    ("exp7c_ecgfounder_v2", "h5", "best_resnet_large_exp7c_ecgfounder_v2.h5", "finetune_exp7c_ecgfounder_v2.py"),
    ("exp7c_ecgfounder_v3", "h5", "best_resnet_large_exp7c_ecgfounder_v3.h5", "finetune_exp7c_ecgfounder_v3.py"),
    ("exp7c_ecgfounder_v4", "h5", "best_resnet_large_exp7c_ecgfounder_v4.h5", "finetune_exp7c_ecgfounder_v4.py"),
    ("DEPLOYED_exp7c_int8", "tflite", "ecg_model_exp7c_int8.tflite", "finetune_exp7c.py"),
    ("exp7c_qat_int8", "tflite", "ecg_model_exp7c_qat_int8.tflite", "qat_exp7c.py"),
    ("ecgfounder_v3_qat_int8", "tflite", "ecg_model_exp7c_ecgfounder_v3_qat_int8.tflite", "qat_exp7c_v3.py"),
    ("ecgfounder_v3b_qat_int8", "tflite", "ecg_model_exp7c_ecgfounder_v3b_qat_int8.tflite", "qat_exp7c_v3b.py"),
    ("ecgfounder_v4_qat_int8", "tflite", "ecg_model_exp7c_ecgfounder_v4_qat_int8.tflite", "qat_exp7c_v4.py"),
    ("ecgfounder_v5_qat_int8", "tflite", "ecg_model_exp7c_ecgfounder_v5_qat_int8.tflite", "qat_exp7c_v5.py"),
]


def leaked_test_records():
    """每个出身脚本见过的、且属于测试患者的记录 (原生 rid, 按 tag)。"""
    test_rids = {tag: set(np.asarray(get_guard(tag).test_record_ids()).tolist())
                 for tag in ("mit_bih", "incart", "ptb")}
    seen_specs, out = {}, {}
    for script, (specs, pre) in SCRIPT_SPECS.items():
        key = (tuple(tuple(s) for s in specs), pre)
        if key not in seen_specs:
            picked = ap.replicate(script, specs, pre)
            seen_specs[key] = {tag: sorted(set(np.asarray(sel).tolist()) & test_rids[tag])
                               for tag, sel in picked.items()}
        out[script] = seen_specs[key]
    # exp7c_v4: INCART 掩码错位放行的测试/验证记录 → 测试部分才是泄漏
    v4 = ap.audit_v4_alignment()
    out["__v4_mask__"] = {
        "mit_bih": [],
        "incart": [int(x) for x in v4["inc_test_records_passing_filter"]],
        "ptb": [],
    }
    return out


def make_predictor(kind, path):
    if kind == "h5":
        model = tf.keras.models.load_model(str(path), compile=False)

        def predict(beats):
            return model.predict(beats[..., np.newaxis].astype(np.float32),
                                 batch_size=512, verbose=0)[:, 1]
        return predict

    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    in_d = it.get_input_details()[0]
    out_d = it.get_output_details()[0]
    is_int8 = in_d["dtype"] == np.int8
    in_scale = float(in_d["quantization_parameters"]["scales"].flatten()[0]) if is_int8 else 1.0
    in_zp = int(in_d["quantization_parameters"]["zero_points"].flatten()[0]) if is_int8 else 0
    out_scale = float(out_d["quantization_parameters"]["scales"].flatten()[0])
    out_zp = int(out_d["quantization_parameters"]["zero_points"].flatten()[0])
    out_shape = out_d["shape"]          # 判断输出维度 (2 → 概率对)
    CHUNK = 8192

    def predict(beats):
        n = len(beats)
        probs = np.zeros(n, dtype=np.float32)
        for s in range(0, n, CHUNK):
            x = beats[s:s + CHUNK][..., np.newaxis].astype(np.float32)
            if is_int8:
                x = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
            it.resize_tensor_input(in_d["index"], [len(x), 250, 1], strict=False)
            it.allocate_tensors()
            it.set_tensor(in_d["index"], x)
            it.invoke()
            y = it.get_tensor(out_d["index"]).astype(np.float32)
            if out_d["dtype"] == np.int8 or out_d["dtype"] == np.uint8:
                y = (y - out_zp) * out_scale
            if y.shape[-1] >= 2:
                probs[s:s + CHUNK] = y[:, 1]     # single-softmax 语义
            else:
                probs[s:s + CHUNK] = y[:, 0]
        return probs
    return predict


def beat_metrics(labels, probs):
    pred = (probs > REF_THETA).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = 2 * sens * prec / (sens + prec) if (sens and prec) else None
    out = {"n_beats": int(len(labels)), "n_abnormal": int((labels == 1).sum()),
           "theta": REF_THETA, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "sensitivity": sens, "precision": prec, "f1": f1}
    if len(np.unique(labels)) == 2:
        out["auc"] = float(roc_auc_score(labels, probs))
    else:
        out["auc"] = None
    return out


def event_metrics(rids, labels, probs, tag):
    symbols = np.full(len(labels), "U", dtype=object)
    pol.THETAS = [REF_THETA]
    pol.POLICIES = [REF_POLICY]
    rows = evaluate_sequence_set(tag, rids, labels, probs, symbols,
                                 DEFAULT_GT_GAP, REF_COOLDOWN)
    if not rows:
        return {}
    r = rows[0]
    return {k: r.get(k) for k in (
        "event_recall", "event_precision", "event_f1", "fp_per_record",
        "fp_per_1000_beats", "alert_rate", "gt_events", "alert_blocks",
        "false_alarm_blocks", "matched_gt_events", "matched_pred_blocks")}


def subset_mask(rids, leaked_mi, leaked_ptb, domain):
    """domain='mit_incart' 时 rids 为合并偏移空间; 'ptb' 为原生空间。"""
    if domain == "mit_incart":
        lk_mit = set(leaked_mi["mit_bih"])
        lk_inc = set(r + 100000 for r in leaked_mi["incart"])
        keep = np.ones(len(rids), dtype=bool)
        for i, r in enumerate(rids):
            r = int(r)
            if (r < 100000 and r in lk_mit) or (r >= 100000 and r in lk_inc):
                keep[i] = False
        return keep
    lk = set(leaked_ptb["ptb"])
    return np.array([int(r) not in lk for r in rids], dtype=bool)


def main():
    t0 = time.time()
    dmi = np.load(MIT_NPZ)
    dpt = np.load(PTB_NPZ)
    mi_beats, mi_labels, mi_rids = (np.asarray(dmi["beats"], dtype=np.float32),
                                    np.asarray(dmi["labels"]).astype(np.int32),
                                    np.asarray(dmi["record_ids"]))
    pt_beats, pt_labels, pt_rids = (np.asarray(dpt["beats"], dtype=np.float32),
                                    np.asarray(dpt["labels"]).astype(np.int32),
                                    np.asarray(dpt["record_ids"]))
    print(f"[CLEAN] MIT+INCART test beats={len(mi_beats)} "
          f"(abn={int((mi_labels==1).sum())}, records={len(np.unique(mi_rids))})", flush=True)
    print(f"[CLEAN] PTB test beats={len(pt_beats)} "
          f"(abn={int((pt_labels==1).sum())}, records={len(np.unique(pt_rids))})", flush=True)

    leaks = leaked_test_records()
    print("[CLEAN] leaked test records per training script:", flush=True)
    for s, m in leaks.items():
        print(f"        {s}: mit={len(m['mit_bih'])} inc={len(m['incart'])} "
              f"ptb={len(m['ptb'])}", flush=True)

    results = {}
    for name, kind, fname, provenance in REGISTRY:
        path = MODELS / fname
        if not path.exists():
            print(f"[CLEAN] SKIP {name}: {fname} 不存在", flush=True)
            continue
        print(f"[CLEAN] {name} ({fname}) ...", flush=True)
        predict = make_predictor(kind, path)

        p_mi = predict(mi_beats)
        p_pt = predict(pt_beats)

        entry = {"file": fname, "kind": kind, "provenance": provenance,
                 "leaked_test_records": leaks.get(provenance)}
        for domain, beats, labels, rids, p, lk in (
                ("mit_incart", mi_beats, mi_labels, mi_rids, p_mi, leaks.get(provenance)),
                ("ptb", pt_beats, pt_labels, pt_rids, p_pt, leaks.get(provenance))):
            full = {"beat": beat_metrics(labels, p),
                    "event": event_metrics(rids, labels, p, f"{domain}_test_full")}
            if lk is not None:
                keep = subset_mask(rids, lk, lk, domain) if domain == "mit_incart" \
                    else subset_mask(rids, None, lk, domain)
                n_drop = int((~keep).sum())
                if keep.sum() and len(np.unique(rids[keep])) > 0:
                    clean = {"beat": beat_metrics(labels[keep], p[keep]),
                             "event": event_metrics(rids[keep], labels[keep], p[keep],
                                                    f"{domain}_test_clean")}
                else:
                    clean = {"error": "clean subset empty"}
                clean["dropped_beats"] = n_drop
                clean["dropped_records"] = sorted(int(r) for r in np.unique(rids[~keep]))
            else:
                clean = {"error": "no provenance mapping"}
            entry[domain] = {"full": full, "clean": clean}
        results[name] = entry

        mi_f = entry["mit_incart"]["full"]["beat"]
        mi_c = entry["mit_incart"]["clean"]["beat"]
        pt_f = entry["ptb"]["full"]["beat"]
        pt_c = entry["ptb"]["clean"]["beat"]
        def g(d, k):
            v = d.get(k) if isinstance(d, dict) else None
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"        MIT+INCART AUC full={g(mi_f,'auc')} clean={g(mi_c,'auc')} | "
              f"F1 full={g(mi_f,'f1')} clean={g(mi_c,'f1')}", flush=True)
        print(f"        PTB        AUC full={g(pt_f,'auc')} clean={g(pt_c,'auc')} | "
              f"F1 full={g(pt_f,'f1')} clean={g(pt_c,'f1')}", flush=True)

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "§97 泄漏审计后的干净测试集重评 (full vs clean 双口径)",
        "reference_policy": {"theta": REF_THETA, "policy": "1-of-5",
                             "cooldown": REF_COOLDOWN, "gt_gap": DEFAULT_GT_GAP},
        "test_sets": {
            "mit_incart": {"beats": int(len(mi_beats)),
                           "records": [int(r) for r in np.unique(mi_rids)]},
            "ptb": {"beats": int(len(pt_beats)),
                    "records": [int(r) for r in np.unique(pt_rids)]},
        },
        "caveats": [
            "clean 子集仅剔除本训练脚本见过的测试记录; 祖先谱系 (exp7/exp7b) 未审计。",
            "DEPLOYED_exp7c_int8 按权重谱系 (finetune_exp7c.py) 剔除; 其 PTQ 校准集出处未审计。",
            "INT8 概率采用 single-softmax 语义 (反量化后直接取 p[:,1])。",
        ],
        "models": results,
    }

    def _native(o):
        if isinstance(o, dict):
            return {k: _native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_native(v) for v in o]
        if isinstance(o, np.ndarray):
            return _native(o.tolist())
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    OUT.write_text(json.dumps(_native(report), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[CLEAN] saved {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
