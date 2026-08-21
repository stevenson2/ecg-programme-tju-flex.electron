#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ptbxl_record_level.py — PTB-XL 记录级验证（完全复刻板上部署链）

与普通 PC 端推理的区别：
  本脚本严格按 ESP32 固件（src/main.cpp + src/filter/filter.cpp +
  src/ai_inference/ai_inference.cpp + tflite_settings.h）的部署链路处理信号：

    原始信号(mV数值)
      → 去偏置 noisyNoDC = x - 1.65
      → 双级 10 抽头梳状滤波 (50Hz/100Hz 陷零)
      → AI 输入链: HP 0.05Hz(fs=500) + LP 40Hz(fs=500)   [因果 IIR, 固件系数]
      → 2:1 抽取 (500Hz→250Hz)
      → AI 链: HP 0.5Hz(fs=250)                           [因果 IIR, 固件系数]
      → 250 点窗口 = 1.0s, 步进 250 = 1.0s, 触发偏移 6
      → 窗口 Z-score
      → INT8 量化 (round(x/scale)+zp, clip [-128,127])
      → TFLite 推理
      → 输出反量化取异常类概率 (不做二次 softmax, 与固件一致)

标签定义（--superclass）：
  NORM : 正常（负类基线）
  MI   : IMI/ASMI/ILMI/AMI/ALMI/INJAS/LMI/INJAL/IPLMI/IPMI/INJIN/INJLA/PMI/INJIL
  STTC : NDT/NST_/DIG/LNGQT/ISC_/ISCAL/ISCIN/ISCIL/ISCAS/ISCLA/ANEUR/EL/ISCAN
  CD   : LAFB/IRBBB/1AVB/IVCD/CRBBB/CLBBB/LPFB/WPW/ILBBB/3AVB/2AVB
  HYP  : LVH/LAO-LAE/RVH/RAO-RAE/SEHYP
  abnormal : 任意非 NORM/SR 诊断码（默认）

用法（WSL）：
  python3 pc_tools/ecg_dl/eval_ptbxl_record_level.py --superclass MI
  python3 pc_tools/ecg_dl/eval_ptbxl_record_level.py --superclass AFIB --aggregate mean
  python3 pc_tools/ecg_dl/eval_ptbxl_record_level.py --n-max 100
  python3 pc_tools/ecg_dl/eval_ptbxl_record_level.py --superclass MI --negative abnormal \
      --aggregate mean --threshold-sweep --tag mi_vs_abnormal

输出：
  pc_tools/ecg_dl/models/ptbxl_record_level_eval.json
"""

import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # 项目根
REPO = Path(__file__).resolve().parent          # pc_tools/ecg_dl
PTBXL_DIR = ROOT / "PTB-XL_ECG"
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"
SCP_CSV = PTBXL_DIR / "scp_statements.csv"
MODELS = REPO / "models"
OUT_JSON = MODELS / "ptbxl_record_level_eval.json"

FS = 500
REC_LEN = 5000          # 10s @500Hz
N_LEADS = 12

# 正常记录允许出现的 SCP 键（负类基线）
NORMAL_KEYS = {"NORM", "SR"}

# ---------------- 板上部署链固件系数（来自 src/filter/filter.cpp） ----------------
# AI 输入链 HP 0.05Hz @500Hz (二阶 Butterworth, Direct Form II Transposed)
AI_CHAIN_HP = {
    "a1": -1.9991114234707954, "a2": 0.9991118180796384,
    "b0": 0.9995558103876084, "b1": -1.9991116207752169, "b2": 0.9995558103876084,
}
# AI 输入链 LP 40Hz @500Hz
AI_CHAIN_LP = {
    "a1": -1.3072850288493234, "a2": 0.4918122372225752,
    "b0": 0.046131802093312926, "b1": 0.09226360418662585, "b2": 0.046131802093312926,
}
# AI 链 HP 0.5Hz @250Hz (抽取后)
AI_HP = {
    "a1": -1.9822289297925284, "a2": 0.98238545061412508,
    "b0": 0.99115359510166301, "b1": -1.982307190203326, "b2": 0.99115359510166301,
}

# 部署参数（来自 include/ai_inference/tflite_settings.h）
DECIMATION = 2
WINDOW = 250
STRIDE = 250
TRIGGER_OFFSET = 6
DC_OFFSET_REMOVE = 1.65
COMB_TAPS = 10


def load_scp_classes():
    """读取 scp_statements.csv，返回 SCP 码 -> diagnostic_class 映射。"""
    mapping = {}
    with open(SCP_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r[""].strip()
            cls = r.get("diagnostic_class", "").strip()
            if code and cls:
                mapping[code] = cls
    return mapping


def parse_scp(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def is_normal_record(scp: dict) -> bool:
    """负类定义：scp_codes 键集 ⊆ {NORM, SR}。"""
    return set(scp.keys()) <= NORMAL_KEYS


def superclass_of(scp: dict, mapping: dict) -> set:
    """返回记录包含的诊断大类集合（如 {'MI','STTC'}）。"""
    cls = set()
    for code in scp.keys():
        c = mapping.get(code)
        if c and c != "NORM":
            cls.add(c)
    return cls


def classify(scp: dict, positive: str, mapping: dict) -> bool:
    if positive == "abnormal":
        return not is_normal_record(scp)
    if positive in ("MI", "STTC", "CD", "HYP"):
        return positive in superclass_of(scp, mapping)
    # 否则当作具体 SCP 码
    return positive in scp


# ---------------- 板上部署链处理器（纯 Python 复刻 C++ 实现） ----------------

class Biquad:
    """Direct Form II Transposed，状态跨样本持续（与固件 applyBiquad 一致）。"""
    def __init__(self, coef):
        self.b0 = coef["b0"]; self.b1 = coef["b1"]; self.b2 = coef["b2"]
        self.a1 = coef["a1"]; self.a2 = coef["a2"]
        self.w1 = 0.0; self.w2 = 0.0

    def process(self, x):
        y = self.b0 * x + self.w1
        self.w1 = self.b1 * x - self.a1 * y + self.w2
        self.w2 = self.b2 * x - self.a2 * y
        return y


class CombFilter:
    """双级 10 抽头滑动平均（50Hz/100Hz 陷零），与固件 applyCombFilter 一致。"""
    def __init__(self, taps=COMB_TAPS):
        self.taps = taps
        self.buf1 = np.zeros(taps, dtype=np.float64)
        self.idx1 = 0; self.sum1 = 0.0
        self.buf2 = np.zeros(taps, dtype=np.float64)
        self.idx2 = 0; self.sum2 = 0.0

    def process(self, x):
        # 第一级
        self.sum1 -= self.buf1[self.idx1]
        self.buf1[self.idx1] = x
        self.sum1 += x
        self.idx1 = (self.idx1 + 1) % self.taps
        y1 = self.sum1 / self.taps
        # 第二级
        self.sum2 -= self.buf2[self.idx2]
        self.buf2[self.idx2] = y1
        self.sum2 += y1
        self.idx2 = (self.idx2 + 1) % self.taps
        return self.sum2 / self.taps


def deploy_preprocess(x500):
    """完整复刻板上 AI 输入链，返回抽取后 250Hz 的滤波序列（float64）。"""
    comb = CombFilter()
    hp05 = Biquad(AI_CHAIN_HP)   # HP 0.05 @500
    lp40 = Biquad(AI_CHAIN_LP)   # LP 40 @500
    ai_hp = Biquad(AI_HP)        # HP 0.5 @250（抽取后流式）

    out = []
    decim_ctr = 0
    for raw in x500:
        # 固件：noisyNoDC = raw - DC_OFFSET_REMOVE
        v = raw - DC_OFFSET_REMOVE
        # 梳状
        v = comb.process(v)
        # AI 输入链：HP0.05 + LP40（在抽取前）
        v = hp05.process(v)
        v = lp40.process(v)
        # 2:1 抽取
        if (decim_ctr % DECIMATION) != 0:
            decim_ctr += 1
            continue
        decim_ctr += 1
        # 抽取后 250Hz 流上的因果 0.5Hz HP
        v = ai_hp.process(v)
        out.append(v)
    return np.asarray(out, dtype=np.float64)


def int8_quantize(x, scale, zp):
    """与固件 fill_input_tensor 一致：round(x/scale + 0.5) + zp, clip [-128,127]。"""
    q = np.floor(x / scale + 0.5).astype(np.int64) + int(zp)
    return np.clip(q, -128, 127).astype(np.int8)


def run_deploy_inference(interpreter, x250):
    """按固件窗口逻辑推理，返回窗口异常概率列表。

    触发点：buffer_idx % STRIDE == TRIGGER_OFFSET 且 buffer_idx >= WINDOW。
    等价于窗口起点 start = TRIGGER_OFFSET + k*STRIDE，取 x250[start-WINDOW+1 : start+1]。
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale = float(np.asarray(input_details["quantization_parameters"]["scales"]).flatten()[0])
    in_zp = int(np.asarray(input_details["quantization_parameters"]["zero_points"]).flatten()[0])
    out_scale = float(np.asarray(output_details["quantization_parameters"]["scales"]).flatten()[0])
    out_zp = int(np.asarray(output_details["quantization_parameters"]["zero_points"]).flatten()[0])

    probs = []
    n = len(x250)
    # 触发点：6, 256, 506, ...
    trigger = TRIGGER_OFFSET
    while trigger < n:
        start = trigger - WINDOW + 1
        if start < 0:
            trigger += STRIDE
            continue
        window = x250[start:trigger + 1]
        if len(window) < WINDOW:
            break
        # Z-score（固件 preprocess_samples：用总体标准差）
        mean = float(np.mean(window))
        std = float(np.sqrt(np.mean((window - mean) ** 2)))
        if std < 1e-6:
            std = 1.0
        norm = (window - mean) / std
        # INT8 量化
        q = int8_quantize(norm, in_scale, in_zp).reshape(1, WINDOW, 1)
        # 推理
        interpreter.set_tensor(input_details["index"], q)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])[0]
        # 固件：反量化取异常类（index=1）概率，不做二次 softmax
        if output_details["dtype"] in (np.int8, np.uint8):
            p_abn = float((out[1].astype(np.float32) - out_zp) * out_scale)
        else:
            p_abn = float(out[1])
        probs.append(max(0.0, min(1.0, p_abn)))
        trigger += STRIDE
    return np.asarray(probs, dtype=np.float64)


def aggregate(probs, method):
    if method == "max":
        return float(probs.max()) if probs.size else 0.0
    if method == "mean":
        return float(probs.mean()) if probs.size else 0.0
    if method == "p95":
        return float(np.percentile(probs, 95)) if probs.size else 0.0
    if method == "abn_ratio":
        return float((probs >= 0.6).mean()) if probs.size else 0.0
    raise ValueError(f"unknown aggregate: {method}")


def load_lead(path, lead_idx=1):
    """读 records500 .dat（12 导联交错 int16, 500Hz）单导联，返回 mV 数值数组。"""
    raw = np.fromfile(path, dtype="<i2")
    sig = raw.reshape(-1, N_LEADS)[:, lead_idx].astype(np.float64) / 1000.0
    return sig[:REC_LEN]


def confusion(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main():
    ap = argparse.ArgumentParser(description="PTB-XL 记录级验证（板上部署链）")
    ap.add_argument("--model", type=str,
                    default=str(MODELS / "ecg_model_exp7c_int8.tflite"),
                    help="TFLite 模型路径（默认 exp7c INT8，与固件一致）")
    ap.add_argument("--lead", type=int, default=1,
                    help="导联索引 0-11（0=I, 1=II, 2=III...）")
    ap.add_argument("--n-max", type=int, default=0,
                    help="每类最多记录数（调试用，0=全部）")
    ap.add_argument("--aggregate", choices=["max", "mean", "p95", "abn_ratio"],
                    default="max", help="窗口概率→记录级分数方法")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="记录级判定阈值（默认 0.6，与固件 INFERENCE_THRESHOLD 一致）")
    ap.add_argument("--superclass", type=str, default="abnormal",
                    choices=["abnormal", "NORM", "MI", "STTC", "CD", "HYP"],
                    help="阳性诊断大类（abnormal=任意异常；或 MI/STTC/CD/HYP）")
    ap.add_argument("--positive", type=str, default=None,
                    help="（可选）直接指定 SCP 码，如 AFIB/PVC/IMI；优先于 --superclass")
    ap.add_argument("--negative", type=str, default="normal",
                    choices=["normal", "abnormal"],
                    help="负类定义：normal=健康对照（默认）；abnormal=排除阳性后的其他异常记录")
    ap.add_argument("--threshold-sweep", action="store_true",
                    help="对记录级分数做全阈值扫描，输出 Se/Sp/PPV/F1 表和最优操作点")
    ap.add_argument("--sweep-step", type=float, default=0.01,
                    help="阈值扫描步长（默认 0.01，范围 0.01~0.99）")
    ap.add_argument("--save-scores", action="store_true",
                    help="保存每条记录的原始分数/标签/patient_id 到 npz，"
                         "供嵌套阈值选择等离线分析（避免重复推理）")
    ap.add_argument("--tag", default="", help="输出 JSON 文件名后缀")
    args = ap.parse_args()

    if not PTBXL_CSV.exists():
        print(f"[ERROR] PTB-XL 数据库不存在: {PTBXL_CSV}")
        sys.exit(1)

    mapping = load_scp_classes() if SCP_CSV.exists() else {}
    print(f"SCP→class 映射载入: {len(mapping)} 条")

    import tensorflow as tf
    print(f"[Inference] Loading TFLite model: {args.model}")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    print(f"[Inference] Input: {in_d['shape']} dtype={in_d['dtype']}")
    print(f"[Inference] Output: {out_d['shape']} dtype={out_d['dtype']}")

    rows = list(csv.DictReader(open(PTBXL_CSV)))
    rows = [r for r in rows if r["validated_by_human"] == "True"]
    print(f"记录总数（validated）: {len(rows)}")

    positive = args.positive or args.superclass
    pos = [r for r in rows if classify(parse_scp(r["scp_codes"]), positive, mapping)]
    if args.negative == "normal":
        neg = [r for r in rows if not classify(parse_scp(r["scp_codes"]), positive, mapping)
               and is_normal_record(parse_scp(r["scp_codes"]))]
        neg_desc = "正常"
    elif args.negative == "abnormal":
        if positive == "abnormal":
            raise SystemExit("[ERROR] --negative abnormal 在 positive=abnormal 下无意义：负类需排除阳性")
        neg = [r for r in rows if not classify(parse_scp(r["scp_codes"]), positive, mapping)
               and not is_normal_record(parse_scp(r["scp_codes"]))]
        neg_desc = "排除阳性后的其他异常"
    else:
        raise SystemExit(f"[ERROR] unknown --negative: {args.negative}")
    print(f"正类({positive}): {len(pos)}  负类({neg_desc}): {len(neg)}")

    n_pos = len(pos) if args.n_max == 0 else min(args.n_max, len(pos))
    n_neg = len(neg) if args.n_max == 0 else min(args.n_max, len(neg))
    print(f"实际采样: 正 {n_pos} / 负 {n_neg}（Lead {args.lead+1}, 部署链）")

    t0 = time.time()
    y_true, scores, patient_ids, fnames = [], [], [], []

    def process(rec):
        fname = PTBXL_DIR / (rec["filename_hr"] + ".dat")
        x = load_lead(fname, args.lead)
        x250 = deploy_preprocess(x)
        probs = run_deploy_inference(interpreter, x250)
        return aggregate(probs, args.aggregate)

    for i, rec in enumerate(pos[:n_pos]):
        scores.append(process(rec))
        y_true.append(1)
        patient_ids.append(rec.get("patient_id", ""))
        fnames.append(rec["filename_hr"])
        if (i + 1) % 200 == 0:
            print(f"  正类 {i+1}/{n_pos}  ({time.time()-t0:.0f}s)")
    for i, rec in enumerate(neg[:n_neg]):
        scores.append(process(rec))
        y_true.append(0)
        patient_ids.append(rec.get("patient_id", ""))
        fnames.append(rec["filename_hr"])
        if (i + 1) % 200 == 0:
            print(f"  负类 {i+1}/{n_neg}  ({time.time()-t0:.0f}s)")

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    # 保存原始分数供离线分析（嵌套阈值选择等，避免重复推理）
    if args.save_scores:
        scores_npz = OUT_JSON.with_suffix(".npz") if not args.tag else \
            OUT_JSON.with_name(f"{OUT_JSON.stem}_{args.tag}.npz")
        np.savez_compressed(
            scores_npz, y_true=y_true, scores=scores,
            patient_ids=np.asarray(patient_ids, dtype=str),
            filenames=np.asarray(fnames, dtype=str))
        print(f"[save-scores] 原始分数已保存: {scores_npz}")

    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
        print("\n[WARN] 样本不足或分数无变化，无法计算 AUC")
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, scores))

    y_pred = (scores >= args.threshold).astype(int)
    cm = confusion(y_true, y_pred)
    if cm["TP"] + cm["FP"] > 0:
        prec = cm["TP"] / (cm["TP"] + cm["FP"])
    else:
        prec = float("nan")
    if cm["TP"] + cm["FN"] > 0:
        rec = cm["TP"] / (cm["TP"] + cm["FN"])
    else:
        rec = float("nan")
    if prec + rec > 0 and not (np.isnan(prec) or np.isnan(rec)):
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = float("nan")
    spec = cm["TN"] / (cm["TN"] + cm["FP"]) if (cm["TN"] + cm["FP"]) > 0 else float("nan")

    # ---------- 阈值扫描 ----------
    sweep = None
    best_youden = None
    best_f1 = None
    if args.threshold_sweep:
        thrs = np.arange(0.01, 1.0, args.sweep_step)
        sweep = []
        n_pos_total = int((y_true == 1).sum())
        n_neg_total = int((y_true == 0).sum())
        for thr in thrs:
            pred = (scores >= thr).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            se = tp / max(1, tp + fn)          # 灵敏度 = 召回
            sp = tn / max(1, tn + fp)          # 特异度
            ppv = tp / max(1, tp + fp)         # 阳性预测值 = 精确率
            f1v = 2 * se * ppv / (se + ppv) if (se + ppv) > 0 else 0.0
            row = {"threshold": round(float(thr), 3), "TP": tp, "FP": fp,
                   "FN": fn, "TN": tn, "sensitivity": round(se, 4),
                   "specificity": round(sp, 4), "precision": round(ppv, 4),
                   "f1": round(f1v, 4)}
            sweep.append(row)
            youden = se + sp
            if best_youden is None or youden > best_youden["youden"]:
                best_youden = {"youden": youden, **row}
            if best_f1 is None or f1v > best_f1["f1"]:
                best_f1 = row

    out_json = OUT_JSON if not args.tag else OUT_JSON.with_name(
        f"{OUT_JSON.stem}_{args.tag}.json")
    report = {
        "config": vars(args),
        "model": args.model,
        "deploy_chain": {
            "comb_taps": COMB_TAPS,
            "ai_chain_hp_hz": 0.05, "ai_chain_lp_hz": 40,
            "decimation": DECIMATION, "ai_hp_hz": 0.5,
            "window": WINDOW, "stride": STRIDE, "trigger_offset": TRIGGER_OFFSET,
            "dc_offset_remove": DC_OFFSET_REMOVE,
            "output": "dequant abnormal class probability (no re-softmax)",
        },
        "label_definition": {"positive": positive, "negative": args.negative,
                             "normal_keys": sorted(NORMAL_KEYS)},
        "n_records": {"positive": n_pos, "negative": n_neg},
        "aggregate": args.aggregate,
        "auc": auc,
        "threshold": args.threshold,
        "confusion_matrix": cm,
        "metrics": {
            "precision": prec, "recall": rec, "f1": f1, "specificity": spec,
        },
        "threshold_sweep": sweep,
        "best_youden_point": best_youden,
        "best_f1_point": best_f1,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"PTB-XL 记录级验证（部署链复刻, positive={positive}, agg={args.aggregate}, thr={args.threshold}）")
    print("=" * 70)
    print(f"样本: 正 {n_pos} / 负 {n_neg}")
    print(f"AUC: {auc:.4f}" if not np.isnan(auc) else "AUC: N/A")
    print(f"混淆矩阵: {cm}")
    print(f"Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} Specificity={spec:.3f}")
    if best_youden:
        b = best_youden
        print("\n-- 最优操作点（Youden = Se+Sp-1 最大）--")
        print(f"  阈值={b['threshold']}  Se={b['sensitivity']:.3f}  Sp={b['specificity']:.3f}  "
              f"PPV={b['precision']:.3f}  F1={b['f1']:.3f}")
        print(f"  TP={b['TP']} FP={b['FP']} FN={b['FN']} TN={b['TN']}")
    if best_f1:
        b = best_f1
        print("\n-- 最优操作点（F1 最大）--")
        print(f"  阈值={b['threshold']}  Se={b['sensitivity']:.3f}  Sp={b['specificity']:.3f}  "
              f"PPV={b['precision']:.3f}  F1={b['f1']:.3f}")
        print(f"  TP={b['TP']} FP={b['FP']} FN={b['FN']} TN={b['TN']}")
    print(f"\n报告已保存: {out_json}")


if __name__ == "__main__":
    main()
