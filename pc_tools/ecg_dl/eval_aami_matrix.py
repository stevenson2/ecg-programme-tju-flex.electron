#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_aami_matrix.py — 批量 AAMI 逐类矩阵评估（患者级 + 部署链）
================================================================
用途：
  在同一患者级测试集和同一部署链数据上，批量评估多个 H5 模型的
  AAMI superclass（N/S/V/F/Q）recall，并输出统一 JSON/CSV，
  用于比较 exp7c、历史 ResNet 模型与跨架构模型的逐类行为。

输入：
  - MIT-BIH + INCART 部署链 npz/npy（默认 suffix=_deploy）
  - 患者级划分 seed=42（与 eval_aami_breakdown.py / 训练一致）
  - Keras H5 模型

输出：
  - models/aami_matrix_deploy_patient.json
  - models/aami_matrix_deploy_patient.csv

口径与审计：
  - 逐类只报 Recall / n / n_abn / AUC；不报类内 precision，
    因为类内可能无负样本，precision 存在恒等式陷阱（AGENTS §30）。
  - 全局报 Precision / Recall / F1 / FAR / Spec，混淆矩阵自洽断言。
  - 出现 0/1 边界值时写入 perfect_value_flags，不直接采信。

用法：
  python3 eval_aami_matrix.py
  python3 eval_aami_matrix.py --list-models
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS_DIR
from data.dataset import set_npz_suffix, add_channel_dim, load_mit_incart_merged
from data.patient_split import (
    build_mit_patient_map,
    build_incart_patient_map,
    patient_level_split,
)
from eval_aami_breakdown import (
    recover_mit_symbols_per_record,
    recover_incart_symbols_per_record,
    align_symbols_to_npz,
)

# 显存按需分配，避免批量评估时贪心占满 GPU。
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

THRESHOLDS = [0.35, 0.50, 0.60, 0.65]
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# 默认矩阵：Tier 1 主叙事模型 + Tier 2 跨架构诊断组。
# 路径均相对 pc_tools/ecg_dl/models/。
DEFAULT_MODELS = [
    {
        "path": "best_resnet_large_exp5_patient_clean.h5",
        "tag": "exp5_patient_clean",
        "role": "patient_level_research_baseline",
        "format": "h5_float32",
    },
    {
        "path": "best_resnet_large_exp6_sgd.h5",
        "tag": "exp6_sgd",
        "role": "deploy_chain_retrain",
        "format": "h5_float32",
    },
    {
        "path": "best_resnet_large_exp7b.h5",
        "tag": "exp7b",
        "role": "pre_real_afe_finetune",
        "format": "h5_float32",
    },
    {
        "path": "best_resnet_large_exp7c.h5",
        "tag": "exp7c_float32",
        "role": "current_deployment_anchor",
        "format": "h5_float32",
    },
    {
        "path": "ecg_model_exp7c_int8.tflite",
        "tag": "exp7c_int8_tflite",
        "role": "current_deployment_anchor",
        "format": "tflite_int8",
    },
    {
        "path": "archived/final_resnet_l_p2a_backup.h5",
        "tag": "p2a_float32",
        "role": "historical_strong_research_model",
        "format": "h5_float32",
    },
    {
        "path": "cross_arch/lstm_cnn_baseline.h5",
        "tag": "lstm_cnn_baseline",
        "role": "cross_arch_baseline",
        "format": "h5_float32",
    },
    {
        "path": "cross_arch/lstm_cnn_deploy.h5",
        "tag": "lstm_cnn_deploy",
        "role": "cross_arch_deploy",
        "format": "h5_float32",
    },
    {
        "path": "cross_arch/cnn_standard_baseline.h5",
        "tag": "cnn_standard_baseline",
        "role": "cross_arch_baseline",
        "format": "h5_float32",
    },
    {
        "path": "cross_arch/cnn_standard_deploy.h5",
        "tag": "cnn_standard_deploy",
        "role": "cross_arch_deploy",
        "format": "h5_float32",
    },
    {
        "path": "cross_arch/resnet1d_baseline.h5",
        "tag": "resnet1d_baseline",
        "role": "cross_arch_baseline",
        "format": "h5_float32",
    },
    {
        "path": "cross_arch/resnet1d_deploy.h5",
        "tag": "resnet1d_deploy",
        "role": "cross_arch_deploy",
        "format": "h5_float32",
    },
]


def sha256_short(path: Path, n_bytes: int = 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n_bytes]


def safe_float(x):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"non-finite metric: {x}")
    return round(x, 6)


def global_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(np.int32)
    cm = np.array([[int(((y_true == 0) & (pred == 0)).sum()),
                    int(((y_true == 0) & (pred == 1)).sum())],
                   [int(((y_true == 1) & (pred == 0)).sum()),
                    int(((y_true == 1) & (pred == 1)).sum())]], dtype=np.int64)
    tn, fp, fn, tp = cm.ravel().tolist()
    n = int(cm.sum())
    assert tn + fp + fn + tp == n, f"confusion matrix does not sum to n: {cm}"
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": thr,
        "precision": safe_float(prec),
        "recall": safe_float(rec),
        "specificity": safe_float(spec),
        "far": safe_float(far),
        "f1": safe_float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
    }


def per_class_metrics(y_true: np.ndarray, prob: np.ndarray,
                      symbols: np.ndarray, aami: str) -> dict:
    mask = symbols == aami
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "n_abn": 0, "n_pos": 0, "n_neg": 0,
                "abn_fraction": None, "auc": None, "recall": {}}
    y_c = y_true[mask]
    p_c = prob[mask]
    n_abn = int((y_c == 1).sum())
    n_pos = n_abn
    n_neg = int((y_c == 0).sum())
    # 若类内没有二分类正例，Recall 无定义；不能报 0.000。
    if n_abn == 0:
        return {
            "n": n,
            "n_abn": 0,
            "n_pos": 0,
            "n_neg": n_neg,
            "abn_fraction": safe_float(0.0),
            "auc": None,
            "recall": {f"thr_{thr:.2f}": None for thr in THRESHOLDS},
        }
    recalls = {}
    for thr in THRESHOLDS:
        pred = (p_c >= thr).astype(np.int32)
        tp = int(((y_c == 1) & (pred == 1)).sum())
        fn = int(((y_c == 1) & (pred == 0)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        recalls[f"thr_{thr:.2f}"] = safe_float(rec)
    auc = None
    if n_pos > 0 and n_neg > 0:
        auc = safe_float(roc_auc_score(y_c, p_c))
    return {
        "n": n,
        "n_abn": n_abn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "abn_fraction": safe_float(n_abn / n),
        "auc": auc,
        "recall": recalls,
    }


def predict_h5(model_path: Path, x: np.ndarray) -> np.ndarray:
    model = tf.keras.models.load_model(str(model_path), compile=False)
    input_shape = tuple(model.input_shape[1:])
    expected = tuple(x.shape[1:])
    if input_shape != expected:
        raise ValueError(f"input shape mismatch: model={input_shape}, data={expected}")
    raw = model.predict(x, batch_size=512, verbose=0)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"unexpected model output: {raw.shape}")
    prob = raw[:, 1]
    if not np.all(np.isfinite(prob)):
        raise ValueError("model produced non-finite probabilities")
    if np.min(prob) < -1e-6 or np.max(prob) > 1 + 1e-6:
        raise ValueError(f"probabilities outside [0,1]: [{prob.min()}, {prob.max()}]")
    return np.clip(prob, 0.0, 1.0)


def predict_tflite_int8(model_path: Path, x: np.ndarray,
                        batch_size: int = 512) -> np.ndarray:
    """批量运行 INT8 TFLite 模型并返回 abnormal probability."""
    if x.ndim != 3 or x.shape[2] != 1:
        raise ValueError(f"expected (n,window,1), got {x.shape}")
    interp = tf.lite.Interpreter(model_path=str(model_path))
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    sig = tuple(int(v) for v in inp["shape_signature"])
    if len(sig) != 3 or sig[1] != x.shape[1] or sig[2] != 1 or sig[0] != -1:
        raise ValueError(f"unexpected TFLite input signature: {sig}")
    interp.resize_tensor_input(inp["index"], (batch_size, x.shape[1], 1),
                               strict=False)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = map(float, inp["quantization"])
    out_scale, out_zp = map(float, out["quantization"])

    n = len(x)
    probs = []
    printed = 0
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size]
        actual = len(xb)
        if actual != batch_size:
            xb = np.concatenate([xb, np.zeros((batch_size - actual, *x.shape[1:]),
                                              dtype=x.dtype)], axis=0)
        xq = np.clip(np.round(xb / in_scale) + in_zp, -128, 127)
        if inp["dtype"] == np.int8:
            xq = xq.astype(np.int8)
        elif inp["dtype"] == np.uint8:
            xq = xq.astype(np.uint8)
        else:
            raise ValueError(f"unsupported TFLite input dtype: {inp['dtype']}")
        interp.set_tensor(inp["index"], xq)
        interp.invoke()
        q = interp.get_tensor(out["index"]).astype(np.float32)[:actual]
        logits = (q - out_zp) * out_scale
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = exp[:, 1] / exp.sum(axis=1)
        probs.append(p)
        if start + batch_size - printed >= 20000:
            done = min(start + batch_size, n)
            print(f"  ... {done}/{n}", flush=True)
            printed = done
    prob = np.concatenate(probs).astype(np.float32)
    if len(prob) != n or not np.all(np.isfinite(prob)):
        raise ValueError("TFLite inference produced invalid probabilities")
    return np.clip(prob, 0.0, 1.0)


def evaluate_model(spec: dict, x_te: np.ndarray, y_te: np.ndarray,
                   sym_te: np.ndarray) -> dict:
    model_path = MODELS_DIR / spec["path"]
    if not model_path.exists():
        raise FileNotFoundError(f"model missing: {model_path}")
    print(f"\n=== {spec['tag']} ===", flush=True)
    print(f"model: {model_path}", flush=True)
    t0 = time.time()
    if spec["format"] == "tflite_int8" or model_path.suffix.lower() == ".tflite":
        prob = predict_tflite_int8(model_path, x_te)
    else:
        prob = predict_h5(model_path, x_te)
    elapsed = time.time() - t0
    global_auc = safe_float(roc_auc_score(y_te, prob))

    per_class = {}
    for c in AAMI_CLASSES:
        per_class[c] = per_class_metrics(y_te, prob, sym_te, c)

    aggregate = {f"thr_{thr:.2f}": global_metrics(y_te, prob, thr)
                 for thr in THRESHOLDS}

    # 完美边界值审计：不阻止产出，但强制下游人工复核。
    perfect_flags = []
    for c, row in per_class.items():
        if row["n"] == 0:
            continue
        for thr_key, rec in row["recall"].items():
            if rec in (0.0, 1.0):
                perfect_flags.append(f"{c}:{thr_key}:recall={rec}")
        if row["auc"] in (0.0, 1.0):
            perfect_flags.append(f"{c}:auc={row['auc']}")
    for thr_key, row in aggregate.items():
        for key in ("precision", "recall", "specificity", "far", "f1"):
            if row[key] in (0.0, 1.0):
                perfect_flags.append(f"ALL:{thr_key}:{key}={row[key]}")
    if global_auc in (0.0, 1.0):
        perfect_flags.append(f"ALL:global_auc={global_auc}")

    return {
        "tag": spec["tag"],
        "role": spec["role"],
        "format": spec["format"],
        "model_path": str(model_path),
        "model_size_bytes": int(model_path.stat().st_size),
        "model_sha256_8": sha256_short(model_path),
        "eval_chain": "deploy",
        "split": "patient_level_60_20_20_seed42_mit_incart",
        "n_test": int(len(y_te)),
        "n_abn": int((y_te == 1).sum()),
        "global_auc": global_auc,
        "per_class": per_class,
        "global": aggregate,
        "perfect_value_flags": perfect_flags,
        "runtime_s": round(elapsed, 1),
    }


def write_csv(rows: list, out_csv: Path) -> None:
    fields = [
        "tag", "role", "format", "aami", "n", "n_abn", "abn_fraction",
        "class_auc", "global_auc",
        *[f"recall@{thr:.2f}" for thr in THRESHOLDS],
        *[f"global_precision@{thr:.2f}" for thr in THRESHOLDS],
        *[f"global_far@{thr:.2f}" for thr in THRESHOLDS],
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in rows:
            for c in AAMI_CLASSES + ["ALL"]:
                r = {"tag": m["tag"], "role": m["role"], "format": m["format"],
                     "aami": c}
                if c == "ALL":
                    r["n"] = m["n_test"]
                    r["n_abn"] = m["n_abn"]
                    r["abn_fraction"] = safe_float(m["n_abn"] / m["n_test"])
                    r["class_auc"] = None
                    r["global_auc"] = m["global_auc"]
                    for thr in THRESHOLDS:
                        key = f"thr_{thr:.2f}"
                        r[f"recall@{thr:.2f}"] = m["global"][key]["recall"]
                        r[f"global_precision@{thr:.2f}"] = m["global"][key]["precision"]
                        r[f"global_far@{thr:.2f}"] = m["global"][key]["far"]
                else:
                    row = m["per_class"][c]
                    r["n"] = row["n"]
                    r["n_abn"] = row["n_abn"]
                    r["abn_fraction"] = row["abn_fraction"]
                    r["class_auc"] = row["auc"]
                    r["global_auc"] = m["global_auc"]
                    for thr in THRESHOLDS:
                        key = f"thr_{thr:.2f}"
                        r[f"recall@{thr:.2f}"] = (
                            row["recall"].get(key) if row["n"] else None
                        )
                        gkey = f"thr_{thr:.2f}"
                        r[f"global_precision@{thr:.2f}"] = m["global"][gkey]["precision"]
                        r[f"global_far@{thr:.2f}"] = m["global"][gkey]["far"]
                w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="批量 AAMI 逐类矩阵评估")
    ap.add_argument("--deploy-suffix", default="_deploy",
                    help="部署链数据后缀，默认 _deploy")
    ap.add_argument("--out-prefix", default="aami_matrix_deploy_patient",
                    help="输出文件名前缀，位于 models/ 下")
    ap.add_argument("--list-models", action="store_true", help="只列出默认模型")
    args = ap.parse_args()

    if args.list_models:
        for i, m in enumerate(DEFAULT_MODELS, 1):
            print(f"{i:02d}. {m['tag']:<24} {m['role']:<36} {m['path']}")
        return 0

    set_npz_suffix(args.deploy_suffix)
    print("=" * 78, flush=True)
    print("加载 MIT+INCART 部署链数据 ...", flush=True)
    data = load_mit_incart_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    if rids is None:
        raise ValueError("record_ids missing; cannot perform patient-level split")

    print("恢复 MIT/INCART AAMI 符号 ...", flush=True)
    per_rec_syms = recover_mit_symbols_per_record()
    incart_dir = Path(__file__).resolve().parent / "data" / "raw" / "incart"
    per_rec_syms.update(recover_incart_symbols_per_record(incart_dir))
    sym_full, n_incart_unknown = align_symbols_to_npz(per_rec_syms, rids, 6)
    if sym_full is None:
        raise RuntimeError("AAMI symbol alignment failed; do not trust outputs")

    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(rids, pmap)

    train_recs = set(np.unique(rids[tr_m]).tolist())
    test_recs = set(np.unique(rids[te_m]).tolist())
    if train_recs & test_recs:
        raise RuntimeError("patient/record leakage detected: train∩test is not empty")

    x_te = add_channel_dim(beats[te_m].astype(np.float32))
    y_te = np.asarray(labels[te_m]).astype(np.int32)
    sym_te = sym_full[te_m]
    print(f"patient split: train={int(tr_m.sum())}, val={int(va_m.sum())}, "
          f"test={int(te_m.sum())}", flush=True)
    print(f"test abnormal: {int((y_te == 1).sum())}/{len(y_te)}", flush=True)
    print(f"INCART unknown-symbol beats: {n_incart_unknown}", flush=True)

    results = []
    for spec in DEFAULT_MODELS:
        results.append(evaluate_model(spec, x_te, y_te, sym_te))
        tf.keras.backend.clear_session()

    out_json = MODELS_DIR / f"{args.out_prefix}.json"
    out_csv = MODELS_DIR / f"{args.out_prefix}.csv"
    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "AAMI superclass matrix under deployment chain",
            "eval_chain": "deploy",
            "deploy_suffix": args.deploy_suffix,
            "split": "patient-level 60/20/20, seed=42, MIT+INCART",
            "thresholds": THRESHOLDS,
            "n_test": int(len(y_te)),
            "n_abn": int((y_te == 1).sum()),
            "n_incart_unknown": int(n_incart_unknown),
            "train_records": len(train_recs),
            "test_records": len(test_recs),
            "train_test_record_intersection": 0,
            "patient_stats": pstats,
            "metric_notes": [
                "Per-class table reports recall/AUC only; classwise precision is omitted because it can be an identity when no negative samples exist (AGENTS §30).",
                "Global precision/FAR are reported separately.",
                "INCART beats without recoverable .atr symbols are excluded from per-class rows but included in global rows.",
            ],
        },
        "models": results,
    }
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    write_csv(results, out_csv)
    print(f"\nSaved: {out_json}", flush=True)
    print(f"Saved: {out_csv}", flush=True)

    all_flags = [(m["tag"], f) for m in results for f in m["perfect_value_flags"]]
    if all_flags:
        print("\nPERFECT/BOUNDARY VALUE FLAGS (audit before use):", flush=True)
        for tag, flag in all_flags:
            print(f"  [{tag}] {flag}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
