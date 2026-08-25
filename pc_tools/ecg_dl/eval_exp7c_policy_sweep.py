#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_exp7c_policy_sweep.py — exp7c 部署报警策略扫描（θ × K-of-N）
================================================================
目的：
  在患者级 MIT+INCART 部署链数据上，评估 exp7c float32 / INT8 分数在
  不同单拍阈值 θ 与部署侧 K-of-N 时间确认策略下的事件级报警表现。

重要口径：
  - 不把 K-of-N 当作训练输入；这是部署报警策略的离线仿真。
  - MIT deploy npz 含 6× 增强块；本脚本只保留每条 MIT 记录的第一块，
    近似还原原始连续心拍序列，避免增强重复拍伪造时间聚集。
  - INCART 本身无增强块，直接保留。
  - 患者级划分仍使用完整 record_ids 上 seed=42 的同一划分。

输出：
  models/exp7c_policy_sweep.json
  models/exp7c_policy_sweep.csv

注意：
  - 心电拍间隔非固定，因此不伪造 FP/hour；报告 FP/record 与 FP/1000 beats。
  - 参数选择只在 validation 上进行；test 结果应视为冻结评估。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS_DIR
from data.dataset import set_npz_suffix, load_mit_incart_merged
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
from eval_aami_matrix import predict_h5, predict_tflite_int8

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

THETAS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
POLICIES = [(1, 1), (1, 3), (1, 5), (1, 7), (2, 3), (2, 4), (3, 5), (4, 7), (5, 5)]
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
# 心律失常常被逐拍标注割裂；评估“多拍事件”时允许短正常间隔回并同一 episode。
DEFAULT_GT_GAP = 5
# 报警后冷却窗：合并近距离重复触发，模拟真实设备的报警保持/不应期。
DEFAULT_ALERT_COOLDOWN = 5
# validation 选择策略时的报警负荷约束。
DEFAULT_MAX_ALERT_RATE = 0.20

MODELS = [
    {
        "path": "best_resnet_large_exp7c.h5",
        "tag": "exp7c_float32",
        "format": "h5_float32",
    },
    {
        "path": "ecg_model_exp7c_int8.tflite",
        "tag": "exp7c_int8_tflite",
        "format": "tflite_int8",
    },
]


def safe_round(x, nd=6):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"non-finite value: {x}")
    return round(x, nd)


def reduce_mit_augmentation(beats, labels, rids):
    """MIT deploy 数据每条记录含 6 个增强块；仅保留第一块原始序列。"""
    keep = []
    n_records = 0
    for rid in np.unique(rids):
        idx = np.flatnonzero(rids == rid)
        if rid < 100000:
            if len(idx) % 6 != 0:
                raise ValueError(
                    f"MIT record {rid}: expected augmented count divisible by 6, got {len(idx)}"
                )
            keep.extend(idx[: len(idx) // 6].tolist())
        else:
            keep.extend(idx.tolist())
        n_records += 1
    keep = np.asarray(keep, dtype=np.int64)
    print(f"reduce augmentation: {len(beats)} → {len(keep)} beats, "
          f"{n_records} records", flush=True)
    return beats[keep], labels[keep], rids[keep], keep


def bool_blocks(mask: np.ndarray):
    """返回闭区间 blocks [(start,end), ...]。"""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def blocks_overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def apply_k_of_n(scores: np.ndarray, theta: float, k: int, n: int) -> np.ndarray:
    cand = scores >= theta
    if n == 1 or len(scores) == 0:
        return cand.copy()
    if k > n or k <= 0:
        raise ValueError(f"invalid K-of-N: {k}-of-{n}")
    c = cand.astype(np.int32)
    cum = np.concatenate(([0], np.cumsum(c)))
    out = np.zeros(len(scores), dtype=bool)
    for i in range(len(scores)):
        left = max(0, i - n + 1)
        out[i] = (cum[i + 1] - cum[left]) >= k
    return out


def merge_blocks_with_gap(blocks: list, max_gap: int) -> list:
    """将间隔不超过 max_gap 的块合并成一个 episode/alert block。"""
    if not blocks:
        return []
    blocks = sorted(blocks)
    merged = [list(blocks[0])]
    for s, e in blocks[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def episode_metrics(labels: np.ndarray, alarms: np.ndarray,
                    symbols: np.ndarray, gt_gap: int,
                    alert_cooldown: int):
    """Episode 级匹配：GT 异常拍按短间隙回并；报警触发按冷却窗回并。"""
    gt_blocks = merge_blocks_with_gap(bool_blocks(labels == 1), gt_gap)
    pred_blocks = merge_blocks_with_gap(bool_blocks(alarms), alert_cooldown)
    matched_gt = set()
    matched_pred = set()
    latencies = []
    for gi, gt in enumerate(gt_blocks):
        hits = [pi for pi, pd in enumerate(pred_blocks) if blocks_overlap(gt, pd)]
        if hits:
            matched_gt.add(gi)
            matched_pred.update(hits)
            first_alarm = min(pred_blocks[pi][0] for pi in hits)
            latencies.append(max(0, first_alarm - gt[0]))
    tp_events = len(matched_gt)
    fp_events = len(pred_blocks) - len(matched_pred)
    prec = tp_events / len(pred_blocks) if pred_blocks else 0.0
    rec = tp_events / len(gt_blocks) if gt_blocks else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "gt_blocks": gt_blocks,
        "pred_blocks": pred_blocks,
        "gt_events": len(gt_blocks),
        "pred_alert_blocks": len(pred_blocks),
        "matched_gt_events": tp_events,
        "matched_pred_blocks": len(matched_pred),
        "event_recall": safe_round(rec),
        "event_precision": safe_round(prec),
        "event_f1": safe_round(f1),
        "false_alarm_blocks": fp_events,
        "latencies": latencies,
    }


def first_event_latencies(gt_blocks, pred_blocks):
    """每个被捕获 GT 事件的首报延迟；未捕获事件不计入。"""
    out = []
    for gt in gt_blocks:
        hits = [pb for pb in pred_blocks if blocks_overlap(gt, pb)]
        if hits:
            first_alarm = min(pb[0] for pb in hits)
            out.append(max(0, first_alarm - gt[0]))
    return out


def class_event_metrics(labels: np.ndarray, alarms: np.ndarray,
                        symbols: np.ndarray, gt_gap: int,
                        alert_cooldown: int):
    """按 GT episode 的多数 AAMI 类统计 confirmed event recall。"""
    gt_blocks = merge_blocks_with_gap(bool_blocks(labels == 1), gt_gap)
    pred_blocks = merge_blocks_with_gap(bool_blocks(alarms), alert_cooldown)
    out = {}
    for cls in AAMI_CLASSES:
        blocks = []
        for s, e in gt_blocks:
            syms = symbols[s:e + 1]
            syms = syms[syms != "U"]
            if len(syms) == 0:
                continue
            vals, counts = np.unique(syms, return_counts=True)
            if str(vals[np.argmax(counts)]) == cls:
                blocks.append((s, e))
        matched = 0
        for blk in blocks:
            if any(blocks_overlap(blk, pb) for pb in pred_blocks):
                matched += 1
        out[cls] = {
            "gt_events": len(blocks),
            "confirmed_events": matched,
            "event_recall": safe_round(matched / len(blocks)) if blocks else None,
        }
    return out


def evaluate_sequence_set(split_name: str, rids: np.ndarray, labels: np.ndarray,
                          probs: np.ndarray, symbols: np.ndarray,
                          gt_gap: int, alert_cooldown: int) -> list:
    # 按记录分组；rids/probs 已保持原始顺序。
    groups = []
    for rid in np.unique(rids):
        idx = np.flatnonzero(rids == rid)
        groups.append((rid, labels[idx], probs[idx], symbols[idx]))

    rows = []
    auc = safe_round(roc_auc_score(labels, probs))
    for theta in THETAS:
        for k, n in POLICIES:
            all_labels = []
            all_alarms = []
            total_gt_events = 0
            total_pred_blocks = 0
            total_matched_gt = 0
            total_matched_pred = 0
            total_fp_blocks = 0
            total_latency = []
            class_acc = {c: {"gt": 0, "hit": 0} for c in AAMI_CLASSES}
            beat_hit = 0
            beat_abn = 0

            for _, y, p, sym in groups:
                alarms = apply_k_of_n(p, theta, k, n)
                m = episode_metrics(y, alarms, sym, gt_gap, alert_cooldown)
                gt_blocks = m["gt_blocks"]
                pred_blocks = m["pred_blocks"]

                total_gt_events += m["gt_events"]
                total_pred_blocks += m["pred_alert_blocks"]
                total_matched_gt += m["matched_gt_events"]
                total_matched_pred += m["matched_pred_blocks"]
                total_fp_blocks += m["false_alarm_blocks"]
                total_latency.extend(first_event_latencies(gt_blocks, pred_blocks))

                abn = y == 1
                beat_abn += int(abn.sum())
                beat_hit += int((abn & alarms).sum())
                all_labels.append(y)
                all_alarms.append(alarms)

                cm = class_event_metrics(y, alarms, sym, gt_gap, alert_cooldown)
                for c in AAMI_CLASSES:
                    class_acc[c]["gt"] += cm[c]["gt_events"]
                    class_acc[c]["hit"] += cm[c]["confirmed_events"]

            y_all = np.concatenate(all_labels)
            a_all = np.concatenate(all_alarms)
            tp = int(((y_all == 1) & a_all).sum())
            fp = int(((y_all == 0) & a_all).sum())
            tn = int(((y_all == 0) & ~a_all).sum())
            fn = int(((y_all == 1) & ~a_all).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            far = fp / (fp + tn) if (fp + tn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            ev_prec = total_matched_pred / total_pred_blocks if total_pred_blocks else 0.0
            ev_rec = total_matched_gt / total_gt_events if total_gt_events else 0.0
            ev_f1 = 2 * ev_prec * ev_rec / (ev_prec + ev_rec) if (ev_prec + ev_rec) else 0.0

            row = {
                "split": split_name,
                "theta": theta,
                "k": k,
                "n_window": n,
                "policy": f"{k}-of-{n}",
                "gt_gap_beats": gt_gap,
                "alert_cooldown_beats": alert_cooldown,
                "global_auc": auc,
                "beat_tp": tp,
                "beat_fp": fp,
                "beat_tn": tn,
                "beat_fn": fn,
                "beat_precision": safe_round(precision),
                "beat_recall": safe_round(recall),
                "beat_far": safe_float_local(far),
                "beat_f1": safe_round(f1),
                "alert_rate": safe_float_local(a_all.mean()),
                "gt_events": total_gt_events,
                "alert_blocks": total_pred_blocks,
                "matched_gt_events": total_matched_gt,
                "matched_pred_blocks": total_matched_pred,
                "false_alarm_blocks": total_fp_blocks,
                "event_precision": safe_round(ev_prec),
                "event_recall": safe_round(ev_rec),
                "event_f1": safe_round(ev_f1),
                "fp_per_record": safe_round(total_fp_blocks / len(groups)),
                "fp_per_1000_beats": safe_round(
                    1000.0 * total_fp_blocks / len(y_all)),
                "median_latency_beats": (
                    safe_round(np.median(total_latency), 2)
                    if total_latency else None),
                "per_class_confirmed_event_recall": {
                    c: safe_round(class_acc[c]["hit"] / class_acc[c]["gt"])
                    if class_acc[c]["gt"] else None
                    for c in AAMI_CLASSES
                },
            }
            rows.append(row)
    return rows


def safe_float_local(x):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("non-finite")
    return round(x, 6)


def write_summary_csv(all_results, out_csv):
    fields = [
        "tag", "format", "split", "theta", "policy", "gt_gap_beats",
        "alert_cooldown_beats", "global_auc",
        "beat_precision", "beat_recall", "beat_far", "beat_f1", "alert_rate",
        "event_precision", "event_recall", "event_f1", "false_alarm_blocks",
        "fp_per_record", "fp_per_1000_beats", "median_latency_beats",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in all_results:
            base = {k: m[k] for k in ["tag", "format"]}
            for row in m["splits"]:
                w.writerow({**base, **{k: row[k] for k in fields if k not in base}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-suffix", default="_deploy")
    ap.add_argument("--gt-gap", type=int, default=DEFAULT_GT_GAP,
                    help="GT 异常拍回并为同一 episode 的最大正常间隙")
    ap.add_argument("--alert-cooldown", type=int, default=DEFAULT_ALERT_COOLDOWN,
                    help="报警触发块回并的冷却/不应期拍数")
    ap.add_argument("--max-alert-rate", type=float, default=DEFAULT_MAX_ALERT_RATE,
                    help="validation 选择策略时允许的最大 confirmed-beat/alert rate")
    ap.add_argument("--out-prefix", default="exp7c_policy_sweep")
    args = ap.parse_args()

    t0 = time.time()
    set_npz_suffix(args.deploy_suffix)
    print("=" * 78, flush=True)
    print("加载并还原 MIT+INCART 原始连续序列 ...", flush=True)
    data = load_mit_incart_merged()
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    if rids is None:
        raise RuntimeError("record_ids missing")
    beats, labels, rids, kept_idx = reduce_mit_augmentation(beats, labels, rids)

    print("恢复 AAMI 符号 ...", flush=True)
    per_rec_syms = recover_mit_symbols_per_record()
    incart_dir = Path(__file__).resolve().parent / "data" / "raw" / "incart"
    per_rec_syms.update(recover_incart_symbols_per_record(incart_dir))
    sym_full, n_unknown = align_symbols_to_npz(
        per_rec_syms,
        data["record_ids"],
        6,
    )
    if sym_full is None:
        raise RuntimeError("symbol alignment failed")
    symbols = sym_full[kept_idx]

    print("患者级划分 ...", flush=True)
    pmap = {}
    pmap.update(build_mit_patient_map())
    pmap.update({rid + 100000: "inc_" + pat
                 for rid, pat in build_incart_patient_map().items()})
    tr_m, va_m, te_m, pstats = patient_level_split(data["record_ids"], pmap)
    tr_red = tr_m[kept_idx]
    va_red = va_m[kept_idx]
    te_red = te_m[kept_idx]
    train_recs = set(np.unique(data["record_ids"][tr_m]).tolist())
    test_recs = set(np.unique(data["record_ids"][te_m]).tolist())
    if train_recs & test_recs:
        raise RuntimeError("train/test record leakage detected")

    x = add_channel_local(beats.astype(np.float32))
    y = np.asarray(labels).astype(np.int32)

    all_results = []
    for spec in MODELS:
        path = MODELS_DIR / spec["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"\n=== inference: {spec['tag']} ===", flush=True)
        if spec["format"] == "tflite_int8":
            probs_full = predict_tflite_int8(path, x)
        else:
            probs_full = predict_h5(path, x)
        model_result = {
            **spec,
            "model_path": str(path),
            "model_sha256_8": None,
            "splits": [],
        }
        from eval_aami_matrix import sha256_short
        model_result["model_sha256_8"] = sha256_short(path)

        for split_name, mask in [("validation", va_red), ("test", te_red)]:
            yy = y[mask]
            pp = probs_full[mask]
            ss = symbols[mask]
            rr = rids[mask]
            print(f"[{spec['tag']}/{split_name}] n={len(yy)}, abn={int(yy.sum())}, "
                  f"records={len(np.unique(rr))}", flush=True)
            rows = evaluate_sequence_set(split_name, rr, yy, pp, ss,
                                         args.gt_gap, args.alert_cooldown)
            model_result["splits"].extend(rows)
        all_results.append(model_result)

    # 只用 exp7c INT8 validation 选部署策略；test 行全部保留但单独标记。
    int8_val = next(m for m in all_results
                    if m["tag"] == "exp7c_int8_tflite")["splits"]
    val_rows = [r for r in int8_val if r["split"] == "validation"]
    if not val_rows:
        raise RuntimeError("missing INT8 validation rows")
    eligible = [r for r in val_rows
                if r["alert_rate"] is not None
                and r["alert_rate"] <= args.max_alert_rate]
    pool = eligible if eligible else val_rows
    selected = sorted(
        pool,
        key=lambda r: (-r["event_recall"], -r["event_precision"],
                       r["fp_per_record"], r["median_latency_beats"]
                       if r["median_latency_beats"] is not None else 1e9),
    )[0]

    out_json = MODELS_DIR / f"{args.out_prefix}.json"
    out_csv = MODELS_DIR / f"{args.out_prefix}.csv"
    output = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "exp7c multi-beat episode alarm-policy sweep",
            "deploy_suffix": args.deploy_suffix,
            "gt_gap_beats": args.gt_gap,
            "alert_cooldown_beats": args.alert_cooldown,
            "max_alert_rate_for_selection": args.max_alert_rate,
            "sequence_note": "MIT 6x augmented blocks reduced to first raw block; INCART kept unchanged.",
            "patient_stats": pstats,
            "train_test_record_intersection": 0,
            "thresholds": THETAS,
            "policies": [f"{k}-of-{n}" for k, n in POLICIES],
            "selection_model": "exp7c_int8_tflite",
            "selection_split": "validation",
            "selection_rule": f"among validation rows with alert_rate <= {args.max_alert_rate}, maximize event_recall then event_precision, then lower fp_per_record/latency",
            "selection_fallback_note": "if no policy satisfies max_alert_rate, the lowest-alert-rate validation row is selected",
            "selected_policy_validation": selected,
            "selected_policy_test": next(
                r for r in int8_val
                if r["split"] == "test"
                and r["theta"] == selected["theta"]
                and r["policy"] == selected["policy"]
            ),
            "incart_unknown_note": "INCART beats lacking .atr symbols are excluded from AAMI class rows.",
        },
        "models": all_results,
    }
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    write_summary_csv(all_results, out_csv)
    print(f"\nSaved: {out_json}", flush=True)
    print(f"Saved: {out_csv}", flush=True)
    print("\nSelected on INT8 validation:", flush=True)
    print(json.dumps(selected, indent=2, ensure_ascii=False), flush=True)
    print(f"elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


def add_channel_local(x):
    return x[..., np.newaxis]


if __name__ == "__main__":
    sys.exit(main())
