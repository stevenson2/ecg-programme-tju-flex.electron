#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_distill_pilot.py — 蒸馏试点 val 侧预注册门 + 双模型同协议评估 (TH §106)
================================================================================
用户授权的小规模蒸馏试点之评估入口。试点问题: 教师 (ECGFounder 1-lead)
tile 特征经训练侧留一记录读出得到的软伪概率, 以 α=0.5 混合目标重训同架构
学生, 能否降低 val 侧正常拍的高置信误报 (P1 症状), 且不伤检出。

模式:
  --prereg-only   只写预注册文件 (任何运行之前执行)
  (默认)          双模型 (v3 基线 + 试点) 同脚本同协议评估 + 门裁定

铁律: 测试集零接触 — 本脚本只使用 val 患者全序列 (deploy_causal 数组
val_mask) 与真实 AFE 留出 40 拍; 禁止加载 mit/ptb_deploy_causal_match.npz。
val 同时被用作早停监控与全部门 → 一切数字是相对指标, 非诚实泛化估计。

前置自检 (任一不过即中止, 不出裁定): 同脚本重算的 v3 基线必须复现
盘上数字 (AUC/event_recall 1e-3; frac 与 alert_blocks 用实测边界带推导的
容差 — GPU/XLA 推理跨进程非确定性, v3 正常拍有数百点压住 θ=0.95 边界,
1e-6 位级复现不可行; 假设门为同进程对比, 不受此噪声影响)。

输出: models/deploy_match/distill_pilot_prereg.json
      models/deploy_match/distill_pilot_eval.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_clean_test import make_predictor
from sweep_val_policy_v3 import val_sequences, pooled_metrics
from probe_val_overalert_v3 import quantiles, per_record_blocks

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CACHE = MODELS / "deploy_match"
MODEL_BASE = MODELS / "best_resnet_large_clean_baseline_v3.h5"
MODEL_PILOT = MODELS / "best_resnet_large_distill_pilot.h5"
TARGETS_NPZ = CACHE / "distill_pilot_teacher_targets.npz"
P1_JSON = CACHE / "p1_val_overalert_v3.json"
SWEEP_JSON = CACHE / "val_policy_sweep_v3.json"
PREREG = CACHE / "distill_pilot_prereg.json"
OUT = CACHE / "distill_pilot_eval.json"

FROZEN = {"theta": 0.95, "k": 1, "n": 5, "cooldown": 20}

GATE_CONSTS = {
    "primary_relative_reduction_min": 0.30,
    "guardrail_auc_max_drop": 0.02,
    "guardrail_event_recall_max_drop": 0.02,
    "guardrail_alert_blocks_min": 3,
    "real_holdout_mean_prob_max": 0.10,
    "real_holdout_frac_gt_0.5_max": 0.0,
}

BASELINE_REF = {
    "p1_mit_frac_gt_0.95": 0.24480296159996495,
    "p1_incart_frac_gt_0.95": 0.2735855129057867,
    "p1_mit_auc": 0.7679994798154276,
    "p1_ptb_auc": 0.8262335343912346,
    "sweep_mit_event_recall": 0.9807,
    "sweep_mit_alert_blocks": 401,
    "sweep_mit_false_alarm_blocks": 188,
    "sweep_mit_fp_per_record": 6.963,
    "sweep_ptb_event_recall": 0.9348,
}

BOOT_N, BOOT_SEED = 2000, 7


def prereg_payload():
    return {
        "gates": GATE_CONSTS,
        "frozen_operating_point": FROZEN,
        "baseline_references_from_disk": BASELINE_REF,
        "verdict_semantics": {
            "PASS": "主门 + 护栏 A/B/C 全过",
            "PARTIAL": "主门过但至少一条护栏破",
            "UNPROVEN": "主门未过 (相对降幅在 0–30% 区间或为负) 记未证明; "
                        "单种子运行, 噪声区间内不称证伪",
        },
        "abort_gate_stage1": "训练侧 LORO 读出 AUC < 0.75 → 中止, 不训练",
        "real_synth_swap_rule": "real_synth 组均值伪概率 > 0.5 → 该组目标"
                                "替换为 EPS (预注册, 防 OOD 教师反噬)",
        "zero_test_contact": "仅使用 val 患者全序列与真实 AFE 留出 40 拍; "
                             "禁止加载 mit/ptb_deploy_causal_match.npz",
        "known_limitations": [
            "val 双重使用 (早停监控 + 全部门): 一切数字为相对指标, 非诚实泛化估计",
            "教师真 10s 读出 0.7794 未过 0.80 GO 线; 本试点教的是偏移后更强的 "
            "tile 视图 (0.8893); 若通过, 结论只覆盖'教师软目标能否降低 val 侧误报'",
            "伪目标含噪 (读出 AUC ≈ 0.89 非 1); α=0.5 保留硬标签锚",
            "单种子运行; 记录级 bootstrap CI 仅报告不进门; 多种子复跑为后续工作",
        ],
        "bootstrap": {"n_iterations": BOOT_N, "seed": BOOT_SEED,
                      "note": "记录级, 仅报告不进门"},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §106 蒸馏试点预注册: 教师 tile 特征软目标 (α=0.5) "
                   "能否降低 v3 A 的 val 侧高置信误报",
    }


def write_prereg():
    CACHE.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(prereg_payload(), indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"[DPE] prereg saved: {PREREG}", flush=True)


def assert_prereg_consistent():
    if not PREREG.exists():
        raise SystemExit("缺少预注册文件; 先运行 --prereg-only")
    reg = json.loads(PREREG.read_text(encoding="utf-8"))
    for k, v in GATE_CONSTS.items():
        got = reg["gates"].get(k)
        if got != v:
            raise SystemExit(f"预注册门与脚本常量不一致: {k}: {got} != {v}")
    if reg["frozen_operating_point"] != FROZEN:
        raise SystemExit("预注册冻结操作点与脚本不一致")
    print("[DPE] prereg consistency PASS", flush=True)


def group_rids(groups):
    """record_groups 返回 (rid, idx); 还原逐拍 rid 数组 (拼接顺序)。"""
    return np.concatenate([np.full(len(idx), rid, dtype=np.int64)
                           for rid, idx in groups])


def domain_split(groups):
    """MIT 原生 rid < 100000, INCART +100000。"""
    rids = group_rids(groups)
    return rids < 100000, rids >= 100000


def per_record_normal_counts(groups, labels, probs):
    rows = []
    for _rid, idx in groups:
        l, p = labels[idx], probs[idx]
        nn = int((l == 0).sum())
        hi = int(((l == 0) & (p > 0.95)).sum())
        rows.append((nn, hi))
    return rows


def record_bootstrap_ci(rows, n_iter=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    nn = np.array([r[0] for r in rows], dtype=np.float64)
    hi = np.array([r[1] for r in rows], dtype=np.float64)
    tot = nn.sum()
    if tot == 0:
        return None
    k = len(rows)
    idx = rng.integers(0, k, size=(n_iter, k))
    fr = hi[idx].sum(axis=1) / np.maximum(nn[idx].sum(axis=1), 1e-12)
    return [float(np.quantile(fr, 0.025)), float(np.quantile(fr, 0.975))]


def evaluate_model(predict, sets):
    out = {}
    for name, (groups, labels, probs) in sets.items():
        m_mask, i_mask = domain_split(groups)
        n_labels = labels == 0
        out[name] = {
            "auc": float(roc_auc_score(labels, probs)),
            "n_beats": int(len(labels)),
            "norm_quantiles": quantiles(probs[n_labels]),
        }
        if name == "mit_incart":
            out["mit_only"] = {
                "n_norm": int((n_labels & m_mask).sum()),
                "frac_gt_0.95": float((probs[m_mask & n_labels] > 0.95).mean())
                if (m_mask & n_labels).any() else None,
                "frac_gt_0.5": float((probs[m_mask & n_labels] > 0.5).mean())
                if (m_mask & n_labels).any() else None,
                "band_0.005_count": int(
                    (np.abs(probs[m_mask & n_labels] - 0.95) < 0.005).sum()),
            }
            out["incart_only"] = {
                "n_norm": int((n_labels & i_mask).sum()),
                "frac_gt_0.95": float((probs[i_mask & n_labels] > 0.95).mean())
                if (i_mask & n_labels).any() else None,
                "frac_gt_0.5": float((probs[i_mask & n_labels] > 0.5).mean())
                if (i_mask & n_labels).any() else None,
                "band_0.005_count": int(
                    (np.abs(probs[i_mask & n_labels] - 0.95) < 0.005).sum()),
            }
            nn_mi = int((n_labels & m_mask).sum())
            nn_in = int((n_labels & i_mask).sum())
            hi_mi = int(((probs > 0.95) & n_labels & m_mask).sum())
            hi_in = int(((probs > 0.95) & n_labels & i_mask).sum())
            out["pooled_norm_frac_gt_0.95"] = (hi_mi + hi_in) / (nn_mi + nn_in)
            out["pooled_bootstrap_ci95"] = record_bootstrap_ci(
                per_record_normal_counts(groups, labels, probs))
        out[name]["frozen"] = pooled_metrics(
            groups, labels, probs, FROZEN["theta"], FROZEN["k"],
            FROZEN["n"], FROZEN["cooldown"])
        out[name]["p1_caliber_per_record"] = per_record_blocks(
            group_rids(groups), labels, probs)
    return out


def summarize_per_record(per):
    clean = {rid: v for rid, v in per.items() if v["abn_beats"] == 0}
    top = sorted(per.items(), key=lambda kv: -kv[1]["alert_blocks"])[:5]
    return {
        "clean_normal_records": len(clean),
        "clean_normal_alert_blocks_total": sum(v["alert_blocks"]
                                               for v in clean.values()),
        "record_212": per.get(212),
        "top5_by_alert_blocks": [{"rid": rid, **v} for rid, v in top],
    }


def main():
    if "--prereg-only" in sys.argv:
        write_prereg()
        return
    t0 = time.time()
    assert_prereg_consistent()
    p1 = json.loads(P1_JSON.read_text(encoding="utf-8"))
    sweep = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))

    predict_base = make_predictor("h5", MODEL_BASE)
    predict_pilot = make_predictor("h5", MODEL_PILOT)
    sets_base, _flat_b = val_sequences(predict_base)
    sets_pilot, _flat_p = val_sequences(predict_pilot)

    base = evaluate_model(predict_base, sets_base)
    pilot = evaluate_model(predict_pilot, sets_pilot)

    # ---------- 真实 AFE 留出 (GUARDRAIL C) ----------
    d = np.load(TARGETS_NPZ)
    x_ho = np.asarray(d["x_real_holdout"], dtype=np.float32)
    holdout = {}
    for tag, pr in (("base", predict_base), ("pilot", predict_pilot)):
        p = pr(x_ho)
        holdout[tag] = {"mean_prob": float(p.mean()),
                        "frac_gt_0.5": float((p > 0.5).mean()),
                        "n": int(len(p))}

    # ---------- 前置自检: 同脚本基线复现盘上数字 ----------
    # 容差修订 (2026-09-01, 执行细则缺陷修复, 假设门未动):
    # GPU/XLA 推理跨进程非确定性 — v3 正常拍有数百点密集压住 θ=0.95 边界带
    # (实测合并 709 点在 ±0.005 内), 不同进程翻转十几个边界点, 1e-6 位级复现
    # 不可行。反证: 独立进程重跑同口径冻结点得 401/0.9807/188 与盘上完全一致。
    # frac 容差 = 边界带点数/该域正常拍数 (数据推导); alert_blocks ±2
    # (每翻转点最多 ±1 块, 实测差异 ≤1)。假设门全部为同进程基线-试点对比,
    # 共享同一推理实现, 不受此噪声影响。
    band_mi = base["mit_only"]["band_0.005_count"]
    band_in = base["incart_only"]["band_0.005_count"]
    mit_tol = band_mi / base["mit_only"]["n_norm"] + 1e-6
    inc_tol = band_in / base["incart_only"]["n_norm"] + 1e-6
    selfcheck = {
        "mit_frac_gt_0.95_vs_p1": (
            base["mit_only"]["frac_gt_0.95"],
            abs(base["mit_only"]["frac_gt_0.95"]
                - BASELINE_REF["p1_mit_frac_gt_0.95"]) <= mit_tol,
            f"tol={mit_tol:.6f} (band={band_mi})"),
        "incart_frac_gt_0.95_vs_p1": (
            base["incart_only"]["frac_gt_0.95"],
            abs(base["incart_only"]["frac_gt_0.95"]
                - BASELINE_REF["p1_incart_frac_gt_0.95"]) <= inc_tol,
            f"tol={inc_tol:.6f} (band={band_in})"),
        "mit_auc_vs_p1": (base["mit_incart"]["auc"],
                          abs(base["mit_incart"]["auc"]
                              - BASELINE_REF["p1_mit_auc"]) < 1e-3,
                          "tol=1e-3"),
        "ptb_auc_vs_p1": (base["ptb"]["auc"],
                          abs(base["ptb"]["auc"]
                              - BASELINE_REF["p1_ptb_auc"]) < 1e-3,
                          "tol=1e-3"),
        "frozen_event_recall_vs_sweep": (
            base["mit_incart"]["frozen"]["event_recall"],
            abs(base["mit_incart"]["frozen"]["event_recall"]
                - BASELINE_REF["sweep_mit_event_recall"]) < 1e-3,
            "tol=1e-3"),
        "frozen_alert_blocks_vs_sweep": (
            base["mit_incart"]["frozen"]["alert_blocks"],
            abs(base["mit_incart"]["frozen"]["alert_blocks"]
                - BASELINE_REF["sweep_mit_alert_blocks"]) <= 2,
            "tol=±2 (边界翻转实测差≤1)"),
    }
    ok = [v[1] for v in selfcheck.values()]
    for k, (v, good, tol) in selfcheck.items():
        print(f"[DPE] selfcheck {k}: {v} ({tol}) "
              f"{'PASS' if good else 'FAIL'}", flush=True)
    if not all(ok):
        OUT.write_text(json.dumps(
            {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
             "status": "ABORTED_SELFCHECK", "selfcheck": selfcheck},
            indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit("[DPE] 前置自检失败: 基线未复现盘上数字, 不出裁定")

    # ---------- 门裁定 ----------
    b_frac = base["pooled_norm_frac_gt_0.95"]
    p_frac = pilot["pooled_norm_frac_gt_0.95"]
    rel_red = (b_frac - p_frac) / b_frac
    b_auc = base["mit_incart"]["auc"]
    p_auc = pilot["mit_incart"]["auc"]
    b_ev = base["mit_incart"]["frozen"]["event_recall"]
    p_ev = pilot["mit_incart"]["frozen"]["event_recall"]
    p_ab = pilot["mit_incart"]["frozen"]["alert_blocks"]
    ho = holdout["pilot"]

    primary = rel_red >= GATE_CONSTS["primary_relative_reduction_min"]
    gA = (b_auc - p_auc) <= GATE_CONSTS["guardrail_auc_max_drop"]
    gB = ((b_ev - p_ev) <= GATE_CONSTS["guardrail_event_recall_max_drop"]
          and p_ab >= GATE_CONSTS["guardrail_alert_blocks_min"])
    gC = (ho["mean_prob"] < GATE_CONSTS["real_holdout_mean_prob_max"]
          and ho["frac_gt_0.5"] <= GATE_CONSTS["real_holdout_frac_gt_0.5_max"])
    if primary and gA and gB and gC:
        verdict = "PASS"
    elif primary:
        verdict = "PARTIAL"
    else:
        verdict = "UNPROVEN"

    gates = {
        "primary_pooled_norm_frac_gt_0.95": {
            "baseline": b_frac, "pilot": p_frac,
            "relative_reduction": rel_red, "passed": primary,
            "criterion": f">={GATE_CONSTS['primary_relative_reduction_min']} "
                         "相对降幅"},
        "guardrail_A_mit_auc": {
            "baseline": b_auc, "pilot": p_auc, "drop": b_auc - p_auc,
            "passed": gA,
            "criterion": f"drop<={GATE_CONSTS['guardrail_auc_max_drop']}"},
        "guardrail_B_frozen_event_recall": {
            "baseline_event_recall": b_ev, "pilot_event_recall": p_ev,
            "pilot_alert_blocks": p_ab, "passed": gB,
            "criterion": f"drop<={GATE_CONSTS['guardrail_event_recall_max_drop']}"
                         f" 且 alert_blocks>={GATE_CONSTS['guardrail_alert_blocks_min']}"},
        "guardrail_C_real_holdout": {
            "pilot": ho, "passed": gC,
            "criterion": f"mean_prob<{GATE_CONSTS['real_holdout_mean_prob_max']}"
                         f" 且 frac_gt_0.5<={GATE_CONSTS['real_holdout_frac_gt_0.5_max']}"},
    }

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "TH §106 蒸馏试点评估: 教师软目标 (α=0.5) vs v3 A 基线, "
                   "val 侧同脚本同协议; 测试集零接触",
        "models": {"baseline": str(MODEL_BASE.relative_to(BASE)),
                   "pilot": str(MODEL_PILOT.relative_to(BASE))},
        "frozen_operating_point": FROZEN,
        "selfcheck": selfcheck,
        "selfcheck_tolerance_revision": {
            "date": time.strftime("%Y-%m-%d"),
            "reason": "GPU/XLA 推理跨进程非确定性; v3 正常拍有数百点密集压住 "
                      "θ=0.95 边界带, 1e-6 位级复现不可行 (独立进程重跑同口径"
                      "冻结点与盘上完全一致, 证管道口径正确)",
            "frac_tolerance": "该域边界带(±0.005)点数 / 该域正常拍数 (数据推导)",
            "alert_blocks_tolerance": "±2",
            "gates_affected": "无 — 全部假设门为同进程基线-试点对比"},
        "gates": gates,
        "verdict": verdict,
        "verdict_semantics": prereg_payload()["verdict_semantics"],
        "baseline_metrics": base,
        "pilot_metrics": pilot,
        "real_afe_holdout": holdout,
        "secondary": {
            "alert_blocks_ratio_pilot_over_baseline":
                p_ab / base["mit_incart"]["frozen"]["alert_blocks"],
            "p1_caliber_summary": {
                "baseline": summarize_per_record(
                    base["mit_incart"]["p1_caliber_per_record"]),
                "pilot": summarize_per_record(
                    pilot["mit_incart"]["p1_caliber_per_record"]),
            },
            "sweep_reference_frozen": {
                "mit_incart": {k: BASELINE_REF["sweep_mit_" + k]
                               for k in ("event_recall", "alert_blocks",
                                         "false_alarm_blocks", "fp_per_record")},
                "ptb_event_recall": BASELINE_REF["sweep_ptb_event_recall"],
            },
        },
        "known_limitations": prereg_payload()["known_limitations"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    # 逐记录表过大, 正文只保留摘要; 完整表另存
    for tag, m in (("baseline", base), ("pilot", pilot)):
        for name in ("mit_incart", "ptb"):
            m[name].pop("p1_caliber_per_record", None)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    for k, g in gates.items():
        print(f"[DPE] gate {k}: passed={g['passed']}", flush=True)
    print(f"[DPE] pooled_norm_frac_gt_0.95: base={b_frac:.4f} pilot={p_frac:.4f} "
          f"rel_red={rel_red:.3f}", flush=True)
    print(f"[DPE] verdict={verdict} ({time.time() - t0:.0f}s) saved {OUT.name}",
          flush=True)


if __name__ == "__main__":
    main()
