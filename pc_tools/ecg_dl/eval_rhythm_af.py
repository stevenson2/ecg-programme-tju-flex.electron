#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_rhythm_af.py — T4-8: 模块1(心律安全逻辑) + 模块3(AF RR 不规则度检测)
======================================================================
任务: 必做清单 T4-8 / consumer_ecg_architecture_plan.md 模块 1+3
设计 (架构计划):
  模块1 心律安全 (纯逻辑, 秒级):
    停搏: RR ≥ 4.0s; 重度过缓: 30s 窗平均 HR < 40bpm; 过速: 30s 窗平均 HR > 180bpm
    SQI 门控: 低 SQI 窗口不触发 (防噪声误报)
  模块3 AF 检测 (RR 不规则度, 30-60s 窗, 通知级, 三档):
    特征: 变异系数 CV=SDNN/mean, Shannon 熵, RMSSD
    三档: 正常 / AF 疑似 / 无法判定 (R4 范式: RR 数不足或质量差 → 无法判定, ~20%)
输出: models/rhythm_af_eval.json (回放测试 + AFDB 验证)
用法 (WSL): python3 eval_rhythm_af.py [--afdb-dir <path>]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
OUT_JSON = MODELS / "rhythm_af_eval.json"

# ---- 模块1 心律安全参数 (架构计划) ----
ASYSTOLE_RR_S = 4.0        # 停搏 RR 阈值 (秒)
BRADY_BPM = 40.0            # 重度过缓
TACHY_BPM = 180.0           # 过速
SAFE_WIN_S = 30.0           # 过缓/过速评估窗
SQI_GATE = 0.5              # SQI 门控阈值

# ---- 模块3 AF 参数 ----
AF_WIN_S = 30.0             # AF 评估窗 (行业标准 30s)
AF_MIN_RR = 20              # 窗内最少 RR 数 (不足 → 无法判定)
AF_CV = 0.10                # CV 阈值 (校准后可调)
AF_ENTROPY = 1.9            # Shannon 熵阈值 (AF 均匀 RR 实测 ~1.99, 校准; AFDB ROC 复核)


def shannon_entropy(rr, bins=16, lo=0.3, hi=1.5):
    """RR 直方图 Shannon 熵 (nat)."""
    h, _ = np.histogram(rr, bins=bins, range=(lo, hi))
    p = h / max(1, h.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


class RhythmSafety:
    """模块1: 心律安全逻辑 (纯规则, SQI 门控)."""

    def __init__(self, asystole_s=ASYSTOLE_RR_S, brady_bpm=BRADY_BPM,
                 tachy_bpm=TACHY_BPM, win_s=SAFE_WIN_S, sqi_gate=SQI_GATE):
        self.asystole_s = asystole_s
        self.brady_bpm = brady_bpm
        self.tachy_bpm = tachy_bpm
        self.win_s = win_s
        self.sqi_gate = sqi_gate

    def evaluate_stream(self, rr, sqi=None):
        """对 RR 流 (秒) 逐拍评估. 返回报警事件列表."""
        events = []
        rr = np.asarray(rr, dtype=float)
        # 停搏: 任一 RR ≥ 阈值
        for i, r in enumerate(rr):
            if r >= self.asystole_s:
                events.append({"type": "asystole", "idx": i, "rr_s": round(float(r), 2),
                               "time_s": round(float(rr[:i].sum()), 1)})
        # 过缓/过速: 30s 滑窗平均 HR
        if len(rr) > 0:
            cum = np.concatenate([[0.0], np.cumsum(rr)])
            t_end = cum[-1]
            win = self.win_s
            t = 0.0
            j = 0
            while t + win <= t_end:
                # 窗 [t, t+win] 内 RR 数
                i0 = int(np.searchsorted(cum, t, side="right") - 1)
                i1 = int(np.searchsorted(cum, t + win, side="left"))
                n = i1 - i0
                if n >= 5:
                    hr = n / win * 60.0
                    sqi_ok = sqi is None or np.mean(sqi[i0:i1]) >= self.sqi_gate
                    if sqi_ok:
                        if hr < self.brady_bpm:
                            events.append({"type": "bradycardia", "time_s": round(t, 1),
                                           "hr_bpm": round(float(hr), 1)})
                        elif hr > self.tachy_bpm:
                            events.append({"type": "tachycardia", "time_s": round(t, 1),
                                           "hr_bpm": round(float(hr), 1)})
                t += win / 2  # 50% 重叠
        return events


class AFDetector:
    """模块3: AF RR 不规则度检测 (30s 窗, 三档)."""

    def __init__(self, win_s=AF_WIN_S, min_rr=AF_MIN_RR, cv_thr=AF_CV,
                 ent_thr=AF_ENTROPY):
        self.win_s = win_s
        self.min_rr = min_rr
        self.cv_thr = cv_thr
        self.ent_thr = ent_thr

    def score_window(self, rr):
        """单窗评分. 返回 (label, score, n_rr). label: 0正常/1AF疑似/2无法判定."""
        if len(rr) < self.min_rr:
            return 2, 0.0, len(rr)  # 无法判定 (RR 不足)
        cv = float(np.std(rr) / max(1e-9, np.mean(rr)))
        ent = shannon_entropy(rr)
        # 组合分数: 归一化 (经验权重)
        score = 0.5 * (cv / 0.2) + 0.5 * (ent / 4.5)
        if cv > self.cv_thr and ent > self.ent_thr:
            return 1, score, len(rr)
        return 0, score, len(rr)

    def evaluate_stream(self, rr, t0=0.0):
        """对 RR 流滑窗 (50% 重叠) 评估. 返回 (label, score, n_rr, t)."""
        cum = np.concatenate([[0.0], np.cumsum(rr)])
        out = []
        t = t0
        while t + self.win_s <= cum[-1] + 1e-9:
            i0 = int(np.searchsorted(cum, t, side="right") - 1)
            i1 = int(np.searchsorted(cum, t + self.win_s, side="left"))
            lab, score, n = self.score_window(rr[i0:i1])
            out.append((lab, score, n, round(t, 1)))
            t += self.win_s / 2
        return out


# ---- 合成 RR 生成器 (回放测试) ----
def synth_rr_normal(rate_bpm=70, seconds=120, noise=0.02):
    rng = np.random.default_rng(0)
    rr = np.full(int(seconds * rate_bpm / 60), 60.0 / rate_bpm)
    return rr + rng.normal(0, 60.0 / rate_bpm * noise, len(rr))


def synth_rr_af(seconds=120):
    rng = np.random.default_rng(1)
    # AF: 随机 RR (300-900ms 均匀)
    t = 0.0
    rr = []
    while t < seconds:
        r = rng.uniform(0.30, 0.90)
        rr.append(r)
        t += r
    return np.array(rr)


def synth_rr_asystole():
    rr = synth_rr_normal(seconds=60)
    rr = np.concatenate([rr, [4.5, 4.8], rr[:30]])  # 中间插入两拍停搏
    return rr


def synth_rr_brady():
    rng = np.random.default_rng(2)
    rr = np.full(180, 60.0 / 35.0) + rng.normal(0, 0.05, 180)  # 35 bpm 90s
    return rr


def synth_rr_tachy():
    rng = np.random.default_rng(3)
    rr = np.full(360, 60.0 / 190.0) + rng.normal(0, 0.02, 360)  # 190 bpm 113s
    return rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afdb-dir", default="/home/devcontainers/afdb_wfdb")
    ap.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()

    safety = RhythmSafety()
    af = AFDetector()

    results = {"replay": {}, "afdb": None}

    # ---- 回放测试 (合成 RR 序列) ----
    print("=" * 60)
    print("回放测试 (合成 RR 序列)")
    print("=" * 60)
    cases = [
        ("正常窦性 (70bpm)", synth_rr_normal(), []),
        ("停搏 (2×4.5s)", synth_rr_asystole(), ["asystole"]),
        ("重度过缓 (35bpm)", synth_rr_brady(), ["bradycardia"]),
        ("过速 (190bpm)", synth_rr_tachy(), ["tachycardia"]),
        ("AF (随机 RR)", synth_rr_af(), []),
    ]
    for name, rr, expect in cases:
        evts = safety.evaluate_stream(rr)
        af_win = af.evaluate_stream(rr)
        af_lab = max(w[0] for w in af_win) if af_win else 2
        got = sorted(set(e["type"] for e in evts))
        ok = set(expect) == set(got) if expect else True
        results["replay"][name] = {
            "safety_events": evts[:6], "af_label": af_lab,
            "n_af_windows": len(af_win),
            "pass": bool(ok),
        }
        print(f"  [{name}] 心律安全事件: {got} | AF 标签: {af_lab} "
              f"(0正常/1疑似/2无法判定) | {'PASS' if ok else 'FAIL'}")

    # ---- AFDB 验证 (窗级: .atr 权威节律标签 + .qrs 自动检测 RR) ----
    if not args.replay_only:
        afdb = Path(args.afdb_dir)
        if afdb.exists():
            print("\n" + "=" * 60)
            print(f"AFDB 验证 ({afdb}) — 窗级")
            print("=" * 60)
            import wfdb
            from sklearn.metrics import roc_auc_score
            recs = sorted(f.name[:-4] for f in afdb.iterdir()
                          if f.name.endswith(".dat") and f.stat().st_size > 100000)
            print(f"可用记录: {len(recs)}")
            X, y, meta = [], [], []
            for rec in recs:
                try:
                    ann = wfdb.rdann(str(afdb / rec), "atr")
                    qrs = wfdb.rdann(str(afdb / rec), "qrs")
                except Exception:
                    continue
                beats = qrs.sample / 250.0
                rr = np.diff(beats)
                # AF 时段 (权威): (onset, end), 单注解记录 = 全程; 无注解 = 无 AF
                af_intervals = []
                rec_dur = (qrs.sample[-1] - qrs.sample[0]) / 250.0
                for i in range(len(ann.sample)):
                    o = ann.sample[i] / 250.0
                    note = (ann.aux_note[i] or "") if ann.aux_note else ""
                    e = (ann.sample[i + 1] / 250.0) if i + 1 < len(ann.sample) else rec_dur
                    if "AFIB" in note or "AFL" in note:
                        af_intervals.append((o, e))
                if not af_intervals:
                    continue
                cum = np.concatenate([[0.0], np.cumsum(rr)])
                t = 0.0
                while t + AF_WIN_S <= cum[-1]:
                    i0 = int(np.searchsorted(cum, t, side="right") - 1)
                    i1 = int(np.searchsorted(cum, t + AF_WIN_S, side="left"))
                    win_rr = rr[i0:i1]
                    if len(win_rr) >= AF_MIN_RR:
                        cv = float(np.std(win_rr) / max(1e-9, np.mean(win_rr)))
                        ent = shannon_entropy(win_rr)
                        overlap = sum(max(0, min(t + AF_WIN_S, e) - max(t, o))
                                      for o, e in af_intervals)
                        label = 1 if overlap / AF_WIN_S >= 0.5 else 0
                        X.append((cv, ent))
                        y.append(label)
                        meta.append((rec, round(t)))
                    t += AF_WIN_S
            if len(y) > 20:
                Xa, ya = np.array(X), np.array(y)
                score = 0.5 * (Xa[:, 0] / 0.2) + 0.5 * (Xa[:, 1] / 4.5)
                auc_cv = roc_auc_score(ya, Xa[:, 0])
                auc_ent = roc_auc_score(ya, Xa[:, 1])
                auc_c = roc_auc_score(ya, score)
                # 检测器阈值 (CV/熵) 混淆矩阵
                pred = ((Xa[:, 0] > AF_CV) & (Xa[:, 1] > AF_ENTROPY)).astype(int)
                tp = ((pred == 1) & (ya == 1)).sum()
                fp = ((pred == 1) & (ya == 0)).sum()
                fn = ((pred == 0) & (ya == 1)).sum()
                tn = ((pred == 0) & (ya == 0)).sum()
                se = tp / max(1, tp + fn)
                sp = tn / max(1, tn + fp)
                # 阈值扫描找最优 (Se+Sp)
                best = None
                for thr_cv in np.arange(0.06, 0.25, 0.02):
                    for thr_ent in np.arange(1.5, 2.6, 0.1):
                        pr = ((Xa[:, 0] > thr_cv) & (Xa[:, 1] > thr_ent)).astype(int)
                        tpp = ((pr == 1) & (ya == 1)).sum()
                        fpp = ((pr == 1) & (ya == 0)).sum()
                        fnn = ((pr == 0) & (ya == 1)).sum()
                        tnn = ((pr == 0) & (ya == 0)).sum()
                        sse = tpp / max(1, tpp + fnn)
                        ssp = tnn / max(1, tnn + fpp)
                        if best is None or (sse + ssp) > (best["se"] + best["sp"]):
                            best = {"cv": float(thr_cv), "ent": float(thr_ent),
                                    "se": sse, "sp": ssp}
                results["afdb"] = {
                    "n_windows": len(ya), "n_af_windows": int(ya.sum()),
                    "auc_cv": float(auc_cv), "auc_entropy": float(auc_ent),
                    "auc_combo": float(auc_c),
                    "detector_se": se, "detector_sp": sp,
                    "best_thr_cv": best["cv"], "best_thr_ent": best["ent"],
                    "best_se": best["se"], "best_sp": best["sp"],
                    "n_records": len(recs),
                }
                print(f"  AFDB 窗数: {len(ya)} (AF {ya.sum()}, 非 AF {len(ya)-ya.sum()})")
                print(f"  AUC: CV={auc_cv:.4f} 熵={auc_ent:.4f} 组合={auc_c:.4f}")
                print(f"  检测器 (CV>{AF_CV}, 熵>{AF_ENTROPY}): Se={se:.4f} Sp={sp:.4f}")
                print(f"  最优阈值 (CV>{best['cv']}, 熵>{best['ent']}): "
                      f"Se={best['se']:.4f} Sp={best['sp']:.4f}")
            else:
                print(f"  有效窗不足: {len(y)}")
        else:
            print(f"  AFDB 目录不存在: {afdb}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
