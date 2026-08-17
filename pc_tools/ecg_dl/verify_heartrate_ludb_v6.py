#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_heartrate_ludb_v6.py — LUDB 心率 v6 参数扫描 (完整状态机仿真)
=====================================================================

v5 基线 (当前固件 heartrate.cpp 能量包络复刻, MIN_CONF_FEAT=1000 禁用形态学)
在 200 条 LUDB 记录上: Se 96.94% / PPV 71.03% / F1 0.820 / BPM MAE 10.17。
FP 峰级诊断 (models/ludb_hr_v5_peaks.csv) 表明双计数结构:
  - T 波双计数: 峰宽 65-80 采样、rise/fall 比 65-70 (真 QRS 6-33) → rf 门最有区分度;
  - 宽 QRS 双计数: 次峰窄(≈19)但幅度 ~0.5× 前拍 → 幅度/前拍比门;
  - 记录开头假峰: 多为大 T/P 波, 无金标准可参考, 不设 gold 专属规则。
固定宽度门 (40-60 采样) 会误杀真宽 QRS (TP 宽可达 120) → 放弃固定宽度口径。

本脚本在 v5 状态机逐样本复刻之上新增可开关门限 (beat_count>=MIN_CONF_FEAT 激活):
  - GATE_RF_MAX         : MWI 峰 rise/fall 比上限 (T 波 rf≈65-70)
  - GATE_WIDTH_RATIO    : 峰半高宽 > ratio × 近期有效峰宽中位数 → 拒 (自适应)
  - GATE_AMP_CONSISTENCY: 峰幅一致性容差 (固件原有机制, 能量域重新标定)
  - GATE_AMP_FRAC_PREV  : rr<0.9s 且峰幅 < frac×前一有效峰 → 拒 (双计数)
  - GATE_RR_RATIO       : rr < ratio×medianRR → 拒 (半 RR 双计数)

所有候选组合都在**全部 200 条记录**上完整仿真 (multiprocessing, 16 核)。
产出:
  - models/ludb_hr_v6_scan.json        全部候选汇总
  - models/ludb_hr_v6_param_table.csv  参数表 (v5 基线 + 全候选)
  - models/ludb_hr_v6_eval.json        选定组合 (--final 指定)
  - models/ludb_hr_v6_detail.csv       选定组合逐记录明细

数字审计 (AGENTS §8): TP+FN==gold / TP+FP==det 断言; 完美值告警写入 audit。
冒烟测试: python3 verify_heartrate_ludb_v6.py --records 1,2,3,4,5
"""
import argparse
import csv
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_heartrate_ludb_v5 import (
    FS, TS, DEFAULT_DATA_DIR,
    HRDetectorV5, HRParamsV5, chain_filter_v5, load_ludb_record,
)

TOLERANCE_SAMP = 75  # 150ms @500Hz (main 中按 --tolerance-ms 更新)
MODELS = Path(__file__).resolve().parent / "models"


# ============================================================================
# v6 参数与检测器
# ============================================================================
class HRParamsV6(HRParamsV5):
    """v5 全部常数 + 新门限 (None=关闭)。旧形态学组件在扫描中全部关闭,
    避免旧导数域标定 (40-80 宽 / rf 0.5-2.0) 污染能量域重标定。"""

    def __init__(self, **kw):
        super().__init__()
        # 起始消隐 (0=关闭): 滤波/MWI 初始化瞬态在样本 ~44-97 产生伪峰,
        # LUDB 最早 TP 检出在样本 306 → 250 样本 (500ms) 消隐安全 (证据:
        # pc_tools/ecg_dl/models/ludb_hr_v5_peaks.csv, TP 首峰分位 0% = 306)
        self.STARTUP_BLANK_SAMP = 0
        # 新门限
        self.GATE_RF_MAX = None
        self.GATE_WIDTH_RATIO = None
        self.GATE_AMP_CONSISTENCY = None
        self.GATE_AMP_FRAC_PREV = None
        self.GATE_RR_RATIO = None
        # 旧形态学组件关闭 (新门限接管)
        self.MIN_QRS_WIDTH = 0
        self.MAX_QRS_WIDTH = 10**9
        self.AMP_CONSISTENCY = 1e9
        self.RR_CONSISTENCY = 1e9
        self.MIN_QRS_WIDTH_MOT = 0
        self.MAX_QRS_WIDTH_MOT = 10**9
        self.RISE_FALL_MIN = 1e-6
        self.RISE_FALL_MAX = 1e9
        self.RISE_FALL_MIN_MOT = 1e-6
        self.RISE_FALL_MAX_MOT = 1e9
        for k, v in kw.items():
            setattr(self, k, v)


class HRDetectorV6(HRDetectorV5):
    """v6 检测器: v5 状态机 + 新门限 (MIN_CONF_FEAT 起激活)。"""

    def reset_state(self):
        super().reset_state()
        self.valid_widths = []
        self.last_width = None
        self.last_rf = None

    def _gate_rf(self, peak_val):
        p = self.p
        if p.GATE_RF_MAX is None:
            return True
        self.last_rf = self.get_rise_fall_ratio()
        return self.last_rf <= p.GATE_RF_MAX

    def _gate_width_ratio(self, peak_val):
        p = self.p
        if p.GATE_WIDTH_RATIO is None:
            return True
        self.last_width = self.get_qrs_width()
        if len(self.valid_widths) < 3:
            return True
        med_w = float(np.median(self.valid_widths[-8:]))
        if med_w < 1:
            return True
        return self.last_width <= p.GATE_WIDTH_RATIO * med_w

    def _gate_amp_consistency(self, peak_val):
        p = self.p
        if p.GATE_AMP_CONSISTENCY is None:
            return True
        if self.peak_hist_count < 3:
            return True
        mean = float(np.mean(self.recent_peaks[: self.peak_hist_count]))
        if mean < 0.0001:
            return True
        return abs(peak_val - mean) / mean <= p.GATE_AMP_CONSISTENCY

    def _gate_amp_frac_prev(self, peak_val, rr_sec):
        p = self.p
        if p.GATE_AMP_FRAC_PREV is None:
            return True
        if self.peak_hist_count < 1 or rr_sec >= 0.9:
            return True
        prev = float(self.recent_peaks[(self.peak_hist_idx - 1) % p.PEAK_HIST_LEN])
        if prev < 0.0001:
            return True
        return peak_val >= p.GATE_AMP_FRAC_PREV * prev

    def _gate_rr_ratio(self, rr_sec):
        p = self.p
        if p.GATE_RR_RATIO is None:
            return True
        if self.rr_count < 3 or self.median_rr < 0.001:
            return True
        return rr_sec >= p.GATE_RR_RATIO * self.median_rr

    def is_qrs_valid(self, peak_val, rr_sec):
        p = self.p
        self.last_width = None
        self.last_rf = None
        # 起始消隐: 滤波链/MWI 初始化瞬态伪峰 (安全上限见 LUDB TP 首峰分布)
        if self.i < p.STARTUP_BLANK_SAMP:
            return False
        if not super().is_qrs_valid(peak_val, rr_sec):
            return False
        if self.beat_count >= p.MIN_CONF_FEAT:
            if not self._gate_rf(peak_val):
                return False
            if not self._gate_width_ratio(peak_val):
                return False
            if not self._gate_amp_consistency(peak_val):
                return False
            if not self._gate_amp_frac_prev(peak_val, rr_sec):
                return False
            if not self._gate_rr_ratio(rr_sec):
                return False
        return True

    def record_valid_peak(self, peak_val):
        super().record_valid_peak(peak_val)
        if self.last_width is not None:
            self.valid_widths.append(self.last_width)
            self.last_width = None
        else:
            self.valid_widths.append(self.get_qrs_width())


# ============================================================================
# 单记录运行 / 汇总 / 审计
# ============================================================================
def run_record_v6(y, gold, p, tol_samp):
    """对已滤波信号 y (chain_filter_v5 输出) 运行检测器, 返回逐记录汇总。"""
    det = HRDetectorV6(p)
    det_idx = []
    bpm_outputs = []
    for i in range(len(y)):
        res = det.process(float(y[i]))
        if res["beatDetected"]:
            det_idx.append(i)
        if res["bpm"] > 0:
            bpm_outputs.append(res["bpm"])

    gold_arr = np.asarray(gold, dtype=np.int64)
    det_arr = np.asarray(det_idx, dtype=np.int64)

    used = set()
    tp = 0
    for g in sorted(gold_arr.tolist()):
        cands = np.where(np.abs(det_arr - g) <= tol_samp)[0]
        if len(cands) == 0:
            continue
        best = min(cands, key=lambda c: abs(int(det_arr[c]) - int(g)))
        if int(best) in used:
            continue
        used.add(int(best))
        tp += 1
    fn = len(gold_arr) - tp
    fp = len(det_arr) - tp

    gold_bpm = 0.0
    if len(gold_arr) >= 2:
        rr = np.diff(gold_arr).astype(np.float64) / FS
        rr = rr[(rr >= 0.3) & (rr <= 2.0)]
        if len(rr) > 0:
            gold_bpm = 60.0 / float(np.median(rr))
    det_bpm = float(np.median(bpm_outputs)) if bpm_outputs else 0.0

    return {
        "record": None, "gold": int(len(gold_arr)), "det": int(len(det_arr)),
        "tp": tp, "fp": fp, "fn": fn,
        "gold_bpm": gold_bpm, "det_bpm": det_bpm,
        "bpm_err": (abs(det_bpm - gold_bpm) if det_bpm > 0 and gold_bpm > 0 else None),
    }


def summarize_v6(results):
    n_gold = sum(r["gold"] for r in results)
    n_det = sum(r["det"] for r in results)
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)

    # AGENTS §8 合理性断言: 产出即校验
    assert tp + fn == n_gold, f"混淆矩阵不自洽: TP+FN={tp+fn} != gold={n_gold}"
    assert tp + fp == n_det, f"混淆矩阵不自洽: TP+FP={tp+fp} != det={n_det}"
    assert all(r["tp"] >= 0 and r["fp"] >= 0 and r["fn"] >= 0 for r in results)

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * se * ppv / (se + ppv)) if (se + ppv) > 0 else 0.0

    errs = [r["bpm_err"] for r in results if r["bpm_err"] is not None]
    bpm_mae = float(np.mean(errs)) if errs else 0.0
    bpm_med = float(np.median(errs)) if errs else 0.0
    p90 = float(np.percentile(errs, 90)) if errs else 0.0
    within3 = 100.0 * np.mean([1.0 if e <= 3.0 else 0.0 for e in errs]) if errs else 0.0
    within5 = 100.0 * np.mean([1.0 if e <= 5.0 else 0.0 for e in errs]) if errs else 0.0

    audit_warnings = []
    for key, val in (("se", se), ("ppv", ppv)):
        if val >= 0.999999 or val <= 1e-9:
            audit_warnings.append(f"{key}={val:.9f} 边界完美值, 需人工核查")
    if bpm_mae == 0.0 and n_gold > 0:
        audit_warnings.append("BPM MAE == 0, 需人工核查")

    return {
        "records": len(results), "gold_beats": n_gold, "det_beats": n_det,
        "tp": tp, "fp": fp, "fn": fn, "se": se, "ppv": ppv, "f1": f1,
        "bpm_mae": bpm_mae, "bpm_median_err": bpm_med, "bpm_p90": p90,
        "bpm_within3": within3, "bpm_within5": within5,
        "audit": {"confusion_ok": True, "warnings": audit_warnings},
    }


def objective_of(s):
    """Se<95% 不可行; 否则先最小化 BPM MAE (向 3.2), 再最大化 PPV。"""
    if s["se"] < 0.95:
        return -1e9
    return -s["bpm_mae"] + 0.001 * s["ppv"]


# ============================================================================
# 候选组合
# ============================================================================
def build_combos():
    c = {}
    c["v5_baseline"] = {}
    c["blank250"] = {"STARTUP_BLANK_SAMP": 250}
    c["rf_c3_40"] = {"MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40}
    c["rf_c4_40"] = {"MIN_CONF_FEAT": 4, "GATE_RF_MAX": 40}
    c["rf_c5_35"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 35}
    c["rf_c5_40"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40}
    c["rf_c5_45"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 45}
    c["rf_c8_35"] = {"MIN_CONF_FEAT": 8, "GATE_RF_MAX": 35}
    c["rf_c8_40"] = {"MIN_CONF_FEAT": 8, "GATE_RF_MAX": 40}
    c["rf_c8_45"] = {"MIN_CONF_FEAT": 8, "GATE_RF_MAX": 45}
    c["rf_c5_40_ac025"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                           "GATE_AMP_CONSISTENCY": 0.25}
    c["rf_c5_40_ac030"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                           "GATE_AMP_CONSISTENCY": 0.30}
    c["rf_c5_40_ac035"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                           "GATE_AMP_CONSISTENCY": 0.35}
    c["rf_c5_40_wr20"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                          "GATE_WIDTH_RATIO": 2.0}
    c["rf_c5_40_wr25"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                          "GATE_WIDTH_RATIO": 2.5}
    c["rf_c5_40_prev045"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                             "GATE_AMP_FRAC_PREV": 0.45}
    c["rf_c5_40_prev055"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                             "GATE_AMP_FRAC_PREV": 0.55}
    c["rf_c5_40_rr055"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                           "GATE_RR_RATIO": 0.55}
    c["rf_c5_40_rr060"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                           "GATE_RR_RATIO": 0.60}
    c["rf_c5_40_ac030_wr25"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                                "GATE_AMP_CONSISTENCY": 0.30,
                                "GATE_WIDTH_RATIO": 2.5}
    c["rf_c5_40_ac030_prev055"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                                   "GATE_AMP_CONSISTENCY": 0.30,
                                   "GATE_AMP_FRAC_PREV": 0.55}
    c["rf_c5_40_rr060_wr25"] = {"MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40,
                                "GATE_RR_RATIO": 0.60,
                                "GATE_WIDTH_RATIO": 2.5}
    c["adaw_c5_20"] = {"MIN_CONF_FEAT": 5, "GATE_WIDTH_RATIO": 2.0}
    c["adaw_c5_25"] = {"MIN_CONF_FEAT": 5, "GATE_WIDTH_RATIO": 2.5}
    c["fixw_c12_10_45"] = {"MIN_CONF_FEAT": 12, "MIN_QRS_WIDTH": 10,
                           "MAX_QRS_WIDTH": 45}
    # 起始消隐 + rf 门组合 (2026-08-16 二轮)
    c["blank200_rf_c3_40"] = {"STARTUP_BLANK_SAMP": 200,
                              "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40}
    c["blank250_rf_c3_35"] = {"STARTUP_BLANK_SAMP": 250,
                              "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 35}
    c["blank250_rf_c3_40"] = {"STARTUP_BLANK_SAMP": 250,
                              "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40}
    c["blank250_rf_c3_45"] = {"STARTUP_BLANK_SAMP": 250,
                              "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 45}
    c["blank250_rf_c4_40"] = {"STARTUP_BLANK_SAMP": 250,
                              "MIN_CONF_FEAT": 4, "GATE_RF_MAX": 40}
    c["blank250_rf_c5_40"] = {"STARTUP_BLANK_SAMP": 250,
                              "MIN_CONF_FEAT": 5, "GATE_RF_MAX": 40}
    c["blank250_rf_c3_40_prev045"] = {"STARTUP_BLANK_SAMP": 250,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.45}
    c["blank250_rf_c3_40_prev055"] = {"STARTUP_BLANK_SAMP": 250,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.55}
    c["blank250_rf_c3_40_rr055"] = {"STARTUP_BLANK_SAMP": 250,
                                    "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                    "GATE_RR_RATIO": 0.55}
    c["blank250_rf_c3_40_rr060"] = {"STARTUP_BLANK_SAMP": 250,
                                    "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                    "GATE_RR_RATIO": 0.60}
    c["blank250_rf_c3_40_wr25"] = {"STARTUP_BLANK_SAMP": 250,
                                   "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                   "GATE_WIDTH_RATIO": 2.5}
    c["blank250_rf_c3_40_wr30"] = {"STARTUP_BLANK_SAMP": 250,
                                   "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                   "GATE_WIDTH_RATIO": 3.0}
    c["blank250_rf_c4_40_prev055"] = {"STARTUP_BLANK_SAMP": 250,
                                      "MIN_CONF_FEAT": 4, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.55}
    c["blank300_rf_c3_40"] = {"STARTUP_BLANK_SAMP": 300,
                              "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40}
    # 三轮微调: 围绕 blank250_rf_c3_40_prev055 (2026-08-16)
    c["blank250_rf_c3_40_prev050"] = {"STARTUP_BLANK_SAMP": 250,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.50}
    c["blank250_rf_c3_40_prev060"] = {"STARTUP_BLANK_SAMP": 250,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.60}
    c["blank250_rf_c3_40_prev055_rr060"] = {"STARTUP_BLANK_SAMP": 250,
                                            "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                            "GATE_AMP_FRAC_PREV": 0.55,
                                            "GATE_RR_RATIO": 0.60}
    c["blank250_rf_c3_40_prev055_rr070"] = {"STARTUP_BLANK_SAMP": 250,
                                            "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                            "GATE_AMP_FRAC_PREV": 0.55,
                                            "GATE_RR_RATIO": 0.70}
    c["blank250_rf_c3_40_prev055_wr30"] = {"STARTUP_BLANK_SAMP": 250,
                                           "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                           "GATE_AMP_FRAC_PREV": 0.55,
                                           "GATE_WIDTH_RATIO": 3.0}
    c["blank260_rf_c3_40_prev055"] = {"STARTUP_BLANK_SAMP": 260,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.55}
    c["blank270_rf_c3_40_prev055"] = {"STARTUP_BLANK_SAMP": 270,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.55}
    c["blank250_rf_c4_40_prev060"] = {"STARTUP_BLANK_SAMP": 250,
                                      "MIN_CONF_FEAT": 4, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.60}
    # 四轮微调: blank 260/275/280 与 rr065 (2026-08-16)
    c["blank250_rf_c3_40_prev055_rr065"] = {"STARTUP_BLANK_SAMP": 250,
                                            "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                            "GATE_AMP_FRAC_PREV": 0.55,
                                            "GATE_RR_RATIO": 0.65}
    c["blank260_rf_c3_40_prev055_rr065"] = {"STARTUP_BLANK_SAMP": 260,
                                            "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                            "GATE_AMP_FRAC_PREV": 0.55,
                                            "GATE_RR_RATIO": 0.65}
    c["blank260_rf_c3_40_prev055_rr070"] = {"STARTUP_BLANK_SAMP": 260,
                                            "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                            "GATE_AMP_FRAC_PREV": 0.55,
                                            "GATE_RR_RATIO": 0.70}
    c["blank260_rf_c3_40_prev060"] = {"STARTUP_BLANK_SAMP": 260,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.60}
    c["blank275_rf_c3_40_prev055"] = {"STARTUP_BLANK_SAMP": 275,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.55}
    c["blank280_rf_c3_40_prev055"] = {"STARTUP_BLANK_SAMP": 280,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.55}
    c["blank260_rf_c3_40_prev050"] = {"STARTUP_BLANK_SAMP": 260,
                                      "MIN_CONF_FEAT": 3, "GATE_RF_MAX": 40,
                                      "GATE_AMP_FRAC_PREV": 0.50}
    return c


def combo_to_config(params):
    """把组合参数导出为固件可对照的配置字典。"""
    return {
        "STARTUP_BLANK_SAMP": params.get("STARTUP_BLANK_SAMP", 0),
        "MIN_CONF_FEAT": params.get("MIN_CONF_FEAT", 1000),
        "GATE_RF_MAX": params.get("GATE_RF_MAX"),
        "GATE_WIDTH_RATIO": params.get("GATE_WIDTH_RATIO"),
        "GATE_AMP_CONSISTENCY": params.get("GATE_AMP_CONSISTENCY"),
        "GATE_AMP_FRAC_PREV": params.get("GATE_AMP_FRAC_PREV"),
        "GATE_RR_RATIO": params.get("GATE_RR_RATIO"),
        "MIN_QRS_WIDTH": params.get("MIN_QRS_WIDTH"),
        "MAX_QRS_WIDTH": params.get("MAX_QRS_WIDTH"),
    }


# ============================================================================
# multiprocessing
# ============================================================================
def run_combo(job):
    name, params, sigs, golds, tol_samp = job
    p = HRParamsV6(**params)
    results = []
    for rid in sigs:
        r = run_record_v6(sigs[rid], golds[rid], p, tol_samp)
        r["record"] = rid
        results.append(r)
    return {"name": name, "params": combo_to_config(params),
            "summary": summarize_v6(results), "details": results}


# ============================================================================
# main
# ============================================================================
def main():
    global TOLERANCE_SAMP

    ap = argparse.ArgumentParser(description="LUDB 心率 v6 参数扫描 (完整状态机)")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--lead", type=str, default="ii")
    ap.add_argument("--gain", type=float, default=1000.0)
    ap.add_argument("--records", type=str, default=None, help="记录子集 (冒烟测试)")
    ap.add_argument("--tolerance-ms", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=0, help="0=min(16, 组合数)")
    ap.add_argument("--final", type=str, default=None,
                    help="选定组合名, 只跑该组合并输出 eval.json/detail.csv")
    ap.add_argument("--json", type=str, default=str(MODELS / "ludb_hr_v6_eval.json"))
    ap.add_argument("--csv", type=str, default=str(MODELS / "ludb_hr_v6_detail.csv"))
    ap.add_argument("--param-csv", type=str,
                    default=str(MODELS / "ludb_hr_v6_param_table.csv"))
    args = ap.parse_args()

    TOLERANCE_SAMP = int(args.tolerance_ms * FS / 1000.0)

    data_dir = Path(args.data_dir)
    if not (data_dir / "1.hea").exists():
        sys.exit(f"错误: 未找到 LUDB 数据 ({data_dir})")
    rec_file = data_dir.parent / "RECORDS"
    all_recs = ([l.strip().split("/")[-1] for l in rec_file.read_text().splitlines()
                 if l.strip()] if rec_file.exists()
                else sorted({f.stem for f in data_dir.glob("*.hea")}))
    if args.records:
        all_recs = [r for r in all_recs if r in args.records.split(",")]

    print(f"LUDB 目录: {data_dir}")
    print(f"导联: {args.lead}   记录数: {len(all_recs)}   增益: {args.gain}   "
          f"容差: {args.tolerance_ms}ms")

    # 预滤波缓存 (fork 后 COW 共享)
    sigs, golds = {}, {}
    for rid in all_recs:
        sig, gold = load_ludb_record(data_dir, rid, args.lead)
        sigs[rid] = chain_filter_v5(sig, args.gain)
        golds[rid] = gold
    print("预滤波完成")

    combos = build_combos()
    if args.final:
        if args.final not in combos:
            sys.exit(f"未知组合 {args.final}; 可用: {sorted(combos)}")
        combos = {args.final: combos[args.final]}

    jobs = [(name, combos[name], sigs, golds, TOLERANCE_SAMP)
            for name in combos]
    workers = args.workers or min(16, len(jobs))
    print(f"候选组合: {len(jobs)}   并行 workers: {workers}")

    if workers > 1:
        with get_context("fork").Pool(processes=workers) as pool:
            results = pool.map(run_combo, jobs, chunksize=1)
    else:
        results = [run_combo(j) for j in jobs]

    scan_path = MODELS / "ludb_hr_v6_scan.json"
    if args.final is None:
        with open(scan_path, "w", encoding="utf-8") as f:
            json.dump({r["name"]: {"params": r["params"],
                                   "summary": {k: v for k, v in r["summary"].items()
                                               if k != "audit"}}
                       for r in results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 112)
    print(f"{'组合':<26} {'Se%':>6} {'PPV%':>6} {'F1':>6} {'MAE':>6} {'中位':>6} "
          f"{'P90':>6} {'±3%':>5} {'±5%':>5} {'TP':>4} {'FP':>4} {'FN':>3}  审计")
    print("=" * 112)
    for r in sorted(results, key=lambda x: -x["summary"]["f1"]):
        s = r["summary"]
        print(f"{r['name']:<26} {s['se']*100:6.2f} {s['ppv']*100:6.2f} {s['f1']:6.4f} "
              f"{s['bpm_mae']:6.2f} {s['bpm_median_err']:6.2f} {s['bpm_p90']:6.2f} "
              f"{s['bpm_within3']:5.1f} {s['bpm_within5']:5.1f} {s['tp']:4d} "
              f"{s['fp']:4d} {s['fn']:3d}  {'OK' if not s['audit']['warnings'] else s['audit']['warnings']}")

    # 参数表 CSV (仅全量扫描时写; --final 只产出选定组合 eval/detail)
    if args.final is None:
        with open(args.param_csv, "w", newline="", encoding="utf-8") as f:
            fields = ["name", "STARTUP_BLANK_SAMP", "MIN_CONF_FEAT", "GATE_RF_MAX",
                      "GATE_WIDTH_RATIO",
                      "GATE_AMP_CONSISTENCY", "GATE_AMP_FRAC_PREV", "GATE_RR_RATIO",
                      "MIN_QRS_WIDTH", "MAX_QRS_WIDTH",
                      "records", "gold_beats", "det_beats", "tp", "fp", "fn",
                      "se", "ppv", "f1", "bpm_mae", "bpm_median_err", "bpm_p90",
                      "bpm_within3", "bpm_within5"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in results:
                row = {"name": r["name"], **r["params"], **r["summary"]}
                w.writerow(row)
        print(f"\n参数表已写: {args.param_csv}")

    # 选定/推荐
    eligible = [r for r in results if r["summary"]["se"] >= 0.95]
    if not eligible:
        eligible = results
    best = max(eligible, key=lambda r: objective_of(r["summary"]))
    print(f"\nSe>=0.95 约束下最佳: {best['name']}  "
          f"(MAE={best['summary']['bpm_mae']:.2f} PPV={best['summary']['ppv']*100:.2f}%)")

    if args.final or True:
        r = best
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "detector": f"energy-envelope v6 ({r['name']}, 2026-08-14)",
                "config": {"lead": args.lead, "gain": args.gain,
                           "tolerance_ms": args.tolerance_ms,
                           "records": len(all_recs), "bandpass": "8-25Hz",
                           "mwi_window": 40, "min_conf_feat": r["params"]["MIN_CONF_FEAT"],
                           **r["params"]},
                "summary": r["summary"],
                "selection_note": ("在 Se>=0.95 约束下, 按 BPM MAE 最小、PPV 次优选择"
                                   if args.final is None else f"--final {args.final}"),
            }, f, indent=2, ensure_ascii=False)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["record", "gold", "det", "tp", "fp", "fn",
                                              "gold_bpm", "det_bpm", "bpm_err"])
            w.writeheader()
            for d in r["details"]:
                w.writerow({k: d[k] for k in w.fieldnames})
        print(f"选定组合输出: {args.json} / {args.csv}")

    print(f"\n扫描 JSON: {scan_path}")


if __name__ == "__main__":
    main()
