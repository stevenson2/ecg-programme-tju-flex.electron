#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_heartrate_ludb_v5.py — LUDB 金标准验证固件能量包络心率算法 (v5, 2026-08-14)
=================================================================================

复现 ESP32 固件当前部署的完整心率检测链路 (heartrate.cpp 2026-08-14 重构版),
用 LUDB (导联 ii, 500Hz, 200 记录 / 1,831 手标 QRS 峰) 评估:

    50Hz 双级 10 抽头梳状 (main.cpp)
  → HP 0.5Hz + LP 40Hz (filter.cpp 完整 double 精度系数)
  → heartrate.cpp v5 状态机:
        QRS 带通 8-25Hz (LP25 + HP8, butter(2,[8,25],fs=500))
        → 能量包络 x² (弃导数平方)
        → MWI 40 采样滑动积分 (80ms)
        → 自适应峰值阈值 (窗口 MWI max × 0.4, beatCount==0 滚动重学)
        → 200ms 不应期 (采样等时)
        → millis 计时 RR (离线 = 样本索引 / fs)
        → 形态学验证关闭 (MIN_CONF_FEAT 1000)

旧版 (v4.2 导数平方 + 5-15Hz 带通) 见 verify_heartrate_ludb.py (Se 72.9% / PPV
82.6% / F1 0.774 / BPM MAE 3.2, 论文表 T12 已脱节, 待本脚本更新)。

LUDB 信号为 mV 级, 固件输入为 V 级, 默认 --gain 1000 缩放 (与旧脚本一致)。

用法 (WSL2 Ubuntu):
  $ python3 pc_tools/ecg_dl/verify_heartrate_ludb_v5.py \
        --json models/ludb_hr_v5_eval.json --csv models/ludb_hr_v5_detail.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import wfdb
except ImportError:
    wfdb = None

FS = 500.0
TS = 1.0 / FS
TOLERANCE_MS = 150  # AAMI 标准匹配容差
TOLERANCE_SAMP = int(TOLERANCE_MS * FS / 1000)  # 75 @500Hz

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    ROOT
    / "ECG-Database"
    / "lobachevsky-university-electrocardiography-database-1.0.1"
    / "lobachevsky-university-electrocardiography-database-1.0.1"
    / "data"
)

# ============ 前置滤波链系数 (filter.cpp 完整 double 精度, fs=500) ============
# HP 0.5Hz (2026-08-13 TH §44: 显示/心率链 HP 0.05→0.5Hz, 基线稳定)
HP = dict(b=(0.9955669720176472, -1.9911339440352944, 0.9955669720176472),
          a=(1.0, -1.9911142922016536, 0.9911535958689355))
# LP 40Hz (完整 double 精度)
LP = dict(b=(0.046131802093312926, 0.09226360418662585, 0.046131802093312926),
          a=(1.0, -1.3072850288493234, 0.4918122372225752))

# ============ QRS 专用带通 8-25Hz 系数 (heartrate.cpp 2026-08-14) ============
# butter(2, [8, 25], 'band', fs=500), Direct-Form-II Transposed
# LP 25Hz
QRS_LP25_A1 = -1.561018
QRS_LP25_A2 = 0.641352
QRS_LP25_B0 = 0.020083
QRS_LP25_B1 = 0.040167
QRS_LP25_B2 = 0.020083
# HP 8Hz
QRS_HP8_A1 = -1.858043
QRS_HP8_A2 = 0.867472
QRS_HP8_B0 = 0.931379
QRS_HP8_B1 = -1.862758
QRS_HP8_B2 = 0.931379


class HRParamsV5:
    """heartrate.cpp 2026-08-14 能量包络版全部算法常数"""

    def __init__(self, **kw):
        self.MWI_WINDOW = 40            # 滑动积分 80ms @500Hz
        self.REFRACTORY_SAMP = 100      # 不应期 200ms @500Hz
        self.RR_BUFFER_SIZE = 8
        self.THRESHOLD_INIT = 0.0001    # 能量包络域 (1e-4)
        self.THRESHOLD_RATIO = 0.30
        self.SIGNAL_WEIGHT = 0.0625
        self.NOISE_WEIGHT = 0.03
        self.SIGNAL_WEIGHT_FAST = 0.25
        self.SIGNAL_WEIGHT_MOT = 0.02
        self.MIN_RR_SEC = 0.480         # 最小 RR (秒), isQRSValid 硬拒
        self.MAX_RR_SEC = 2.000         # 最大 RR (秒)
        self.MIN_RR_SAMP = 240          # 480ms @500Hz (addRRInterval)
        self.MAX_RR_SAMP = 1000         # 2000ms @500Hz
        self.TIMEOUT_SAMP = 1500        # 3 秒无 QRS → 软复位
        self.HOLD_SAMP = 500            # 1 秒无新拍 → 停止输出旧 BPM
        self.MIN_CONF_BEATS = 5
        self.MIN_CONF_FEAT = 1000       # 形态学验证禁用 (1000 拍内永不触发)
        self.MIN_PEAK_RATIO = 1.2
        self.MIN_PEAK_RATIO_MOT = 1.2
        self.ACT_WIN_SAMP = 500
        self.ACT_THRESHOLD = 0.005
        self.ACT_TIMEOUT_CNT = 2
        self.ADAPT_INIT_SAMP = 100
        self.ADAPT_INIT_PEAK_FACTOR = 0.4   # 峰基法 max×0.4 (固件硬编码)
        self.SQI_EMA_WEIGHT = 0.05
        self.SQI_MOTION_ENTER = 0.35
        self.SQI_MOTION_EXIT = 0.55
        self.SQI_SNR_FLOOR = 0.0001
        self.MOTION_BPM_HOLD = 1500
        self.MOTION_ENTER_CNT = 250
        self.MOTION_EXIT_CNT = 100
        self.MWI_HIST_LEN = 120
        self.PEAK_HIST_LEN = 8
        self.MIN_QRS_WIDTH = 25
        self.MAX_QRS_WIDTH = 100
        self.AMP_CONSISTENCY = 0.35
        self.RR_CONSISTENCY = 0.30
        self.RISE_FALL_MIN = 0.5
        self.RISE_FALL_MAX = 2.0
        self.MIN_QRS_WIDTH_MOT = 30
        self.MAX_QRS_WIDTH_MOT = 100
        self.RISE_FALL_MIN_MOT = 0.35
        self.RISE_FALL_MAX_MOT = 2.5
        self.BPM_EMA_WEIGHT_FAST = 0.30
        self.BPM_EMA_WEIGHT_SLOW = 0.10
        self.BPM_EMA_FADE_STEPS = 500
        self.BPM_EMA_WEIGHT_ANOM = 0.05
        self.BPM_ANOMALY_THRESH = 0.40
        self.BPM_SLEW_MAX = 3.0
        self.BPM_CONFIRMED_LEN = 5
        self.BPM_REJECT_DEV = 0.30
        self.BPM_REJECT_DEV_MOT = 0.45
        self.BPM_REJECT_MIN_CNT = 3
        for k, v in kw.items():
            setattr(self, k, v)


class HRDetectorV5:
    """heartrate.cpp 2026-08-14 能量包络状态机逐样本复现"""

    def __init__(self, p: HRParamsV5, verbose=False):
        self.p = p
        self.verbose = verbose
        self.i = 0  # 全局样本索引 (millis() 等时, 500Hz 精确)
        self.reset_state()
        self.bpm_out = []
        self.sqi_out = []
        self.motion_out = []

    # ---------- 全量初始化 (对应 hrInit + hrFullReset) ----------
    def reset_state(self):
        p = self.p
        self.qrs_lp_w1 = 0.0
        self.qrs_lp_w2 = 0.0
        self.qrs_hp_w1 = 0.0
        self.qrs_hp_w2 = 0.0
        self.mwi_buf = np.zeros(p.MWI_WINDOW)
        self.mwi_idx = 0
        self.mwi_sum = 0.0
        self.mwi_prev = 0.0
        self.mwi_prevprev = 0.0

        self.state = "LEARNING"
        self.last_beat_sample = 0
        self.refract_until_sample = 0
        self.signal_peak = p.THRESHOLD_INIT
        self.noise_peak = p.THRESHOLD_INIT * 0.3
        self.threshold = p.THRESHOLD_INIT

        self.rr_buf = np.zeros(p.RR_BUFFER_SIZE)
        self.rr_idx = 0
        self.rr_count = 0
        self.median_rr = 0.0
        self.last_rr = 0.0

        self.beat_count = 0

        self.signal_present = True
        self.win_max_abs = 0.0
        self.win_count = 0
        self.no_signal_seconds = 0

        self.sqi = 0.5
        self.motion_active = False
        self.motion_confirmed = False
        self.motion_low_count = 0
        self.motion_high_count = 0
        self.motion_recover_cnt = 0
        self.motion_hold_sp = 0.0
        self.motion_hold_np = 0.0

        self.mwi_history = np.zeros(p.MWI_HIST_LEN)
        self.mwi_hist_idx = 0
        self.recent_peaks = np.zeros(p.PEAK_HIST_LEN)
        self.peak_hist_idx = 0
        self.peak_hist_count = 0

        self.bpm_ema = 0.0
        self.bpm_ema_fade_cnt = 0
        self.last_output_bpm = 0.0

        self.confirmed_bpm = np.zeros(p.BPM_CONFIRMED_LEN)
        self.confirmed_bpm_idx = 0
        self.confirmed_bpm_count = 0

        self.adapt_init_done = False
        self.adapt_init_count = 0
        self.adapt_win_max = 0.0

    # ---------- 复位 (对应 hrReset, 保留 signal_present) ----------
    def hr_reset(self):
        sp = self.signal_present
        self.reset_state()
        self.signal_present = sp
        self.state = "LEARNING"

    # ---------- 软复位 (对应 hrSoftReset, 保留自适应阈值) ----------
    def hr_soft_reset(self):
        hold_sp = self.signal_peak
        hold_np = self.noise_peak
        hold_th = self.threshold
        self.hr_reset()
        self.signal_peak = hold_sp
        self.noise_peak = hold_np
        self.threshold = hold_th

    # ---------- 工具函数 ----------
    def apply_qrs_bandpass(self, x):
        # LP 25Hz (DF2T)
        w_lp = x - QRS_LP25_A1 * self.qrs_lp_w1 - QRS_LP25_A2 * self.qrs_lp_w2
        y_lp = (QRS_LP25_B0 * w_lp + QRS_LP25_B1 * self.qrs_lp_w1
                + QRS_LP25_B2 * self.qrs_lp_w2)
        self.qrs_lp_w2 = self.qrs_lp_w1
        self.qrs_lp_w1 = w_lp
        # HP 8Hz (DF2T)
        w_hp = y_lp - QRS_HP8_A1 * self.qrs_hp_w1 - QRS_HP8_A2 * self.qrs_hp_w2
        y_hp = (QRS_HP8_B0 * w_hp + QRS_HP8_B1 * self.qrs_hp_w1
                + QRS_HP8_B2 * self.qrs_hp_w2)
        self.qrs_hp_w2 = self.qrs_hp_w1
        self.qrs_hp_w1 = w_hp
        return y_hp

    def compute_mwi(self, squared):
        p = self.p
        self.mwi_sum -= self.mwi_buf[self.mwi_idx]
        self.mwi_buf[self.mwi_idx] = squared
        self.mwi_sum += squared
        self.mwi_idx = (self.mwi_idx + 1) % p.MWI_WINDOW
        return self.mwi_sum / float(p.MWI_WINDOW)

    def update_threshold(self, peak_val, is_signal):
        p = self.p
        if is_signal:
            if self.motion_confirmed:
                weight = p.SIGNAL_WEIGHT_MOT
            elif self.motion_recover_cnt > 0:
                weight = p.SIGNAL_WEIGHT_FAST
            else:
                weight = p.SIGNAL_WEIGHT
            self.signal_peak = weight * peak_val + (1.0 - weight) * self.signal_peak
        else:
            self.noise_peak = p.NOISE_WEIGHT * peak_val + (1.0 - p.NOISE_WEIGHT) * self.noise_peak

        delta = self.signal_peak - self.noise_peak
        if delta < 0.0001:
            delta = 0.0001
        self.threshold = self.noise_peak + p.THRESHOLD_RATIO * delta
        if self.threshold < p.THRESHOLD_INIT:
            self.threshold = p.THRESHOLD_INIT

    def update_signal_peak(self, peak_val):
        # 2026-08-14: 高于阈值但被 RR/形态门拒的峰仅喂 signalPeak (不触碰阈值/噪声)
        w = self.p.SIGNAL_WEIGHT
        self.signal_peak = w * peak_val + (1.0 - w) * self.signal_peak

    def update_sqi(self):
        p = self.p
        denom = self.signal_peak + self.noise_peak + p.SQI_SNR_FLOOR
        raw = self.signal_peak / denom
        raw = min(1.0, max(0.0, raw))
        self.sqi = p.SQI_EMA_WEIGHT * raw + (1.0 - p.SQI_EMA_WEIGHT) * self.sqi

    def update_motion_state(self):
        p = self.p
        if not self.motion_confirmed:
            if self.sqi < p.SQI_MOTION_ENTER:
                self.motion_low_count += 1
                self.motion_high_count = 0
            else:
                self.motion_low_count = 0
            if self.motion_low_count >= p.MOTION_ENTER_CNT:
                self.motion_confirmed = True
                self.motion_active = True
                self.motion_low_count = 0
                self.motion_hold_sp = self.signal_peak
                self.motion_hold_np = self.noise_peak
                self.motion_recover_cnt = 0
                self.bpm_ema_fade_cnt = 0
        else:
            if self.sqi > p.SQI_MOTION_EXIT:
                self.motion_high_count += 1
                self.motion_low_count = 0
            else:
                self.motion_high_count = 0
            if self.motion_high_count >= p.MOTION_EXIT_CNT:
                self.motion_confirmed = False
                self.motion_active = False
                self.motion_high_count = 0
                self.motion_recover_cnt = p.MOTION_BPM_HOLD
                self.bpm_ema_fade_cnt = p.BPM_EMA_FADE_STEPS
                if self.motion_hold_sp > self.signal_peak:
                    self.signal_peak = self.motion_hold_sp
                if self.motion_hold_np < self.noise_peak:
                    self.noise_peak = self.motion_hold_np

        if self.motion_recover_cnt > 0:
            self.motion_recover_cnt -= 1
        if self.bpm_ema_fade_cnt > 0:
            self.bpm_ema_fade_cnt -= 1

    # ---------- 记录 + 形态学 (形态学验证已禁用 MIN_CONF_FEAT=1000, 保留死代码) ----------
    def record_valid_peak(self, peak_val):
        p = self.p
        self.recent_peaks[self.peak_hist_idx] = peak_val
        self.peak_hist_idx = (self.peak_hist_idx + 1) % p.PEAK_HIST_LEN
        if self.peak_hist_count < p.PEAK_HIST_LEN:
            self.peak_hist_count += 1

    def is_amplitude_consistent(self, peak_val):
        p = self.p
        if self.peak_hist_count < 3:
            return True
        mean = np.mean(self.recent_peaks[: self.peak_hist_count])
        if mean < 0.0001:
            return True
        return abs(peak_val - mean) / mean <= p.AMP_CONSISTENCY

    def get_qrs_width(self):
        p = self.p
        peak_idx = (self.mwi_hist_idx - 2 + p.MWI_HIST_LEN) % p.MWI_HIST_LEN
        peak_val = self.mwi_history[peak_idx]
        if peak_val < 0.00001:
            return 999
        half = peak_val * 0.5

        rise_count = 0
        scan = (peak_idx - 1 + p.MWI_HIST_LEN) % p.MWI_HIST_LEN
        while rise_count < (p.MWI_HIST_LEN - 2):
            val = self.mwi_history[scan]
            nxt = (scan - 1 + p.MWI_HIST_LEN) % p.MWI_HIST_LEN
            next_val = self.mwi_history[nxt]
            if val <= half:
                if next_val > half:
                    rise_count += 1
                break
            if val > next_val:
                if next_val <= half:
                    rise_count += 1
            rise_count += 1
            scan = nxt
            if scan == peak_idx:
                break

        fall_idx = (peak_idx + 1) % p.MWI_HIST_LEN
        fall_val = self.mwi_history[fall_idx]
        fall_count = 0
        if fall_val >= half:
            f = fall_idx
            while fall_count < (p.MWI_HIST_LEN - 2):
                v = self.mwi_history[f]
                if v <= half:
                    break
                fall_count += 1
                f = (f + 1) % p.MWI_HIST_LEN
                if f == self.mwi_hist_idx:
                    break
        else:
            grad = peak_val - fall_val
            if grad > 0.00001:
                fall_count = max(1, int((peak_val - half) / grad + 0.5))
            else:
                fall_count = 1

        return rise_count + fall_count + 1

    def get_rise_fall_ratio(self):
        p = self.p
        peak_idx = (self.mwi_hist_idx - 2 + p.MWI_HIST_LEN) % p.MWI_HIST_LEN
        peak_val = self.mwi_history[peak_idx]
        if peak_val < 0.00001:
            return 1.0
        half = peak_val * 0.5

        rise = 0
        scan = (peak_idx - 1 + p.MWI_HIST_LEN) % p.MWI_HIST_LEN
        while rise < (p.MWI_HIST_LEN - 2):
            if self.mwi_history[scan] <= half:
                break
            rise += 1
            scan = (scan - 1 + p.MWI_HIST_LEN) % p.MWI_HIST_LEN
            if scan == peak_idx:
                break

        fall = 0
        f = (peak_idx + 1) % p.MWI_HIST_LEN
        while fall < (p.MWI_HIST_LEN - 2):
            if self.mwi_history[f] <= half:
                break
            fall += 1
            f = (f + 1) % p.MWI_HIST_LEN
            if f == self.mwi_hist_idx:
                break
        if fall == 0:
            fall_val = self.mwi_history[(peak_idx + 1) % p.MWI_HIST_LEN]
            grad = peak_val - fall_val
            if grad > 0.00001:
                fall = max(1, int((peak_val - half) / grad + 0.5))
            else:
                fall = 1

        fall = max(1, fall)
        rise = max(1, rise)
        return float(rise) / float(fall)

    def is_rr_consistent(self, rr_sec):
        p = self.p
        if self.motion_confirmed:
            return True
        if self.rr_count < 3:
            return True
        if self.median_rr < 0.001:
            return True
        return abs(rr_sec - self.median_rr) / self.median_rr <= p.RR_CONSISTENCY

    def get_confirmed_bpm_median(self):
        if self.confirmed_bpm_count == 0:
            return 0.0
        return float(np.median(self.confirmed_bpm[: self.confirmed_bpm_count]))

    def compute_median_rr(self):
        if self.rr_count == 0:
            return 0.0
        return float(np.median(self.rr_buf[: self.rr_count]))

    def is_qrs_valid(self, peak_val, rr_sec):
        p = self.p
        if not self.signal_present:
            return False
        if self.state == "REFRACTORY":
            return False
        if peak_val <= self.threshold:
            return False
        # 2026-08-14: RR 范围用秒, 硬拒绝 (首拍 beat_count==0 跳过)
        if self.beat_count > 0:
            if rr_sec < p.MIN_RR_SEC:
                return False
            if rr_sec > p.MAX_RR_SEC:
                return False

        ratio = p.MIN_PEAK_RATIO_MOT if self.motion_confirmed else p.MIN_PEAK_RATIO
        if peak_val < self.noise_peak * ratio:
            return False

        # 形态学验证 (MIN_CONF_FEAT=1000 禁用, LUDB 拍数远达不到)
        if self.beat_count >= p.MIN_CONF_FEAT:
            if self.motion_confirmed:
                w = self.get_qrs_width()
                if w < p.MIN_QRS_WIDTH_MOT or w > p.MAX_QRS_WIDTH_MOT:
                    return False
                r = self.get_rise_fall_ratio()
                if r < p.RISE_FALL_MIN_MOT or r > p.RISE_FALL_MAX_MOT:
                    return False
            else:
                if not self.is_amplitude_consistent(peak_val):
                    return False
                w = self.get_qrs_width()
                if w < p.MIN_QRS_WIDTH or w > p.MAX_QRS_WIDTH:
                    return False
                if not self.is_rr_consistent(rr_sec):
                    return False

        if self.confirmed_bpm_count >= p.BPM_REJECT_MIN_CNT:
            med = self.get_confirmed_bpm_median()
            if med > 1.0 and rr_sec > 0.001:
                dev = abs(60.0 / rr_sec - med) / med
                rej = p.BPM_REJECT_DEV_MOT if self.motion_confirmed else p.BPM_REJECT_DEV
                if dev > rej:
                    return False

        return True

    def check_signal_activity(self, filtered_sample):
        p = self.p
        if abs(filtered_sample) > self.win_max_abs:
            self.win_max_abs = abs(filtered_sample)
        self.win_count += 1

        if self.win_count >= p.ACT_WIN_SAMP:
            if self.win_max_abs < p.ACT_THRESHOLD:
                self.no_signal_seconds += 1
            else:
                self.no_signal_seconds = 0
                if not self.signal_present:
                    self.signal_present = True
                    self.hr_reset()
                    self.state = "LEARNING"
                    self.signal_present = True
            if self.no_signal_seconds >= p.ACT_TIMEOUT_CNT:
                if self.signal_present:
                    self.signal_present = False
                    self.hr_reset()
                    self.state = "LEARNING"
            self.win_max_abs = 0.0
            self.win_count = 0

    def add_rr_interval(self, rr_seconds):
        p = self.p
        rr_samp = int(rr_seconds / TS + 0.5)
        if rr_samp < p.MIN_RR_SAMP or rr_samp > p.MAX_RR_SAMP:
            return

        self.rr_buf[self.rr_idx] = rr_seconds
        self.rr_idx = (self.rr_idx + 1) % p.RR_BUFFER_SIZE
        if self.rr_count < p.RR_BUFFER_SIZE:
            self.rr_count += 1

        old_median = self.median_rr
        self.median_rr = self.compute_median_rr()
        self.last_rr = rr_seconds

        inst_bpm = 60.0 / rr_seconds
        if self.bpm_ema < 1.0:
            self.bpm_ema = inst_bpm
        else:
            anomalous = False
            if old_median > 0.001:
                med_bpm = 60.0 / old_median
                if abs(inst_bpm - med_bpm) / med_bpm > p.BPM_ANOMALY_THRESH:
                    anomalous = True
            if anomalous:
                weight = p.BPM_EMA_WEIGHT_ANOM
            elif self.motion_confirmed or self.motion_recover_cnt > 0:
                weight = p.BPM_EMA_WEIGHT_FAST
            else:
                weight = p.BPM_EMA_WEIGHT_SLOW
            self.bpm_ema = weight * inst_bpm + (1.0 - weight) * self.bpm_ema

    def compute_output_bpm(self):
        p = self.p
        if self.median_rr < 0.001:
            return 0
        med_bpm = 60.0 / self.median_rr
        if self.motion_confirmed:
            bpm = self.bpm_ema
        elif self.bpm_ema_fade_cnt > 0 and p.BPM_EMA_FADE_STEPS > 0:
            frac = float(self.bpm_ema_fade_cnt) / float(p.BPM_EMA_FADE_STEPS)
            bpm = frac * self.bpm_ema + (1.0 - frac) * med_bpm
        else:
            bpm = med_bpm

        if self.last_output_bpm > 1.0:
            delta = bpm - self.last_output_bpm
            if delta > p.BPM_SLEW_MAX:
                bpm = self.last_output_bpm + p.BPM_SLEW_MAX
            elif delta < -p.BPM_SLEW_MAX:
                bpm = self.last_output_bpm - p.BPM_SLEW_MAX
        self.last_output_bpm = bpm

        bpm_raw = int(bpm + 0.5)
        if bpm_raw < 30 or bpm_raw > 200:
            return 0
        return bpm_raw

    # ---------- 主处理 (对应 hrProcess) ----------
    def process(self, filtered_sample):
        p = self.p
        i = self.i
        result = {"beatDetected": False, "bpm": 0, "confidence": 0.0,
                  "sqi": 0.0, "motionActive": False, "beatCount": 0, "rrInterval": 0.0}

        self.check_signal_activity(filtered_sample)
        self.update_sqi()
        self.update_motion_state()

        qrs_signal = self.apply_qrs_bandpass(filtered_sample)

        # 能量包络 x² (弃导数平方) + MWI 40
        energy = qrs_signal * qrs_signal
        mwi = self.compute_mwi(energy)

        # 自适应初始阈值 (峰基法, beatCount==0 时每 ADAPT_INIT_SAMP 滚动重学)
        if self.beat_count == 0:
            self.adapt_init_count += 1
            if mwi > self.adapt_win_max:
                self.adapt_win_max = mwi
            if self.adapt_init_count >= p.ADAPT_INIT_SAMP:
                adaptive = self.adapt_win_max * p.ADAPT_INIT_PEAK_FACTOR
                if adaptive < p.THRESHOLD_INIT:
                    adaptive = p.THRESHOLD_INIT
                if self.sqi >= 0.45 or adaptive < self.threshold:
                    self.threshold = adaptive
                    self.signal_peak = adaptive
                    self.noise_peak = adaptive * 0.3
                self.adapt_init_count = 0
                self.adapt_win_max = 0.0
                self.adapt_init_done = True

        self.mwi_history[self.mwi_hist_idx] = mwi
        self.mwi_hist_idx = (self.mwi_hist_idx + 1) % p.MWI_HIST_LEN

        is_peak = (self.mwi_prev > self.mwi_prevprev) and (self.mwi_prev > mwi)

        if is_peak:
            peak_val = self.mwi_prev
            rr_sec = (i - self.last_beat_sample) / FS if self.beat_count > 0 else 0.0

            if self.is_qrs_valid(peak_val, rr_sec):
                self.add_rr_interval(rr_sec)
                self.update_threshold(peak_val, True)
                self.record_valid_peak(peak_val)

                self.state = "REFRACTORY"
                self.last_beat_sample = i
                self.refract_until_sample = i + p.REFRACTORY_SAMP
                self.beat_count += 1

                result["beatDetected"] = True
                result["beatCount"] = self.beat_count
                result["rrInterval"] = self.last_rr

                if self.state == "LEARNING" and self.beat_count >= p.MIN_CONF_BEATS:
                    self.state = "TRACKING"

                if self.state == "TRACKING" and self.median_rr > 0.001:
                    bpm_raw = self.compute_output_bpm()
                    if 30 <= bpm_raw <= 200:
                        result["bpm"] = bpm_raw
                        self.confirmed_bpm[self.confirmed_bpm_idx] = float(bpm_raw)
                        self.confirmed_bpm_idx = (self.confirmed_bpm_idx + 1) % p.BPM_CONFIRMED_LEN
                        if self.confirmed_bpm_count < p.BPM_CONFIRMED_LEN:
                            self.confirmed_bpm_count += 1
                    buf_conf = float(self.rr_count) / float(p.RR_BUFFER_SIZE)
                    sqi_w = (self.sqi / 0.4) if self.sqi < 0.4 else 1.0
                    motion_f = 0.5 if self.motion_confirmed else 1.0
                    result["confidence"] = min(1.0, buf_conf * sqi_w * motion_f)
            else:
                # 不应期内/无信号不更新; 否则按峰相对阈值分路
                if self.state != "REFRACTORY" and self.signal_present:
                    if peak_val < self.threshold:
                        self.update_threshold(peak_val, False)   # 噪声峰
                    else:
                        self.update_signal_peak(peak_val)        # 真 QRS 被拒, 仍喂 signalPeak

        # s_sampSinceBeat++ (v5 帧率无关 RR, 该计数已弃用, 略)

        if self.state == "REFRACTORY" and i >= self.refract_until_sample:
            self.state = "TRACKING" if self.beat_count >= p.MIN_CONF_BEATS else "IDLE"

        if self.signal_present and (i - self.last_beat_sample) > p.TIMEOUT_SAMP:
            self.hr_soft_reset()
            self.state = "LEARNING"

        self.mwi_prevprev = self.mwi_prev
        self.mwi_prev = mwi

        since_beat = i - self.last_beat_sample
        if (not result["beatDetected"]) and self.beat_count > 0 and self.median_rr > 0.001                 and self.signal_present and since_beat < p.HOLD_SAMP:
            bpm_raw = self.compute_output_bpm()
            if 30 <= bpm_raw <= 200:
                result["bpm"] = bpm_raw
            decay = 1.0 - float(since_beat) / float(p.HOLD_SAMP)
            buf_conf = float(self.rr_count) / float(p.RR_BUFFER_SIZE)
            sqi_w = (self.sqi / 0.4) if self.sqi < 0.4 else 1.0
            motion_f = 0.5 if self.motion_confirmed else 1.0
            result["confidence"] = min(0.8, buf_conf * sqi_w * motion_f) * decay

        result["sqi"] = self.sqi
        result["motionActive"] = self.motion_active
        result["beatCount"] = self.beat_count
        result["rrInterval"] = self.last_rr

        self.i += 1
        return result


def chain_filter_v5(x, gain=1000.0):
    """main.cpp 50Hz 双级梳状 + filter.cpp HP 0.5Hz / LP 40Hz (完整 double 精度)"""
    from scipy.signal import lfilter

    x = np.asarray(x, dtype=np.float64) * gain
    # 双级 10 抽头滑动平均 (50Hz 陷零), 因果
    kernel = np.ones(10) / 10.0
    x = np.convolve(x, kernel, mode="full")[: len(x)]
    x = np.convolve(x, kernel, mode="full")[: len(x)]
    # HP 0.5Hz → LP 40Hz (filter.cpp)
    y = lfilter(HP["b"], HP["a"], x)
    y = lfilter(LP["b"], LP["a"], y)
    return y


def load_ludb_record(data_dir, rid, lead="ii"):
    """返回 (signal_mV, gold_qrs_samples)"""
    if wfdb is None:
        sys.exit("错误: 需要 wfdb 库, 请先 pip install wfdb")
    hdr = wfdb.rdheader(str(data_dir / rid))
    lead_idx = hdr.sig_name.index(lead)
    rec = wfdb.rdrecord(str(data_dir / rid), channels=[lead_idx])
    sig = rec.p_signal[:, 0].astype(np.float64)
    ann = wfdb.rdann(str(data_dir / rid), lead)
    gold = [int(s) for s, sym in zip(ann.sample, ann.symbol) if sym == "N"]
    return sig, gold


def run_record(sig, gold, p, gain):
    """对单条记录运行完整链路, 返回评估结果"""
    y = chain_filter_v5(sig, gain)

    det = HRDetectorV5(p)
    det_idx = []
    bpm_outputs = []
    for i in range(len(y)):
        res = det.process(y[i])
        if res["beatDetected"]:
            det_idx.append(i)
        if res["bpm"] > 0:
            bpm_outputs.append(res["bpm"])

    # ---- 匹配 (±150ms 容差, 贪婪互斥: 一个 det 只匹配一个 gold) ----
    gold_arr = np.asarray(gold, dtype=np.int64)
    det_arr = np.asarray(det_idx, dtype=np.int64)

    tp = 0
    used_det = set()
    for g in sorted(gold_arr):
        cands = np.where(np.abs(det_arr - g) <= TOLERANCE_SAMP)[0]
        if len(cands) == 0:
            continue
        best = min(cands, key=lambda c: abs(int(det_arr[c]) - int(g)))
        if best in used_det:
            continue
        used_det.add(best)
        tp += 1
    fn = len(gold_arr) - tp
    fp = len(det_arr) - tp

    # ---- BPM ----
    gold_bpm = 0.0
    if len(gold_arr) >= 2:
        rr = np.diff(gold_arr).astype(np.float64) / FS
        rr = rr[(rr >= 0.3) & (rr <= 2.0)]
        if len(rr) > 0:
            gold_bpm = 60.0 / float(np.median(rr))
    det_bpm = float(np.median(bpm_outputs)) if bpm_outputs else 0.0

    return {
        "record": None,
        "gold": len(gold_arr),
        "det": len(det_arr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold_bpm": gold_bpm,
        "det_bpm": det_bpm,
        "bpm_err": (abs(det_bpm - gold_bpm) if det_bpm > 0 and gold_bpm > 0 else None),
    }


def summarize(results):
    n_rec = len(results)
    n_gold = sum(r["gold"] for r in results)
    n_det = sum(r["det"] for r in results)
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * se * ppv / (se + ppv)) if (se + ppv) > 0 else 0.0

    errs = [r["bpm_err"] for r in results if r["bpm_err"] is not None]
    bpm_mae = float(np.mean(errs)) if errs else 0.0
    bpm_med = float(np.median(errs)) if errs else 0.0
    p90 = float(np.percentile(errs, 90)) if errs else 0.0
    within3 = 100.0 * np.mean([1.0 if e <= 3.0 else 0.0 for e in errs]) if errs else 0.0
    within5 = 100.0 * np.mean([1.0 if e <= 5.0 else 0.0 for e in errs]) if errs else 0.0

    rec_se = []
    rec_ppv = []
    for r in results:
        t = r["tp"]; f = r["fn"]; p_ = r["fp"]
        if t + f > 0:
            rec_se.append(t / (t + f))
        if t + p_ > 0:
            rec_ppv.append(t / (t + p_))

    return {
        "records": n_rec,
        "gold_beats": n_gold,
        "det_beats": n_det,
        "tp": tp, "fp": fp, "fn": fn,
        "se": se, "ppv": ppv, "f1": f1,
        "bpm_mae": bpm_mae,
        "bpm_median_err": bpm_med,
        "bpm_p90": p90,
        "bpm_within3": within3,
        "bpm_within5": within5,
        "rec_se_mean": float(np.mean(rec_se)) if rec_se else 0.0,
        "rec_ppv_mean": float(np.mean(rec_ppv)) if rec_ppv else 0.0,
        "rec_se_median": float(np.median(rec_se)) if rec_se else 0.0,
        "rec_ppv_median": float(np.median(rec_ppv)) if rec_ppv else 0.0,
    }


def print_summary(s, label="结果"):
    print("=" * 64)
    print(f"LUDB 心率验证 (v5 能量包络) — {label}")
    print("=" * 64)
    print(f"记录数: {s['records']}   金标准拍: {s['gold_beats']}   检测拍: {s['det_beats']}")
    print(f"TP={s['tp']}  FP={s['fp']}  FN={s['fn']}")
    print(f"Se = {s['se'] * 100:.2f}%   PPV = {s['ppv'] * 100:.2f}%   F1 = {s['f1']:.4f}")
    print(f"BPM 误差: MAE={s['bpm_mae']:.2f}  中位={s['bpm_median_err']:.2f}  "
          f"P90={s['bpm_p90']:.2f}  ±3BPM={s['bpm_within3']:.1f}%  ±5BPM={s['bpm_within5']:.1f}%")
    print(f"每记录 Se: 均值={s['rec_se_mean'] * 100:.1f}% 中位={s['rec_se_median'] * 100:.1f}%")
    print(f"每记录 PPV: 均值={s['rec_ppv_mean'] * 100:.1f}% 中位={s['rec_ppv_median'] * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="LUDB 金标准验证固件能量包络心率算法 (v5)")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                    help="LUDB data 目录")
    ap.add_argument("--lead", type=str, default="ii", help="导联 (默认 ii)")
    ap.add_argument("--gain", type=float, default=1000.0, help="mV→V 缩放 (默认 1000)")
    ap.add_argument("--records", type=str, default=None, help="记录子集, 逗号分隔")
    ap.add_argument("--tolerance-ms", type=float, default=TOLERANCE_MS)
    ap.add_argument("--csv", type=str, default=None, help="每记录明细 CSV 输出")
    ap.add_argument("--json", type=str, default=None, help="汇总 JSON 输出")
    args = ap.parse_args()

    global TOLERANCE_SAMP
    TOLERANCE_SAMP = int(args.tolerance_ms * FS / 1000)

    data_dir = Path(args.data_dir)
    if not (data_dir / "1.hea").exists():
        sys.exit(f"错误: 未找到 LUDB 数据 ({data_dir}). 请用 --data-dir 指定.")

    rec_file = data_dir.parent / "RECORDS"
    if rec_file.exists():
        all_recs = [l.strip().split("/")[-1] for l in rec_file.read_text().splitlines() if l.strip()]
    else:
        all_recs = sorted({f.stem for f in data_dir.glob("*.hea")})

    if args.records:
        all_recs = [r for r in all_recs if r in args.records.split(",")]

    print(f"LUDB 目录: {data_dir}")
    print(f"导联: {args.lead}   记录数: {len(all_recs)}   增益: {args.gain}   容差: {args.tolerance_ms}ms")

    p = HRParamsV5()
    results = []
    for rid in all_recs:
        sig, gold = load_ludb_record(data_dir, rid, args.lead)
        r = run_record(sig, gold, p, args.gain)
        r["record"] = rid
        results.append(r)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["record", "gold", "det", "tp", "fp", "fn",
                                              "gold_bpm", "det_bpm", "bpm_err"])
            w.writeheader()
            for r in results:
                row = {k: r[k] for k in w.fieldnames}
                w.writerow(row)
        print(f"明细已写: {args.csv}")

    s = summarize(results)
    print_summary(s)

    if args.json:
        out = {
            "detector": "energy-envelope v5 (2026-08-14)",
            "config": {
                "lead": args.lead, "gain": args.gain,
                "tolerance_ms": args.tolerance_ms, "records": len(all_recs),
                "bandpass": "8-25Hz", "mwi_window": 40,
                "threshold_init": p.THRESHOLD_INIT,
                "min_conf_feat": p.MIN_CONF_FEAT,
            },
            "summary": s,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"JSON 已写: {args.json}")


if __name__ == "__main__":
    main()
