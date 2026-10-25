"""Firmware-matched constants for the AF RR-source experiment.

Values are copied from the ESP-IDF sources (not guessed):
  af_detect.h, heartrate.cpp (v6 gates around isQRSValid).
"""
from __future__ import annotations

FW_FS = 500
FW_TS_US = 2000

# af_detect.h defaults (10 s fast-check)
AF_WIN_S = 10.0
AF_MIN_RR = 6
AF_CV_THR = 0.08
AF_ENT_THR = 1.2
AF_SQI_GATE = 0.5
AF_RR_BUF = 120
AF_MAX_RR_S = 3.0
AF_ENT_LO = 0.3
AF_ENT_HI = 1.5
AF_ENT_BINS = 16

# heartrate.cpp active v6 gates (MIN_CONF_FEAT=1000 keeps the old block off)
MIN_RR_SEC = 0.480
MAX_RR_SEC = 2.000
RR_RATIO_MIN = 0.65
AMP_FRAC_PREV = 0.55
AMP_FRAC_PREV_RR_MAX = 0.9
REFRACTORY_MS = 200
AIQRS_REFRACTORY_MS = 250

PEAK_MATCH_TOL_S = 0.150
MIN_EPISODE_S = 10.0
ALARM_MERGE_GAP_S = 5.0
WINDOW_OVERLAP_POS = 0.5

BOOTSTRAP_N = 1000
SEED = 20261021

# Calibration grid (development set only)
CAL_CV_GRID = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
CAL_ENT_GRID = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
CAL_MIN_ANALYZABLE = 0.30

QRS_BEAT_SYMBOLS = frozenset(
    list("NLRAJSVFEEe/fQj") + ["a"]
)

AFIB_CODES = frozenset({"AFIB", "AF"})
AFL_CODES = frozenset({"AFL", "AFLT"})

# MIT-BIH known same-patient pairs
MIT_PATIENT_MAP = {"201": "201_202", "202": "201_202"}
