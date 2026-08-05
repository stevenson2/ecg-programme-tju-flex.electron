#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_ondevice_bench.py — On-Device Benchmark Analysis
========================================================

Parses CSV output from ESP32-S3 on-device benchmark measurements and
produces a summary table (latency distribution, power, INT8 agreement,
thermal curve).

Usage:
    python analyze_ondevice_bench.py --latency ondevice_latency_20260803.csv \\
                                     --power ondevice_power_20260803.csv \\
                                     --int8 ondevice_int8_20260803.csv \\
                                     --thermal ondevice_thermal_20260803.csv \\
                                     --output ondevice_bench_summary.json

Input CSVs are produced by following the protocol in
docs/hardware/ondevice_bench_protocol.md.  This script runs AFTER the
student collects data — it does NOT connect to hardware.

Measures:
  1. Inference latency: mean, std, P95, P99, min, max (us & ms)
  2. Power: idle / AI-active / BLE-active current + power (mA / mW)
  3. INT8 consistency: AUC rank preservation, mean|Δp|, agreement rate
  4. Thermal: T_start, T_max, ΔT, per-minute averages
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# Constants
# ============================================================

EXPECTED_LATENCY_RANGE_US = (80_000, 120_000)  # 80–120 ms per window
VOLTAGE_V = 3.3  # ESP32-S3 operating voltage
INT8_THRESHOLDS = [0.35, 0.50]
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 42

# Historical baselines (from TUNING_HISTORY §13.4)
HIST_MEAN_ABS_DELTA_P = 0.25
HIST_MAX_ABS_DELTA_AUC = 0.025


# ============================================================
# 1. Latency parser
# ============================================================

def parse_latency_csv(path: str) -> np.ndarray:
    """Parse ondevice_latency_<date>.csv.

    Expected format (one of two):
      Format A (firmware patched):  LAT,<inference_count>,<latency_us>
      Format B (plain us):          <latency_us>  (one column)

    Returns 1D float64 array of latency values in microseconds.
    Skips the first 5 rows (cold-start warmup).
    """
    raw = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3 and parts[0].strip() == "LAT":
                # Format A: LAT,count,latency_us
                try:
                    raw.append(float(parts[2]))
                except ValueError:
                    continue
            else:
                # Format B: single float per line
                try:
                    raw.append(float(parts[0]))
                except ValueError:
                    continue

    if not raw:
        raise ValueError(f"No latency values found in {path}")

    arr = np.array(raw, dtype=np.float64)
    # Skip first 5 samples (cold start)
    if len(arr) > 5:
        arr = arr[5:]
    return arr


def compute_latency_stats(arr_us: np.ndarray) -> dict:
    """Compute latency distribution stats from microsecond array."""
    n = len(arr_us)
    if n == 0:
        return {"error": "no data after warmup exclusion"}

    arr_ms = arr_us / 1000.0  # us -> ms

    return {
        "n_windows": int(n),
        "mean_us": float(np.mean(arr_us)),
        "mean_ms": float(np.mean(arr_ms)),
        "std_us": float(np.std(arr_us)),
        "std_ms": float(np.std(arr_ms)),
        "min_us": float(np.min(arr_us)),
        "max_us": float(np.max(arr_us)),
        "p50_us": float(np.percentile(arr_us, 50)),
        "p95_us": float(np.percentile(arr_us, 95)),
        "p99_us": float(np.percentile(arr_us, 99)),
        "p95_ms": float(np.percentile(arr_ms, 95)),
        "p99_ms": float(np.percentile(arr_ms, 99)),
        "within_expected": bool(
            EXPECTED_LATENCY_RANGE_US[0]
            <= np.percentile(arr_us, 95)
            <= EXPECTED_LATENCY_RANGE_US[1]
        ),
    }


# ============================================================
# 2. Power parser
# ============================================================

def parse_power_csv(path: str) -> dict:
    """Parse ondevice_power_<date>.csv.

    Expected format:
      mode,voltage_v,current_ma,power_mw,notes
      idle_baseline,5.12,42.0,215.0,...
      ai_active,,,,
      ble_active,,,,
    """
    records = {}
    with open(path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("#"):
                    continue
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                header = parts
                continue
            mode = parts[0]
            rec = {}
            for i, h in enumerate(header):
                if i >= len(parts):
                    continue
                val = parts[i]
                if val:
                    try:
                        rec[h] = float(val)
                    except ValueError:
                        rec[h] = val
            if mode:
                records[mode] = rec

    summary = {}
    for mode in ["idle_baseline", "ai_active", "ble_active"]:
        if mode in records:
            r = records[mode]
            current_ma = r.get("current_ma", None)
            power_mw = r.get("power_mw", None)
            if current_ma is not None and power_mw is None:
                power_mw = current_ma * VOLTAGE_V
            summary[mode] = {
                "voltage_v": r.get("voltage_v", None),
                "current_ma": current_ma,
                "power_mw": power_mw,
            }

    # Compute deltas
    if "idle_baseline" in summary and "ai_active" in summary:
        idle_p = summary["idle_baseline"].get("power_mw")
        ai_p = summary["ai_active"].get("power_mw")
        if idle_p and ai_p:
            summary["ai_delta_mw"] = ai_p - idle_p
    if "ai_active" in summary and "ble_active" in summary:
        ai_p = summary["ai_active"].get("power_mw")
        ble_p = summary["ble_active"].get("power_mw")
        if ai_p and ble_p:
            summary["ble_delta_mw"] = ble_p - ai_p

    return summary


# ============================================================
# 3. INT8 consistency parser
# ============================================================

def parse_int8_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ondevice_int8_<date>.csv.

    Expected format:
      beat_index,label,pc_prob_abnormal,esp32_prob_abnormal,abs_delta
    """
    beats = []
    labels = []
    pc_probs = []
    esp32_probs = []
    with open(path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                header = parts
                continue
            try:
                beats.append(int(parts[0]))
                labels.append(int(parts[1]))
                pc_probs.append(float(parts[2]))
                esp32_probs.append(float(parts[3]))
            except (ValueError, IndexError):
                continue

    if not beats:
        raise ValueError(f"No INT8 data found in {path}")

    return (
        np.array(labels, dtype=np.int32),
        np.array(pc_probs, dtype=np.float64),
        np.array(esp32_probs, dtype=np.float64),
    )


def compute_int8_stats(
    labels: np.ndarray, pc_probs: np.ndarray, esp32_probs: np.ndarray
) -> dict:
    """Compute INT8 consistency metrics."""
    from sklearn.metrics import roc_auc_score

    n = len(labels)
    if n == 0:
        return {"error": "no data"}

    abs_delta = np.abs(esp32_probs - pc_probs)
    mean_dp = float(np.mean(abs_delta))
    max_dp = float(np.max(abs_delta))

    # AUC
    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2:
        auc_pc = float(roc_auc_score(labels, pc_probs))
        auc_esp = float(roc_auc_score(labels, esp32_probs))
        delta_auc = auc_esp - auc_pc
    else:
        auc_pc = auc_esp = delta_auc = None

    # Agreement rate at thresholds
    agreement = {}
    for thr in INT8_THRESHOLDS:
        pred_pc = (pc_probs >= thr).astype(int)
        pred_esp = (esp32_probs >= thr).astype(int)
        agree = float(np.mean(pred_pc == pred_esp))
        agreement[f"thr_{thr:.2f}"] = agree

    # Verdict
    verdict_hard = "PASS" if (
        (delta_auc is not None and abs(delta_auc) <= 0.01)
        and max_dp <= 0.05
        and agreement.get("thr_0.50", 0) >= 0.99
    ) else "FAIL"

    verdict_expected = "PASS" if (
        (delta_auc is not None and abs(delta_auc) <= HIST_MAX_ABS_DELTA_AUC)
        and agreement.get("thr_0.50", 0) >= 0.94
    ) else "FAIL"

    return {
        "n_beats": int(n),
        "n_normal": int(np.sum(labels == 0)),
        "n_abnormal": int(np.sum(labels == 1)),
        "mean_abs_delta_p": mean_dp,
        "max_abs_delta_p": max_dp,
        "auc_pc": auc_pc,
        "auc_esp32": auc_esp,
        "delta_auc": delta_auc,
        "agreement": agreement,
        "verdict_strict": verdict_hard,
        "verdict_expected": verdict_expected,
        "historical_mean_dp": HIST_MEAN_ABS_DELTA_P,
    }


# ============================================================
# 4. Thermal parser
# ============================================================

def parse_thermal_csv(path: str) -> np.ndarray:
    """Parse ondevice_thermal_<date>.csv.

    Expected format:
      time_s,temperature_c
    """
    times = []
    temps = []
    with open(path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                header = parts
                continue
            try:
                times.append(float(parts[0]))
                temps.append(float(parts[1]))
            except (ValueError, IndexError):
                continue

    if not temps:
        raise ValueError(f"No temperature data found in {path}")

    return np.array(temps, dtype=np.float64), np.array(times, dtype=np.float64)


def compute_thermal_stats(
    temps: np.ndarray, times: np.ndarray
) -> dict:
    """Compute thermal curve stats."""
    n = len(temps)
    duration_s = float(times[-1] - times[0]) if n > 1 else 0.0

    t_start = float(temps[0])
    t_max = float(np.max(temps))
    t_min = float(np.min(temps))
    t_mean = float(np.mean(temps))
    delta_t = t_max - t_start

    # Overheat check
    overheated = t_max > 65.0
    warned = t_max > 55.0

    # Per-minute averages
    per_minute = []
    if duration_s >= 60:
        n_minutes = max(1, int(duration_s / 60))
        for m in range(n_minutes):
            lo = m * 60
            hi = min((m + 1) * 60, n)
            mask = (times >= lo) & (times < hi)
            if np.any(mask):
                per_minute.append({
                    "minute": m + 1,
                    "mean_temp_c": float(np.mean(temps[mask])),
                    "min_temp_c": float(np.min(temps[mask])),
                    "max_temp_c": float(np.max(temps[mask])),
                })

    return {
        "n_samples": int(n),
        "duration_s": round(duration_s, 1),
        "t_start_c": round(t_start, 2),
        "t_max_c": round(t_max, 2),
        "t_min_c": round(t_min, 2),
        "t_mean_c": round(t_mean, 2),
        "delta_t_c": round(delta_t, 2),
        "overheated": overheated,
        "high_temp_warning": warned,
        "per_minute": per_minute,
    }


# ============================================================
# Summary & output
# ============================================================

def build_summary(
    latency_stats: Optional[dict],
    power_stats: Optional[dict],
    int8_stats: Optional[dict],
    thermal_stats: Optional[dict],
) -> dict:
    """Assemble the full benchmark summary."""
    return {
        "meta": {
            "protocol_version": "v1.0",
            "protocol_doc": "docs/hardware/ondevice_bench_protocol.md",
            "board": "ESP32-S3-SUPERMINI (ESP32S3FH4R2)",
            "cpu_freq_mhz": 240,
            "voltage_v": VOLTAGE_V,
            "model": "exp5_clean_int8.tflite (or as specified)",
            "notes": (
                "All measurements performed by student following protocol; "
                "analysis script runs post-collection."
            ),
        },
        "1_inference_latency": latency_stats,
        "2_power": power_stats,
        "3_int8_consistency": int8_stats,
        "4_thermal": thermal_stats,
    }


def print_summary_table(summary: dict) -> None:
    """Print a human-readable summary table to stdout."""
    print("=" * 72)
    print("  ESP32-S3 ON-DEVICE BENCHMARK SUMMARY")
    print("=" * 72)

    # Latency
    lat = summary.get("1_inference_latency", {})
    if lat and "error" not in lat:
        print(f"\n  [1] INFERENCE LATENCY  ({lat['n_windows']} windows)")
        print(f"      Mean: {lat['mean_ms']:.2f} ms  "
              f"Std: {lat['std_ms']:.2f} ms")
        print(f"      P50: {lat['p50_us'] / 1000:.2f} ms  "
              f"P95: {lat['p95_ms']:.2f} ms  "
              f"P99: {lat['p99_ms']:.2f} ms")
        print(f"      Range: [{lat['min_us'] / 1000:.2f}, {lat['max_us'] / 1000:.2f}] ms")
        print(f"      Expected (80–120ms): {'PASS' if lat['within_expected'] else 'FAIL'}")

    # Power
    pwr = summary.get("2_power", {})
    if pwr:
        print(f"\n  [2] POWER")
        for mode in ["idle_baseline", "ai_active", "ble_active"]:
            if mode in pwr:
                r = pwr[mode]
                print(f"      {mode:20s}: {r.get('current_ma','?'):>6} mA"
                      f"  {r.get('power_mw','?'):>7} mW")
        if "ai_delta_mw" in pwr:
            print(f"      AI inference delta: +{pwr['ai_delta_mw']:.1f} mW")
        if "ble_delta_mw" in pwr:
            print(f"      BLE active delta:   +{pwr['ble_delta_mw']:.1f} mW")

    # INT8
    int8 = summary.get("3_int8_consistency", {})
    if int8 and "error" not in int8:
        print(f"\n  [3] INT8 CONSISTENCY  ({int8['n_beats']} beats: "
              f"{int8['n_normal']} N, {int8['n_abnormal']} A)")
        if int8["auc_pc"] is not None:
            print(f"      AUC_PC: {int8['auc_pc']:.4f}  "
                  f"AUC_ESP32: {int8['auc_esp32']:.4f}  "
                  f"ΔAUC: {int8['delta_auc']:+.5f}")
        print(f"      mean|Δp|: {int8['mean_abs_delta_p']:.4f}  "
              f"max|Δp|: {int8['max_abs_delta_p']:.4f}")
        for thr_key, agree in int8.get("agreement", {}).items():
            print(f"      Agree@{thr_key.split('_')[1]}: {agree * 100:.1f}%")
        print(f"      Verdict (expected, |ΔAUC|≤0.025, agree≥94%): "
              f"{int8['verdict_expected']}")

    # Thermal
    thm = summary.get("4_thermal", {})
    if thm and "error" not in thm:
        print(f"\n  [4] THERMAL  ({thm['n_samples']} samples, "
              f"{thm['duration_s']:.0f}s)")
        print(f"      T_start: {thm['t_start_c']:.1f}°C  "
              f"T_max: {thm['t_max_c']:.1f}°C  "
              f"ΔT: {thm['delta_t_c']:.1f}°C")
        print(f"      Mean: {thm['t_mean_c']:.1f}°C  "
              f"Range: [{thm['t_min_c']:.1f}, {thm['t_max_c']:.1f}]°C")
        if thm["overheated"]:
            print(f"      ⚠ OVERHEAT: exceeded 65°C threshold")
        if thm.get("per_minute"):
            print(f"      Per-minute averages:")
            for pm in thm["per_minute"]:
                print(f"        min {pm['minute']:>2d}: "
                      f"mean={pm['mean_temp_c']:.1f}°C  "
                      f"[{pm['min_temp_c']:.1f}–{pm['max_temp_c']:.1f}]°C")

    print("\n" + "=" * 72)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ESP32-S3 on-device benchmark CSVs"
    )
    parser.add_argument(
        "--latency",
        type=str,
        default=None,
        help="Path to ondevice_latency_<date>.csv",
    )
    parser.add_argument(
        "--power",
        type=str,
        default=None,
        help="Path to ondevice_power_<date>.csv",
    )
    parser.add_argument(
        "--int8",
        type=str,
        default=None,
        help="Path to ondevice_int8_<date>.csv",
    )
    parser.add_argument(
        "--thermal",
        type=str,
        default=None,
        help="Path to ondevice_thermal_<date>.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON summary path (default: stdout only)",
    )
    args = parser.parse_args()

    latency_stats = None
    power_stats = None
    int8_stats = None
    thermal_stats = None

    if args.latency:
        try:
            arr = parse_latency_csv(args.latency)
            latency_stats = compute_latency_stats(arr)
            print(f"[OK] Latency: {latency_stats['n_windows']} windows parsed")
        except Exception as e:
            latency_stats = {"error": str(e)}
            print(f"[ERR] Latency parse failed: {e}")

    if args.power:
        try:
            power_stats = parse_power_csv(args.power)
            print(f"[OK] Power: {len(power_stats)} modes parsed")
        except Exception as e:
            power_stats = {"error": str(e)}
            print(f"[ERR] Power parse failed: {e}")

    if args.int8:
        try:
            labels, pc_p, esp_p = parse_int8_csv(args.int8)
            int8_stats = compute_int8_stats(labels, pc_p, esp_p)
            print(f"[OK] INT8: {int8_stats['n_beats']} beats parsed")
        except Exception as e:
            int8_stats = {"error": str(e)}
            print(f"[ERR] INT8 parse failed: {e}")

    if args.thermal:
        try:
            temps, times = parse_thermal_csv(args.thermal)
            thermal_stats = compute_thermal_stats(temps, times)
            print(f"[OK] Thermal: {thermal_stats['n_samples']} samples, "
                  f"{thermal_stats['duration_s']:.0f}s")
        except Exception as e:
            thermal_stats = {"error": str(e)}
            print(f"[ERR] Thermal parse failed: {e}")

    summary = build_summary(latency_stats, power_stats, int8_stats, thermal_stats)
    print_summary_table(summary)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
