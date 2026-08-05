#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTB 原始数据库预处理 (PhysioNet ptbdb, 1000Hz 12导联)
-> 统一 250Hz beat 级数据集

Pipeline:
  1. 读取 WFDB 记录 (1000Hz, 12导联, 取 II 导联)
  2. 重采样 1000 -> 250Hz
  3. ESP32 匹配滤波 (HP 0.5 + LP 40 + Notch 50)
  4. XQRS 自动 R 峰检测 (PTB 无 .atr 标注)
  5. R 峰中心 250 点窗口切拍 + z-score
  6. 标签: CONTROLS(健康对照 80 条记录) -> Normal(0), 患者记录 -> Abnormal(1)
     (记录级映射: MI/缺血等为持续性形态改变, 比阵发性心律失常更可信)

输出: data/processed/ptb_processed.npz

PTB Database:
  - 549 记录 / 294 患者, ~38s/条, 1000Hz, 12导联+Frank
  - 148 例 MI + 52 健康对照 + 其他心脏疾病
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, TARGET_FS, BEAT_WINDOW_SAMPLES
from data.preprocess_incart import apply_filters, resample_ecg

PTB_DIR = None
for cand in [
    Path(r"C:\Users\cai\OneDrive\Desktop\Fe programme 25261\ecg-programme-tju-flex.electron-master\ECG-Database"),
    Path("/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/ECG-Database"),
]:
    if cand.exists():
        PTB_DIR = cand
        break
if PTB_DIR is None:
    raise RuntimeError("PTB 数据库目录未找到, 请放到 ECG-Database/")


def load_records():
    """从 RECORDS 文件读取 549 条记录列表."""
    recs = [l.strip() for l in open(PTB_DIR / "RECORDS") if l.strip()]
    return recs


def load_controls():
    """CONTROLS 文件: 80 条健康对照记录."""
    ctrl = {l.strip() for l in open(PTB_DIR / "CONTROLS") if l.strip()}
    return ctrl


def detect_r_peaks(sig_f):
    """XQRS 自动 R 峰检测 (250Hz 滤波后信号)."""
    from wfdb.processing import xqrs_detect
    return xqrs_detect(sig_f.astype(np.float64), fs=TARGET_FS, verbose=False)


def process_record(record_name, controls):
    """单条记录 -> (beats, label)."""
    import wfdb
    rec = wfdb.rdrecord(str(PTB_DIR / record_name))
    fs = rec.fs
    lead = rec.p_signal[:, 1]  # II 导联
    sig250 = resample_ecg(lead, fs, TARGET_FS)
    sig_f = apply_filters(sig250, TARGET_FS)

    r_idx = detect_r_peaks(sig_f)
    label = 0 if record_name in controls else 1

    half = BEAT_WINDOW_SAMPLES // 2
    beats = []
    for ri in r_idx:
        lo, hi = ri - half, ri - half + BEAT_WINDOW_SAMPLES
        if lo < 0 or hi > len(sig_f):
            continue
        beat = sig_f[lo:hi]
        s = beat.std()
        if s < 1e-8:
            continue
        beat = (beat - beat.mean()) / s
        beats.append(beat)
    return np.array(beats, dtype=np.float32), label


def process_all(records=None, test_mode=False):
    if records is None:
        records = load_records()
    if test_mode:
        records = records[:3]
        print(f"[PTB] [TEST] 仅处理前 {len(records)} 条记录")

    controls = load_controls()
    all_beats, all_labels, all_rec_ids = [], [], []
    failed = []
    for i, rec_name in enumerate(records):
        print(f"[PTB] [{i+1}/{len(records)}] {rec_name}...", end=" ")
        try:
            beats, label = process_record(rec_name, controls)
            if len(beats) == 0:
                print("[SKIP] no beats"); failed.append(rec_name); continue
            all_beats.append(beats)
            all_labels.append(np.full(len(beats), label, dtype=np.int32))
            # record_id: 400000 + 记录序号 (避免与 MIT/INCART 冲突)
            all_rec_ids.append(np.full(len(beats), 400000 + i, dtype=np.int32))
            print(f"[OK] {len(beats)}拍 (label={label})")
        except Exception as e:
            print(f"[FAIL] {e}"); failed.append(rec_name)

    if not all_beats:
        raise RuntimeError("[PTB] 所有记录处理失败!")
    beats = np.concatenate(all_beats)
    labels = np.concatenate(all_labels)
    rec_ids = np.concatenate(all_rec_ids)
    nN, nA = int((labels == 0).sum()), int((labels == 1).sum())
    print(f"\n[PTB] Total: {len(beats)} beats, N={nN}, A={nA}")
    if failed:
        print(f"[PTB] Failed: {failed}")
    return beats, labels, rec_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PTB preprocessing")
    parser.add_argument("--test", action="store_true", help="测试模式 (前3条)")
    args = parser.parse_args()

    beats, labels, rec_ids = process_all(test_mode=args.test)
    out = PROCESSED_DIR / "ptb_processed.npz"
    np.savez_compressed(out, beats=beats, labels=labels, record_ids=rec_ids)
    print(f"[PTB] Saved: {out} ({out.stat().st_size/1024/1024:.1f} MB)")
    print("\n[DONE] 下一步: dataset.py 加载 + train.py --ptb-beat")
