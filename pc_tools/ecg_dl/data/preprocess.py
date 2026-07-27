#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT-BIH 心电数据预处理
- 读取原始 .dat/.hea/.atr 文件
- 心拍分割 (以 R 峰为中心, 250 点窗口)
- 重采样 360Hz -> 250Hz
- 标签映射 (15类 -> 二分类)
- 数据增强
- 保存为 NumPy 数组 (.npy)
"""

import sys
import os
from pathlib import Path
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, List, Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MIT_BIH_RECORDS, MIT_BIH_LOCAL_DIR, MIT_BIH_TEST_DIR, MIT_BIH_TEST_RECORDS,
    PROCESSED_DIR, TARGET_FS, ORIGINAL_FS, BEAT_WINDOW_SAMPLES,
    AAMI_CLASSES, CLASS_NAMES, TRAIN_CONFIG,
)


def find_record_path(record_name: str) -> Path:
    """
    查找记录文件路径 (支持多个数据目录)
    """
    test_path = MIT_BIH_TEST_DIR / record_name
    if test_path.with_suffix('.dat').exists():
        return test_path
    
    main_path = MIT_BIH_LOCAL_DIR / record_name
    if main_path.with_suffix('.dat').exists():
        return main_path
    
    raise FileNotFoundError(
        f"[预处理] 记录 {record_name} 未找到\n"
        f"   尝试路径:\n"
        f"     - {test_path.with_suffix('.dat')}\n"
        f"     - {main_path.with_suffix('.dat')}"
    )


def load_mit_bih_record(record_name: str) -> Tuple:
    """
    加载一条 MIT-BIH 记录
    """
    record_path = find_record_path(record_name)
    
    try:
        import wfdb
        record = wfdb.rdrecord(str(record_path))
        annotation = wfdb.rdann(str(record_path), 'atr')
        
        signal = record.p_signal.astype(np.float32)
        fs = record.fs
        ann_indices = annotation.sample
        ann_symbols = [s.decode('utf-8') if isinstance(s, bytes) else s 
                       for s in annotation.symbol]
        
        return signal, ann_indices, ann_symbols, fs
        
    except ImportError:
        print("[预处理] wfdb 未安装, 尝试手动读取...")
        return _load_mit_bih_manual(record_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[预处理] 记录 {record_name} 未找到: {e}")


def _load_mit_bih_manual(record_path: Path):
    """手动加载 MIT-BIH 数据"""
    hea_file = record_path.with_suffix('.hea')
    dat_file = record_path.with_suffix('.dat')
    
    with open(hea_file, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    n_leads = int(header[1])
    fs = int(header[2])
    n_samples = int(header[3])
    
    with open(dat_file, 'rb') as f:
        raw_data = f.read()
    
    n_pairs = len(raw_data) // 3
    signal = np.zeros((n_pairs * 2,), dtype=np.float32)
    
    for i in range(n_pairs):
        byte0 = raw_data[i * 3]
        byte1 = raw_data[i * 3 + 1]
        byte2 = raw_data[i * 3 + 2]
        sample0 = byte0 | ((byte1 & 0x0F) << 8)
        sample1 = byte2 | ((byte1 & 0xF0) << 4)
        if sample0 >= 0x800:
            sample0 -= 0x1000
        if sample1 >= 0x800:
            sample1 -= 0x1000
        signal[i * 2] = float(sample0)
        signal[i * 2 + 1] = float(sample1)
    
    signal = signal[:n_samples * 2].reshape(-1, 2)
    print(f"[预处理] 手动模式无法读取标注, 返回空标注")
    return signal, np.array([], dtype=int), [], fs


def resample_ecg(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """对 ECG 信号重采样"""
    n_samples = signal.shape[0]
    n_target = int(n_samples * target_fs / orig_fs)
    resampled = scipy_signal.resample(signal, n_target, axis=0)
    return resampled.astype(np.float32)


def extract_beats(
    signal: np.ndarray,
    ann_indices: np.ndarray,
    ann_symbols: List[str],
    orig_fs: int,
    target_fs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """以 R 峰为中心提取心拍"""
    resampled = resample_ecg(signal, orig_fs, target_fs)
    n_resampled = resampled.shape[0]
    resample_ratio = target_fs / orig_fs
    ann_indices_resampled = (ann_indices * resample_ratio).astype(int)
    
    beats = []
    labels = []
    half_window = BEAT_WINDOW_SAMPLES // 2
    
    for idx, symbol in zip(ann_indices_resampled, ann_symbols):
        if symbol not in AAMI_CLASSES:
            continue
        label = AAMI_CLASSES[symbol]
        start = max(0, idx - half_window)
        end = min(n_resampled, idx + half_window)
        if end - start < BEAT_WINDOW_SAMPLES * 0.5:
            continue
        beat = resampled[start:end, 0]
        if len(beat) < BEAT_WINDOW_SAMPLES:
            pad_before = (BEAT_WINDOW_SAMPLES - len(beat)) // 2
            pad_after = BEAT_WINDOW_SAMPLES - len(beat) - pad_before
            beat = np.pad(beat, (pad_before, pad_after), mode='constant')
        elif len(beat) > BEAT_WINDOW_SAMPLES:
            center = len(beat) // 2
            beat = beat[center - half_window:center + half_window]
        beats.append(beat)
        labels.append(label)
    
    if len(beats) == 0:
        raise ValueError("未提取到任何心拍! 请检查标注文件。")
    
    beats = np.stack(beats, axis=0)
    beats = (beats - beats.mean(axis=1, keepdims=True)) / (beats.std(axis=1, keepdims=True) + 1e-8)
    labels = np.array(labels, dtype=np.int32)
    
    return beats, labels


def augment_data(beats: np.ndarray, labels: np.ndarray, config: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """数据增强"""
    if config is None:
        config = TRAIN_CONFIG['augmentation']
    
    augmented_beats = [beats]
    augmented_labels = [labels]
    
    for noise_std in config['noise_std']:
        noisy = beats + np.random.randn(*beats.shape) * noise_std
        augmented_beats.append(noisy)
        augmented_labels.append(labels)
    
    scale_low, scale_high = config['amplitude_scale_range']
    for _ in range(2):
        scale = np.random.uniform(scale_low, scale_high)
        scaled = beats * scale
        augmented_beats.append(scaled)
        augmented_labels.append(labels)
    
    drift_amp = config.get('baseline_drift_amplitude', 0.1)
    for _ in range(2):
        t = np.linspace(0, 2 * np.pi, BEAT_WINDOW_SAMPLES)
        freq = np.random.uniform(0.1, 0.5)
        drift = drift_amp * np.sin(freq * t)[np.newaxis, :]
        drifted = beats + drift
        augmented_beats.append(drifted)
        augmented_labels.append(labels)
    
    beats_all = np.concatenate(augmented_beats, axis=0)
    labels_all = np.concatenate(augmented_labels, axis=0)
    return beats_all, labels_all


def process_all_records(
    records: List[int] = None,
    augment: bool = True,
    test_mode: bool = False
) -> Dict[str, np.ndarray]:
    """处理所有 MIT-BIH 记录"""
    if records is None:
        records = MIT_BIH_RECORDS
    
    if test_mode:
        records = MIT_BIH_TEST_RECORDS
        print(f"[预处理] [TEST] 测试模式: 仅处理 {len(records)} 条记录: {records}")
    
    all_beats = []
    all_labels = []
    all_rec_ids = []
    failed_records = []
    
    for i, rec_id in enumerate(records):
        rec_name = str(rec_id)
        print(f"[预处理] [{i+1}/{len(records)}] 处理记录 {rec_name}...", end=" ")
        
        try:
            signal, ann_indices, ann_symbols, fs = load_mit_bih_record(rec_name)
            beats, labels = extract_beats(signal, ann_indices, ann_symbols, orig_fs=fs, target_fs=TARGET_FS)
            
            if augment:
                beats_aug, labels_aug = augment_data(beats, labels)
            else:
                beats_aug, labels_aug = beats, labels
            
            all_beats.append(beats_aug)
            all_labels.append(labels_aug)
            all_rec_ids.append(np.full(len(beats_aug), rec_id, dtype=np.int32))
            
            n_normal = int((labels_aug == 0).sum())
            n_abnormal = int((labels_aug == 1).sum())
            print(f"[OK] {n_normal}正常 / {n_abnormal}异常")
            
        except Exception as e:
            print(f"[FAIL] {e}")
            failed_records.append(rec_name)
    
    if len(all_beats) == 0:
        raise RuntimeError("所有记录处理失败!")
    
    final_beats = np.concatenate(all_beats, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    final_rec_ids = np.concatenate(all_rec_ids, axis=0)
    
    n_normal = int((final_labels == 0).sum())
    n_abnormal = int((final_labels == 1).sum())
    
    print(f"\n[预处理] ======== 处理完成 ========")
    print(f"[预处理] 总心拍数: {len(final_beats)}")
    print(f"[预处理]   Normal: {n_normal} ({n_normal/len(final_beats)*100:.1f}%)")
    print(f"[预处理]   Abnormal: {n_abnormal} ({n_abnormal/len(final_beats)*100:.1f}%)")
    
    if failed_records:
        print(f"[预处理] 以下记录处理失败: {failed_records}")
    
    return {"beats": final_beats, "labels": final_labels, "record_ids": final_rec_ids}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MIT-BIH 心电数据预处理")
    parser.add_argument("--test", action="store_true", help="测试模式 (仅处理前3条记录)")
    parser.add_argument("--no-augment", action="store_true", help="不进行数据增强")
    args = parser.parse_args()
    
    result = process_all_records(
        test_mode=args.test,
        augment=not args.no_augment
    )
    
    output_path = PROCESSED_DIR / "mit_bih_processed.npz"
    np.savez_compressed(
        output_path,
        beats=result["beats"],
        labels=result["labels"],
        record_ids=result["record_ids"]
    )
    print(f"[预处理] [OK] 已保存到: {output_path}")
    print(f"[预处理]   文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")