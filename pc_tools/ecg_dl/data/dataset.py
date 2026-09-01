#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow Dataset 构建器
从预处理后的 .npz 文件构建训练/验证/测试集
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DIR, TRAIN_CONFIG, CLASS_NAMES, INFERENCE_CONFIG,
    BEAT_WINDOW_SAMPLES,
)

# 部署链数据源开关 (阶段 1.5, TUNING_HISTORY 十三章):
# train.py --deploy-chain 时调用 set_npz_suffix("_deploy"),
# 三个基础加载器改读 *_deploy.npz (部署链重建数据), 默认 "" 行为完全不变。
_NPZ_SUFFIX = ""


def set_npz_suffix(suffix: str) -> None:
    global _NPZ_SUFFIX
    _NPZ_SUFFIX = suffix
    print(f"[数据集] npz 后缀切换: '{suffix}' (部署链数据源)")


def _load_arrays(npz_path: Path):
    """优先 mmap 加载独立 .npy (ECG_PROCESSED_DIR 本地数据, TUNING_HISTORY 十三章);
    否则回退 npz 常规加载."""
    stem = npz_path.stem  # e.g. mit_bih_processed_deploy
    base = npz_path.parent / stem
    npy_beats = base.with_name(stem + "_beats.npy")
    if npy_beats.exists():
        beats = np.load(npy_beats, mmap_mode="r")
        labels = np.load(base.with_name(stem + "_labels.npy"), mmap_mode="r")
        rec = base.with_name(stem + "_record_ids.npy")
        record_ids = np.load(rec, mmap_mode="r") if rec.exists() else None
        print(f"[数据集] 加载 (mmap): {npy_beats.name}")
        return beats, labels, record_ids
    data = np.load(npz_path)
    return data["beats"], data["labels"], data.get("record_ids", None)


def load_processed_data() -> dict:
    """
    加载预处理后的数据
    
    Returns:
        {"beats": np.ndarray (n, 250), "labels": np.ndarray (n,)}
    """
    npz_path = PROCESSED_DIR / f"mit_bih_processed{_NPZ_SUFFIX}.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(
            f"预处理数据未找到: {npz_path}\n"
            f"请先运行: python data/preprocess.py"
        )
    
    beats, labels, record_ids = _load_arrays(npz_path)
    
    print(f"[数据集] 加载数据: {npz_path}")
    print(f"[数据集]   心拍数: {len(beats)}")
    print(f"[数据集]   形状: {beats.shape}")
    print(f"[数据集]   类别分布:")
    for i, name in enumerate(CLASS_NAMES):
        count = int((labels == i).sum())
        print(f"[数据集]     {name}: {count} ({count/len(labels)*100:.1f}%)")
    
    return {"beats": beats, "labels": labels, "record_ids": record_ids}


def load_ptbxl_data() -> dict:
    """
    Load preprocessed PTB-XL data.

    Returns:
        {"beats": np.ndarray, "labels": np.ndarray, "record_ids": np.ndarray}
    """
    npz_path = PROCESSED_DIR / "ptbxl_processed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"PTB-XL 预处理数据未找到: {npz_path}\n"
            f"请先运行: python data/preprocess_ptbxl.py"
        )
    data = np.load(npz_path)
    beats = data["beats"]
    labels = data["labels"]
    record_ids = data.get("record_ids", None)
    print(f"[PTB-XL] 加载: {len(beats)} 心拍, 形状: {beats.shape}")
    for i, name in enumerate(CLASS_NAMES):
        count = int((labels == i).sum())
        print(f"[PTB-XL]   {name}: {count} ({count/len(labels)*100:.1f}%)")
    return {"beats": beats, "labels": labels, "record_ids": record_ids}


def load_incart_data() -> dict:
    """
    Load preprocessed INCART data.

    Returns:
        {"beats": np.ndarray, "labels": np.ndarray, "record_ids": np.ndarray}
    """
    npz_path = PROCESSED_DIR / f"incart_processed{_NPZ_SUFFIX}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"INCART 预处理数据未找到: {npz_path}\n"
            f"请先运行: python data/preprocess_incart.py"
        )
    beats, labels, record_ids = _load_arrays(npz_path)
    print(f"[INCART] 加载: {len(beats)} 心拍, 形状: {beats.shape}")
    for i, name in enumerate(CLASS_NAMES):
        count = int((labels == i).sum())
        print(f"[INCART]   {name}: {count} ({count/len(labels)*100:.1f}%)")
    return {"beats": beats, "labels": labels, "record_ids": record_ids}


def load_ptb_data() -> dict:
    """
    Load preprocessed PTB database data (beat-level, 250Hz).

    Returns:
        {"beats": np.ndarray, "labels": np.ndarray, "record_ids": np.ndarray}
    """
    npz_path = PROCESSED_DIR / f"ptb_processed{_NPZ_SUFFIX}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"PTB 预处理数据未找到: {npz_path}\n"
            f"请先运行: python data/preprocess_ptb.py"
        )
    beats, labels, record_ids = _load_arrays(npz_path)
    print(f"[PTB] 加载: {len(beats)} 心拍, 形状: {beats.shape}")
    for i, name in enumerate(CLASS_NAMES):
        count = int((labels == i).sum())
        print(f"[PTB]   {name}: {count} ({count/len(labels)*100:.1f}%)")
    return {"beats": beats, "labels": labels, "record_ids": record_ids}


def load_mit_incart_ptb_merged() -> dict:
    """
    Load MIT-BIH + INCART + PTB merged dataset (beat-level).

    Returns:
        {"beats": np.ndarray, "labels": np.ndarray, "record_ids": np.ndarray}
    """
    mit_inc = load_mit_incart_merged()
    ptb = load_ptb_data()
    # PTB record IDs already offset to 400000+ in preprocessing
    beats = np.concatenate([mit_inc["beats"], ptb["beats"]], axis=0)
    labels = np.concatenate([mit_inc["labels"], ptb["labels"]], axis=0)
    rids = (np.concatenate([mit_inc["record_ids"], ptb["record_ids"]])
            if mit_inc.get("record_ids") is not None
            and ptb.get("record_ids") is not None else None)
    nN, nA = int((labels == 0).sum()), int((labels == 1).sum())
    print(f"\n[MIT+INCART+PTB] {len(beats)} beats "
          f"(N={nN}, A={nA}, {nA/len(labels)*100:.1f}% abnormal)")
    return {"beats": beats, "labels": labels, "record_ids": rids}


def load_mit_incart_merged() -> dict:
    """
    Load MIT-BIH + INCART merged dataset.

    Returns:
        {"beats": np.ndarray, "labels": np.ndarray, "record_ids": np.ndarray}
    """
    mit = load_processed_data()
    inc = load_incart_data()

    # INCART record IDs are I01-I75, convert to int offset
    # to avoid collision with MIT-BIH IDs (100-234)
    if inc["record_ids"] is not None:
        inc["record_ids"] = inc["record_ids"] + 100000

    beats = np.concatenate([mit["beats"], inc["beats"]], axis=0)
    labels = np.concatenate([mit["labels"], inc["labels"]], axis=0)

    if mit.get("record_ids") is not None and inc.get("record_ids") is not None:
        rids = np.concatenate([mit["record_ids"], inc["record_ids"]], axis=0)
    else:
        rids = None

    print(f"\n[合并数据集] MIT-BIH + INCART")
    print(f"  总心拍: {len(beats)}")
    for i, name in enumerate(CLASS_NAMES):
        count = int((labels == i).sum())
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    return {"beats": beats, "labels": labels, "record_ids": rids}


def load_svdb_data() -> dict:
    npz_path = PROCESSED_DIR / "svdb_processed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Run: python data/preprocess_svdb.py")
    data = np.load(npz_path)
    beats, labels, rids = data["beats"], data["labels"], data["record_ids"]
    print(f"[SVDB] {len(beats)} beats (N={(labels==0).sum()}, A={(labels==1).sum()})")
    return {"beats": beats, "labels": labels, "record_ids": rids}


def load_mit_incart_svdb_merged() -> dict:
    mit_inc = load_mit_incart_merged()
    svdb = load_svdb_data()
    svdb["record_ids"] = svdb["record_ids"] + 200000
    beats = np.concatenate([mit_inc["beats"], svdb["beats"]])
    labels = np.concatenate([mit_inc["labels"], svdb["labels"]])
    rids = np.concatenate([mit_inc["record_ids"], svdb["record_ids"]])
    nN, nA = (labels==0).sum(), (labels==1).sum()
    print(f"\n[MIT+INCART+SVDB] {len(beats)} beats (N={nN}, A={nA}, {nA/len(labels)*100:.1f}%)")
    return {"beats": beats, "labels": labels, "record_ids": rids}


def load_3beat_merged() -> dict:
    """Load MIT-BIH + INCART 3-beat merged dataset (Phase 2B).

    Honors _NPZ_SUFFIX (deploy-chain 数据源): 读 mit_incart_3beat{_deploy}.npz.
    """
    npz = PROCESSED_DIR / f"mit_incart_3beat{_NPZ_SUFFIX}.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"3-beat 预处理数据未找到: {npz}\n"
            f"请先运行: python data/preprocess_3beat.py"
        )
    beats, labels, record_ids = _load_arrays(npz)
    print(f"[3-beat] 加载: {len(beats)} 序列, 形状: {beats.shape}")
    for i, name in enumerate(CLASS_NAMES):
        c = int((labels == i).sum())
        print(f"[3-beat]   {name}: {c} ({c/len(labels)*100:.1f}%)")
    return {"beats": beats, "labels": labels, "record_ids": record_ids}


def load_ecg1000_data() -> dict:
    """Load preprocessed ECG1000 data."""
    npz_path = PROCESSED_DIR / "ecg1000_processed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"ECG1000 data not found: {npz_path}\n"
            f"Run: python data/preprocess_ecg1000.py"
        )
    data = np.load(npz_path)
    beats, labels = data["beats"], data["labels"]
    rids = data.get("record_ids", None)
    print(f"[ECG1000] Loaded: {len(beats)} beats, shape: {beats.shape}")
    for i, name in enumerate(CLASS_NAMES):
        c = int((labels == i).sum())
        print(f"[ECG1000]   {name}: {c} ({c/len(labels)*100:.1f}%)")
    return {"beats": beats, "labels": labels, "record_ids": rids}


def load_ptbxl_rhythm_data() -> dict:
    """Load PTB-XL rhythm-only preprocessed data."""
    npz = PROCESSED_DIR / "ptbxl_rhythm_processed.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Run: python data/preprocess_ptbxl_rhythm.py")
    d = np.load(npz)
    print(f"[PTBXL-R] {len(d['beats'])} beats (N={(d['labels']==0).sum()}, A={(d['labels']==1).sum()})")
    return {"beats": d["beats"], "labels": d["labels"], "record_ids": d.get("record_ids")}


def load_all_three_merged() -> dict:
    """MIT-BIH + INCART + PTB-XL Rhythm (all beat-applicable labels)."""
    mit_inc = load_mit_incart_merged()
    ptb = load_ptbxl_rhythm_data()
    # Offset PTB-XL IDs to avoid collision (MIT=100-234, INCART=1-75+100000, PTBXL=300000+)
    if ptb["record_ids"] is not None:
        ptb["record_ids"] = ptb["record_ids"] + 300000
    beats = np.concatenate([mit_inc["beats"], ptb["beats"]])
    labels = np.concatenate([mit_inc["labels"], ptb["labels"]])
    rids = (np.concatenate([mit_inc["record_ids"], ptb["record_ids"]])
            if mit_inc.get("record_ids") is not None and ptb.get("record_ids") is not None
            else None)
    nN, nA = (labels==0).sum(), (labels==1).sum()
    print(f"\n[Merged ALL] {len(beats)} beats (N={nN}, A={nA}, {nA/len(labels)*100:.1f}% abnormal)")
    return {"beats": beats, "labels": labels, "record_ids": rids}


def load_mit_ecg1000_merged() -> dict:
    """Load MIT-BIH + ECG1000 merged dataset."""
    mit = load_processed_data()
    ecg = load_ecg1000_data()
    if ecg["record_ids"] is not None:
        ecg["record_ids"] = ecg["record_ids"] + 200000
    beats = np.concatenate([mit["beats"], ecg["beats"]], axis=0)
    labels = np.concatenate([mit["labels"], ecg["labels"]], axis=0)
    rids = (np.concatenate([mit["record_ids"], ecg["record_ids"]])
            if mit.get("record_ids") is not None and ecg.get("record_ids") is not None
            else None)
    print(f"\n[Merged] MIT-BIH + ECG1000: {len(beats)} beats")
    for i, name in enumerate(CLASS_NAMES):
        c = int((labels == i).sum())
        print(f"  {name}: {c} ({c/len(labels)*100:.1f}%)")
    return {"beats": beats, "labels": labels, "record_ids": rids}


def load_merged_data() -> dict:
    """
    Load MIT-BIH + PTB-XL merged dataset.

    Returns:
        {"beats": np.ndarray, "labels": np.ndarray, "record_ids": np.ndarray}
    """
    mit = load_processed_data()
    ptb = load_ptbxl_data()

    # Offset PTB-XL record IDs to avoid collision
    if ptb["record_ids"] is not None:
        offset = 100000  # MIT-BIH IDs are 100-234
        ptb["record_ids"] = ptb["record_ids"] + offset

    beats = np.concatenate([mit["beats"], ptb["beats"]], axis=0)
    labels = np.concatenate([mit["labels"], ptb["labels"]], axis=0)

    if mit.get("record_ids") is not None and ptb.get("record_ids") is not None:
        rids = np.concatenate([mit["record_ids"], ptb["record_ids"]], axis=0)
    else:
        rids = None

    print(f"\n[合并数据集] MIT-BIH + PTB-XL")
    print(f"  总心拍: {len(beats)}")
    for i, name in enumerate(CLASS_NAMES):
        count = int((labels == i).sum())
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    return {"beats": beats, "labels": labels, "record_ids": rids}



def train_val_test_split(
    beats: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray = None,
    val_split: float = None,
    test_split: float = None,
    random_seed: int = None
) -> dict:
    """
    按记录号划分训练/验证/测试集 (防止数据泄露)
    
    同一病人的心拍不会同时出现在训练和测试集中
    
    Args:
        beats: 心拍数据 (n, 250)
        labels: 标签 (n,)
        record_ids: 每条心拍的来源记录 ID (n,)
        val_split: 验证集比例
        test_split: 测试集比例
        random_seed: 随机种子
        
    Returns:
        {"train": (x, y), "val": (x, y), "test": (x, y)}
    """
    if val_split is None:
        val_split = TRAIN_CONFIG['validation_split']
    if test_split is None:
        test_split = TRAIN_CONFIG['test_split']
    if random_seed is None:
        random_seed = TRAIN_CONFIG['random_seed']
    
    np.random.seed(random_seed)
    
    if record_ids is not None:
        # ★ 按记录号分组 (防泄露)
        unique_records = np.unique(record_ids)
        np.random.shuffle(unique_records)
        n_total = len(unique_records)
        n_test = max(1, int(n_total * test_split))
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_test - n_val
        
        test_recs = set(unique_records[:n_test])
        val_recs = set(unique_records[n_test:n_test+n_val])
        train_recs = set(unique_records[n_test+n_val:])
        
        train_mask = np.isin(record_ids, list(train_recs))
        val_mask = np.isin(record_ids, list(val_recs))
        test_mask = np.isin(record_ids, list(test_recs))
        
        x_train, y_train = beats[train_mask], labels[train_mask]
        x_val, y_val = beats[val_mask], labels[val_mask]
        x_test, y_test = beats[test_mask], labels[test_mask]
        
        print(f"[数据集] 按记录号分组 (防数据泄露):")
        print(f"[数据集]   训练: {len(train_recs)} 条记录, {len(x_train)} 样本")
        print(f"[数据集]   验证: {len(val_recs)} 条记录, {len(x_val)} 样本")
        print(f"[数据集]   测试: {len(test_recs)} 条记录, {len(x_test)} 样本")
    else:
        # 回退: 随机划分 (仅用于兼容旧数据)
        from sklearn.model_selection import train_test_split
        x_temp, x_test, y_temp, y_test = train_test_split(
            beats, labels, test_size=test_split,
            random_state=random_seed, stratify=labels)
        val_ratio = val_split / (1.0 - test_split)
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, test_size=val_ratio,
            random_state=random_seed, stratify=y_temp)
        print(f"[数据集] 随机划分 (警告: 可能数据泄露):")
        print(f"[数据集]   训练集: {len(x_train)} 样本")
        print(f"[数据集]   验证集: {len(x_val)} 样本")
        print(f"[数据集]   测试集: {len(x_test)} 样本")
    
    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test)
    }


def add_channel_dim(beats: np.ndarray) -> np.ndarray:
    """
    添加通道维度 (n, 250) → (n, 250, 1)
    Conv1D 需要通道维度
    """
    return beats[..., np.newaxis]


def apply_sliding_window_augmentation(
    x: np.ndarray,
    y: np.ndarray,
    dup: int = 1,
    max_shift: int = 40,
    seed: int = None,
    both_classes: bool = True,
) -> tuple:
    """
    滑窗相位增强 (参考: 张异凡等, 哈尔滨工业大学学报 2019)。

    对心拍沿时间轴随机平移窗口生成 dup 个新视图:
    - 模拟 R 峰在 ESP32 滑动窗口内的不同相位位置 (部署时窗口每 125
      样本滑动一次, 心拍相位不固定), 缩小训练/部署域差异;
    - 平移产生的边缘空隙用 reflect 填充, 近似邻近心拍的波形延续;
    - 不伪造波形形态 (每个视图都是同一心拍的真实相位视图)。

    ★ 必须对两个类别做同等移位 (both_classes=True):
      若只移位 Abnormal, "窗口相位"会成为训练集上完美的类别捷径,
      模型学到"R 峰不在中心 ⇒ 异常", 对验证集(全部居中)灾难性误报
      (实测 P(abn|Normal) 0.13 → 0.67, val_auc 0.95 → 0.32, 已复现)。

    Args:
        x: 心拍数据 (n, 250)
        y: 标签 (n,)
        dup: 每个心拍额外生成的移位视图数 (0 = 关闭)
        max_shift: 最大平移量 (采样点, 默认 40 = 160ms @250Hz)
        seed: 随机种子
        both_classes: True=所有类别同等移位 (变体保留原标签, 推荐);
                      False=仅 Abnormal 移位 (危险, 仅用于对照实验)

    Returns:
        (x_new, y_new): 扩充后的数据
    """
    if seed is None:
        seed = TRAIN_CONFIG['random_seed']
    rng = np.random.default_rng(seed)

    n_before = len(x)
    if dup <= 0 or max_shift <= 0 or n_before == 0:
        print("[滑窗增强] 未启用 (dup=0 或样本为空)")
        return x, y

    if both_classes:
        sel_idx = np.arange(n_before)
        sel_labels = y
        label_desc = "全部心拍(双类)"
    else:
        sel_idx = np.where(y == 1)[0]
        sel_labels = None
        label_desc = "仅异常心拍"
    n_sel = len(sel_idx)
    x_sel = x[sel_idx]

    # reflect 填充, 使正负偏移都能在窗口内取到完整 250 点
    xp = np.pad(x_sel, ((0, 0), (max_shift, max_shift)), mode='reflect')
    rows = np.arange(n_sel)
    cols = np.arange(BEAT_WINDOW_SAMPLES)

    variants = []
    for _ in range(dup):
        mag = rng.integers(1, max_shift + 1, size=n_sel)
        sign = rng.choice([-1, 1], size=n_sel)
        shifts = mag * sign
        start = max_shift + shifts
        variants.append(xp[rows[:, None], start[:, None] + cols[None, :]])

    x_v = np.concatenate(variants, axis=0).astype(np.float32)
    # 重新 Z-score (与 ESP32 推理窗口归一化一致)
    x_v = (x_v - x_v.mean(axis=1, keepdims=True)) / (
        x_v.std(axis=1, keepdims=True) + 1e-8
    )
    if both_classes:
        y_v = np.concatenate([y] * dup, axis=0)
    else:
        y_v = np.ones(len(x_v), dtype=y.dtype)

    x_new = np.concatenate([x, x_v], axis=0)
    y_new = np.concatenate([y, y_v], axis=0)

    n_after = len(x_new)
    n_abn_after = int((y_new == 1).sum())
    print(f"[滑窗增强] 对{label_desc}生成 {dup} 个移位视图/拍 (max_shift={max_shift} 点)")
    print(f"[滑窗增强] 训练集: {n_before} → {n_after} "
          f"(异常占比 {int((y==1).sum())/n_before*100:.1f}% → "
          f"{n_abn_after/n_after*100:.1f}%)")
    return x_new, y_new


def make_domain_balanced_dataset(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    batch_size: int = None,
    frac_b: float = 0.20,
    weight_b: float = 0.5,
    buffer_size: int = 10000,
) -> tf.data.Dataset:
    """
    域平衡采样 Dataset (Phase 3B 方案 A):
    每个 batch 固定 frac_b 比例来自 B 域 (PTB), 其余来自 A 域 (MIT+INCART)。
    B 域样本的 loss 权重 = weight_b (记录级标签噪声降权)。
    返回 (x, y_onehot, sample_weight) 三元组, Keras fit 直接支持。

    目标: 单模型同时学到两个域 (心律失常 + MI), 不被任一域主导。
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']
    nb = max(1, int(round(batch_size * frac_b)))
    na = batch_size - nb
    if nb >= batch_size or na <= 0:
        raise ValueError(f"frac_b={frac_b} 导致 batch 划分无效")

    x_a = add_channel_dim(np.asarray(x_a, dtype=np.float32))
    x_b = add_channel_dim(np.asarray(x_b, dtype=np.float32))
    y_a = tf.keras.utils.to_categorical(y_a, num_classes=2)
    y_b = tf.keras.utils.to_categorical(y_b, num_classes=2)

    ds_a = tf.data.Dataset.from_tensor_slices((x_a, y_a)).shuffle(
        min(buffer_size, len(x_a))).repeat().batch(na)
    ds_b = tf.data.Dataset.from_tensor_slices((x_b, y_b)).shuffle(
        min(buffer_size, len(x_b))).repeat().batch(nb)

    ds = tf.data.Dataset.zip((ds_a, ds_b))

    def merge(ab, bb):
        x = tf.concat([ab[0], bb[0]], axis=0)
        y = tf.concat([ab[1], bb[1]], axis=0)
        sw = tf.concat([tf.ones(na, tf.float32), tf.ones(nb, tf.float32) * weight_b], axis=0)
        return x, y, sw

    ds = ds.map(merge)

    def shuffle_batch(x, y, sw):
        idx = tf.random.shuffle(tf.range(batch_size))
        return tf.gather(x, idx), tf.gather(y, idx), tf.gather(sw, idx)

    ds = ds.map(shuffle_batch)
    # 每 epoch 覆盖 A 域一遍 (B 域按 frac 比例过采样)
    steps = int(np.ceil(len(x_a) / na))
    ds = ds.take(steps)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_domain_balanced_dataset_kd(
    x_a: np.ndarray,
    y_a: np.ndarray,
    z_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    z_b: np.ndarray,
    batch_size: int = None,
    frac_b: float = 0.20,
    weight_b: float = 0.5,
    buffer_size: int = 10000,
) -> tf.data.Dataset:
    """
    域平衡 KD 采样: 同 make_domain_balanced_dataset 但 y 为 (N,4) = concat([onehot, teacher_logits]).

    y_a / y_b 为 caller 预 one-hot 的 (N,2) 标签; z_a / z_b 为 (N,2) teacher logits;
    函数内部 concat 出 (N,4) KD 目标 (前 2 维 onehot, 后 2 维 teacher logits),
    供 kd_loss(y_true=(B,4), y_pred=(B,2)) 直接消费。
    其余流程 (nb/na 划分、zip、merge、shuffle_batch、steps、take、prefetch)
    与 make_domain_balanced_dataset 完全一致。
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']
    nb = max(1, int(round(batch_size * frac_b)))
    na = batch_size - nb
    if nb >= batch_size or na <= 0:
        raise ValueError(f"frac_b={frac_b} 导致 batch 划分无效")

    x_a = add_channel_dim(np.asarray(x_a, dtype=np.float32))
    x_b = add_channel_dim(np.asarray(x_b, dtype=np.float32))
    # y_a/y_b 已是 (N,2) onehot (caller 已 to_categorical), 不再次 one-hot; cast float32 后
    # 与 z_a/z_b (N,2) concat 产出 (N,4) KD 目标: [onehot(2) | teacher_logits(2)]
    y_a = np.concatenate(
        [y_a.astype(np.float32), z_a.astype(np.float32)], axis=-1)
    y_b = np.concatenate(
        [y_b.astype(np.float32), z_b.astype(np.float32)], axis=-1)

    ds_a = tf.data.Dataset.from_tensor_slices((x_a, y_a)).shuffle(
        min(buffer_size, len(x_a))).repeat().batch(na)
    ds_b = tf.data.Dataset.from_tensor_slices((x_b, y_b)).shuffle(
        min(buffer_size, len(x_b))).repeat().batch(nb)

    ds = tf.data.Dataset.zip((ds_a, ds_b))

    def merge(ab, bb):
        x = tf.concat([ab[0], bb[0]], axis=0)
        y = tf.concat([ab[1], bb[1]], axis=0)
        sw = tf.concat([tf.ones(na, tf.float32), tf.ones(nb, tf.float32) * weight_b], axis=0)
        return x, y, sw

    ds = ds.map(merge)

    def shuffle_batch(x, y, sw):
        idx = tf.random.shuffle(tf.range(batch_size))
        return tf.gather(x, idx), tf.gather(y, idx), tf.gather(sw, idx)

    ds = ds.map(shuffle_batch)
    # 每 epoch 覆盖 A 域一遍 (B 域按 frac 比例过采样)
    steps = int(np.ceil(len(x_a) / na))
    ds = ds.take(steps)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_balanced_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = None,
    buffer_size: int = 10000
) -> tf.data.Dataset:
    """
    类别均衡 Dataset：每个 batch 中 Normal/Abnormal 数量相等。

    通过 oversample 少数类实现，避免模型被多数类淹没。
    用于训练集可以显著提升 Abnormal Recall。
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']

    half = batch_size // 2

    x = add_channel_dim(x)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=2)

    # Split by class
    mask_n = (y == 0)
    mask_a = (y == 1)

    x_n, y_n = x[mask_n], y_onehot[mask_n]
    x_a, y_a = x[mask_a], y_onehot[mask_a]

    n_n, n_a = len(x_n), len(x_a)
    print(f"[均衡采样] Normal: {n_n}, Abnormal: {n_a}, "
          f"每 batch {half}+{half}")

    if n_a == 0:
        # Fallback: no abnormal samples
        ds = tf.data.Dataset.from_tensor_slices((x, y_onehot))
        return ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_n = tf.data.Dataset.from_tensor_slices(x_n).repeat()
    ds_n = ds_n.shuffle(buffer_size).batch(half)
    ds_n_labels = tf.data.Dataset.from_tensor_slices(y_n).repeat()
    ds_n_labels = ds_n_labels.shuffle(buffer_size).batch(half)
    ds_n = tf.data.Dataset.zip((ds_n, ds_n_labels))

    ds_a = tf.data.Dataset.from_tensor_slices(x_a).repeat()
    ds_a = ds_a.shuffle(buffer_size).batch(half)
    ds_a_labels = tf.data.Dataset.from_tensor_slices(y_a).repeat()
    ds_a_labels = ds_a_labels.shuffle(buffer_size).batch(half)
    ds_a = tf.data.Dataset.zip((ds_a, ds_a_labels))

    # Interleave: alternate normal and abnormal batches
    ds = tf.data.Dataset.zip((ds_n, ds_a))
    ds = ds.map(lambda n, a: (
        tf.concat([n[0], a[0]], axis=0),
        tf.concat([n[1], a[1]], axis=0)
    ))

    # Shuffle the combined batch internally
    def shuffle_batch(x_batch, y_batch):
        idx = tf.random.shuffle(tf.range(batch_size))
        return tf.gather(x_batch, idx), tf.gather(y_batch, idx)

    ds = ds.map(shuffle_batch)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Steps per epoch: roughly n_n / half (cover all normal samples)
    return ds


def make_multitask_dataset(
    x: np.ndarray,
    y_cls: np.ndarray,
    y_bpm: np.ndarray,
    y_sqi: np.ndarray,
    batch_size: int = None,
    shuffle: bool = True,
    buffer_size: int = 10000,
    augment: bool = False
) -> tf.data.Dataset:
    """
    Build TF Dataset for multi-task learning.

    Produces (x, (y_cls_onehot, y_bpm, y_sqi)) tuple for multi-output models.
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']

    x = add_channel_dim(x)
    y_cls_onehot = tf.keras.utils.to_categorical(y_cls, num_classes=2)
    y_bpm = y_bpm.reshape(-1, 1).astype(np.float32)
    y_sqi = y_sqi.reshape(-1, 1).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (x, (y_cls_onehot, y_bpm, y_sqi)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(buffer_size, len(x)))
    dataset = dataset.batch(batch_size)

    if augment and TRAIN_CONFIG['augmentation']['enabled']:
        from losses.focal_loss import mixup_1d, apply_mild_augmentation

        aug_cfg = TRAIN_CONFIG['augmentation']
        aug_prob = aug_cfg.get('apply_prob', 0.80)

        def augment_batch(x_batch, y_tuple):
            y_cls_batch, y_bpm_batch, y_sqi_batch = y_tuple
            x_batch = tf.cast(x_batch, tf.float32)
            y_cls_batch = tf.cast(y_cls_batch, tf.float32)
            y_bpm_batch = tf.cast(y_bpm_batch, tf.float32)
            y_sqi_batch = tf.cast(y_sqi_batch, tf.float32)
            x_aug = apply_mild_augmentation(
                x_batch, prob=aug_prob,
                phase_max_shift=aug_cfg.get('phase_shift', 0))
            if TRAIN_CONFIG['mixup']['enabled']:
                mixup_prob = TRAIN_CONFIG['mixup']['prob']
                if tf.random.uniform(()) < mixup_prob:
                    x_aug, y_cls_batch = mixup_1d(
                        x_aug, y_cls_batch,
                        alpha=TRAIN_CONFIG['mixup']['alpha']
                    )
            return x_aug, (y_cls_batch, y_bpm_batch, y_sqi_batch)

        dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = None,
    shuffle: bool = True,
    buffer_size: int = 10000,
    augment: bool = False
) -> tf.data.Dataset:
    """
    构建 TensorFlow Dataset

    Args:
        x: 特征数据 (n, 250)
        y: 标签 (n,)
        batch_size: 批大小
        shuffle: 是否打乱
        buffer_size: 打乱缓冲区大小
        augment: 是否应用数据增强 (仅训练集)

    Returns:
        tf.data.Dataset
    """
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']

    x = add_channel_dim(x)
    y = tf.keras.utils.to_categorical(y, num_classes=2)

    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(buffer_size, len(x)))

    dataset = dataset.batch(batch_size)

    # Phase 2A: 训练时应用 Mixup + 温和 ECG 数据增强（在 batch 后）
    if augment and TRAIN_CONFIG['augmentation']['enabled']:
        from losses.focal_loss import mixup_1d, apply_mild_augmentation

        aug_cfg = TRAIN_CONFIG['augmentation']
        aug_prob = aug_cfg.get('apply_prob', 0.80)

        def augment_batch(x_batch, y_batch):
            x_batch = tf.cast(x_batch, tf.float32)
            y_batch = tf.cast(y_batch, tf.float32)
            x_aug = apply_mild_augmentation(
                x_batch, prob=aug_prob,
                phase_max_shift=aug_cfg.get('phase_shift', 0))
            if TRAIN_CONFIG['mixup']['enabled']:
                mixup_prob = TRAIN_CONFIG['mixup']['prob']
                if tf.random.uniform(()) < mixup_prob:
                    x_aug, y_batch = mixup_1d(
                        x_aug, y_batch,
                        alpha=TRAIN_CONFIG['mixup']['alpha']
                    )
            return x_aug, y_batch

        dataset = dataset.map(
            augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def prepare_datasets(
    augment: bool = False,
    batch_size: int = None,
    use_ptbxl: bool = False,
    use_merged: bool = False,
    use_incart: bool = False,
    use_ecg1000: bool = False,
    use_ptbxl_rhythm: bool = False,
    use_ptb_beat: bool = False,
    ptb_abn_max: int = 10000,
    domain_balanced: bool = False,
    ptb_batch_frac: float = 0.20,
    ptb_loss_weight: float = 0.5,
    use_balanced: bool = False,
    use_3beat: bool = False,
    sliding_dup: int = 0,
    sliding_max_shift: int = 40,
    input_shape_override: tuple = None,
    patient_split: bool = False,        # 4.4-4: 患者级划分 (发表级严谨)
) -> dict:
    """
    一站式准备所有数据集。

    Args:
        use_ptbxl: 仅用 PTB-XL.
        use_merged: MIT-BIH + PTB-XL.
        use_incart: MIT-BIH + INCART.
        use_ecg1000: MIT-BIH + ECG1000.
        use_ptb_beat: MIT-BIH + INCART 原协议划分 (val/test 与历史可比),
                      PTB beat 级数据**只进训练集** (避免 PTB 信号风格污染评估),
                      异常拍按 ptb_abn_max 限量配比.
        ptb_abn_max: PTB 异常拍最大数量 (默认 10000, 防 MI 形态主导).
        domain_balanced: 域平衡采样 (每 batch 固定 ptb_batch_frac 比例 PTB 拍,
                          PTB 拍 loss 权重 ptb_loss_weight), 需配合 use_ptb_beat.
        ptb_batch_frac: 每 batch 中 PTB 拍占比 (默认 0.20).
        ptb_loss_weight: PTB 拍 loss 权重 (默认 0.5, 记录级标签降权).
        use_balanced: 训练集 50/50 类别均衡 oversample.
        sliding_dup: 异常类滑窗采样增强, 每个异常心拍生成的移位视图数
                     (0 = 关闭, 仅作用于训练集, 参考哈工大学报 2019).
        sliding_max_shift: 滑窗最大平移量 (采样点).
    """
    if use_3beat:
        data = load_3beat_merged()
    elif use_ptb_beat:
        # ★ 必须以 MIT+INCART 为基础 (否则退化为 MIT 单独, PTB 占比过高)
        data = load_mit_incart_merged()
    elif use_ptbxl_rhythm:
        data = load_all_three_merged()
    elif use_ecg1000:
        data = load_mit_ecg1000_merged()
    elif use_incart:
        data = load_mit_incart_merged()
    elif use_merged:
        data = load_merged_data()
    elif use_ptbxl:
        data = load_ptbxl_data()
    else:
        data = load_processed_data()
    if patient_split:
        # 4.4-4 患者级划分 (发表级严谨): 同一患者的所有记录/心拍不跨划分。
        # 与 eval_patient_split_all.py 使用完全相同的映射与 seed=42 划分,
        # 保证训练/评估的患者分区一致。注意 INCART record_id 在合并时 +100000。
        from data.patient_split import (build_mit_patient_map,
                                        build_incart_patient_map,
                                        patient_level_split)
        _pmap = {}
        _pmap.update(build_mit_patient_map())
        # 与 eval 脚本完全一致 (含双前缀, 保证排序→permutation→测试患者完全相同)
        _pmap.update({rid + 100000: "inc_" + pat
                      for rid, pat in build_incart_patient_map().items()})
        tr_m, va_m, te_m, pstats = patient_level_split(data["record_ids"], _pmap)
        splits = {"train": (data["beats"][tr_m], data["labels"][tr_m]),
                  "val":   (data["beats"][va_m], data["labels"][va_m]),
                  "test":  (data["beats"][te_m], data["labels"][te_m])}
        print(f"[数据集] 患者级划分 (seed=42): 患者 {pstats['n_patients']} = "
              f"train {pstats['n_train']} / val {pstats['n_val']} / test {pstats['n_test']}")
        print(f"[数据集]   拍数: train {pstats['beats_train']:,} / "
              f"val {pstats['beats_val']:,} / test {pstats['beats_test']:,}")
    else:
        splits = train_val_test_split(data["beats"], data["labels"],
                                       record_ids=data.get("record_ids"))


    # 划分结果已按布尔掩码拷贝成独立数组；尽快释放合并后的全量数组，
    # 再加载 PTB / 构建 tf.data，避免约 512MB+ 的同 dtype 峰值内存。
    del data

    if use_ptb_beat and not domain_balanced:
        # PTB 受控配比进训练集: val/test 保持 MIT+INCART 原协议。
        # (domain_balanced 模式下 PTB 由域平衡分支单独处理)
        # 实验验证: 全量 PTB 异常拍(59K, 占异常60%)会主导模型形态学习
        # (手动 val AUC 0.63); 仅正常拍安全 (0.92)。故异常拍限量配比,
        # 让模型学到 MI 形态而不被主导。
        ptb = load_ptb_data()
        if patient_split:
            # 4.4-4: 训练侧消除泄漏 — PTB 拍仅取 train 患者,
            # test/val 患者的拍绝不进训练集 (历史 exp5: seed42 全患者抽拍,
            # ~17% 测试拍训练时见过)
            from data.patient_split import (build_ptb_patient_map,
                                            patient_level_split)
            _trp, _, _, _ps = patient_level_split(ptb["record_ids"],
                                                  build_ptb_patient_map())
            ptb = {"beats": ptb["beats"][_trp], "labels": ptb["labels"][_trp],
                   "record_ids": ptb["record_ids"][_trp]}
            print(f"[数据集] 患者级清洁: PTB 训练拍仅取 train 患者 "
                  f"({_ps['n_train']}/{_ps['n_patients']} 患者, 剩 {len(ptb['beats']):,} 拍)")
        mask_n = ptb["labels"] == 0
        x_ptb_n = ptb["beats"][mask_n]
        n_ptb_n = len(x_ptb_n)
        idx_a = np.where(ptb["labels"] == 1)[0]
        if len(idx_a) > ptb_abn_max:
            rng = np.random.default_rng(42)
            idx_a = rng.choice(idx_a, ptb_abn_max, replace=False)
        x_ptb_a = ptb["beats"][idx_a]
        n_ptb_a = len(x_ptb_a)
        n_ptb = n_ptb_n + n_ptb_a
        print(f"[数据集] PTB 进训练集: 正常+{n_ptb_n}, 异常+{n_ptb_a} "
              f"(限量 {ptb_abn_max}, 共 {len(ptb['beats'])} 拍)")
        x_tr_all = np.concatenate([splits["train"][0], x_ptb_n, x_ptb_a], axis=0)
        y_tr_all = np.concatenate([
            splits["train"][1],
            np.zeros(n_ptb_n, dtype=splits["train"][1].dtype),
            np.ones(n_ptb_a, dtype=splits["train"][1].dtype)], axis=0)
        print(f"[数据集] 训练集: {len(splits['train'][0])} → {len(x_tr_all)} "
              f"(异常占比 {np.mean(y_tr_all==1)*100:.1f}%)")
        # ★ 全局打乱: 防止尾部数据集 (PTB) 超过 shuffle buffer 造成
        #   每 epoch 尾部纯 PTB 批次, 权重在域间震荡 (val_auc 0.25~0.68 跳变)
        perm = np.random.default_rng(0).permutation(len(x_tr_all))
        splits["train"] = (x_tr_all[perm], y_tr_all[perm])

    if domain_balanced:
        # ★ 域平衡采样: 不合并, 分域构建 batch (A=MIT+INCART, B=PTB)
        if not use_ptb_beat:
            raise ValueError("domain_balanced 需配合 --ptb-beat")
        ptb = load_ptb_data()
        if patient_split:
            # 4.4-4: 训练侧消除泄漏 (域平衡模式同理)
            from data.patient_split import (build_ptb_patient_map,
                                            patient_level_split)
            _trp, _, _, _ps = patient_level_split(ptb["record_ids"],
                                                  build_ptb_patient_map())
            ptb = {"beats": ptb["beats"][_trp], "labels": ptb["labels"][_trp],
                   "record_ids": ptb["record_ids"][_trp]}
            print(f"[数据集] 患者级清洁: PTB 训练拍仅取 train 患者 "
                  f"({_ps['n_train']}/{_ps['n_patients']} 患者, 剩 {len(ptb['beats']):,} 拍)")
        mask_n = ptb["labels"] == 0
        x_ptb = ptb["beats"][mask_n]
        y_ptb = np.zeros(len(x_ptb), dtype=np.int32)
        idx_a = np.where(ptb["labels"] == 1)[0]
        if len(idx_a) > ptb_abn_max:
            rng = np.random.default_rng(42)
            idx_a = rng.choice(idx_a, ptb_abn_max, replace=False)
        x_ptb = np.concatenate([x_ptb, ptb["beats"][idx_a]], axis=0)
        y_ptb = np.concatenate([y_ptb, np.ones(len(idx_a), dtype=np.int32)], axis=0)
        n_ptb_a = len(idx_a)
        n_ptb_n = len(mask_n[mask_n])
        print(f"[数据集] 域平衡采样: A域(MIT+INCART) {len(splits['train'][0])} 拍, "
              f"B域(PTB) {len(x_ptb)} 拍 (正常{n_ptb_n}+异常{n_ptb_a})")
        print(f"[数据集] 每 batch {ptb_batch_frac*100:.0f}% PTB, PTB loss 权重 {ptb_loss_weight}")
        train_ds = make_domain_balanced_dataset(
            splits["train"][0], splits["train"][1],
            x_ptb, y_ptb,
            batch_size=batch_size,
            frac_b=ptb_batch_frac,
            weight_b=ptb_loss_weight,
        )
        val_ds = make_tf_dataset(
            splits["val"][0], splits["val"][1],
            batch_size=batch_size, shuffle=False
        )
        test_ds = make_tf_dataset(
            splits["test"][0], splits["test"][1],
            batch_size=batch_size, shuffle=False
        )
        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "data": splits,
            "class_names": CLASS_NAMES,
            "input_shape": input_shape_override if input_shape_override
                           else (750 if use_3beat else INFERENCE_CONFIG['window_size'], 1)
        }

    if sliding_dup > 0:
        x_tr, y_tr = apply_sliding_window_augmentation(
            splits["train"][0], splits["train"][1],
            dup=sliding_dup, max_shift=sliding_max_shift
        )
        splits["train"] = (x_tr, y_tr)

    if use_balanced:
        print("[数据集] 使用类别均衡采样 (50/50 per batch)")
        train_ds = make_balanced_dataset(
            splits["train"][0], splits["train"][1],
            batch_size=batch_size
        )
    else:
        train_ds = make_tf_dataset(
            splits["train"][0], splits["train"][1],
            batch_size=batch_size, shuffle=True, augment=True
        )
    val_ds = make_tf_dataset(
        splits["val"][0], splits["val"][1],
        batch_size=batch_size, shuffle=False
    )
    test_ds = make_tf_dataset(
        splits["test"][0], splits["test"][1],
        batch_size=batch_size, shuffle=False
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "data": splits,
        "class_names": CLASS_NAMES,
        "input_shape": input_shape_override if input_shape_override
                       else (750 if use_3beat else INFERENCE_CONFIG['window_size'], 1)
    }


if __name__ == "__main__":
    print("[数据集] 测试数据集加载与构建...")
    
    # 先尝试加载预处理数据
    try:
        data = load_processed_data()
        print(f"[数据集] ✅ 数据加载成功")
    except FileNotFoundError as e:
        print(f"[数据集] ❌ {e}")
        sys.exit(1)
    
    # 测试划分
    splits = train_val_test_split(data["beats"], data["labels"])
    
    # 测试 TF Dataset 构建
    for name, (x, y) in splits.items():
        ds = make_tf_dataset(x, y, batch_size=32, shuffle=(name == "train"))
        print(f"[数据集] {name} dataset: {ds}")
        for batch_x, batch_y in ds.take(1):
            print(f"[数据集]   batch_x: {batch_x.shape}, batch_y: {batch_y.shape}")
    
    print("[数据集] ✅ 数据集构建测试通过")
