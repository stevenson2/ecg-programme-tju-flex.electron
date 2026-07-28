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
    PROCESSED_DIR, TRAIN_CONFIG, CLASS_NAMES, INFERENCE_CONFIG
)


def load_processed_data() -> dict:
    """
    加载预处理后的数据
    
    Returns:
        {"beats": np.ndarray (n, 250), "labels": np.ndarray (n,)}
    """
    npz_path = PROCESSED_DIR / "mit_bih_processed.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(
            f"预处理数据未找到: {npz_path}\n"
            f"请先运行: python data/preprocess.py"
        )
    
    data = np.load(npz_path)
    beats = data["beats"]
    labels = data["labels"]
    record_ids = data.get("record_ids", None)
    
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
    npz_path = PROCESSED_DIR / "incart_processed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"INCART 预处理数据未找到: {npz_path}\n"
            f"请先运行: python data/preprocess_incart.py"
        )
    data = np.load(npz_path)
    beats = data["beats"]
    labels = data["labels"]
    record_ids = data.get("record_ids", None)
    print(f"[INCART] 加载: {len(beats)} 心拍, 形状: {beats.shape}")
    for i, name in enumerate(CLASS_NAMES):
        count = int((labels == i).sum())
        print(f"[INCART]   {name}: {count} ({count/len(labels)*100:.1f}%)")
    return {"beats": beats, "labels": labels, "record_ids": record_ids}


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
        
        train_mask = np.array([rid in train_recs for rid in record_ids])
        val_mask = np.array([rid in val_recs for rid in record_ids])
        test_mask = np.array([rid in test_recs for rid in record_ids])
        
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


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = None,
    shuffle: bool = True,
    buffer_size: int = 10000
) -> tf.data.Dataset:
    """
    构建 TensorFlow Dataset
    
    Args:
        x: 特征数据 (n, 250)
        y: 标签 (n,)
        batch_size: 批大小
        shuffle: 是否打乱
        buffer_size: 打乱缓冲区大小
        
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
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def prepare_datasets(
    augment: bool = False,
    batch_size: int = None,
    use_ptbxl: bool = False,
    use_merged: bool = False,
    use_incart: bool = False,
    use_ecg1000: bool = False,
    use_ptbxl_rhythm: bool = False,
) -> dict:
    """
    一站式准备所有数据集。

    Args:
        use_ptbxl: 仅用 PTB-XL.
        use_merged: MIT-BIH + PTB-XL.
        use_incart: MIT-BIH + INCART.
        use_ecg1000: MIT-BIH + ECG1000.
    """
    if use_ptbxl_rhythm:
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
    splits = train_val_test_split(data["beats"], data["labels"],
                                   record_ids=data.get("record_ids"))
    
    train_ds = make_tf_dataset(
        splits["train"][0], splits["train"][1],
        batch_size=batch_size, shuffle=True
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
        "input_shape": (INFERENCE_CONFIG['window_size'], 1)
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