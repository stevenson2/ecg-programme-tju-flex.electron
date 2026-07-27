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
    batch_size: int = None
) -> dict:
    """
    一站式准备所有数据集（从原始预处理数据到TF Dataset）
    
    Args:
        augment: 是否使用数据增强后的数据
        batch_size: 批大小
        
    Returns:
        {
            "train_ds": tf.data.Dataset,
            "val_ds": tf.data.Dataset,
            "test_ds": tf.data.Dataset,
            "data": {"train": (x,y), "val": (x,y), "test": (x,y)}
        }
    """
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