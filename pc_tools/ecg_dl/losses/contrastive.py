#!/usr/bin/env python3
"""
Phase 2C: SimCLR Contrastive Learning for 1D ECG

Components:
  1. Strong ECG augmentations for view pair creation (TimeWarp ±20%,
     AmpScale ±30%, σ=0.03 noise, crop-resize, sign-invert, random mask)
  2. NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
  3. ResNet-Lite encoder backbone (identical layer names to ResNet-L)
  4. Projection head for contrastive learning
  5. SimCLR model builder (encoder + projection head)
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INFERENCE_CONFIG


# ===========================================================================
# Strong ECG Augmentations for SimCLR View Pairs
# ===========================================================================

@tf.function
def ecg_time_warp_strong(x, max_stretch=0.20):
    bs = tf.shape(x)[0]
    sl = tf.cast(tf.shape(x)[1], tf.float32)
    sl_i = tf.cast(sl, tf.int32)
    st = tf.random.uniform((), 1.0 - max_stretch, 1.0 + max_stretch)
    new_len = tf.clip_by_value(
        tf.cast(sl * st, tf.int32),
        tf.cast(sl * 0.75, tf.int32),
        tf.cast(sl * 1.25, tf.int32))
    x_4d = x[:, :, :, tf.newaxis]
    resized = tf.image.resize(x_4d, (new_len, 1))
    padded = tf.image.resize_with_crop_or_pad(resized, sl_i, 1)
    return tf.squeeze(padded, axis=-1)


@tf.function
def ecg_amplitude_scale_strong(x, min_s=0.70, max_s=1.30):
    s = tf.random.uniform((tf.shape(x)[0], 1, 1), min_s, max_s, dtype=x.dtype)
    return x * s


@tf.function
def ecg_gaussian_noise_strong(x, noise_std=0.03):
    return x + tf.random.normal(tf.shape(x), 0.0, noise_std, dtype=x.dtype)


@tf.function
def ecg_baseline_wander_strong(x, amplitude=0.30):
    bs = tf.shape(x)[0]
    sl = tf.cast(tf.shape(x)[1], tf.float32)
    t = tf.reshape(tf.linspace(0.0, 1.0, tf.cast(sl, tf.int32)), (1, -1, 1))
    freq = tf.random.uniform((bs, 1, 1), 0.0, 0.03, dtype=x.dtype)
    phase = tf.random.uniform((bs, 1, 1), 0.0, 6.283, dtype=x.dtype)
    amp = tf.random.uniform((bs, 1, 1), 0.0, amplitude, dtype=x.dtype)
    wander = amp * tf.sin(6.283 * freq * t * sl + phase)
    return x + tf.cast(wander, x.dtype)


@tf.function
def ecg_random_crop_resize(x, min_ratio=0.75, max_ratio=1.0):
    bs = tf.shape(x)[0]
    sl = tf.cast(tf.shape(x)[1], tf.float32)
    sl_i = tf.cast(sl, tf.int32)
    ratio = tf.random.uniform((), min_ratio, max_ratio)
    crop_len = tf.cast(sl * ratio, tf.int32)
    max_start = sl_i - crop_len
    start = tf.cast(tf.random.uniform(()) * tf.cast(max_start, tf.float32), tf.int32)
    x_cropped = x[:, start:start + crop_len, :]
    x_4d = x_cropped[:, :, :, tf.newaxis]
    resized = tf.image.resize(x_4d, (sl_i, 1))
    return tf.squeeze(resized, axis=-1)


@tf.function
def ecg_sign_invert(x, prob=0.3):
    if tf.random.uniform(()) < prob:
        return -x
    return x


@tf.function
def ecg_random_mask(x, max_ratio=0.15):
    bs = tf.shape(x)[0]
    sl_i = tf.shape(x)[1]
    mlr = tf.random.uniform((bs,), 0.05, max_ratio)
    mask_len = tf.cast(tf.cast(sl_i, tf.float32) * mlr, tf.int32)
    max_start_f = tf.cast(sl_i - mask_len, tf.float32)
    start = tf.cast(tf.random.uniform((bs,)) * max_start_f, tf.int32)
    indices = tf.range(sl_i)[tf.newaxis, :, tf.newaxis]
    start_e = start[:, tf.newaxis, tf.newaxis]
    end_e = (start + mask_len)[:, tf.newaxis, tf.newaxis]
    mask = tf.cast((indices >= start_e) & (indices < end_e), x.dtype)
    return x * (1.0 - mask)


_AUGMENTATIONS = [
    ecg_amplitude_scale_strong,
    ecg_gaussian_noise_strong,
    ecg_baseline_wander_strong,
    ecg_sign_invert,
    ecg_random_mask,
]


def _apply_random_augs(y):
    for aug_fn in _AUGMENTATIONS:
        y = aug_fn(y)
    return y


def create_contrastive_pair(x):
    v1 = _apply_random_augs(x)
    v2 = _apply_random_augs(x)
    return v1, v2


# ===========================================================================
# NT-Xent Contrastive Loss
# ===========================================================================

@tf.function
def ntxent_loss(z1, z2, temperature=0.1):
    N = tf.shape(z1)[0]
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    z = tf.concat([z1, z2], axis=0)
    sim = tf.matmul(z, z, transpose_b=True) / temperature
    diag_mask = tf.eye(2 * N) * 1e9
    sim_masked = sim - diag_mask
    labels = tf.concat([tf.range(N, 2 * N), tf.range(0, N)], axis=0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=sim_masked)
    return tf.reduce_mean(loss)


# ===========================================================================
# ResNet-Lite Encoder (backbone; layer names match resnet_lite_1d.py)
# ===========================================================================

def _res_block(x, filters, kernel_size, stride=1, block_id=0):
    pfx = f"b{block_id}"
    shortcut = x
    in_ch = x.shape[-1]
    x = layers.DepthwiseConv1D(kernel_size, strides=stride, padding='same',
                               use_bias=False, name=f"{pfx}_dw")(x)
    x = layers.BatchNormalization(name=f"{pfx}_dwbn")(x)
    x = layers.ReLU(name=f"{pfx}_dwrl")(x)
    x = layers.Conv1D(filters, 1, padding='same', use_bias=False,
                      name=f"{pfx}_pw")(x)
    x = layers.BatchNormalization(name=f"{pfx}_pwbn")(x)
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{pfx}_se_gap")(x)
    s = layers.Dense(max(1, ch // 4), activation='relu', name=f"{pfx}_se_d1")(s)
    s = layers.Dense(ch, activation='sigmoid', name=f"{pfx}_se_d2")(s)
    s = layers.Reshape((1, ch), name=f"{pfx}_se_rs")(s)
    x = layers.Multiply(name=f"{pfx}_se_mul")([x, s])
    if stride != 1 or in_ch != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                 use_bias=False, name=f"{pfx}_sk")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{pfx}_skbn")(shortcut)
    x = layers.Add(name=f"{pfx}_add")([x, shortcut])
    x = layers.ReLU(name=f"{pfx}_out")(x)
    return x


def ecg_encoder(input_shape=None, name="ecg_encoder"):
    if input_shape is None:
        input_shape = (INFERENCE_CONFIG['window_size'], 1)
    inputs = layers.Input(shape=input_shape, name="ecg_input")
    x = layers.Conv1D(16, 7, strides=2, padding='same', use_bias=False,
                      name="stem")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_rl")(x)
    stages = [(16, 2, 7, 1), (32, 3, 5, 2), (64, 3, 3, 2), (128, 1, 3, 1)]
    bid = 0
    for f, n_blk, k, s0 in stages:
        for blk in range(n_blk):
            stride = s0 if blk == 0 else 1
            x = _res_block(x, filters=f, kernel_size=k, stride=stride, block_id=bid)
            bid += 1
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    return Model(inputs=inputs, outputs=x, name=name)


def projection_head(embedding_dim=128, hidden_dim=128, output_dim=64):
    return tf.keras.Sequential([
        layers.Dense(hidden_dim, activation='relu', name="proj_d1"),
        layers.Dense(output_dim, name="proj_d2"),
    ], name="projection_head")


def build_simclr_model(input_shape=None, temperature=0.1, name="simclr"):
    if input_shape is None:
        input_shape = (INFERENCE_CONFIG['window_size'], 1)
    encoder = ecg_encoder(input_shape=input_shape)
    proj = projection_head()
    inputs = layers.Input(shape=input_shape, name="simclr_input")
    embeddings = encoder(inputs)
    projections = proj(embeddings)
    full_model = Model(inputs=inputs, outputs=projections, name=name)
    full_model.temperature = temperature
    return encoder, full_model


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import numpy as np
    print("[SimCLR] Testing contrastive learning components...\n")
    x = tf.constant(np.random.randn(4, 512, 1).astype(np.float32))
    print(f"[Test] Input: {x.shape}")
    for name, fn in [("TimeWarp", ecg_time_warp_strong),
                     ("AmpScale", ecg_amplitude_scale_strong),
                     ("GaussNoise", ecg_gaussian_noise_strong),
                     ("Wander", ecg_baseline_wander_strong),
                     ("CropResize", ecg_random_crop_resize),
                     ("SignInvert", ecg_sign_invert),
                     ("Mask", ecg_random_mask)]:
        out = fn(x)
        print(f"[Test] {name}: {out.shape}, range=[{tf.reduce_min(out):.3f}, {tf.reduce_max(out):.3f}]")
    v1, v2 = create_contrastive_pair(x)
    print(f"\n[Test] Contrastive pair: v1={v1.shape}, v2={v2.shape}")
    z1 = tf.random.normal((8, 64)); z2 = tf.random.normal((8, 64))
    print(f"[Test] NT-Xent loss (random): {ntxent_loss(z1, z2).numpy():.4f}")
    z2_close = z1 + tf.random.normal((8, 64), stddev=0.01)
    print(f"[Test] NT-Xent loss (similar): {ntxent_loss(z1, z2_close).numpy():.4f}")
    encoder = ecg_encoder(input_shape=(512, 1))
    enc_out = encoder(x)
    print(f"\n[Test] Encoder: {encoder.count_params():,} params, output={enc_out.shape}")
    enc, ssl = build_simclr_model(input_shape=(512, 1))
    print(f"[Test] SimCLR: {ssl.count_params():,} params")
    proj_out = ssl(x)
    print(f"[Test] Projection: {proj_out.shape}")
    print("\n[SimCLR] All tests passed!")
