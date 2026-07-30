#!/usr/bin/env python3
"""
SSL Performance Profiler v2 — test XLA compilation.

Usage:
  python3 profile_ssl.py
"""

import sys, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf
from losses.contrastive import (build_simclr_model,
                                create_contrastive_pair, ntxent_loss)

B, T = 512, 512
WARMUP, REPEAT = 2, 10


def benchmark(name, fn, warmup=WARMUP, repeat=REPEAT):
    try:
        for _ in range(warmup):
            fn()
        t0 = time.time()
        for _ in range(repeat):
            fn()
        ms = (time.time() - t0) / repeat * 1000
        print(f"  {name:<40} {ms:>8.2f} ms")
        return ms
    except Exception as e:
        print(f"  {name:<40} {'FAILED':>8s}  ({e})")
        return None


def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    x = tf.constant(np.random.randn(B, T, 1).astype(np.float32))

    encoder, simclr = build_simclr_model(input_shape=(T, 1))
    print(f"\n{'='*60}")
    print(f"  SSL Bottleneck Profiler (batch={B})")
    print(f"  Encoder: {encoder.count_params():,} params")
    print(f"{'='*60}")

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    print(f"\n--- @tf.function (standard) ---")
    @tf.function
    def step_std(x):
        with tf.GradientTape() as tape:
            v1, v2 = create_contrastive_pair(x)
            z = simclr(tf.concat([v1, v2], axis=0), training=True)
            z1, z2 = tf.split(z, 2, axis=0)
            loss = ntxent_loss(z1, z2, 0.1)
        grads = tape.gradient(loss, simclr.trainable_weights)
        optimizer.apply_gradients(zip(grads, simclr.trainable_weights))
        return loss
    ms_std = benchmark("full step (standard)", lambda: step_std(x))

    print(f"\n--- @tf.function(jit_compile=True) XLA ---")
    @tf.function(jit_compile=True)
    def step_xla(x):
        with tf.GradientTape() as tape:
            v1, v2 = create_contrastive_pair(x)
            z = simclr(tf.concat([v1, v2], axis=0), training=True)
            z1, z2 = tf.split(z, 2, axis=0)
            loss = ntxent_loss(z1, z2, 0.1)
        grads = tape.gradient(loss, simclr.trainable_weights)
        optimizer.apply_gradients(zip(grads, simclr.trainable_weights))
        return loss
    ms_xla = benchmark("full step (XLA)", lambda: step_xla(x))

    print(f"\n--- Forward only (no grad, no optimizer) ---")
    @tf.function
    def fwd(x):
        v1, v2 = create_contrastive_pair(x)
        z = simclr(tf.concat([v1, v2], axis=0), training=True)
        z1, z2 = tf.split(z, 2, axis=0)
        return ntxent_loss(z1, z2, 0.1)
    ms_fwd = benchmark("forward + loss only", lambda: fwd(x))

    print(f"\n--- Backward pass only (no optimizer update) ---")
    @tf.function
    def bwd(x):
        with tf.GradientTape() as tape:
            v1, v2 = create_contrastive_pair(x)
            z = simclr(tf.concat([v1, v2], axis=0), training=True)
            z1, z2 = tf.split(z, 2, axis=0)
            loss = ntxent_loss(z1, z2, 0.1)
        grads = tape.gradient(loss, simclr.trainable_weights)
        return grads
    ms_bwd = benchmark("forward + gradients", lambda: bwd(x))

    print(f"\n--- Optimizer update only (no grad computation) ---")
    grads = bwd(x)
    @tf.function
    def update():
        optimizer.apply_gradients(zip(grads, simclr.trainable_weights))
    ms_opt = benchmark("apply_gradients only", lambda: update())

    print(f"\n--- SGD optimizer (no AdamW overhead) ---")
    sgd = tf.keras.optimizers.SGD(learning_rate=1e-3)
    @tf.function
    def step_sgd(x):
        with tf.GradientTape() as tape:
            v1, v2 = create_contrastive_pair(x)
            z = simclr(tf.concat([v1, v2], axis=0), training=True)
            z1, z2 = tf.split(z, 2, axis=0)
            loss = ntxent_loss(z1, z2, 0.1)
        grads = tape.gradient(loss, simclr.trainable_weights)
        sgd.apply_gradients(zip(grads, simclr.trainable_weights))
        return loss
    benchmark("full step (SGD)", lambda: step_sgd(x))

    if ms_std and ms_fwd:
        print(f"\n{'='*60}")
        print(f"  Breakdown (standard @tf.function):")
        print(f"    Forward + loss:    {ms_fwd:>7.1f} ms")
        print(f"    + Gradients:       {ms_bwd - ms_fwd:>+7.1f} ms" if ms_bwd else "    + Gradients:       N/A")
        print(f"    + Optimizer update: {ms_opt:>+7.1f} ms" if ms_opt else "    + Optimizer update: N/A")
        print(f"    = Total:            {ms_std:>7.1f} ms")
        if ms_xla:
            print(f"\n  XLA speedup: {ms_std/ms_xla:.1f}x  "
                  f"({ms_std:.0f}ms → {ms_xla:.0f}ms)")
            print(f"  Est. epoch time: {ms_xla*125:.1f}s (XLA) vs "
                  f"{ms_std*125:.1f}s (std)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
