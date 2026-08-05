#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for losses/kd_loss.py.

Plain-assert script: prints per-test status, sys.exit(1) on first
failure, sys.exit(0) when all pass. Run from the ecg_dl directory:

    cd pc_tools/ecg_dl
    python3 tests/test_kd_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly: tests/ sits next to losses/.
_HERE = Path(__file__).resolve().parent
_ECG_DL = _HERE.parent
if str(_ECG_DL) not in sys.path:
    sys.path.insert(0, str(_ECG_DL))

import numpy as np
import tensorflow as tf
from losses.kd_loss import make_kd_loss, SlicedAUC  # noqa: E402


_FAILURES: list[str] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"  -- {detail}"
    print(line, flush=True)
    if not condition:
        _FAILURES.append(name)


def _make_inputs(B: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = np.zeros((B, 4), dtype=np.float32)
    # Alternate one-hot ground truth.
    y_true[0::2, 0] = 1.0
    y_true[1::2, 1] = 1.0
    y_true[:, 2:] = rng.normal(size=(B, 2))  # teacher raw logits
    # Valid softmax predictions.
    raw = rng.normal(size=(B, 2))
    y_pred = tf.nn.softmax(raw).numpy().astype(np.float32)
    return y_true, y_pred


def test_shape_and_finite() -> None:
    """(a) Output shape is (B,) and all entries are finite."""
    y_true, y_pred = _make_inputs(B=5)
    loss = make_kd_loss(alpha=0.5, temperature=3.0)
    out = loss(y_true, y_pred)
    arr = np.asarray(out)
    _check(
        "shape_and_finite: shape == (B,)",
        arr.shape == (5,),
        f"got shape {arr.shape}",
    )
    _check(
        "shape_and_finite: all finite",
        np.all(np.isfinite(arr)),
        f"got min={arr.min():.3e} max={arr.max():.3e}",
    )


def test_alpha_zero_equals_ce() -> None:
    """(b) alpha=0 reduces to plain categorical cross-entropy (atol 1e-6)."""
    y_true, y_pred = _make_inputs(B=6, seed=1)
    loss = make_kd_loss(alpha=0.0, temperature=3.0)
    got = np.asarray(loss(y_true, y_pred))
    ref = tf.keras.losses.categorical_crossentropy(
        y_true[:, :2], y_pred, from_logits=False
    ).numpy()
    _check(
        "alpha_zero_equals_ce: per-sample match",
        np.allclose(got, ref, atol=1e-6),
        f"max abs diff = {np.max(np.abs(got - ref)):.3e}",
    )


def test_T_1000_kl_approximately_zero() -> None:
    """(c) T=1000 makes both softened distributions ~ uniform => raw KL ~ 0.

    Hinton 2015 T^2 scaling deliberately amplifies the soft-target term
    so its gradient magnitude stays on the same order as the unscaled
    CE; the *raw* KL is what is supposed to be ~0 here, not the
    T^2-scaled return value.
    """
    y_true, y_pred = _make_inputs(B=8, seed=2)
    # Reference pure KL with the exact softened distributions.
    T = 1000.0
    s_soft = y_pred ** (1.0 / T)
    s_soft = s_soft / s_soft.sum(axis=-1, keepdims=True)
    t_soft = tf.nn.softmax(y_true[:, 2:] / T).numpy()
    kl_ref = np.sum(
        t_soft
        * (
            np.log(np.clip(t_soft, 1e-7, 1.0))
            - np.log(np.clip(s_soft, 1e-7, 1.0))
        ),
        axis=-1,
    )
    # Cross-check: compute the kl exactly the way the loss does it.
    loss = make_kd_loss(alpha=1.0, temperature=T)
    out = np.asarray(loss(y_true, y_pred))
    # The loss returns kl * T^2; with raw kl ~ 1e-7 and T^2 = 1e6, the
    # scaled value is ~ 0.1. So we check the unscaled reference kl is
    # tiny, and that the loss's T^2-scaled output matches it (sanity).
    _check(
        "T_1000_kl: raw KL ~ 0",
        np.all(np.abs(kl_ref) < 1e-3),
        f"max |kl_ref| = {np.max(np.abs(kl_ref)):.3e}",
    )
    # The implementation must produce (raw KL) * T^2; verify that.
    expected_scaled = kl_ref * (T * T)
    _check(
        "T_1000_kl: loss output == raw_kl * T^2 (atol 1e-3)",
        np.allclose(out, expected_scaled, atol=1e-3),
        f"max diff = {np.max(np.abs(out - expected_scaled)):.3e}",
    )


def test_T1_onehot_teacher_kl_sane() -> None:
    """(d) T=1 with one-hot teacher => KL finite, > 0, not huge."""
    rng = np.random.default_rng(3)
    B = 4
    y_true = np.zeros((B, 4), dtype=np.float32)
    y_true[:, 0] = 1.0                                  # one-hot
    y_true[:, 2] = 20.0                                 # ~ one-hot teacher
    y_true[:, 3] = -20.0
    raw = rng.normal(size=(B, 2))
    y_pred = tf.nn.softmax(raw).numpy().astype(np.float32)

    loss = make_kd_loss(alpha=1.0, temperature=1.0)
    got = np.asarray(loss(y_true, y_pred))
    # KL * T^2 with T=1, teacher ~ one-hot => KL ~ -log(p_class0).
    # p_class0 is the student's first-class prob; should be in (0,1).
    # So KL is in (0, ~10]. Loss is KL (alpha=1) * 1^2 = KL.
    _check(
        "T1_onehot_teacher_kl: all finite",
        np.all(np.isfinite(got)),
        f"got {got}",
    )
    _check(
        "T1_onehot_teacher_kl: strictly > 0",
        bool(np.all(got > 0.0)),
        f"got {got}",
    )
    _check(
        "T1_onehot_teacher_kl: not pathologically large",
        bool(np.all(got < 50.0)),
        f"got {got}",
    )
    # Sanity: -log(p_class0) reference.
    p0 = y_pred[:, 0]
    ref = -np.log(np.clip(p0, 1e-7, 1.0))
    _check(
        "T1_onehot_teacher_kl: matches -log p_class0 (atol 1e-3)",
        np.allclose(got, ref, atol=1e-3),
        f"max diff = {np.max(np.abs(got - ref)):.3e}",
    )


def _build_tiny_model(input_dim: int = 4) -> tf.keras.Model:
    """A reproducible 2-layer softmax model for the sample_weight test."""
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(8, activation="relu", name="dense1")(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="dense2")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _snapshot_initial_weights() -> list:
    """Build a model and snapshot its initial weights for re-use."""
    m = _build_tiny_model()
    return m.get_weights()


def test_sample_weight_plumbing() -> None:
    """(e) sw=1 == unweighted loss, sw=0 ~ 0.

    To compare like-for-like, every model in this test is built with the
    same architecture and then forced to share the *exact same initial
    weights* via ``set_weights``. This isolates the effect of
    ``sample_weight`` from random initialization noise.
    """
    x = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.0, 0.2]],
                 dtype=np.float32)
    y_true = np.array(
        [[1.0, 0.0,  3.0, -1.0],
         [0.0, 1.0, -2.0,  4.0]],
        dtype=np.float32,
    )
    init_w = _snapshot_initial_weights()

    def _fresh_compiled_model() -> tf.keras.Model:
        m = _build_tiny_model()
        m.set_weights(init_w)
        m.compile(loss=make_kd_loss(0.5, 3.0), optimizer="adam")
        return m

    # Reference: no sample_weight.
    m_ref = _fresh_compiled_model()
    loss_ref = float(m_ref.train_on_batch(x, y_true))

    # sw = ones: should match the unweighted result (sample mean).
    m_ones = _fresh_compiled_model()
    loss_ones = float(
        m_ones.train_on_batch(x, y_true, sample_weight=np.ones(2))
    )

    # sw = zeros: should be ~ 0.
    m_zero = _fresh_compiled_model()
    loss_zero = float(
        m_zero.train_on_batch(x, y_true, sample_weight=np.zeros(2))
    )

    _check(
        "sample_weight: sw=ones == unweighted",
        abs(loss_ones - loss_ref) < 1e-5,
        f"loss_ref={loss_ref:.6e}  loss_ones={loss_ones:.6e}",
    )
    _check(
        "sample_weight: sw=zeros ~ 0",
        abs(loss_zero) < 1e-5,
        f"loss_zero={loss_zero:.6e}",
    )


def test_gradients_finite() -> None:
    """(f) tf.GradientTape on y_pred: grads finite, no NaN."""
    y_true, y_pred_np = _make_inputs(B=4, seed=7)
    y_pred = tf.Variable(y_pred_np, dtype=tf.float32)

    loss = make_kd_loss(alpha=0.5, temperature=3.0)
    with tf.GradientTape() as tape:
        loss_val = loss(y_true, y_pred)
    grads = tape.gradient(loss_val, y_pred).numpy()

    _check(
        "gradients: no NaN",
        not np.any(np.isnan(grads)),
        f"any NaN: {bool(np.any(np.isnan(grads)))}",
    )
    _check(
        "gradients: all finite",
        np.all(np.isfinite(grads)),
        f"min={grads.min():.3e}  max={grads.max():.3e}",
    )
    _check(
        "gradients: shape matches y_pred",
        grads.shape == y_pred.shape,
        f"got {grads.shape}",
    )


def test_sliced_auc_metric() -> None:
    """Sanity: SlicedAUC slices y_true[:, :2] and reports a metric under key 'auc'."""
    rng = np.random.default_rng(11)
    y_true = np.zeros((64, 4), dtype=np.float32)
    y_true[:32, 0] = 1.0
    y_true[32:, 1] = 1.0
    # Add teacher logits in cols 2-3; should NOT affect the AUC.
    y_true[:, 2:] = rng.normal(size=(64, 2)) * 5.0
    # Predictions that are clearly class-correlated.
    raw = rng.normal(size=(64, 2))
    y_pred = tf.nn.softmax(raw).numpy().astype(np.float32)
    y_pred[:32, 0] += 0.5
    y_pred[32:, 1] += 0.5
    y_pred = y_pred / y_pred.sum(axis=-1, keepdims=True)

    metric = SlicedAUC(name="auc")
    metric.update_state(y_true, y_pred)
    val = float(metric.result().numpy())
    _check(
        "SlicedAUC: result is finite and in [0, 1]",
        np.isfinite(val) and 0.0 <= val <= 1.0,
        f"got {val}",
    )
    _check(
        "SlicedAUC: result > 0.5 (predictions are class-correlated)",
        val > 0.5,
        f"got {val}",
    )


def main() -> int:
    print("=== test_kd_loss ===", flush=True)
    test_shape_and_finite()
    test_alpha_zero_equals_ce()
    test_T_1000_kl_approximately_zero()
    test_T1_onehot_teacher_kl_sane()
    test_sample_weight_plumbing()
    test_gradients_finite()
    test_sliced_auc_metric()

    print("=" * 30, flush=True)
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} test(s): {_FAILURES}", flush=True)
        return 1
    print("All tests PASSED.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
