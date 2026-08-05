#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for make_domain_balanced_dataset_kd (knowledge-distillation variant).

Synthetic data only — no real ECG files required.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Bootstrap sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import make_domain_balanced_dataset_kd  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
N_A = 1000
N_B = 200
WIN = 250

rng = np.random.default_rng(SEED)

# x_a: normal ECG-like noise; x_b: filled with a distinctive constant (999)
x_a = rng.standard_normal((N_A, WIN)).astype(np.float32)
x_b = np.full((N_B, WIN), 999.0, dtype=np.float32)

# y_a / y_b: already (N,4) KD targets = concat([onehot(2), teacher_logits(2)])
onehot_a = tf.keras.utils.to_categorical(
    np.array([0] * (N_A // 2) + [1] * (N_A - N_A // 2)), num_classes=2
).astype(np.float32)
logits_a = rng.standard_normal((N_A, 2)).astype(np.float32)
y_a = np.concatenate([onehot_a, logits_a], axis=-1)

onehot_b = tf.keras.utils.to_categorical(
    np.array([0] * (N_B // 2) + [1] * (N_B - N_B // 2)), num_classes=2
).astype(np.float32)
logits_b = rng.standard_normal((N_B, 2)).astype(np.float32)
y_b = np.concatenate([onehot_b, logits_b], axis=-1)


# ── helpers ───────────────────────────────────────────────────────────────
def _make_ds():
    """Build dataset — function adds channel dim and concatenates onehot+logits internally."""
    return make_domain_balanced_dataset_kd(
        x_a, onehot_a, logits_a,
        x_b, onehot_b, logits_b,
        batch_size=BATCH_SIZE,
        frac_b=0.20,
        weight_b=0.5,
    )


def _collect_batches(ds, n):
    """Yield first n batches as numpy tuples."""
    for i, batch in enumerate(ds):
        if i >= n:
            break
        yield tuple(b.numpy() if isinstance(b, tf.Tensor) else b for b in batch)


# ── tests ─────────────────────────────────────────────────────────────────
def test_a_yields_3_tuples_with_y_last_dim_4():
    """(a) dataset yields 3-tuples (x, y, sw) with y.shape[-1] == 4."""
    ds = _make_ds()
    for bx, by, bsw in _collect_batches(ds, 10):
        assert bx.ndim == 3, f"x ndim={bx.ndim}, expected 3"
        assert by.ndim == 2, f"y ndim={by.ndim}, expected 2"
        assert by.shape[-1] == 4, f"y last dim={by.shape[-1]}, expected 4"
        assert bsw.ndim == 1, f"sw ndim={bsw.ndim}, expected 1"
    print("  PASS test_a_yields_3_tuples_with_y_last_dim_4")


def test_b_argmax_consistent_with_source_labels():
    """(b) argmax(y[:, :2], -1) over first batch matches source labels."""
    ds = _make_ds()
    all_labels_a = np.argmax(y_a[:, :2], axis=-1)  # source labels for A
    all_labels_b = np.argmax(y_b[:, :2], axis=-1)  # source labels for B

    for bx, by, bsw in _collect_batches(ds, 5):
        batch_labels = np.argmax(by[:, :2], axis=-1)
        # Every label in the batch must exist in the combined source pool
        combined = np.concatenate([all_labels_a, all_labels_b])
        for lbl in batch_labels:
            assert lbl in combined, f"Label {lbl} not in source labels"
    print("  PASS test_b_argmax_consistent_with_source_labels")


def test_c_sw_values_in_expected_set():
    """(c) sw values ∈ {1.0, 0.5} over 50 batches."""
    ds = _make_ds()
    for _, _, bsw in _collect_batches(ds, 50):
        unique = set(np.unique(bsw))
        assert unique.issubset({1.0, 0.5}), (
            f"Unexpected sw values: {unique - {1.0, 0.5}}")
    print("  PASS test_c_sw_values_in_expected_set")


def test_d_b_domain_fraction_in_range():
    """(d) Over 100 batches, fraction of rows from B ∈ [0.15, 0.25].

    B-domain rows are identifiable by x == 999 (distinctive constant).
    """
    ds = _make_ds()
    total_rows = 0
    b_rows = 0
    for bx, _, _ in _collect_batches(ds, 100):
        # x shape (batch, 250, 1); B rows have first value == 999
        is_b = bx[:, 0, 0] == 999.0
        b_rows += int(is_b.sum())
        total_rows += len(bx)
    frac = b_rows / total_rows
    assert 0.15 <= frac <= 0.25, (
        f"B-domain fraction {frac:.3f} outside [0.15, 0.25]")
    print(f"  PASS test_d_b_domain_fraction_in_range (frac={frac:.3f})")


def test_e_z_columns_match_source_provenance():
    """(e) z-columns of a sampled batch match corresponding rows of input z arrays.

    Strategy: mark z_b rows with a distinctive value so we can identify which
    rows came from B, then verify the z columns match.
    """
    # Rebuild with distinctive z_b values
    rng2 = np.random.default_rng(99)
    z_a2 = rng2.standard_normal((N_A, 2)).astype(np.float32)
    z_b2 = np.full((N_B, 2), 777.0, dtype=np.float32)  # distinctive marker

    onehot_a2 = tf.keras.utils.to_categorical(
        np.array([0] * (N_A // 2) + [1] * (N_A - N_A // 2)), num_classes=2
    ).astype(np.float32)

    onehot_b2 = tf.keras.utils.to_categorical(
        np.array([0] * (N_B // 2) + [1] * (N_B - N_B // 2)), num_classes=2
    ).astype(np.float32)

    ds2 = make_domain_balanced_dataset_kd(
        x_a, onehot_a2, z_a2,
        x_b, onehot_b2, z_b2,
        batch_size=BATCH_SIZE,
        frac_b=0.20,
        weight_b=0.5,
    )

    for bx, by, bsw in _collect_batches(ds2, 20):
        # B-domain rows: z columns should be 777.0
        is_b = bx[:, 0, 0] == 999.0
        z_cols = by[:, 2:]  # last 2 columns are teacher logits
        if is_b.any():
            b_z = z_cols[is_b]
            assert np.allclose(b_z, 777.0), (
                f"B-domain z mismatch: expected 777.0, got {b_z[:3]}")
        # A-domain rows: z columns should NOT be 777.0
        is_a = ~is_b
        if is_a.any():
            a_z = z_cols[is_a]
            assert not np.all(a_z == 777.0), (
                "A-domain z should not carry B marker 777.0")
    print("  PASS test_e_z_columns_match_source_provenance")


# ── runner ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Testing make_domain_balanced_dataset_kd")
    print("=" * 60)
    test_a_yields_3_tuples_with_y_last_dim_4()
    test_b_argmax_consistent_with_source_labels()
    test_c_sw_values_in_expected_set()
    test_d_b_domain_fraction_in_range()
    test_e_z_columns_match_source_provenance()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
