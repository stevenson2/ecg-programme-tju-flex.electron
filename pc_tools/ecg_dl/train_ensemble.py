#!/usr/bin/env python3
"""
Phase 2C: 3-Seed Ensemble Training for ECG Anomaly Detection

Trains 3 identical ResNet-L models with different random seeds,
then evaluates the soft-voting ensemble (averaged predicted probabilities).

Expected improvement over single model: Acc +1-2%, Recall +2-4%

Usage:
  python train_ensemble.py                                    # Train 3 seeds + evaluate
  python train_ensemble.py --eval-only                        # Evaluate existing ensemble
  python train_ensemble.py --seeds 42 123 456 --epochs 100
  python train_ensemble.py --model resnet_m                   # Use ResNet-M instead
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf
from config import (TRAIN_CONFIG, MODELS_DIR, PROCESSED_DIR,
                    BEAT_WINDOW_SAMPLES, CLASS_NAMES)
from data.dataset import (load_mit_incart_merged, train_val_test_split,
                          make_tf_dataset, add_channel_dim)
from losses.focal_loss import FocalLoss
from models.resnet_lite_1d import (build_ecg_resnet_lite_large,
                                   build_ecg_resnet_lite_medium,
                                   model_summary_table)

ABNORMAL_IDX = 1


def build_model(model_type="resnet_l"):
    if model_type == "resnet_l":
        return build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    elif model_type == "resnet_m":
        return build_ecg_resnet_lite_medium(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    else:
        raise ValueError(f"Unknown model: {model_type}")


def compile_model(model, learning_rate=5e-4):
    fl = FocalLoss(gamma=TRAIN_CONFIG['focal_loss']['gamma'],
                   alpha=TRAIN_CONFIG['focal_loss']['alpha'],
                   label_smoothing=TRAIN_CONFIG['focal_loss']['label_smoothing'])
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                    weight_decay=1e-4)
    model.compile(optimizer=opt, loss=fl, metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ])
    return model


def get_callbacks(model_name, patience=20):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=patience,
                                         mode='max', restore_best_weights=True,
                                         verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                             patience=10, min_lr=1e-6,
                                             mode='max', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / model_name),
            monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    ]


def compute_metrics(y_true, prob_ab, threshold=0.5):
    y_pred = (prob_ab >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-8)
    spec = tn / max(tn + fp, 1)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, prob_ab))
    return {"acc": acc, "prec": prec, "recall": recall, "f1": f1,
            "spec": spec, "auc": auc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def train_single_seed(seed, data_splits, model_type, epochs, batch_size,
                      learning_rate):
    print(f"\n{'='*55}")
    print(f"  Seed {seed}: Training ResNet-L")
    print(f"{'='*55}\n")

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    x_train, y_train = data_splits["train"]
    x_val, y_val = data_splits["val"]

    train_ds = make_tf_dataset(x_train, y_train, batch_size=batch_size,
                               shuffle=True, augment=True)
    val_ds = make_tf_dataset(x_val, y_val, batch_size=batch_size,
                             shuffle=False)

    model_name = f"ensemble_seed{seed}.h5"
    model = build_model(model_type)
    model = compile_model(model, learning_rate=learning_rate)
    model_summary_table(model)

    cbs = get_callbacks(model_name)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=cbs, verbose=2)

    tf.keras.backend.clear_session()
    return MODELS_DIR / model_name


def evaluate_ensemble(model_paths, x_test, y_test, threshold=0.50):
    print(f"\n{'='*55}")
    print(f"  Ensemble Evaluation ({len(model_paths)} models)")
    print(f"{'='*55}")

    models = []
    prob_ab_list = []
    x_in = add_channel_dim(x_test)

    for mp in model_paths:
        print(f"  Loading {mp.name}...")
        m = tf.keras.models.load_model(str(mp), compile=False)
        models.append(m)
        y_prob = m.predict(x_in, verbose=0)
        prob_ab_list.append(y_prob[:, ABNORMAL_IDX])

    prob_stack = np.stack(prob_ab_list, axis=0)
    prob_ensemble = prob_stack.mean(axis=0)

    print(f"\n{'─'*50}")
    print(f"  {'Model':<18} {'Acc':>7} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'─'*50}")

    best_recall = 0.0
    for i, (mp, prob_ab) in enumerate(zip(model_paths, prob_ab_list)):
        m = compute_metrics(y_test, prob_ab, threshold)
        name = f"Seed {mp.stem.replace('ensemble_seed','')}"
        print(f"  {name:<18} {m['acc']:>6.2%} {m['auc']:>7.4f} {m['prec']:>6.2%} "
              f"{m['recall']:>6.2%} {m['f1']:>6.2%}")
        best_recall = max(best_recall, m['recall'])

    m_ens = compute_metrics(y_test, prob_ensemble, threshold)
    print(f"  {'─'*50}")
    print(f"  {'ENSEMBLE (avg)':<18} {m_ens['acc']:>6.2%} {m_ens['auc']:>7.4f} "
          f"{m_ens['prec']:>6.2%} {m_ens['recall']:>6.2%} {m_ens['f1']:>6.2%}")
    print(f"  {'─'*50}")

    delta_recall = (m_ens['recall'] - best_recall) * 100
    delta_auc = (m_ens['auc'] - max(compute_metrics(y_test, p, threshold)['auc']
                                     for p in prob_ab_list)) * 1000
    print(f"\n  Ensemble ΔRecall: {delta_recall:+.1f}% vs best single model")
    print(f"  Ensemble TP={m_ens['tp']} FP={m_ens['fp']} "
          f"FN={m_ens['fn']} TN={m_ens['tn']}")
    print(f"{'='*55}")

    for m in models:
        tf.keras.backend.clear_session()

    return m_ens


def train_ensemble(seeds=(42, 123, 456), model_type="resnet_l",
                   epochs=150, batch_size=64, learning_rate=5e-4):
    print(f"\n{'='*55}")
    print(f"  3-Seed Ensemble Training")
    print(f"  Model: {model_type}  |  Seeds: {seeds}")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}")
    print(f"{'='*55}")

    data = load_mit_incart_merged()
    splits = train_val_test_split(data["beats"], data["labels"],
                                  record_ids=data.get("record_ids"))
    nN = int((splits["test"][1] == 0).sum())
    nA = int((splits["test"][1] == 1).sum())
    print(f"  Test set: {len(splits['test'][0])} samples "
          f"(N={nN}, A={nA}, {nA/(nN+nA)*100:.1f}%)")

    model_paths = []
    for seed in seeds:
        mp = train_single_seed(seed, splits, model_type, epochs,
                               batch_size, learning_rate)
        model_paths.append(mp)

    x_test, y_test = splits["test"]
    evaluate_ensemble(model_paths, x_test, y_test, threshold=0.35)

    # Also evaluate at threshold 0.5 for comparison with Phase 1/2A baselines
    print("\n  At threshold 0.50 (baseline comparison):")
    evaluate_ensemble(model_paths, x_test, y_test, threshold=0.50)

    return model_paths


def eval_only(seeds=(42, 123, 456)):
    model_paths = [MODELS_DIR / f"ensemble_seed{s}.h5" for s in seeds]
    missing = [mp for mp in model_paths if not mp.exists()]
    if missing:
        print(f"[ERROR] Models not found: {missing}")
        print("  Run train_ensemble.py without --eval-only first.")
        return

    data = load_mit_incart_merged()
    splits = train_val_test_split(data["beats"], data["labels"],
                                  record_ids=data.get("record_ids"))
    x_test, y_test = splits["test"]
    evaluate_ensemble(model_paths, x_test, y_test, threshold=0.35)
    print("\n  At threshold 0.50 (baseline comparison):")
    evaluate_ensemble(model_paths, x_test, y_test, threshold=0.50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3-Seed Ensemble Training for ECG Anomaly Detection")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate existing ensemble models")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds (default: 42 123 456)")
    parser.add_argument("--model", type=str, default="resnet_l",
                        choices=["resnet_l", "resnet_m"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)

    args = parser.parse_args()

    if args.eval_only:
        eval_only(seeds=tuple(args.seeds))
    else:
        train_ensemble(seeds=tuple(args.seeds), model_type=args.model,
                       epochs=args.epochs, batch_size=args.batch_size,
                       learning_rate=args.lr)

    print("\n[DONE]")
