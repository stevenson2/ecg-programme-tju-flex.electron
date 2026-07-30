#!/usr/bin/env python3
"""
Phase 2C+: SSL Encoder Classification Head Fine-tuning

Strategy:
  1. Freeze SSL encoder (AUC 0.98, excellent representations)
  2. Train only classification head with aggressive FocalLoss
  3. Optionally unfreeze top layers for joint fine-tuning

This pushes the decision boundary toward higher recall while preserving
the encoder's learned ECG representations.

Usage:
  python3 train_cls_head.py                                    # Default: head only
  python3 train_cls_head.py --unfreeze                         # Head + unfreeze encoder
  python3 train_cls_head.py --alpha 0.85 --epochs1 30 --epochs2 30
"""

import sys, os, argparse, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf
from config import (TRAIN_CONFIG, MODELS_DIR, BEAT_WINDOW_SAMPLES, CLASS_NAMES)
from data.dataset import (load_mit_incart_merged, train_val_test_split,
                          make_tf_dataset, make_balanced_dataset, add_channel_dim)
from losses.contrastive import ecg_encoder
from losses.focal_loss import FocalLoss
from models.resnet_lite_1d import (build_ecg_resnet_lite_large,
                                   model_summary_table)

ENCODER_WEIGHTS_DEFAULT = str(MODELS_DIR / "ssl_encoder.weights.h5")


def transfer_encoder_weights(model, encoder_weights_path):
    encoder_weights_path = Path(encoder_weights_path)
    print(f"[Transfer] Loading SSL encoder: {encoder_weights_path}")
    try:
        ssl_enc = ecg_encoder(input_shape=(BEAT_WINDOW_SAMPLES, 1))
        ssl_enc.build((None, BEAT_WINDOW_SAMPLES, 1))
        ssl_enc.load_weights(str(encoder_weights_path))
    except FileNotFoundError:
        print("[Transfer] WARNING: SSL encoder not found, starting from scratch")
        return 0
    ssl_names = {l.name: l for l in ssl_enc.layers}
    transferred = 0
    for layer in model.layers:
        if layer.name in ssl_names:
            try:
                layer.set_weights(ssl_names[layer.name].get_weights())
                transferred += 1
            except Exception:
                pass
    print(f"[Transfer] {transferred} layers transferred from SSL encoder")
    return transferred


def freeze_encoder(model, freeze=True):
    encoder_layer_names = {"stem", "stem_bn", "stem_rl", "gap"}
    for i in range(20):
        for suffix in ["_dw", "_dwbn", "_dwrl", "_pw", "_pwbn",
                       "_se_gap", "_se_d1", "_se_d2", "_se_rs",
                       "_se_mul", "_sk", "_skbn", "_add", "_out"]:
            encoder_layer_names.add(f"b{i}{suffix}")
    frozen = 0
    for layer in model.layers:
        if layer.name in encoder_layer_names:
            layer.trainable = not freeze
            if freeze:
                frozen += 1
    print(f"[Freeze] {'Froze' if freeze else 'Unfroze'} {frozen} encoder layers")


def train_head(alpha=0.85, gamma=1.0, epochs=30, batch_size=64, lr=1e-3,
               encoder_weights_path=ENCODER_WEIGHTS_DEFAULT, freeze=True,
               balanced=False):
    print(f"\n{'='*60}")
    print(f" Phase 1: Train Classification Head (encoder frozen)")
    print(f"{'='*60}")
    print(f"  FocalLoss: α={alpha}, γ={gamma}")
    print(f"  LR: {lr}, Epochs: {epochs}")

    data = load_mit_incart_merged()
    splits = train_val_test_split(data["beats"], data["labels"],
                                  record_ids=data.get("record_ids"))
    if balanced:
        train_ds = make_balanced_dataset(splits["train"][0], splits["train"][1],
                                         batch_size=batch_size)
        print(f"[Train] Using BALANCED sampling (50/50 per batch)")
    else:
        train_ds = make_tf_dataset(splits["train"][0], splits["train"][1],
                                   batch_size=batch_size, shuffle=True, augment=True)
    val_ds = make_tf_dataset(splits["val"][0], splits["val"][1],
                             batch_size=batch_size, shuffle=False)

    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    model.build((None, BEAT_WINDOW_SAMPLES, 1))
    transfer_encoder_weights(model, encoder_weights_path)
    freeze_encoder(model, freeze=freeze)
    if freeze:
        print(f"[Train] Encoder FROZEN — training classification head only")
    else:
        print(f"[Train] Encoder UNFROZEN — full model training with SSL init")
    model_summary_table(model)

    fl = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=0.05)
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    model.compile(optimizer=opt, loss=fl, metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ])

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=10,
                                         mode='max', restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_recall', factor=0.5,
                                             patience=5, min_lr=1e-6, mode='max', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / 'best_head_tuned.h5'),
            monitor='val_recall', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(MODELS_DIR / 'head_tune_history.csv')),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=cbs, verbose=2)
    return model, splits


def unfreeze_train(model, splits, alpha=0.85, gamma=1.0, epochs=30,
                   batch_size=64, lr=1e-4):
    print(f"\n{'='*60}")
    print(f" Phase 2: Joint Fine-tuning (encoder unfrozen)")
    print(f"{'='*60}")
    print(f"  LR: {lr} (lower for stable encoder)")

    train_ds = make_tf_dataset(splits["train"][0], splits["train"][1],
                               batch_size=batch_size, shuffle=True, augment=True)
    val_ds = make_tf_dataset(splits["val"][0], splits["val"][1],
                             batch_size=batch_size, shuffle=False)

    freeze_encoder(model, freeze=False)

    fl = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=0.05)
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    model.compile(optimizer=opt, loss=fl, metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ])

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=15,
                                         mode='max', restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_recall', factor=0.5,
                                             patience=8, min_lr=1e-7, mode='max', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / 'best_head_unfrozen.h5'),
            monitor='val_recall', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(MODELS_DIR / 'head_unfreeze_history.csv')),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=cbs, verbose=2)
    return model


def evaluate_final(model, splits):
    x_test, y_test = splits["test"]
    x_in = add_channel_dim(x_test)
    y_prob = model.predict(x_in, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_onehot = tf.keras.utils.to_categorical(y_test, 2)
    loss, acc, prec, rec, auc = model.evaluate(x_in, y_onehot, verbose=0)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    a_recall = tp / max(tp + fn, 1)
    a_prec = tp / max(tp + fp, 1)
    a_f1 = 2 * a_prec * a_recall / max(a_prec + a_recall, 1e-8)

    from sklearn.metrics import roc_auc_score
    prob_ab = y_prob[:, 1]
    full_auc = float(roc_auc_score(y_test, prob_ab))

    print(f"\n{'='*45}")
    print(f"  Final Test Results")
    print(f"{'='*45}")
    print(f"  Accuracy:        {acc*100:.2f}%")
    print(f"  AUC:             {full_auc:.4f}")
    print(f"  Abnormal Recall: {a_recall*100:.2f}%")
    print(f"  Abnormal Prec:   {a_prec*100:.2f}%")
    print(f"  Abnormal F1:     {a_f1:.4f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={int(((y_pred == 0) & (y_test == 0)).sum())}")

    # Optimal threshold sweep
    best_f1, best_th, best_recall = 0, 0.5, 0
    for th in np.arange(0.05, 0.90, 0.005):
        yp = (prob_ab >= th).astype(int)
        tp_t = int(((yp == 1) & (y_test == 1)).sum())
        fp_t = int(((yp == 1) & (y_test == 0)).sum())
        fn_t = int(((yp == 0) & (y_test == 1)).sum())
        prec_t = tp_t / max(tp_t + fp_t, 1)
        rec_t = tp_t / max(tp_t + fn_t, 1)
        f1_t = 2 * prec_t * rec_t / max(prec_t + rec_t, 1e-8)
        if f1_t > best_f1:
            best_f1, best_th, best_recall = f1_t, th, rec_t

    print(f"\n  Best F1 threshold: {best_th:.4f}")
    print(f"    → Recall: {best_recall*100:.2f}%  F1: {best_f1:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="SSL Encoder Classification Head Tuning")
    parser.add_argument("--unfreeze", action="store_true",
                        help="After head training, unfreeze encoder for joint tuning")
    parser.add_argument("--alpha", type=float, default=0.85,
                        help="FocalLoss alpha (abnormal class weight)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="FocalLoss gamma")
    parser.add_argument("--epochs1", type=int, default=30,
                        help="Phase 1 epochs (head only)")
    parser.add_argument("--epochs2", type=int, default=30,
                        help="Phase 2 epochs (joint fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-freeze", action="store_true",
                        help="Do NOT freeze encoder (use when adding new data)")
    parser.add_argument("--balanced", action="store_true",
                        help="Use balanced 50/50 normal/abnormal batch sampling")
    parser.add_argument("--encoder-weights", type=str,
                        default=ENCODER_WEIGHTS_DEFAULT)

    args = parser.parse_args()

    model, splits = train_head(alpha=args.alpha, gamma=args.gamma,
                               epochs=args.epochs1, batch_size=args.batch_size,
                               encoder_weights_path=args.encoder_weights,
                               freeze=not args.no_freeze,
                               balanced=args.balanced)
    evaluate_final(model, splits)

    if args.unfreeze:
        model = unfreeze_train(model, splits, alpha=args.alpha, gamma=args.gamma,
                               epochs=args.epochs2, batch_size=args.batch_size)
        evaluate_final(model, splits)

    final_path = MODELS_DIR / "final_cls_tuned.h5"
    model.save(str(final_path))
    print(f"\n[DONE] {final_path}")


if __name__ == "__main__":
    main()
