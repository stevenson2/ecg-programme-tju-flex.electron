#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route F: Multi-task Learning Training Script.

Shared ResNet encoder + 3 task heads:
  1. Classification: Normal/Abnormal (FocalLoss)
  2. BPM regression:  Heart rate (MSE)
  3. SQI regression:  Signal quality (MSE)

Usage:
  python train_multitask.py --incart --epochs 200
  python train_multitask.py --incart --epochs 200 --w-cls 1.0 --w-bpm 0.3 --w-sqi 0.2
"""

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (TRAIN_CONFIG, MULTITASK_CONFIG, TTA_CONFIG,
                     MODELS_DIR, CLASS_NAMES, INFERENCE_CONFIG)

tf.random.set_seed(TRAIN_CONFIG['random_seed'])
np.random.seed(TRAIN_CONFIG['random_seed'])


def prepare_multitask_data(
    use_incart=True, use_ptbxl_rhythm=False, use_ecg1000=False,
    val_split=None, test_split=None
):
    from data.dataset import (
        load_processed_data, load_incart_data, load_ptbxl_rhythm_data,
        load_ecg1000_data, train_val_test_split,
        load_mit_incart_merged, load_all_three_merged,
        load_mit_ecg1000_merged, make_multitask_dataset,
    )
    from utils.multitask_labels import compute_multitask_labels

    if use_ptbxl_rhythm:
        data = load_all_three_merged()
    elif use_ecg1000:
        data = load_mit_ecg1000_merged()
    elif use_incart:
        data = load_mit_incart_merged()
    else:
        data = load_processed_data()

    beats = data["beats"]
    labels = data["labels"]
    record_ids = data.get("record_ids")

    print("[多任务] Computing pseudo-labels (BPM + SQI)...")
    bpm_labels, sqi_labels = compute_multitask_labels(beats, fs=250)
    print(f"  BPM range: [{bpm_labels.min():.1f}, {bpm_labels.max():.1f}]")
    print(f"  SQI range: [{sqi_labels.min():.3f}, {sqi_labels.max():.3f}]")

    splits = train_val_test_split(
        beats, labels, record_ids=record_ids,
        val_split=val_split, test_split=test_split)
    bpm_splits = train_val_test_split(
        bpm_labels, labels, record_ids=record_ids,
        val_split=val_split, test_split=test_split)
    sqi_splits = train_val_test_split(
        sqi_labels, labels, record_ids=record_ids,
        val_split=val_split, test_split=test_split)

    # Extract test record_ids for per-record multi-beat confirmation
    test_record_ids = None
    if record_ids is not None:
        # Use the same split logic: identify test records
        np.random.seed(TRAIN_CONFIG['random_seed'])
        unique_records = np.unique(record_ids)
        np.random.shuffle(unique_records)
        n_total = len(unique_records)
        n_test = max(1, int(n_total * (test_split or TRAIN_CONFIG['test_split'])))
        test_recs = set(unique_records[:n_test])
        test_mask = np.array([rid in test_recs for rid in record_ids])
        test_record_ids = record_ids[test_mask]

    train_ds = make_multitask_dataset(
        splits["train"][0], splits["train"][1],
        bpm_splits["train"][0], sqi_splits["train"][0],
        shuffle=True, augment=True)
    val_ds = make_multitask_dataset(
        splits["val"][0], splits["val"][1],
        bpm_splits["val"][0], sqi_splits["val"][0],
        shuffle=False)
    test_ds = make_multitask_dataset(
        splits["test"][0], splits["test"][1],
        bpm_splits["test"][0], sqi_splits["test"][0],
        shuffle=False)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "data": splits,
        "test_labels": splits["test"][1],
        "test_beats": splits["test"][0],
        "test_record_ids": test_record_ids,
        "class_names": CLASS_NAMES,
        "input_shape": (INFERENCE_CONFIG['window_size'], 1),
    }


def evaluate_multitask(model, datasets):
    print(f"\n{'='*60}")
    print("  Multi-Task Model Evaluation")
    print(f"{'='*60}")

    x_test, y_test = datasets["test_beats"], datasets["test_labels"]
    x_test_input = x_test[..., np.newaxis]

    cls_pred, bpm_preds, sqi_preds = model.predict(x_test_input, verbose=0)
    if isinstance(cls_pred, list):
        cls_probs = cls_pred[0] if isinstance(cls_pred, list) else cls_pred
    else:
        cls_probs = cls_pred
    bpm_preds = np.asarray(bpm_preds).flatten()
    sqi_preds = np.asarray(sqi_preds).flatten()

    cls_pred_labels = np.argmax(cls_probs, axis=1)
    abnormal_probs = cls_probs[:, 1]

    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, roc_auc_score, f1_score)

    print(f"\n[Classification Head]")
    for th in [0.50, 0.35]:
        preds = (abnormal_probs > th).astype(int)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, abnormal_probs)
        print(f"  Threshold={th:.2f}: Acc={acc:.4f} Prec={prec:.4f} "
              f"Recall={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

    print(f"\n[BPM Regression Head]")
    from utils.multitask_labels import compute_multitask_labels, bpm_denormalize
    bpm_true, sqi_true = compute_multitask_labels(x_test, fs=250, normalize_bpm=True)
    bpm_preds_denorm = bpm_denormalize(bpm_preds)
    bpm_true_denorm = bpm_denormalize(bpm_true)
    bpm_mae = np.mean(np.abs(bpm_preds_denorm - bpm_true_denorm))
    bpm_rmse = np.sqrt(np.mean((bpm_preds_denorm - bpm_true_denorm) ** 2))
    print(f"  MAE={bpm_mae:.2f} BPM, RMSE={bpm_rmse:.2f} BPM")

    print(f"\n[SQI Regression Head]")
    sqi_mae = np.mean(np.abs(sqi_preds - sqi_true))
    print(f"  MAE={sqi_mae:.4f}")

    return cls_pred, abnormal_probs


def evaluate_tta(model, datasets):
    print(f"\n{'='*60}")
    print("  TTA Evaluation (Route G)")
    print(f"{'='*60}")

    x_test, y_test = datasets["test_beats"], datasets["test_labels"]

    # Only evaluate classification head for TTA
    class ModelWrapper:
        def __init__(self, mt_model):
            self.mt_model = mt_model
        def predict(self, x, **kwargs):
            cls_out, _, _ = self.mt_model.predict(x, **kwargs)
            return cls_out

    wrapped = ModelWrapper(model)

    from inference.tta import tta_evaluate

    # Get record_ids from test split for per-record multi-beat confirmation
    test_rids = datasets.get("test_record_ids", None)

    results = tta_evaluate(
        wrapped, x_test, y_test,
        threshold=TTA_CONFIG['threshold'],
        use_sliding=TTA_CONFIG['sliding_window']['enabled'],
        use_augmented=TTA_CONFIG['augmentation']['enabled'],
        n_confirm=TTA_CONFIG['multi_beat_confirm']['n_confirm'],
        stride_samples=TTA_CONFIG['sliding_window']['stride_samples'],
        n_views=TTA_CONFIG['sliding_window']['n_views'],
        n_aug=TTA_CONFIG['augmentation']['n_aug'],
        record_ids=test_rids,
    )

    for mode, metrics in results.items():
        print(f"\n  [{mode.upper()}]")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    return results


def train(
    use_incart=True,
    use_ptbxl_rhythm=False,
    use_ecg1000=False,
    epochs=None,
    batch_size=None,
    w_cls=1.0,
    w_bpm=0.3,
    w_sqi=0.2,
    focal_gamma=1.0,
    focal_alpha=0.75,
    learning_rate=0.0005,
    skip_eval=False,
    skip_tta=False,
    pretrained=None,
    freeze_encoder_ratio=0.5,
    suffix="",
):
    tag = f"_{suffix}" if suffix else ""
    print(f"\n{'='*60}")
    ds_tag = ""
    if use_ptbxl_rhythm:
        ds_tag = "MIT+INCART+PTBXL"
    elif use_incart:
        ds_tag = "MIT-BIH+INCART"
    elif use_ecg1000:
        ds_tag = "MIT-BIH+ECG1000"
    else:
        ds_tag = "MIT-BIH"
    print(f"  Route F: Multi-Task Learning [{ds_tag}]")
    print(f"  Loss weights: cls={w_cls}, bpm={w_bpm}, sqi={w_sqi}")
    print(f"{'='*60}\n")

    # Step 1: Data
    print("[1/4] Preparing multi-task data...")
    datasets = prepare_multitask_data(
        use_incart=use_incart,
        use_ptbxl_rhythm=use_ptbxl_rhythm,
        use_ecg1000=use_ecg1000,
    )

    # Step 2: Model
    print("\n[2/4] Building multi-task ResNet...")
    from models.resnet_multitask_1d import (
        build_ecg_resnet_multitask, compile_multitask_model,
        model_summary_table, get_multitask_callbacks,
    )
    model = build_ecg_resnet_multitask(
        input_shape=datasets["input_shape"],
        n_classes=len(CLASS_NAMES),
    )
    model = compile_multitask_model(
        model,
        learning_rate=learning_rate,
        gamma=focal_gamma,
        alpha=focal_alpha,
        w_cls=w_cls,
        w_bpm=w_bpm,
        w_sqi=w_sqi,
    )
    model_summary_table(model)

    # Load pretrained encoder weights (Route K Stage 2)
    if pretrained:
        print(f"\n[*] Loading pretrained weights: {pretrained}")
        pretrained_model = tf.keras.models.load_model(
            pretrained, compile=False)
        # Transfer encoder layer weights by name (stem + res_blocks + pooling)
        transferred = 0
        for layer in model.layers:
            if layer.name in ['ecg_input', 'cls_out', 'bpm_out', 'sqi_out',
                              'cls_fc1', 'cls_fc2', 'cls_bn', 'cls_do', 'cls_do2',
                              'bpm_fc1', 'bpm_fc2', 'sqi_fc1', 'sqi_fc2']:
                continue
            try:
                src_layer = pretrained_model.get_layer(layer.name)
                layer.set_weights(src_layer.get_weights())
                transferred += 1
            except Exception:
                pass
        print(f"  Transferred {transferred} layers from pretrained model")

        # Freeze early layers
        if freeze_encoder_ratio > 0:
            encoder_layers = [l for l in model.layers
                            if 'cls_' not in l.name
                            and 'bpm_' not in l.name
                            and 'sqi_' not in l.name
                            and l.name != 'ecg_input']
            n_freeze = int(len(encoder_layers) * freeze_encoder_ratio)
            for layer in encoder_layers[:n_freeze]:
                layer.trainable = False
            print(f"  Frozen {n_freeze}/{len(encoder_layers)} encoder layers")

    # Step 3: Train
    print("\n[3/4] Training...")
    callbacks = get_multitask_callbacks(
        model_name=f"best_resnet_multitask{tag}.h5",
        csv_name=f"multitask_history{tag}.csv")

    history = model.fit(
        datasets["train_ds"],
        validation_data=datasets["val_ds"],
        epochs=epochs or TRAIN_CONFIG['epochs'],
        callbacks=callbacks,
        verbose=2,
    )

    # Save final model
    model.save(str(MODELS_DIR / f"final_resnet_multitask{tag}.h5"))
    print(f"[Training] Model saved: {MODELS_DIR / f'final_resnet_multitask{tag}.h5'}")

    # Step 4: Evaluate
    if not skip_eval:
        print("\n[4/4] Evaluation...")
        evaluate_multitask(model, datasets)

        if not skip_tta:
            evaluate_tta(model, datasets)

    print(f"\n{'='*60}")
    print("  Route F+G complete!")
    print(f"{'='*60}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Route F+G: Multi-Task Learning + TTA")
    parser.add_argument("--incart", action="store_true",
                        help="MIT-BIH + INCART (default)")
    parser.add_argument("--ptbxl-r", action="store_true",
                        help="MIT+INCART+PTBXL rhythm")
    parser.add_argument("--ecg1000", action="store_true",
                        help="MIT-BIH + ECG1000")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--w-cls", type=float, default=1.0,
                        help="Classification loss weight")
    parser.add_argument("--w-bpm", type=float, default=0.3,
                        help="BPM regression loss weight")
    parser.add_argument("--w-sqi", type=float, default=0.2,
                        help="SQI regression loss weight")
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-tta", action="store_true")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained .h5 model (Route K Stage 2)")
    parser.add_argument("--freeze", type=float, default=0.5,
                        help="Fraction of encoder layers to freeze (0-1)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for output files (parallel runs)")

    args = parser.parse_args()

    train(
        use_incart=args.incart or not (args.ptbxl_r or args.ecg1000),
        use_ptbxl_rhythm=args.ptbxl_r,
        use_ecg1000=args.ecg1000,
        epochs=args.epochs,
        batch_size=args.batch_size,
        w_cls=args.w_cls,
        w_bpm=args.w_bpm,
        w_sqi=args.w_sqi,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        learning_rate=args.lr,
        skip_eval=args.skip_eval,
        skip_tta=args.skip_tta,
        pretrained=args.pretrained,
        freeze_encoder_ratio=args.freeze,
        suffix=args.suffix,
    )
