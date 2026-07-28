#!/usr/bin/env python3
"""
Two-stage training: PTB-XL pretrain -> MIT-BIH+INCART finetune

Stage 1: PTB-XL rhythm subset (160K beats) — learn general ECG features
Stage 2: MIT-BIH+INCART (263K beats) — fine-tune decision boundary (lr×0.1)
"""
import sys, os, numpy as np, tensorflow as tf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_CONFIG, MODELS_DIR, CLASS_NAMES
from data.dataset import (
    load_mit_incart_merged, train_val_test_split,
    make_tf_dataset, add_channel_dim,
)
from models.cnn_1d import (
    build_ecg_cnn_1d_v2, compile_model, get_callbacks, model_summary_table,
)
tf.random.set_seed(TRAIN_CONFIG['random_seed']); np.random.seed(TRAIN_CONFIG['random_seed'])
PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"



def load_ptbxl_rhythm():
    npz = PROCESSED_DIR / "ptbxl_rhythm_processed.npz"
    if not npz.exists():
        raise FileNotFoundError("Run preprocess_ptbxl_rhythm.py first")
    d = np.load(npz)
    nN, nA = (d['labels']==0).sum(), (d['labels']==1).sum()
    print(f"[PTB-XL] {len(d['beats'])} beats (N={nN}, A={nA})")
    return {"beats": d["beats"], "labels": d["labels"], "record_ids": d.get("record_ids")}

def pretrain_ptbxl(epochs=30):
    """Stage 1: Pre-train on PTB-XL rhythm data."""
    print(f"\n{'='*50}\n STAGE 1: PTB-XL Pre-training\n{'='*50}")
    data = load_ptbxl_rhythm()
    s = train_val_test_split(data["beats"], data["labels"], record_ids=data.get("record_ids"))
    bs = TRAIN_CONFIG['batch_size']
    train_ds = make_tf_dataset(s["train"][0], s["train"][1], batch_size=bs, shuffle=True)
    val_ds = make_tf_dataset(s["val"][0], s["val"][1], batch_size=bs, shuffle=False)
    model = build_ecg_cnn_1d_v2(input_shape=(250,1), n_classes=2)
    model = compile_model(model, learning_rate=TRAIN_CONFIG['learning_rate'])
    model_summary_table(model)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=get_callbacks(), verbose=2)
    x_test, y_test = s["test"]
    loss, acc, prec, rec, auc = model.evaluate(
        add_channel_dim(x_test), tf.keras.utils.to_categorical(y_test,2), verbose=0)
    print(f"\n[Stage 1 Test] Acc={acc:.4f} AUC={auc:.4f} Rec={rec:.4f}")
    pt = MODELS_DIR / "pretrained_ptbxl.weights.h5"
    model.save_weights(str(pt))
    print(f"[Stage 1] Saved: {pt}")
    return model


def finetune_incart(epochs=50):
    """Stage 2: Fine-tune on MIT-BIH+INCART."""
    print(f"\n{'='*50}\n STAGE 2: MIT-BIH+INCART Fine-tuning\n{'='*50}")
    data = load_mit_incart_merged()
    s = train_val_test_split(data["beats"], data["labels"], record_ids=data.get("record_ids"))
    bs = TRAIN_CONFIG['batch_size']
    train_ds = make_tf_dataset(s["train"][0], s["train"][1], batch_size=bs, shuffle=True)
    val_ds = make_tf_dataset(s["val"][0], s["val"][1], batch_size=bs, shuffle=False)
    model = build_ecg_cnn_1d_v2(input_shape=(250,1), n_classes=2)
    pt = MODELS_DIR / "pretrained_ptbxl.weights.h5"
    if pt.exists():
        model.build((None, 250, 1)); model.load_weights(str(pt))
        print(f"[Stage 2] Loaded pre-trained weights")
    else:
        print("[Stage 2] Training from scratch (no Stage 1 weights)")
    ft_lr = TRAIN_CONFIG['learning_rate'] / 10
    model = compile_model(model, learning_rate=ft_lr)
    cb = get_callbacks(); cb[0].patience = 30
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb, verbose=2)
    x_test, y_test = s["test"]
    x_in = add_channel_dim(x_test)
    y_prob = model.predict(x_in, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    loss, acc, prec, rec, auc = model.evaluate(
        x_in, tf.keras.utils.to_categorical(y_test,2), verbose=0)
    print(f"\n{'='*40}\n  Stage 2 Results\n{'='*40}")
    print(f"  Acc: {acc:.4f} ({acc*100:.2f}%)  AUC: {auc:.4f}")
    print(f"  Prec: {prec:.4f}  Recall: {rec:.4f}")
    tp = ((y_pred==1)&(y_test==1)).sum(); fp = ((y_pred==1)&(y_test==0)).sum()
    fn = ((y_pred==0)&(y_test==1)).sum()
    print(f"  A.Recall: {tp/max(tp+fn,1):.4f}  A.Prec: {tp/max(tp+fp,1):.4f}")
    model.save(str(MODELS_DIR / "final_model.h5"))
    print(f"\n[DONE] Model: {MODELS_DIR}/final_model.h5")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-only", action="store_true")
    p.add_argument("--stage2-only", action="store_true")
    p.add_argument("--epochs1", type=int, default=30)
    p.add_argument("--epochs2", type=int, default=50)
    args = p.parse_args()
    if not args.stage2_only:
        pretrain_ptbxl(epochs=args.epochs1)
    if not args.stage1_only:
        finetune_incart(epochs=args.epochs2)
    print("\n[DONE]")


