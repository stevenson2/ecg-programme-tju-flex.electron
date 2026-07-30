#!/usr/bin/env python3
"""
Phase 2C: SimCLR Self-Supervised Pre-training for ECG

Stage 1: SSL pre-training on PTB-XL (no labels, NT-Xent loss)
Stage 2: Fine-tuning on MIT-BIH+INCART (FocalLoss, Encoder+Classifier)

Usage:
  python train_ssl.py                           # Both stages
  python train_ssl.py --stage1-only              # SSL only
  python train_ssl.py --stage2-only              # Fine-tune only (loads saved encoder)
  python train_ssl.py --ssl-epochs 150 --ft-epochs 60

Expected GPU time:
  Stage 1: ~3-5h (100 epochs, RTX 5070)
  Stage 2: ~20min (50 epochs)
"""

import sys, os, ast, argparse, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (TRAIN_CONFIG, MODELS_DIR, PROCESSED_DIR, TARGET_FS,
                    BEAT_WINDOW_SAMPLES, CLASS_NAMES)
from data.dataset import (load_mit_incart_merged, train_val_test_split,
                          make_tf_dataset, add_channel_dim)
from losses.contrastive import (ecg_encoder, projection_head, build_simclr_model,
                                create_contrastive_pair, ntxent_loss)
from losses.focal_loss import FocalLoss
from models.resnet_lite_1d import (build_ecg_resnet_lite_large,
                                   compile_model as compile_resnet,
                                   get_callbacks, model_summary_table)

import tensorflow as tf
tf.random.set_seed(TRAIN_CONFIG['random_seed'])
np.random.seed(TRAIN_CONFIG['random_seed'])


# ===========================================================================
# PTB-XL Raw Signal Loader
# ===========================================================================

_RAW = r"C:\Users\cai\OneDrive\Desktop\ecg-programme-tju-flex.electron-master\PTB-XL_ECG"
_WSL = "/mnt/c/Users/cai/OneDrive/Desktop/ecg-programme-tju-flex.electron-master/PTB-XL_ECG"
PTBXL_DIR = Path(_WSL if os.path.exists(_WSL) else _RAW)
PTBXL_CSV = PTBXL_DIR / "ptbxl_database.csv"

RHYTHM_NORMAL = {"NORM", "SR", "SBRAD", "STACH", "SARRH"}
RHYTHM_ABNORMAL = {
    "AFIB", "AFLT", "SVARR", "SVTAC", "PSVT",
    "PVC", "PAC", "BIGU", "TRIGU",
    "1AVB", "2AVB", "3AVB", "PACE", "WPW", "LNGQT",
}
STRUCTURAL_EXCLUDE = {
    "IMI", "ASMI", "AMI", "ALMI", "ILMI", "LMI", "IPLMI", "PMI",
    "LVH", "RVH", "LAO/LAE", "RAO/RAE", "SEHYP", "VCLVH",
    "ISC_", "ISCAL", "ISCIN", "ISCIL", "ISCAS", "ISCLA",
    "INJAS", "INJAL", "INJIN", "INJIL",
    "NST_", "DIG", "LOWT", "NT_", "INVT", "TAB_", "STE_", "STD_",
    "ABQRS", "QWAVE", "LVOLT", "HVOLT", "NDT", "ANEUR", "EL", "DTI",
    "CRBBB", "CLBBB", "IRBBB", "IVCD", "LAFB", "LPFB", "LPR",
}


def _classify_rhythm(scp_str):
    try:
        codes = set(ast.literal_eval(scp_str).keys())
    except Exception:
        return -1
    has_rhythm_ab = bool(codes & RHYTHM_ABNORMAL)
    has_structural = bool(codes & STRUCTURAL_EXCLUDE)
    if has_structural and not has_rhythm_ab:
        return -1
    if has_rhythm_ab:
        return 1
    if codes.issubset(RHYTHM_NORMAL):
        return 0
    return -1


def _esp32_filter(sig, fs):
    from scipy import signal as scipy_signal
    bh, ah = scipy_signal.butter(2, 0.5 / (0.5 * fs), btype='high')
    bl, al = scipy_signal.butter(2, 40.0 / (0.5 * fs), btype='low')
    bn, an = scipy_signal.iirnotch(50.0, 20.0, fs)
    sig = scipy_signal.filtfilt(bh, ah, sig)
    sig = scipy_signal.filtfilt(bl, al, sig)
    sig = scipy_signal.filtfilt(bn, an, sig)
    return sig.astype(np.float32)


def load_ptbxl_raw_signals(max_records=None):
    import pandas as pd, wfdb
    from scipy import signal as scipy_signal
    print(f"[SSL-Data] Index: {PTBXL_CSV}")
    db = pd.read_csv(PTBXL_CSV)
    db = db[db["validated_by_human"] == True].copy()
    db["rhythm_label"] = db["scp_codes"].apply(_classify_rhythm)
    db = db[db["rhythm_label"] >= 0].copy()
    nN, nA = (db["rhythm_label"] == 0).sum(), (db["rhythm_label"] == 1).sum()
    print(f"[SSL-Data] {len(db)} rhythm records (N={nN}, A={nA})")
    if max_records:
        db = db.head(max_records)

    signals, failed, loaded = [], 0, 0
    for _, row in db.iterrows():
        try:
            rec_path = str(PTBXL_DIR / row["filename_hr"])
            rec = wfdb.rdrecord(rec_path, channels=[1])
            sig = rec.p_signal[:, 0].astype(np.float64)
            n_tgt = int(len(sig) * TARGET_FS / rec.fs)
            sig250 = scipy_signal.resample(sig, n_tgt)
            sig_f = _esp32_filter(sig250, TARGET_FS)
            signals.append(sig_f)
            loaded += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  [SKIP] {row['ecg_id']}: {e}")
        if (loaded + failed) % 2000 == 0:
            print(f"  [{loaded}] loaded, {failed} failed")
    total_seconds = sum(len(s) for s in signals) / TARGET_FS
    print(f"[SSL-Data] {len(signals)} records ({total_seconds/3600:.1f}h total, "
          f"{total_seconds/len(signals):.1f}s avg)")
    return signals


# ===========================================================================
# SSL Dataset (random segment sampler)
# ===========================================================================

def ssl_dataset(signals, window_size=512, batch_size=128, steps_per_epoch=500,
                buffer_multiplier=4):
    lengths = [len(s) for s in signals]
    cumlen = np.cumsum(lengths)
    all_sig = np.concatenate(signals).astype(np.float32)
    all_sig = (all_sig - all_sig.mean()) / (all_sig.std() + 1e-8)
    n_recs = len(signals)
    del signals

    n_per_epoch = steps_per_epoch * batch_size
    segments = np.empty((n_per_epoch, window_size), dtype=np.float32)
    for i in range(n_per_epoch):
        rec = np.random.randint(0, n_recs)
        rec_len = lengths[rec]
        rec_start = cumlen[rec - 1] if rec > 0 else 0
        start = rec_start + np.random.randint(0, rec_len - window_size)
        segments[i] = all_sig[start:start + window_size]
    del all_sig

    return segments.reshape(steps_per_epoch, batch_size, window_size, 1)


def _refresh_dataset(segments_pool, n_per_epoch, lengths, cumlen, all_sig, n_recs):
    """Re-sample segments from raw signal for a new epoch."""
    for i in range(n_per_epoch):
        rec = np.random.randint(0, n_recs)
        rec_len = lengths[rec]
        rec_start = cumlen[rec - 1] if rec > 0 else 0
        start = rec_start + np.random.randint(0, rec_len - window_size)
        segments_pool[i] = all_sig[start:start + window_size]


# ===========================================================================
# Stage 1: SimCLR SSL Pre-training
# ===========================================================================

def train_stage1_ssl(ssl_epochs=100, ssl_window=512, batch_size=512,
                     steps_per_epoch=125, temperature=0.1,
                     learning_rate=1e-2, max_records=None):
    print(f"\n{'='*60}")
    print(f" STAGE 1: SimCLR SSL Pre-training ({ssl_epochs} epochs)")
    print(f"{'='*60}")
    print(f"  Window: {ssl_window} samples ({ssl_window/TARGET_FS:.1f}s)")
    print(f"  Batch:  {batch_size}")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Total samples/epoch: {batch_size * steps_per_epoch}")
    print(f"  Temperature: {temperature}")
    print(f"  LR: {learning_rate}")

    signals = load_ptbxl_raw_signals(max_records=max_records)
    if not signals:
        raise RuntimeError("No PTB-XL records loaded!")
    print(f"[SSL] Pre-sampling epoch data...")
    batch_data = ssl_dataset(signals, window_size=ssl_window,
                             batch_size=batch_size,
                             steps_per_epoch=steps_per_epoch)
    n_batches = batch_data.shape[0]
    print(f"[SSL] Data ready: {batch_data.shape} ({batch_data.nbytes/1024/1024:.0f} MB)")

    print(f"[SSL] Building SimCLR model...")
    encoder, simclr_model = build_simclr_model(
        input_shape=(ssl_window, 1), temperature=temperature)
    model_summary_table(simclr_model)

    total_steps = ssl_epochs * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=max(1, total_steps // 4),
        t_mul=2.0, m_mul=0.5, alpha=0.0)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=0.9)

    @tf.function
    def train_step(x_batch):
        with tf.GradientTape() as tape:
            v1, v2 = create_contrastive_pair(x_batch)
            combined = tf.concat([v1, v2], axis=0)
            z = simclr_model(combined, training=True)
            z1, z2 = tf.split(z, 2, axis=0)
            loss = ntxent_loss(z1, z2, temperature)
        grads = tape.gradient(loss, simclr_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, simclr_model.trainable_weights))
        return loss

    ssl_patience = TRAIN_CONFIG.get('ssl_patience', 25)
    best_loss = float('inf')
    best_weights_path = MODELS_DIR / "ssl_encoder_best.weights.h5"
    wait = 0
    stopped_epoch = ssl_epochs

    csv_path = MODELS_DIR / "ssl_stage1_history.csv"
    csv_file = open(str(csv_path), 'w')
    csv_file.write("epoch,loss,lr\n")
    loss_history = []

    print(f"\n[SSL] Training ({ssl_epochs} epochs × {steps_per_epoch} steps)")
    print(f"[SSL] Early stopping patience: {ssl_patience} epochs")
    print(f"[SSL] Log: {csv_path}")
    print(f"      {'='*45}")

    import time as _time
    for epoch in range(ssl_epochs):
        t0 = _time.time()
        np.random.shuffle(batch_data)
        total_loss, n_steps = 0.0, 0
        for i in range(n_batches):
            loss = train_step(batch_data[i])
            total_loss += float(loss)
            n_steps += 1
        elapsed = _time.time() - t0
        avg_loss = total_loss / max(n_steps, 1)
        lr_val = float(optimizer.learning_rate.numpy())
        loss_history.append(avg_loss)
        csv_file.write(f"{epoch+1},{avg_loss:.6f},{lr_val:.8f}\n")
        csv_file.flush()

        improved = avg_loss < best_loss
        marker = " *" if improved else f"  ({wait+1}/{ssl_patience})"
        print(f"  Epoch {epoch+1:3d}/{ssl_epochs}  "
              f"Loss: {avg_loss:.4f}  LR: {lr_val:.2e}  "
              f"{elapsed:.1f}s{marker}")

        if improved:
            best_loss = avg_loss
            wait = 0
            encoder.save_weights(str(best_weights_path))
        else:
            wait += 1

        if (epoch + 1) % 20 == 0:
            print(f"    └─ Checkpoint: epoch {epoch+1}")

        if wait >= ssl_patience:
            print(f"\n[SSL] Early stopping at epoch {epoch+1} "
                  f"(no improvement for {ssl_patience} epochs)")
            stopped_epoch = epoch + 1
            break

    csv_file.close()

    if wait > 0 and not improved:
        print(f"[SSL] Restoring best weights (epoch {stopped_epoch - wait}, "
              f"Loss: {best_loss:.4f})")
        encoder.load_weights(str(best_weights_path))
    else:
        print(f"[SSL] Best Loss: {best_loss:.4f} at epoch {stopped_epoch - wait}")

    _plot_ssl_history(csv_path, loss_history, stopped_epoch, best_loss)

    final_weights = MODELS_DIR / "ssl_encoder.weights.h5"
    encoder.save_weights(str(final_weights))
    encoder.save(str(MODELS_DIR / "ssl_encoder.h5"))
    print(f"\n[SSL] Complete in {stopped_epoch} epochs. Encoder: {final_weights}")
    return encoder


def _plot_ssl_history(csv_path, loss_history, stopped_epoch, best_loss):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(loss_history) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, loss_history, color="#2E86AB", linewidth=1.5)
        ax1.axhline(y=best_loss, color="#C73E1D", linestyle="--", alpha=0.6,
                    label=f"Best: {best_loss:.4f}")
        ax1.scatter(stopped_epoch - 1, loss_history[stopped_epoch - 1],
                    color="#C73E1D", s=80, zorder=5)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("NT-Xent Loss")
        ax1.set_title(f"SimCLR Stage 1 — SSL Pre-training ({stopped_epoch} epochs)")
        ax1.legend(); ax1.grid(alpha=0.15, linestyle="--")

        ax2.semilogy(epochs, loss_history, color="#2E86AB", linewidth=1.5)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (log scale)")
        ax2.set_title("Loss (Log Scale)")
        ax2.grid(alpha=0.15, linestyle="--")

        plt.tight_layout()
        plot_path = csv_path.with_suffix(".png")
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"[SSL] Plot saved: {plot_path}")
    except Exception as e:
        print(f"[SSL] Plot warning: {e}")


# ===========================================================================
# Stage 2: Fine-tuning on MIT-BIH+INCART
# ===========================================================================

def train_stage2_finetune(encoder_weights=None, ft_epochs=50,
                          batch_size=64, learning_rate=1e-4):
    print(f"\n{'='*60}")
    print(f" STAGE 2: Fine-tuning MIT-BIH+INCART ({ft_epochs} epochs)")
    print(f"{'='*60}")

    data = load_mit_incart_merged()
    splits = train_val_test_split(data["beats"], data["labels"],
                                  record_ids=data.get("record_ids"))
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    print(f"[Fine-tune] Train: {len(x_train)}, Val: {len(x_val)}, "
          f"Abnormal: {y_train.mean():.1%}")

    train_ds = make_tf_dataset(x_train, y_train, batch_size=batch_size,
                               shuffle=True, augment=True)
    val_ds = make_tf_dataset(x_val, y_val, batch_size=batch_size,
                             shuffle=False)

    model = build_ecg_resnet_lite_large(input_shape=(BEAT_WINDOW_SAMPLES, 1))
    model.build((None, BEAT_WINDOW_SAMPLES, 1))
    model_summary_table(model)

    if encoder_weights:
        wpath = Path(encoder_weights)
        if wpath.exists():
            print(f"[Fine-tune] Loading SSL encoder: {wpath}")
            ssl_enc = ecg_encoder(input_shape=(BEAT_WINDOW_SAMPLES, 1))
            ssl_enc.build((None, BEAT_WINDOW_SAMPLES, 1))
            ssl_enc.load_weights(str(wpath))
            transferred = 0
            ssl_names = {l.name: l for l in ssl_enc.layers}
            for layer in model.layers:
                if layer.name in ssl_names:
                    try:
                        layer.set_weights(ssl_names[layer.name].get_weights())
                        transferred += 1
                    except Exception:
                        pass
            print(f"[Fine-tune] Transferred {transferred} layers from SSL encoder")
            if transferred == 0:
                print("[Fine-tune] WARNING: NO layers transferred! Training from scratch.")
        else:
            print(f"[Fine-tune] WARNING: {wpath} not found, training from scratch")

    fl = FocalLoss(gamma=TRAIN_CONFIG['focal_loss']['gamma'],
                   alpha=TRAIN_CONFIG['focal_loss']['alpha'],
                   label_smoothing=TRAIN_CONFIG['focal_loss']['label_smoothing'])
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    model.compile(optimizer=opt, loss=fl, metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=20,
                                         mode='max', restore_best_weights=True,
                                         verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                             patience=10, min_lr=1e-6,
                                             mode='max', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / 'best_ssl_ft.h5'),
            monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(MODELS_DIR / 'ssl_ft_history.csv')),
    ]

    print("\n[Fine-tune] Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs,
                        callbacks=callbacks, verbose=2)

    x_test, y_test = splits["test"]
    x_in = add_channel_dim(x_test)
    y_prob = model.predict(x_in, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_onehot = tf.keras.utils.to_categorical(y_test, 2)
    loss, acc, prec, rec, auc = model.evaluate(x_in, y_onehot, verbose=0)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    a_recall = tp / max(tp + fn, 1)
    a_prec = tp / max(tp + fp, 1)
    a_f1 = 2 * a_prec * a_recall / max(a_prec + a_recall, 1e-8)

    print(f"\n{'='*45}")
    print(f"  Stage 2 Test Results")
    print(f"{'='*45}")
    print(f"  Accuracy:         {acc*100:.2f}%")
    print(f"  AUC:              {auc:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  Abnormal Recall:  {a_recall*100:.2f}%")
    print(f"  Abnormal Prec:    {a_prec*100:.2f}%")
    print(f"  Abnormal F1:      {a_f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*45}")

    final_path = MODELS_DIR / "final_ssl_finetuned.h5"
    model.save(str(final_path))
    print(f"\n[DONE] {final_path}")
    return model


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2C: SimCLR SSL Pre-training + Fine-tuning")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--ft-epochs", type=int, default=50)
    parser.add_argument("--ssl-window", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="SSL batch size")
    parser.add_argument("--ft-batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ssl-lr", type=float, default=1e-2,
                        help="SSL learning rate (SGD needs ~0.01)")
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-weights", type=str,
                        default=str(MODELS_DIR / "ssl_encoder.weights.h5"))
    parser.add_argument("--max-records", type=int, default=None,
                        help="Limit PTB-XL records (for testing)")
    parser.add_argument("--steps-per-epoch", type=int, default=125,
                        help="SSL training steps per epoch")

    args = parser.parse_args()
    run_s1 = not args.stage2_only
    run_s2 = not args.stage1_only

    if run_s1:
        train_stage1_ssl(
            ssl_epochs=args.ssl_epochs,
            ssl_window=args.ssl_window,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            temperature=args.temperature,
            learning_rate=args.ssl_lr,
            max_records=args.max_records)

    if run_s2:
        train_stage2_finetune(
            encoder_weights=args.encoder_weights,
            ft_epochs=args.ft_epochs,
            batch_size=args.ft_batch_size,
            learning_rate=args.ft_lr)

    if run_s1 and run_s2:
        print(f"\n{'='*60}")
        print(f" Phase 2C Complete!")
        print(f" SSL Encoder:     {MODELS_DIR / 'ssl_encoder.weights.h5'}")
        print(f" Fine-tuned:      {MODELS_DIR / 'final_ssl_finetuned.h5'}")
        print(f"{'='*60}")

    print("\n[DONE]")
