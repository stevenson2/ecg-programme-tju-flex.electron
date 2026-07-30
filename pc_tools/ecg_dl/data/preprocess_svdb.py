#!/usr/bin/env python3
"""SVDB + MIT-BIH unified preprocessing with ESP32 filters."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, TARGET_FS, BEAT_WINDOW_SAMPLES, AAMI_CLASSES
from data.preprocess import resample_ecg, apply_esp32_filters

DATA_DIR = Path(__file__).resolve().parent / "raw"

def process_svdb():
    import wfdb
    svdb_dir = DATA_DIR / "svdb"
    beats, labels, rids = [], [], []
    for f in sorted(svdb_dir.glob("*.dat")):
        rid = f.stem
        try:
            rec = wfdb.rdrecord(str(svdb_dir / rid))
            ann = wfdb.rdann(str(svdb_dir / rid), 'atr')
            fs = rec.fs
            # Use lead 0
            sig = rec.p_signal[:, 0].astype(np.float64)
            n = int(len(sig) * TARGET_FS / fs)
            sig250 = resample_ecg(sig.reshape(-1,1), fs, TARGET_FS)[:,0]
            sig_f = apply_esp32_filters(sig250, TARGET_FS)
            ratio = TARGET_FS / fs
            for idx, sym in zip(ann.sample, ann.symbol):
                sym = sym.decode() if isinstance(sym, bytes) else sym
                if sym not in AAMI_CLASSES: continue
                ri = int(idx * ratio)
                lo = ri - BEAT_WINDOW_SAMPLES//2
                hi = lo + BEAT_WINDOW_SAMPLES
                if lo >= 0 and hi <= len(sig_f):
                    beat = sig_f[lo:hi]
                    beat = (beat - beat.mean()) / (beat.std() + 1e-8)
                    beats.append(beat)
                    labels.append(AAMI_CLASSES[sym])
                    rids.append(int(rid))
        except Exception as e:
            print(f"  SKIP {rid}: {e}")
    fb = np.array(beats, dtype=np.float32)
    fl = np.array(labels, dtype=np.int32)
    fr = np.array(rids, dtype=np.int32)
    print(f"SVDB: {len(fb)} beats, N={(fl==0).sum()}, A={(fl==1).sum()}")
    return fb, fl, fr

if __name__ == "__main__":
    b_svdb, l_svdb, r_svdb = process_svdb()
    out = PROCESSED_DIR / "svdb_processed.npz"
    np.savez_compressed(out, beats=b_svdb, labels=l_svdb, record_ids=r_svdb)
    nN, nA = (l_svdb==0).sum(), (l_svdb==1).sum()
    print(f"SVDB: {len(b_svdb)} beats, N={nN}, A={nA}")
    print(f"Saved: {out}")
